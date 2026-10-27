"""Limpieza y transformación del ``listings.csv.gz`` de Inside Airbnb.

Orden documentado (N10): perfilar (en ``calidad``) → convertir tipos → derivar
(baños, amenities, longitudes de texto, moneda) → eliminar columnas → duplicados →
flags de precio, outliers, estancias, coordenadas → imputación documentada.

Ninguna función muta su argumento: todas devuelven objetos nuevos. Cada regla
registra cuántas filas afecta en el ``registro`` que acaba en ``calidad_datos.json``.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.config import Ciudad
from src.errores import ErrorDatos
from src.geo import dentro_bbox

logger = logging.getLogger(__name__)

COLUMNAS_BOOLEANAS: tuple[str, ...] = (
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "has_availability", "instant_bookable",
)
COLUMNAS_FECHA: tuple[str, ...] = (
    "host_since", "first_review", "last_review", "last_scraped", "calendar_last_scraped",
    "price_quote_checkin_date", "price_quote_checkout_date",
)
COLUMNAS_PORCENTAJE: tuple[str, ...] = ("host_response_rate", "host_acceptance_rate")

ELIMINAR_URLS: tuple[str, ...] = (
    "listing_url", "picture_url", "host_url", "host_profile_url", "host_picture_url", "host_thumbnail_url",
)
ELIMINAR_IDS_TECNICOS: tuple[str, ...] = ("scrape_id", "source", "host_profile_id")
# Minimización de datos personales: ni el nombre ni la ubicación declarada del anfitrión se usan
# en el análisis ni en el dashboard. host_id se conserva porque es imprescindible para agrupar
# anuncios por anfitrión (errores agrupados, validación por grupos, composición de la oferta).
ELIMINAR_PERSONALES: tuple[str, ...] = ("host_name", "host_location")
ELIMINAR_TEXTO_LARGO: tuple[str, ...] = ("description", "host_about")
ELIMINAR_CRUDOS: tuple[str, ...] = ("price_quote_raw", "amenities", "bathrooms_text")

# Política de nulos declarativa: (patrón regex sobre el nombre, decisión, justificación).
# La última regla es el comodín (N15): cualquier otra columna conserva NaN.
POLITICA_NULOS: tuple[tuple[str, str, str], ...] = (
    (r"^price$", "conservar NaN; precio_valido=False",
     "Es la variable objetivo: imputarla fabricaría el resultado. Los nulos se concentran en anuncios "
     "sin fechas reservables (sin cotización), así que no son aleatorios."),
    (r"^(review_scores_.*|first_review|last_review|reviews_per_month)$", "conservar NaN; flag tiene_resenas",
     "Ausencia de reseñas no equivale a puntuación 0."),
    (r"^(minimum_nights|maximum_nights|minimum_minimum_nights|maximum_minimum_nights|"
     r"minimum_maximum_nights|maximum_maximum_nights)$", "conservar NaN",
     "Faltan en anuncios sin calendario; imputar mezclaría anuncios inactivos con activos."),
    (r"^(bedrooms|beds)$", "imputar mediana por (room_type, accommodates); flag *_imputado",
     "Pocos nulos, muy determinados por la capacidad y usados como covariables; el flag permite auditar."),
    (r"^bathrooms$", "conservar NaN", "Texto de baños no parseable en pocos casos; OLS usa casos completos."),
    (r"^license$", "conservar NaN; license se resume en licencia_tipo",
     "La ausencia es informativa (p. ej. «Sin licencia»)."),
    (r"^(price_quote_.*|estimated_revenue_l365d)$", "conservar NaN",
     "Derivan de la cotización de precio: faltan por el mismo motivo que price."),
    (r".*", "conservar NaN", "Regla por defecto: no se imputa nada sin justificación."),
)


def regla_nulos(columna: str) -> tuple[str, str, str]:
    """Devuelve la regla de ``POLITICA_NULOS`` que aplica a ``columna`` (siempre hay una)."""
    for patron, decision, justificacion in POLITICA_NULOS:
        if re.match(patron, columna):
            return patron, decision, justificacion
    raise AssertionError("POLITICA_NULOS debe terminar con un comodín")  # pragma: no cover


# --------------------------------------------------------------------------
# Conversores (puros y vectorizados)
# --------------------------------------------------------------------------
# Formato de precio de Inside Airbnb: símbolo o código de moneda opcional, signo opcional (antes o
# después del símbolo), coma como separador de miles (grupos de tres) y punto decimal: «$1,234.00».
# Política explícita: cualquier otra cosa (coma decimal «25,00», texto, varios puntos) NO se adivina
# y queda como no convertible (NaN), contabilizada en calidad_datos.json.
_PATRON_PRECIO = re.compile(
    r"^(?P<s1>-)?\s*(?:[$€]|[A-Z]{3})?\s*(?P<s2>-)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)$")


def parse_precio(s: pd.Series) -> pd.Series:
    """Convierte textos tipo ``'$1,234.00'`` a ``float64`` conservando el signo.

    Valida el formato completo (``_PATRON_PRECIO``): ``'$-10.00'`` → −10,0; ``'1,234.50'`` → 1.234,5;
    ``'EUR -25,00'`` (coma decimal), ``'abc123xyz'`` o ``'$1.2.3'`` → NaN (no convertible).
    """
    partes = s.astype("string").str.strip().str.extract(_PATRON_PRECIO)
    numero = pd.to_numeric(partes["num"].str.replace(",", "", regex=False), errors="coerce").astype("float64")
    negativo = partes["s1"].notna() | partes["s2"].notna()
    return numero.where(~negativo, -numero).astype("float64")


def parse_porcentaje(s: pd.Series) -> pd.Series:
    """Convierte ``'95%'`` a ``95.0`` (escala 0-100); ``'N/A'`` o fuera de [0, 100] → NaN."""
    texto = s.astype("string").str.replace("%", "", regex=False).str.strip()
    valores = pd.to_numeric(texto, errors="coerce").astype("float64")
    return valores.where(valores.between(0, 100))


def a_booleano(s: pd.Series) -> pd.Series:
    """Convierte ``'t'/'f'`` a ``bool`` (sin nulos) o ``boolean`` nullable (con nulos)."""
    mapa = {"t": True, "f": False, True: True, False: False}
    res = s.map(mapa).astype("boolean")
    if res.isna().any():
        return res
    return res.astype(bool)


def a_fecha(s: pd.Series) -> pd.Series:
    """Convierte fechas ISO ``YYYY-MM-DD`` a ``datetime64``; no parseables → NaT."""
    return pd.to_datetime(s.astype("string"), errors="coerce", format="%Y-%m-%d")


def parse_banios(s: pd.Series) -> pd.DataFrame:
    """Extrae nº de baños y si es compartido desde ``bathrooms_text``.

    ``'1.5 baths'`` → 1,5; ``'Half-bath'`` → 0,5; ``'Shared half-bath'`` → 0,5 compartido;
    ``'1 shared bath'`` → 1 compartido; NaN → NaN.
    """
    texto = s.astype("string").str.lower()
    numero = pd.to_numeric(texto.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    es_medio = texto.str.contains("half", na=False)
    numero = numero.where(~(numero.isna() & es_medio), 0.5).astype("float64")
    compartido = texto.str.contains("shared", na=False).astype(bool)
    return pd.DataFrame({"bathrooms": numero, "bano_compartido": compartido}, index=s.index)


def parse_amenities(valor: Any) -> list[str]:
    """Parsea una lista JSON de amenities en texto. Nunca lanza excepción.

    Plan B ante JSON malformado: extraer las cadenas entre comillas; si no hay, lista vacía.
    """
    if not isinstance(valor, str) or not valor.strip():
        return []
    try:
        datos = json.loads(valor)
        if isinstance(datos, list):
            return [str(x) for x in datos]
    except (json.JSONDecodeError, ValueError):
        pass
    return re.findall(r'"([^"]+)"', valor)


def amenities_validas(s: pd.Series) -> pd.Series:
    """True si el texto de amenities es una lista JSON válida."""
    def _ok(v: Any) -> bool:
        if not isinstance(v, str):
            return False
        try:
            return isinstance(json.loads(v), list)
        except (json.JSONDecodeError, ValueError):
            return False
    return s.map(_ok).astype(bool)


def normalizar_amenity(texto: str) -> str:
    """Minúsculas, espacios (incluido el no separable) colapsados y sin espacios en los extremos."""
    return re.sub(r"\s+", " ", texto.replace("\xa0", " ")).strip().lower()


def flags_amenities(s: pd.Series) -> pd.DataFrame:
    """Flags booleanos de amenities: parsea la lista JSON y aplica cada patrón a cada elemento normalizado.

    Comparar elemento a elemento (y no sobre el texto crudo) evita falsos positivos entre elementos,
    p. ej. que «Dishwasher» active la lavadora o «Pool table» la piscina.
    """
    elementos = s.map(parse_amenities).explode()
    elementos = elementos.dropna().astype("string").map(normalizar_amenity).astype("string")
    salida = {}
    for col, patron in config.PATRONES_AMENITIES.items():
        coincide = elementos.str.contains(patron, regex=True)
        por_fila = coincide.groupby(level=0).any() if len(coincide) else pd.Series(dtype=bool)
        salida[col] = por_fila.reindex(s.index, fill_value=False).astype(bool)
    return pd.DataFrame(salida, index=s.index)


def comprobar_prevalencias(prevalencias: dict[str, float], n_filas: int) -> list[str]:
    """Patrones de amenities con prevalencia 0 o fuera de ``COTAS_PREVALENCIA_AMENITIES``.

    Las cotas solo se aplican con al menos ``UMBRAL_FILAS`` anuncios; la prevalencia 0 siempre.
    """
    rotos = []
    for col, valor in prevalencias.items():
        lo, hi = config.COTAS_PREVALENCIA_AMENITIES[col]
        if valor == 0 or (n_filas >= config.UMBRAL_FILAS and not lo <= valor <= hi):
            rotos.append(f"{col} = {valor:.2f} % (rango plausible {lo}-{hi} %)")
    return rotos


def analizar_cotizaciones(raw: pd.Series) -> pd.DataFrame:
    """Extrae de ``price_quote_raw`` (JSON con clave ``quote``) descuento, subtotal y tarifa mensual.

    Devuelve un DataFrame por fila con ``parseada``, ``descuento``, ``subtotal_noches``,
    ``es_tarifa_mensual`` y ``tipos_descuento`` (descripciones de las partidas de descuento).
    """
    def _uno(valor: Any) -> tuple[bool, float, float, bool, tuple[str, ...]]:
        if not isinstance(valor, str):
            return False, np.nan, np.nan, False, ()
        try:
            q = json.loads(valor).get("quote") or {}
        except (json.JSONDecodeError, ValueError, AttributeError):
            return False, np.nan, np.nan, False, ()
        if not isinstance(q, dict):
            return False, np.nan, np.nan, False, ()

        def _num(clave: str) -> float:
            try:
                return float(q.get(clave)) if q.get(clave) is not None else np.nan
            except (TypeError, ValueError):
                return np.nan
        partidas = q.get("raw_price_line_items") or []
        tipos = tuple(str(p.get("description", "")) for p in partidas
                      if isinstance(p, dict) and p.get("item_type") == "discount_amount")
        return True, _num("discount_amount"), _num("nightly_subtotal"), bool(q.get("is_monthly_priced")), tipos

    filas = [_uno(v) for v in raw]
    return pd.DataFrame(filas, index=raw.index,
                        columns=["parseada", "descuento", "subtotal_noches", "es_tarifa_mensual", "tipos_descuento"])


def resumen_cotizaciones(cot: pd.DataFrame, precio: pd.Series, minimum_nights: pd.Series,
                         noches: pd.Series) -> dict[str, Any]:
    """Cifras de la naturaleza del precio calculadas por código desde ``price_quote_raw``.

    Todas sobre anuncios con precio positivo y cotización parseable.
    """
    base = cot["parseada"] & precio.gt(0)
    n = _contar(base)
    con_desc = base & cot["descuento"].gt(0)
    pct_desc = (cot.loc[con_desc, "descuento"] / cot.loc[con_desc, "subtotal_noches"] * 100)
    mensual = base & cot["es_tarifa_mensual"]
    min30 = base & minimum_nights.ge(config.MIN_NIGHTS_LARGA_ESTANCIA)
    comparables = base & noches.notna() & minimum_nights.notna()
    tipos = cot.loc[con_desc, "tipos_descuento"].explode().dropna()
    tipos = tipos[tipos.astype(str).str.len() > 0]
    return {
        "n_cotizaciones_con_precio": n,
        "n_con_descuento": _contar(con_desc),
        "pct_con_descuento": float(con_desc.sum() / n * 100) if n else float("nan"),
        "mediana_pct_descuento": float(pct_desc.median()) if len(pct_desc) else float("nan"),
        "n_tarifa_mensual": _contar(mensual),
        "pct_tarifa_mensual": float(mensual.sum() / n * 100) if n else float("nan"),
        "n_min30": _contar(min30),
        "pct_mensual_entre_min30": float((mensual & min30).sum() / max(min30.sum(), 1) * 100),
        "n_noches_comparables": _contar(comparables),
        "pct_noches_igual_minimo": float((noches[comparables] == minimum_nights[comparables]).mean() * 100)
        if comparables.any() else float("nan"),
        "partidas_descuento": {str(k): int(v) for k, v in tipos.value_counts().head(8).items()},
    }


def comprobar_columnas(df: pd.DataFrame, columnas: tuple[str, ...], contexto: str) -> None:
    """Lanza ``ErrorDatos`` (accionable, no ``KeyError``) si falta alguna columna requerida."""
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ErrorDatos(f"Faltan columnas obligatorias para {contexto}: {', '.join(faltan)}. "
                         "Puede que Inside Airbnb haya cambiado el formato del snapshot; revisa "
                         "config.COLUMNAS_OBLIGATORIAS.")


def limpiar_html(s: pd.Series) -> pd.Series:
    """Quita etiquetas HTML y desescapa entidades (``<br />`` → espacio)."""
    sin_tags = s.astype("string").str.replace(r"<[^>]+>", " ", regex=True)
    return sin_tags.map(lambda t: html.unescape(t) if isinstance(t, str) else t).astype("string").str.strip()


# --------------------------------------------------------------------------
# Reglas
# --------------------------------------------------------------------------
def vallas_iqr_log(precio: pd.Series, k: float = config.IQR_K) -> tuple[float, float]:
    """Vallas de Tukey calculadas sobre ``ln(precio)`` y devueltas en euros."""
    log_p = np.log(precio.dropna())
    q1, q3 = np.quantile(log_p, [0.25, 0.75])
    iqr = q3 - q1
    return float(np.exp(q1 - k * iqr)), float(np.exp(q3 + k * iqr))


def imputar_por_grupo(df: pd.DataFrame, columna: str) -> tuple[pd.Series, pd.Series]:
    """Imputa ``columna`` con la mediana por (room_type, accommodates), luego por room_type y luego global.

    Devuelve (serie imputada, flag de imputación).
    """
    original = df[columna]
    faltan = original.isna()
    med1 = df.groupby(["room_type", "accommodates"])[columna].transform("median")
    med2 = df.groupby("room_type")[columna].transform("median")
    imputada = original.fillna(med1).fillna(med2).fillna(original.median())
    return imputada.astype("float64"), faltan.astype(bool)


def extraer_moneda(raw: pd.Series) -> pd.Series:
    """Extrae el código ``currency`` del JSON de ``price_quote_raw`` (NaN si no aparece)."""
    return raw.astype("string").str.extract(r'"currency":\s*"([A-Z]{3})"', expand=False)


def _contar(mascara: pd.Series) -> int:
    """Número de True en una máscara (como int de Python)."""
    return int(mascara.sum())


def limpiar(df_raw: pd.DataFrame, barrios: pd.DataFrame, ciudad: Ciudad) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica todas las conversiones y reglas. Devuelve (DataFrame limpio, registro de limpieza)."""
    comprobar_columnas(df_raw, config.COLUMNAS_OBLIGATORIAS, "la limpieza")
    df = df_raw.copy()
    n_raw = len(df)
    registro: dict[str, Any] = {"conversiones": {}, "reglas": {}, "columnas_eliminadas": {},
                                "columnas_reconstruidas": {}}

    # 1. Conversión de tipos ------------------------------------------------
    def _registrar_conversion(col: str, antes: pd.Series, despues: pd.Series, tipo: str) -> None:
        no_convertibles = _contar(antes.notna() & despues.isna())
        registro["conversiones"][col] = {"a": tipo, "no_nulos_antes": int(antes.notna().sum()),
                                         "no_convertibles": no_convertibles}
        if antes.notna().sum() and no_convertibles / antes.notna().sum() > 0.01:
            logger.warning("%s: %d valores no convertibles a %s", col, no_convertibles, tipo)

    if "price" in df:
        nuevo = parse_precio(df["price"])
        _registrar_conversion("price", df["price"], nuevo, "float64 (€)")
        df["price"] = nuevo
    for col in COLUMNAS_PORCENTAJE:
        if col in df:
            nuevo = parse_porcentaje(df[col])
            _registrar_conversion(col, df[col], nuevo, "float64 [0, 100]")
            df[col] = nuevo
    for col in COLUMNAS_BOOLEANAS:
        if col in df:
            nuevo = a_booleano(df[col])
            _registrar_conversion(col, df[col], nuevo, str(nuevo.dtype))
            df[col] = nuevo
    for col in COLUMNAS_FECHA:
        if col in df:
            nuevo = a_fecha(df[col])
            _registrar_conversion(col, df[col], nuevo, "datetime64")
            df[col] = nuevo

    # 2. Derivaciones que necesitan columnas que luego se eliminan ------------
    banios = parse_banios(df["bathrooms_text"]) if "bathrooms_text" in df else None
    if banios is not None:
        if "bathrooms" in df and df["bathrooms"].notna().any():
            df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(banios["bathrooms"])
        else:
            df["bathrooms"] = banios["bathrooms"]
        df["bano_compartido"] = banios["bano_compartido"]
        registro["columnas_reconstruidas"]["bathrooms"] = "reconstruida desde bathrooms_text"
        registro["conversiones"]["bathrooms_text"] = {
            "a": "bathrooms (float) + bano_compartido (bool)",
            "no_nulos_antes": int(df_raw["bathrooms_text"].notna().sum()),
            "no_convertibles": _contar(df_raw["bathrooms_text"].notna() & df["bathrooms"].isna()),
        }
    validas = amenities_validas(df["amenities"])
    df["n_amenities"] = df["amenities"].map(lambda v: len(parse_amenities(v))).astype("int64")
    df = pd.concat([df, flags_amenities(df["amenities"])], axis=1)
    registro["conversiones"]["amenities"] = {"a": "n_amenities + flags", "malformadas_json": _contar(~validas)}
    prevalencias = {c: float(df[c].mean() * 100) for c in config.PATRONES_AMENITIES}
    rotos = comprobar_prevalencias(prevalencias, len(df))
    if rotos:
        raise ErrorDatos("Patrones de amenities con prevalencia nula o fuera de rango plausible "
                         f"(¿formato cambiado?): {'; '.join(rotos)}")
    registro["prevalencia_amenities"] = prevalencias

    nombre_limpio = limpiar_html(df["name"])
    df["nombre_longitud"] = nombre_limpio.str.len().fillna(0).astype("int64")
    if "description" in df:
        desc = limpiar_html(df["description"])
        df["descripcion_longitud"] = desc.str.len().fillna(0).astype("int64")
        df["descripcion_palabras"] = desc.str.split().str.len().fillna(0).astype("int64")
    if "host_about" in df:
        df["host_tiene_descripcion"] = df["host_about"].notna().astype(bool)

    if "price_quote_raw" in df:
        moneda = extraer_moneda(df["price_quote_raw"])
        conteo = moneda.value_counts()
        registro["moneda"] = {
            "moneda": str(conteo.index[0]) if len(conteo) else "desconocida",
            "pct_moneda": float(conteo.iloc[0] / conteo.sum() * 100) if len(conteo) else float("nan"),
            "n_cotizaciones_con_moneda": int(conteo.sum()),
            "n_cotizaciones_sin_moneda": _contar(df["price_quote_raw"].notna() & moneda.isna()),
            "monedas_distintas": int(len(conteo)),
        }
    df["noches_cotizadas"] = (df["price_quote_checkout_date"] - df["price_quote_checkin_date"]).dt.days.astype("float64")
    registro["precio_naturaleza"] = {
        "checkin_min": df["price_quote_checkin_date"].min(),
        "checkin_max": df["price_quote_checkin_date"].max(),
    }
    if "price_quote_price_per_night" in df:
        ambos = df["price"].notna() & df["price_quote_price_per_night"].notna()
        iguales = np.isclose(df.loc[ambos, "price"], df.loc[ambos, "price_quote_price_per_night"])
        registro["precio_naturaleza"].update({
            "n_ambos": _contar(ambos),
            "pct_precio_igual_cotizacion": float(iguales.mean() * 100) if ambos.any() else float("nan"),
        })
    if "price_quote_raw" in df:
        cot = analizar_cotizaciones(df["price_quote_raw"])
        df["es_tarifa_mensual"] = cot["es_tarifa_mensual"].astype(bool)
        registro["precio_naturaleza"].update(
            resumen_cotizaciones(cot, df["price"], df["minimum_nights"], df["noches_cotizadas"]))
    else:
        df["es_tarifa_mensual"] = False

    # 3. Eliminación de columnas (motivo documentado) -------------------------
    eliminar: dict[str, str] = {}
    for col in df.columns:
        if df[col].isna().all():
            eliminar[col] = "100 % nulas en este snapshot"
    for grupo, motivo in ((ELIMINAR_URLS, "URL sin valor analítico"),
                          (ELIMINAR_IDS_TECNICOS, "identificador técnico del scrape"),
                          (ELIMINAR_PERSONALES, "dato personal innecesario (privacidad)"),
                          (ELIMINAR_TEXTO_LARGO, "texto largo sustituido por métricas de longitud"),
                          (ELIMINAR_CRUDOS, "texto crudo ya convertido en columnas derivadas")):
        for col in grupo:
            if col in df and col not in eliminar:
                eliminar[col] = motivo
    df = df.drop(columns=list(eliminar))
    registro["columnas_eliminadas"] = eliminar

    # 4. Duplicados ------------------------------------------------------------
    dup_id = df["id"].duplicated()
    registro["duplicados"] = {"por_id": _contar(dup_id),
                              "filas_completas": int(df_raw.duplicated().sum())}
    df = df.loc[~dup_id].reset_index(drop=True)
    registro["duplicados"]["eliminados"] = _contar(dup_id)
    if n_raw - registro["duplicados"]["eliminados"] != len(df):
        raise ErrorDatos("Contabilidad de filas rota tras eliminar duplicados.")

    # 5. Precio: nulos, cero, outliers ------------------------------------------
    precio = df["price"]
    df["precio_valido"] = (precio.notna() & (precio > 0)).astype(bool)
    sin_disp = df["availability_365"] == 0
    registro["precio"] = {
        "nulo": _contar(precio.isna()),
        "pct_nulo": float(precio.isna().mean() * 100),
        "cero": _contar(precio == 0),
        "negativo": _contar(precio < 0),
        "validos": _contar(df["precio_valido"]),
        "nulo_y_sin_disponibilidad": _contar(precio.isna() & sin_disp),
        "pct_nulos_sin_disponibilidad": float((precio.isna() & sin_disp).sum() / max(precio.isna().sum(), 1) * 100),
    }
    validos = precio[df["precio_valido"]]
    valla_inf, valla_sup = vallas_iqr_log(validos)
    df["es_outlier_bajo"] = (df["precio_valido"] & (precio < valla_inf)).astype(bool)
    df["es_outlier_alto"] = (df["precio_valido"] & (precio > valla_sup)).astype(bool)
    df["es_outlier_precio"] = (df["es_outlier_bajo"] | df["es_outlier_alto"]).astype(bool)
    registro["outliers"] = {
        "metodo": "vallas de Tukey (k = 1,5) sobre ln(precio)",
        "k": config.IQR_K,
        "valla_inf": valla_inf, "valla_sup": valla_sup,
        "n_bajo": _contar(df["es_outlier_bajo"]), "n_alto": _contar(df["es_outlier_alto"]),
        "n_total": _contar(df["es_outlier_precio"]),
        "pct_total": float(df["es_outlier_precio"].sum() / max(len(validos), 1) * 100),
        "p1": float(validos.quantile(0.01)), "p99": float(validos.quantile(0.99)),
        "min": float(validos.min()), "max": float(validos.max()),
    }

    # 6. Estancias mínimas -----------------------------------------------------------
    df["minimum_nights_extremo"] = (df["minimum_nights"] > config.MIN_NIGHTS_EXTREMO).astype(bool)
    df["alquiler_larga_estancia"] = (df["minimum_nights"] >= config.MIN_NIGHTS_LARGA_ESTANCIA).astype(bool)
    registro["minimum_nights"] = {
        "extremo_mayor_365": _contar(df["minimum_nights_extremo"]),
        "larga_estancia_30_o_mas": _contar(df["alquiler_larga_estancia"]),
        "max": float(df["minimum_nights"].max()),
        "nulos": _contar(df["minimum_nights"].isna()),
    }

    # 7. Coordenadas --------------------------------------------------------------
    df["coords_validas"] = dentro_bbox(df["latitude"], df["longitude"], ciudad.bbox)
    registro["coordenadas"] = {"bbox": list(ciudad.bbox), "fuera_bbox_o_nulas": _contar(~df["coords_validas"])}

    # 8. Cruce con barrios oficiales ------------------------------------------------
    oficiales = set(barrios["neighbourhood"])
    en_listings = set(df["neighbourhood_cleansed"].dropna())
    no_oficiales = sorted(en_listings - oficiales)
    sin_anuncios = sorted(oficiales - en_listings)
    if no_oficiales:
        logger.warning("Barrios del listing que no están en neighbourhoods.csv: %s", no_oficiales)
    registro["barrios"] = {"barrios_oficiales": len(oficiales), "barrios_con_anuncios": len(en_listings),
                           "no_oficiales": no_oficiales, "oficiales_sin_anuncios": sin_anuncios,
                           "distritos": int(df["neighbourhood_group_cleansed"].nunique())}

    # 9. Imputación documentada de bedrooms / beds -------------------------------------
    imputados = {}
    for col in ("bedrooms", "beds"):
        if col in df:
            serie, flag = imputar_por_grupo(df, col)
            df[col] = serie
            df[f"{col}_imputado"] = flag
            imputados[col] = _contar(flag)
    registro["imputados"] = imputados

    registro["reglas"] = {
        "precio_nulo_o_no_positivo (precio_valido=False)": _contar(~df["precio_valido"]),
        "precio negativo (precio_valido=False)": registro["precio"]["negativo"],
        "outlier_precio (flag, no se borra)": registro["outliers"]["n_total"],
        "minimum_nights > 365 (flag)": registro["minimum_nights"]["extremo_mayor_365"],
        "minimum_nights >= 30 (flag larga estancia)": registro["minimum_nights"]["larga_estancia_30_o_mas"],
        "coordenadas fuera del bbox o nulas (excluidas de mapa y distancia)": registro["coordenadas"]["fuera_bbox_o_nulas"],
        "duplicados por id (eliminados)": registro["duplicados"]["eliminados"],
        **{f"{c} imputado con mediana por grupo": n for c, n in imputados.items()},
    }
    registro["contabilidad_filas"] = {"filas_raw": n_raw, "duplicados_eliminados": registro["duplicados"]["eliminados"],
                                      "filas_procesadas": len(df)}
    registro["politica_nulos"] = [
        {"columna": col, "nulos": int(df[col].isna().sum()), "regla": regla_nulos(col)[0],
         "decision": regla_nulos(col)[1], "justificacion": regla_nulos(col)[2]}
        for col in df.columns if df[col].isna().any() or (col in df_raw and df_raw[col].isna().any())
    ]
    return df, registro


def validar_procesado(df: pd.DataFrame, documentadas: set[str] | None = None) -> list[str]:
    """Comprueba tipos y rangos del DataFrame procesado antes de escribirlo.

    Lanza ``ErrorDatos`` con todas las violaciones. Devuelve la lista de columnas no
    documentadas en el diccionario (solo aviso, N15).
    """
    errores: list[str] = []
    tipos = pd.api.types
    if not tipos.is_float_dtype(df["price"]):
        errores.append("price no es float")
    if not tipos.is_bool_dtype(df["host_is_superhost"]):
        errores.append("host_is_superhost no es bool")
    for col in ("last_scraped", "first_review", "last_review"):
        if col in df and not tipos.is_datetime64_any_dtype(df[col]):
            errores.append(f"{col} no es datetime64")
    if "no_disponible_365" in df and not df["no_disponible_365"].dropna().between(0, 1).all():
        errores.append("no_disponible_365 fuera de [0, 1]")
    if "distancia_centro_km" in df and (df["distancia_centro_km"].dropna() < 0).any():
        errores.append("distancia_centro_km negativa")
    if "n_amenities" in df and (not tipos.is_integer_dtype(df["n_amenities"]) or (df["n_amenities"] < 0).any()):
        errores.append("n_amenities no es entero no negativo")
    if "precio_por_persona" in df and (df["precio_por_persona"].dropna() <= 0).any():
        errores.append("precio_por_persona no positivo")
    if df["id"].duplicated().any():
        errores.append("id duplicado")
    for col in config.COLUMNAS_DASHBOARD:
        if col not in df:
            errores.append(f"falta la columna del dashboard {col}")
    if errores:
        raise ErrorDatos("Validación del procesado fallida: " + "; ".join(errores))
    sin_documentar: list[str] = []
    if documentadas is not None:
        sin_documentar = sorted(set(df.columns) - documentadas)
        for col in sin_documentar:
            logger.warning("Columna sin documentar en el diccionario: %s", col)
    return sin_documentar
