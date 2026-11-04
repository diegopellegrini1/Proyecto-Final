"""Generación de README, informe y diccionario desde ``findings.json`` mediante plantillas Jinja2.

Toda cifra de los documentos se inserta con ``cifra(ruta, formato)``, que la envuelve en un
marcador invisible ``<!--f:ruta:formato-->valor<!--/f-->``. El verificador independiente
(``scripts/verificar_consistencia.py``) parsea esos marcadores y recalcula cada valor desde
``findings.json``. Las frases cualitativas dependientes de los datos se registran en
``AFIRMACIONES`` y se evalúan antes de escribir ``findings.json``: las ``INVARIANTES`` detienen el
pipeline si fallan; las ``INTERPRETATIVAS`` falsas solo avisan y la plantilla usa un texto
alternativo (``cumple``) o muestra una advertencia visible (``afirmacion``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import jinja2
import pandas as pd

from src import config, diccionario, formato
from src.errores import ErrorInforme

logger = logging.getLogger(__name__)

MARCADOR = re.compile(r"<!--f:([^:>]+):(\w+)-->(.*?)<!--/f-->")
PLANTILLAS: dict[str, Path] = {
    "README.md.j2": config.RUTA_README,
    "informe.md.j2": config.RUTA_INFORME,
}


def obtener(findings: dict[str, Any], ruta: str) -> Any:
    """Valor de ``findings`` en la ruta con puntos (admite índices de lista)."""
    actual: Any = findings
    for parte in ruta.split("."):
        try:
            actual = actual[int(parte)] if isinstance(actual, list) else actual[parte]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ErrorInforme(f"Ruta inexistente en findings: {ruta}") from exc
    return actual


def formatear(valor: Any, nombre_formato: str) -> str:
    """Aplica el formateador es-ES ``nombre_formato`` (o ``texto``)."""
    if nombre_formato == "texto":
        return str(valor)
    if nombre_formato not in formato.FILTROS_JINJA:
        raise ErrorInforme(f"Formato desconocido: {nombre_formato}")
    return formato.FILTROS_JINJA[nombre_formato](valor)


def _coef(f: dict[str, Any], termino: str) -> dict[str, Any]:
    """Coeficiente OLS por término."""
    return next(c for c in f["estadistica"]["ols"]["coeficientes"] if c["termino"] == termino)


E = lambda f: f["estadistica"]  # noqa: E731
D = lambda f: f["descriptivo"]  # noqa: E731
RP = lambda f: f["estadistica"]["renta_precio"]  # noqa: E731
OLS = lambda f: f["estadistica"]["ols"]  # noqa: E731
_Afirmacion = tuple[str, str, Callable[[dict[str, Any]], bool]]


def _signif_pos(c: dict[str, Any]) -> bool:
    return c["pct"] > 0 and c["significativo"]


def _signif_neg(c: dict[str, Any]) -> bool:
    return c["pct"] < 0 and c["significativo"]


# INVARIANTES ESTRUCTURALES: supuestos sin los que el análisis o el texto no tienen sentido
# (requisitos del enunciado, moneda, naturaleza del precio, unión de las dos fuentes, validación sin fuga).
# Si alguna falla, ``evaluar_afirmaciones`` lanza ErrorInforme y el pipeline se detiene (exit 4).
INVARIANTES: list[_Afirmacion] = [
    ("dataset_cumple_requisitos", "El conjunto final cumple ≥ 50.000 filas y ≥ 20 columnas",
     lambda f: f["dataset"]["requisitos"]["cumple_filas"] and f["dataset"]["requisitos"]["cumple_columnas"]),
    ("union_mayoritaria", "Más del 95 % de los anuncios recibe datos del INE",
     lambda f: f["dataset"]["union"]["total"]["pct_distrito"] > 95),
    ("moneda_eur", "La moneda real de las cotizaciones es EUR", lambda f: f["moneda"]["moneda"] == "EUR"),
    ("precio_es_cotizacion", "price coincide con la cotización en casi todos los casos",
     lambda f: f["precio_naturaleza"]["pct_precio_igual_cotizacion"] > 95),
    ("ml_particiones_sin_solape", "La validación del ML no comparte anfitriones entre train y test",
     lambda f: (not E(f)["ml"]["ejecutado"]) or "GroupKFold" in E(f)["ml"]["validacion"]),
]

# AFIRMACIONES INTERPRETATIVAS: dependen del sentido de un resultado de ESTOS snapshots. Si alguna deja
# de cumplirse, no se aborta: se registra un aviso y la plantilla muestra un texto alternativo
# (``cumple()``) o una advertencia visible junto a la frase (``afirmacion()``).
INTERPRETATIVAS: list[_Afirmacion] = [
    ("nulos_precio_sin_disponibilidad", "La mayoría de precios nulos son anuncios sin disponibilidad",
     lambda f: f["limpieza"]["precio"]["pct_nulos_sin_disponibilidad"] > 50),
    ("precio_asimetrico", "El precio es asimétrico a la derecha (media > mediana)",
     lambda f: D(f)["precio"]["asimetria"] > 1 and D(f)["precio"]["media"] > D(f)["precio"]["mediana"]),
    ("log_mas_simetrico", "ln(precio) es mucho más simétrico que el precio",
     lambda f: abs(D(f)["log_precio"]["asimetria"]) < abs(D(f)["precio"]["asimetria"]) / 10),
    ("normalidad_rechazada", "Se rechaza la normalidad del precio y de ln(precio)",
     lambda f: E(f)["normalidad_precio"]["significativo"] and E(f)["normalidad_log_precio"]["significativo"]),
    ("tipo_mayor_efecto_que_ciudad", "El ε² del tipo de habitación supera al de la ciudad",
     lambda f: E(f)["kruskal_tipo"]["tamano_efecto"]["valor"] > E(f)["kruskal_ciudad"]["tamano_efecto"]["valor"]),
    ("kruskal_ciudad_no_despreciable", "Kruskal por ciudad significativo con efecto al menos pequeño",
     lambda f: E(f)["kruskal_ciudad"]["significativo"]
     and E(f)["kruskal_ciudad"]["tamano_efecto"]["interpretacion"] != "despreciable"),
    ("posthoc_ciudad_mayoria", "La mayoría de pares de ciudades difieren",
     lambda f: E(f)["posthoc_ciudad"]["n_pares_significativos"] > E(f)["posthoc_ciudad"]["n_pares"] / 2),
    ("top_ciudad_distinguible", "La ciudad más cara se distingue de la segunda (IC por anfitrión sin solape)",
     lambda f: E(f)["bootstrap"]["top_ciudad_distinguible"]),
    ("superhost_significativo_despreciable", "Mann-Whitney superhost: significativo pero efecto despreciable",
     lambda f: E(f)["mann_whitney_superhost"]["significativo"]
     and E(f)["mann_whitney_superhost"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    ("superhost_ols_pequeno", "Al controlar, la prima de superhost es pequeña (< 10 %) y no sobrevive a Holm",
     lambda f: abs(OLS(f)["coef_superhost"]["pct"]) < 10 and not OLS(f)["coef_superhost"]["significativo_holm"]),
    ("multianuncio_ols_no_significativo", "Ser multi-anuncio no se asocia con el precio al controlar",
     lambda f: not OLS(f)["coef_multianuncio"]["significativo"]),
    ("sensibilidad_misma_decision", "Con un anuncio por anfitrión se mantienen todas las decisiones",
     lambda f: E(f)["sensibilidad_un_anuncio_por_host"]["todas_mismas_decisiones"]),
    ("chi2_ciudad_tipo_no_despreciable", "Chi² ciudad × tipo: asociación no despreciable",
     lambda f: E(f)["chi2_ciudad_tipo"]["significativo"]
     and E(f)["chi2_ciudad_tipo"]["tamano_efecto"]["interpretacion"] != "despreciable"),
    ("chi2_tipo_superhost_despreciable", "Chi² tipo × superhost: significativo pero despreciable",
     lambda f: E(f)["chi2_tipo_superhost"]["significativo"]
     and E(f)["chi2_tipo_superhost"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    # Relación renta-precio
    ("renta_global_debil", "La ρ global renta-precio (anuncio a anuncio) es débil (|ρ| < 0,1)",
     lambda f: abs(RP(f)["global_distrito"]["rho"]) < 0.1),
    ("renta_global_no_significativa", "La ρ global no es significativa remuestreando distritos",
     lambda f: not RP(f)["global_distrito"]["cluster_zona"]["significativo"]),
    ("renta_global_en_el_limite", "El p bootstrap de la ρ global está en el límite de la significación (p cerca "
                                  "de α o IC que roza el 0)",
     lambda f: RP(f)["global_distrito"]["cluster_zona"]["en_el_limite"]),
    ("renta_global_significativa_anfitrion", "Remuestreando solo anfitriones la ρ global sí sería significativa",
     lambda f: RP(f)["global_distrito"]["cluster_anfitrion"]["significativo"]),
    ("renta_ciudades_todas_positivas", "Dentro de cada ciudad la ρ renta-precio es positiva",
     lambda f: all(c["distrito"]["rho"] > 0 for c in RP(f)["por_ciudad"] if c["distrito"].get("evaluable"))),
    ("renta_ecologico_global_no_significativo", "Entre distritos de toda España no hay correlación significativa",
     lambda f: not RP(f)["ecologico_global"]["significativo"]),
    ("renta_rho_sensibilidades_signo", "Las sensibilidades conservan el signo de la ρ global",
     lambda f: RP(f)["sensibilidades_mismo_signo"]),
    ("renta_ols_positiva", "En la regresión con efectos fijos de ciudad la renta se asocia con más precio "
                           "(significativo también agrupando por distrito)",
     lambda f: _signif_pos(OLS(f)["coef_renta"]) and OLS(f)["coef_renta"]["significativo_distrito"]),
    ("renta_aporta_poco_r2", "Las variables del INE añaden menos de 5 puntos de R² ajustado",
     lambda f: OLS(f)["delta_r2_ajustado_ine"] < 0.05),
    ("renta_ols_sensibilidades_signo", "El coeficiente de la renta conserva el signo en todas las sensibilidades",
     lambda f: OLS(f)["sensibilidades_renta_mismo_signo"]),
    ("renta_menor_con_distancia", "En las ciudades urbanas, controlar por la distancia al centro reduce la "
                                  "asociación de la renta",
     lambda f: OLS(f)["sensibilidades"]["solo_urbanas_con_distancia"]["renta"]["pct_por_10pct_renta"]
     < OLS(f)["coef_renta"]["pct_por_10pct_renta"]),
    ("distancia_negativa_urbanas", "En las ciudades urbanas, más distancia al centro se asocia con menor precio",
     lambda f: OLS(f)["sensibilidades"]["solo_urbanas_con_distancia"]["distancia"]["pct"] < 0
     and OLS(f)["sensibilidades"]["solo_urbanas_con_distancia"]["distancia"]["significativo"]),
    ("ml_ine_aporta_poco", "En el ML las variables del INE añaden menos de 5 puntos de R²",
     lambda f: (not E(f)["ml"]["ejecutado"]) or E(f)["ml"]["aporte_r2_ine"] < 0.05),
    # Resto del modelo
    ("capacidad_positiva", "Más capacidad se asocia con mayor precio", lambda f: _signif_pos(_coef(f, "accommodates"))),
    ("privada_negativa", "La habitación privada se asocia con menor precio que el entero",
     lambda f: _signif_neg(OLS(f)["coef_privada"])),
    ("larga_estancia_negativa", "Estancia mínima ≥ 30 noches se asocia con menor precio por noche",
     lambda f: _signif_neg(_coef(f, "alquiler_larga_estancia"))),
    ("bano_compartido_negativo", "El baño compartido se asocia con menor precio",
     lambda f: _signif_neg(_coef(f, "bano_compartido"))),
    ("piscina_aire_positivos", "Piscina y aire acondicionado se asocian con mayor precio",
     lambda f: _signif_pos(_coef(f, "tiene_piscina")) and _signif_pos(_coef(f, "tiene_aire_acondicionado"))),
    ("heterocedasticidad", "Breusch-Pagan detecta heterocedasticidad",
     lambda f: OLS(f)["diagnostico"]["heterocedasticidad"]),
    ("residuos_no_normales", "Jarque-Bera rechaza normalidad de residuos",
     lambda f: OLS(f)["diagnostico"]["jarque_bera_p"] < f["meta"]["alfa"]),
    ("vif_bajo", "Sin dummies de ciudad ni de mes, ningún VIF supera 5", lambda f: OLS(f)["n_vif_mayor_5"] == 0),
    ("ols_sin_influyentes_signos", "Los signos significativos se mantienen sin observaciones influyentes",
     lambda f: OLS(f)["sensibilidad_sin_influyentes"]["signos_coinciden"]),
    ("ols_sin_outliers_signos", "Los signos significativos se mantienen sin outliers",
     lambda f: OLS(f)["sensibilidad_con_outliers_signos"]["signos_coinciden"]),
    ("ml_supera_base", "El modelo ML supera a la línea base en validación agrupada",
     lambda f: (not E(f)["ml"]["ejecutado"]) or (E(f)["ml"]["cv"]["r2_media"] > E(f)["ml"]["cv_base"]["r2_media"]
                                                 and E(f)["ml"]["cv"]["mae_log_media"] < E(f)["ml"]["cv_base"]["mae_log_media"])),
    ("ml_aleatorio_optimista", "La partición aleatoria da un R² mayor que la agrupada por anfitrión",
     lambda f: (not E(f)["ml"]["ejecutado"]) or E(f)["ml"]["caida_r2_aleatorio_a_grupos"] > 0),
    ("ml_hgb_supera_lineal_misma_muestra", "En las mismas filas, particiones y variables, el HGB supera al lineal",
     lambda f: (not E(f)["ml"]["ejecutado"]) or (
         E(f)["ml"]["comparacion_misma_muestra"]["hgb_vars_ols"]["r2_media"]
         > E(f)["ml"]["comparacion_misma_muestra"]["lineal_vars_ols"]["r2_media"])),
    # Descriptivo
    ("cara_no_mas_rica", "La ciudad más cara no es la de mayor renta mediana del distrito",
     lambda f: D(f)["ciudad_mas_cara"]["renta_dis_mediana"] < max(c["renta_dis_mediana"] for c in D(f)["por_ciudad"])),
    ("capacidad_creciente", "La mediana de precio crece con cada tramo de capacidad",
     lambda f: all(a["mediana"] < b["mediana"] for a, b in zip(D(f)["por_capacidad"], D(f)["por_capacidad"][1:]))),
    ("multianuncio_mayoria", "La mayoría de anuncios son de anfitriones multi-anuncio",
     lambda f: D(f)["hosts"]["pct_anuncios_multianuncio"] > 50),
    ("mayoria_hosts_un_anuncio", "La mayoría de anfitriones tiene un solo anuncio",
     lambda f: D(f)["hosts"]["pct_hosts_un_anuncio"] > 50),
    ("no_disponible_mayor", "El % de días no disponibles supera la ocupación estimada por Inside Airbnb",
     lambda f: D(f)["ocupacion"]["no_disponible_media"] > D(f)["ocupacion"]["ocupacion_inside_airbnb_media"]),
    ("capacidad_mayor_correlacion", "La capacidad es la variable más correlacionada con el precio",
     lambda f: D(f)["correlaciones"]["con_precio"][0]["variable"] == "accommodates"),
    ("renta_correlacion_menor_que_capacidad", "La renta del distrito se correlaciona con el precio mucho menos que "
                                               "la capacidad",
     lambda f: abs(D(f)["correlaciones"]["con_precio_por_variable"]["ine_dis_renta_neta_persona"]["spearman"])
     < abs(D(f)["correlaciones"]["con_precio_por_variable"]["accommodates"]["spearman"]) / 2),
]
AFIRMACIONES: list[_Afirmacion] = INVARIANTES + INTERPRETATIVAS
IDS_INVARIANTES: frozenset[str] = frozenset(i for i, _, _ in INVARIANTES)
DESCRIPCIONES: dict[str, str] = {i: d for i, d, _ in AFIRMACIONES}


def evaluar_afirmaciones(findings: dict[str, Any]) -> dict[str, bool]:
    """Evalúa invariantes e interpretativas.

    Lanza ``ErrorInforme`` si una invariante es falsa o si cualquier comprobación no es evaluable
    (ruta inexistente). Las interpretativas falsas solo generan un aviso en el log: la plantilla
    las resuelve con texto condicional.
    """
    resultado: dict[str, bool] = {}
    for ident, _, comprobar in AFIRMACIONES:
        try:
            resultado[ident] = bool(comprobar(findings))
        except (KeyError, IndexError, TypeError, StopIteration) as exc:
            raise ErrorInforme(f"Afirmación «{ident}» no evaluable: {exc}") from exc
    invariantes_falsas = [k for k in IDS_INVARIANTES if not resultado[k]]
    if invariantes_falsas:
        raise ErrorInforme("Invariantes estructurales falsas con estos datos (el análisis no es válido tal cual): "
                           + ", ".join(sorted(invariantes_falsas)))
    for k, v in resultado.items():
        if not v:
            logger.warning("Afirmación interpretativa no cumplida con este snapshot: %s (%s); la plantilla usa "
                           "el texto alternativo o un aviso visible.", k, DESCRIPCIONES[k])
    return resultado


AVISO_AFIRMACION = " **[Aviso automático: con este snapshot no se cumple «{}»; la frase siguiente debe revisarse.]** "

def crear_entorno(findings: dict[str, Any], directorio: Path = config.DIR_PLANTILLAS) -> jinja2.Environment:
    """Entorno Jinja2 estricto con las funciones ``cifra``, ``valor`` y ``afirmacion``."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(directorio)), undefined=jinja2.StrictUndefined,
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, autoescape=False,
    )

    def cifra(ruta: str, nombre_formato: str) -> str:
        return f"<!--f:{ruta}:{nombre_formato}-->{formatear(obtener(findings, ruta), nombre_formato)}<!--/f-->"

    def valor(ruta: str) -> Any:
        return obtener(findings, ruta)

    def cumple(ident: str) -> bool:
        afirmaciones = findings.get("afirmaciones", {})
        if ident not in afirmaciones:
            raise ErrorInforme(f"La plantilla usa la afirmación «{ident}», que no existe en AFIRMACIONES.")
        return bool(afirmaciones[ident])

    def afirmacion(ident: str) -> str:
        """Ancla de una frase interpretativa: vacía si se cumple; aviso visible si no."""
        if cumple(ident):
            return ""
        if ident in IDS_INVARIANTES:  # no debería ocurrir: evaluar_afirmaciones ya habría abortado
            raise ErrorInforme(f"Invariante «{ident}» falsa al renderizar.")
        return AVISO_AFIRMACION.format(DESCRIPCIONES[ident])

    env.globals.update(cifra=cifra, valor=valor, afirmacion=afirmacion, cumple=cumple, f=findings)
    env.filters.update(formato.FILTROS_JINJA)
    return env


def render_todos(findings: dict[str, Any], filas_diccionario: list[dict[str, Any]],
                 directorio: Path = config.DIR_PLANTILLAS) -> dict[Path, str]:
    """Renderiza todos los documentos en memoria (solo lectura sobre ``findings``)."""
    env = crear_entorno(findings, directorio)
    salida: dict[Path, str] = {}
    for nombre, destino in PLANTILLAS.items():
        try:
            salida[destino] = env.get_template(nombre).render()
        except jinja2.TemplateError as exc:
            raise ErrorInforme(f"Error al renderizar {nombre}: {exc}") from exc
    salida[config.RUTA_DICCIONARIO] = diccionario.generar_markdown(filas_diccionario, formato.pct1, formato.n0)
    return salida


def escribir_documentos(findings: dict[str, Any], df: pd.DataFrame) -> list[Path]:
    """Renderiza y escribe README, informe y diccionario con saltos de línea LF."""
    documentos = render_todos(findings, diccionario.tabla(df))
    for ruta, texto in documentos.items():
        with open(ruta, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto)
    return list(documentos)
