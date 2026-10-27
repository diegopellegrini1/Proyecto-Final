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
_Afirmacion = tuple[str, str, Callable[[dict[str, Any]], bool]]

# INVARIANTES ESTRUCTURALES: supuestos sin los que el análisis o el texto no tienen sentido
# (moneda, naturaleza del precio, umbrales del dataset, validación sin fuga por anfitrión).
# Si alguna falla, ``evaluar_afirmaciones`` lanza ErrorInforme y el pipeline se detiene (exit 4).
INVARIANTES: list[_Afirmacion] = [
    ("dataset_cumple_umbrales", "El dataset cumple los umbrales mínimos",
     lambda f: all(f["dataset"]["requisitos"][k] for k in ("cumple_filas", "cumple_columnas", "cumple_tipos"))),
    ("moneda_eur", "La moneda real de las cotizaciones es EUR", lambda f: f["moneda"]["moneda"] == "EUR"),
    ("precio_es_cotizacion", "price coincide con la cotización en casi todos los casos",
     lambda f: f["precio_naturaleza"]["pct_precio_igual_cotizacion"] > 95),
    ("ml_particiones_sin_solape", "La validación del ML no comparte anfitriones entre train y test",
     lambda f: (not E(f)["ml"]["ejecutado"]) or "GroupKFold" in E(f)["ml"]["validacion"]),
]

# AFIRMACIONES INTERPRETATIVAS: dependen del sentido de un resultado de ESTE snapshot. Si alguna deja
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
    ("superhost_mediana_mayor", "Los superhosts tienen mayor mediana bruta",
     lambda f: E(f)["mann_whitney_superhost"]["extra"]["mediana_a"] > E(f)["mann_whitney_superhost"]["extra"]["mediana_b"]),
    ("superhost_significativo_despreciable", "Mann-Whitney superhost: significativo pero efecto despreciable",
     lambda f: E(f)["mann_whitney_superhost"]["significativo"]
     and E(f)["mann_whitney_superhost"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    ("superhost_no_significativo_ols", "Superhost no es significativo en OLS al controlar (el IC incluye el 0)",
     lambda f: not E(f)["ols"]["coef_superhost"]["significativo"]),
    ("superhost_evidencia_debil", "La prima de superhost al controlar es pequeña y no robusta (no sobrevive a Holm "
                                  "o a quitar el control de mes)",
     lambda f: abs(E(f)["ols"]["coef_superhost"]["pct"]) < 10 and (
         not E(f)["ols"]["coef_superhost"]["significativo"] or not E(f)["ols"]["coef_superhost"]["significativo_holm"]
         or not E(f)["ols"]["sensibilidad_sin_mes"]["coeficientes"]["host_is_superhost"]["significativo_alternativo"])),
    ("superhost_hc3_significativo", "Con HC3 (sin agrupar) el coeficiente de superhost sí es significativo",
     lambda f: E(f)["ols"]["coef_superhost"]["p_hc3"] < f["meta"]["alfa"]),
    ("sensibilidad_misma_decision", "Con un anuncio por anfitrión se mantienen todas las decisiones",
     lambda f: E(f)["sensibilidad_un_anuncio_por_host"]["todas_mismas_decisiones"]),
    ("entero_mas_caro", "El alojamiento entero es el tipo más caro",
     lambda f: E(f)["kruskal_tipo"]["extra"]["grupo_mas_caro"] == "Alojamiento entero"),
    ("kruskal_tipo_grande", "Kruskal por tipo: significativo y efecto grande",
     lambda f: E(f)["kruskal_tipo"]["significativo"] and E(f)["kruskal_tipo"]["tamano_efecto"]["interpretacion"] == "grande"),
    ("tipo_mayor_efecto_que_distrito", "El ε² del tipo de habitación supera al del distrito",
     lambda f: E(f)["kruskal_tipo"]["tamano_efecto"]["valor"] > E(f)["kruskal_distrito"]["tamano_efecto"]["valor"]),
    ("kruskal_distrito_significativo", "Kruskal por distrito significativo con efecto al menos pequeño",
     lambda f: E(f)["kruskal_distrito"]["significativo"]
     and E(f)["kruskal_distrito"]["tamano_efecto"]["interpretacion"] != "despreciable"),
    ("posthoc_tipo_todos", "Todos los pares de tipos difieren",
     lambda f: E(f)["posthoc_tipo"]["n_pares_significativos"] == E(f)["posthoc_tipo"]["n_pares"]),
    ("posthoc_distrito_mayoria", "La mayoría de pares de distritos difieren",
     lambda f: E(f)["posthoc_distrito"]["n_pares_significativos"] > E(f)["posthoc_distrito"]["n_pares"] / 2),
    ("top_distritos_empate", "Los dos distritos de mayor mediana no se distinguen estadísticamente",
     lambda f: E(f)["top_distritos_empate"]["empate"]),
    ("chi2_tipo_superhost_despreciable", "Chi² tipo × superhost: significativo pero despreciable",
     lambda f: E(f)["chi2_tipo_superhost"]["significativo"]
     and E(f)["chi2_tipo_superhost"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    ("chi2_multi_tipo_despreciable", "Chi² multi-anuncio × tipo: significativo pero despreciable",
     lambda f: E(f)["chi2_multi_tipo"]["significativo"]
     and E(f)["chi2_multi_tipo"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    ("chi2_distrito_tipo_mayor", "Chi² distrito × tipo: asociación no despreciable",
     lambda f: E(f)["chi2_distrito_tipo"]["significativo"]
     and E(f)["chi2_distrito_tipo"]["tamano_efecto"]["interpretacion"] != "despreciable"),
    ("ic_entero_no_solapa_privada", "Los IC (por anfitrión) de la mediana de entero y privada no se solapan",
     lambda f: E(f)["bootstrap"]["tipo"][0]["ic_inf_cluster"]
     > max(g["ic_sup_cluster"] for g in E(f)["bootstrap"]["tipo"][1:])),
    ("distancia_negativa", "Más distancia al centro se asocia con menor precio",
     lambda f: _coef(f, "distancia_centro_km")["pct"] < 0 and _coef(f, "distancia_centro_km")["significativo"]),
    ("capacidad_positiva", "Más capacidad se asocia con mayor precio",
     lambda f: _coef(f, "accommodates")["pct"] > 0 and _coef(f, "accommodates")["significativo"]),
    ("privada_negativa", "La habitación privada se asocia con menor precio que el entero",
     lambda f: E(f)["ols"]["coef_privada"]["pct"] < 0 and E(f)["ols"]["coef_privada"]["significativo"]),
    ("larga_estancia_negativa", "Estancia mínima ≥ 30 noches se asocia con menor precio por noche",
     lambda f: _coef(f, "alquiler_larga_estancia")["pct"] < 0 and _coef(f, "alquiler_larga_estancia")["significativo"]),
    ("aire_positivo", "El aire acondicionado se asocia con mayor precio",
     lambda f: _coef(f, "tiene_aire_acondicionado")["pct"] > 0 and _coef(f, "tiene_aire_acondicionado")["significativo"]),
    ("bano_compartido_negativo", "El baño compartido se asocia con menor precio",
     lambda f: _coef(f, "bano_compartido")["pct"] < 0 and _coef(f, "bano_compartido")["significativo"]),
    ("distrito_top_condicional", "El distrito con mayor coeficiente positivo tiene una mediana bruta menor que la referencia",
     lambda f: E(f)["ols"]["top_distritos"][0]["pct"] > 0
     and E(f)["ols"]["top_distritos"][0]["mediana_bruta"] < E(f)["ols"]["referencia_distrito"]["mediana_bruta"]),
    ("heterocedasticidad", "Breusch-Pagan detecta heterocedasticidad",
     lambda f: E(f)["ols"]["diagnostico"]["heterocedasticidad"]),
    ("residuos_no_normales", "Jarque-Bera rechaza normalidad de residuos",
     lambda f: E(f)["ols"]["diagnostico"]["jarque_bera_p"] < f["meta"]["alfa"]),
    ("vif_bajo", "Sin dummies de distrito ni de mes, ningún VIF supera 5", lambda f: E(f)["ols"]["n_vif_mayor_5"] == 0),
    ("vif_distancia_alto_con_distritos", "Con la matriz completa el VIF de la distancia supera 5",
     lambda f: E(f)["ols"]["vif_completo_distancia"] > 5),
    ("ols_sensibilidad_signos", "Los signos significativos se mantienen con outliers",
     lambda f: E(f)["ols"]["sensibilidad_con_outliers"]["signos_coinciden"]),
    ("ols_sin_influyentes_signos", "Los signos significativos se mantienen sin observaciones influyentes",
     lambda f: E(f)["ols"]["sensibilidad_sin_influyentes"]["signos_coinciden"]),
    ("ols_sin_mensuales_signos", "Sin tarifas mensuales los coeficientes clave conservan el signo",
     lambda f: all(c["mismo_signo"] for c in E(f)["ols"]["sensibilidad_sin_mensuales"]["coeficientes"].values())),
    ("sin_mensuales_mismo_top2", "Sin tarifas mensuales los dos distritos más caros son los mismos",
     lambda f: D(f)["sin_mensuales"]["mismo_top2"]),
    ("ml_supera_base", "El modelo ML supera a la línea base en validación agrupada",
     lambda f: (not E(f)["ml"]["ejecutado"]) or (E(f)["ml"]["cv"]["r2_media"] > E(f)["ml"]["cv_base"]["r2_media"]
                                                 and E(f)["ml"]["cv"]["mae_log_media"] < E(f)["ml"]["cv_base"]["mae_log_media"])),
    ("ml_aleatorio_optimista", "La partición aleatoria da un R² mayor que la agrupada por anfitrión",
     lambda f: (not E(f)["ml"]["ejecutado"]) or E(f)["ml"]["caida_r2_aleatorio_a_grupos"] > 0),
    ("ml_hgb_supera_lineal_misma_muestra", "En las mismas filas, particiones y variables, el HGB supera al lineal",
     lambda f: (not E(f)["ml"]["ejecutado"]) or (
         E(f)["ml"]["comparacion_misma_muestra"]["hgb_vars_ols"]["r2_media"]
         > E(f)["ml"]["comparacion_misma_muestra"]["lineal_vars_ols"]["r2_media"])),
    ("capacidad_creciente", "La mediana de precio crece con cada tramo de capacidad",
     lambda f: all(a["mediana"] < b["mediana"] for a, b in zip(D(f)["por_capacidad"], D(f)["por_capacidad"][1:]))),
    ("multianuncio_mayoria", "La mayoría de anuncios son de anfitriones multi-anuncio",
     lambda f: D(f)["hosts"]["pct_anuncios_multianuncio"] > 50),
    ("mayoria_hosts_un_anuncio", "La mayoría de anfitriones tiene un solo anuncio",
     lambda f: D(f)["hosts"]["pct_hosts_un_anuncio"] > 50),
    ("no_disponible_mayor", "El % de días no disponibles supera la ocupación estimada por Inside Airbnb",
     lambda f: D(f)["ocupacion_ingresos"]["no_disponible_media"]
     > D(f)["ocupacion_ingresos"]["ocupacion_inside_airbnb_media"]),
    ("ingreso_hipotetico_mayor", "El ingreso hipotético propio supera la estimación de Inside Airbnb (mismas filas)",
     lambda f: D(f)["ocupacion_ingresos"]["cociente_medianas"] > 1),
    ("capacidad_mayor_correlacion", "La capacidad es la variable más correlacionada con el precio",
     lambda f: D(f)["correlaciones"]["con_precio"][0]["variable"] == "accommodates"),
    ("salamanca_o_centro_top", "El distrito más caro por mediana coincide en descriptivo y Kruskal",
     lambda f: D(f)["por_distrito"][0]["grupo"] == E(f)["kruskal_distrito"]["extra"]["grupo_mas_caro"]),
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
    salida[config.RUTA_DICCIONARIO] = diccionario.generar_markdown(filas_diccionario, formato.pct1)
    return salida


def escribir_documentos(findings: dict[str, Any], df: pd.DataFrame) -> list[Path]:
    """Renderiza y escribe README, informe y diccionario con saltos de línea LF."""
    documentos = render_todos(findings, diccionario.tabla(df))
    for ruta, texto in documentos.items():
        with open(ruta, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto)
    return list(documentos)
