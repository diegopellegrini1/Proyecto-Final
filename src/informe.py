"""Generación de README, informe y diccionario desde ``findings.json`` mediante plantillas Jinja2.

Toda cifra de los documentos se inserta con ``cifra(ruta, formato)``, que la envuelve en un
marcador invisible ``<!--f:ruta:formato-->valor<!--/f-->``. El verificador independiente
(``scripts/verificar_consistencia.py``) parsea esos marcadores y recalcula cada valor desde
``findings.json``. Las frases cualitativas dependientes de los datos se registran en
``AFIRMACIONES`` y se evalúan antes de escribir ``findings.json`` (M3).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import jinja2
import pandas as pd

from src import config, diccionario, formato
from src.errores import ErrorInforme

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

# (identificador, descripción, comprobación). Cada frase cualitativa de las plantillas
# que depende del sentido de un resultado tiene aquí su comprobación.
AFIRMACIONES: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
    ("dataset_cumple_umbrales", "El dataset cumple los umbrales mínimos",
     lambda f: all(f["dataset"]["requisitos"][k] for k in ("cumple_filas", "cumple_columnas", "cumple_tipos"))),
    ("moneda_eur", "La moneda real de las cotizaciones es EUR", lambda f: f["moneda"]["moneda"] == "EUR"),
    ("precio_es_cotizacion", "price coincide con la cotización en casi todos los casos",
     lambda f: f["precio_naturaleza"]["pct_precio_igual_cotizacion"] > 95),
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
    ("superhost_no_significativo_ols", "Superhost no es significativo en OLS al controlar",
     lambda f: not E(f)["ols"]["coef_superhost"]["significativo"]),
    ("superhost_hc3_significativo", "Con HC3 (sin agrupar) el coeficiente de superhost sí es significativo",
     lambda f: E(f)["ols"]["coef_superhost"]["p_hc3"] < f["meta"]["alfa"]),
    ("sensibilidad_misma_decision", "Con un anuncio por anfitrión se mantienen las decisiones",
     lambda f: E(f)["sensibilidad_un_anuncio_por_host"]["misma_decision_superhost"]
     and E(f)["sensibilidad_un_anuncio_por_host"]["misma_decision_tipo"]),
    ("entero_mas_caro", "El alojamiento entero es el tipo más caro",
     lambda f: E(f)["kruskal_tipo"]["extra"]["grupo_mas_caro"] == "Alojamiento entero"),
    ("kruskal_tipo_grande", "Kruskal por tipo: significativo y efecto grande",
     lambda f: E(f)["kruskal_tipo"]["significativo"] and E(f)["kruskal_tipo"]["tamano_efecto"]["interpretacion"] == "grande"),
    ("kruskal_distrito_significativo", "Kruskal por distrito significativo con efecto al menos pequeño",
     lambda f: E(f)["kruskal_distrito"]["significativo"]
     and E(f)["kruskal_distrito"]["tamano_efecto"]["interpretacion"] != "despreciable"),
    ("posthoc_tipo_todos", "Todos los pares de tipos difieren",
     lambda f: E(f)["posthoc_tipo"]["n_pares_significativos"] == E(f)["posthoc_tipo"]["n_pares"]),
    ("posthoc_distrito_mayoria", "La mayoría de pares de distritos difieren",
     lambda f: E(f)["posthoc_distrito"]["n_pares_significativos"] > E(f)["posthoc_distrito"]["n_pares"] / 2),
    ("chi2_tipo_superhost_despreciable", "Chi² tipo × superhost: significativo pero despreciable",
     lambda f: E(f)["chi2_tipo_superhost"]["significativo"]
     and E(f)["chi2_tipo_superhost"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    ("chi2_multi_tipo_despreciable", "Chi² multi-anuncio × tipo: significativo pero despreciable",
     lambda f: E(f)["chi2_multi_tipo"]["significativo"]
     and E(f)["chi2_multi_tipo"]["tamano_efecto"]["interpretacion"] == "despreciable"),
    ("chi2_distrito_tipo_mayor", "Chi² distrito × tipo: asociación no despreciable",
     lambda f: E(f)["chi2_distrito_tipo"]["significativo"]
     and E(f)["chi2_distrito_tipo"]["tamano_efecto"]["interpretacion"] != "despreciable"),
    ("ic_entero_no_solapa_privada", "Los IC de la mediana de entero y privada no se solapan",
     lambda f: E(f)["bootstrap"]["tipo"][0]["ic_inf"] > max(g["ic_sup"] for g in E(f)["bootstrap"]["tipo"][1:])),
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
    ("heterocedasticidad", "Breusch-Pagan detecta heterocedasticidad",
     lambda f: E(f)["ols"]["diagnostico"]["heterocedasticidad"]),
    ("residuos_no_normales", "Jarque-Bera rechaza normalidad de residuos",
     lambda f: E(f)["ols"]["diagnostico"]["jarque_bera_p"] < f["meta"]["alfa"]),
    ("vif_bajo", "Ningún VIF supera 5", lambda f: E(f)["ols"]["n_vif_mayor_5"] == 0),
    ("ols_sensibilidad_signos", "Los signos significativos se mantienen con outliers",
     lambda f: E(f)["ols"]["sensibilidad_con_outliers"]["signos_coinciden"]),
    ("ml_supera_base", "El modelo ML supera a la línea base",
     lambda f: (not E(f)["ml"]["ejecutado"]) or (E(f)["ml"]["r2_test"] > E(f)["ml"]["r2_base"]
                                                 and E(f)["ml"]["mae_log_test"] < E(f)["ml"]["mae_log_base"])),
    ("ml_mejor_que_ols", "El R² en test del ML supera al R² ajustado del OLS",
     lambda f: (not E(f)["ml"]["ejecutado"]) or E(f)["ml"]["r2_test"] > E(f)["ols"]["r2_ajustado"]),
    ("capacidad_creciente", "La mediana de precio crece con cada tramo de capacidad",
     lambda f: all(a["mediana"] < b["mediana"] for a, b in zip(D(f)["por_capacidad"], D(f)["por_capacidad"][1:]))),
    ("multianuncio_mayoria", "La mayoría de anuncios son de anfitriones multi-anuncio",
     lambda f: D(f)["hosts"]["pct_anuncios_multianuncio"] > 50),
    ("mayoria_hosts_un_anuncio", "La mayoría de anfitriones tiene un solo anuncio",
     lambda f: D(f)["hosts"]["pct_hosts_un_anuncio"] > 50),
    ("ocupacion_propia_mayor", "La ocupación propia supera a la de Inside Airbnb",
     lambda f: D(f)["ocupacion_ingresos"]["ocupacion_estimada_media"]
     > D(f)["ocupacion_ingresos"]["ocupacion_inside_airbnb_media"]),
    ("ingresos_propios_mayores", "La estimación propia de ingresos supera a la de Inside Airbnb",
     lambda f: D(f)["ocupacion_ingresos"]["cociente_medianas"] > 1),
    ("capacidad_mayor_correlacion", "La capacidad es la variable más correlacionada con el precio",
     lambda f: D(f)["correlaciones"]["con_precio"][0]["variable"] == "accommodates"),
    ("salamanca_o_centro_top", "El distrito más caro por mediana coincide en descriptivo y Kruskal",
     lambda f: D(f)["por_distrito"][0]["grupo"] == E(f)["kruskal_distrito"]["extra"]["grupo_mas_caro"]),
]


def evaluar_afirmaciones(findings: dict[str, Any]) -> dict[str, bool]:
    """Evalúa todas las afirmaciones. Lanza ``ErrorInforme`` si alguna es falsa o no evaluable."""
    resultado: dict[str, bool] = {}
    for ident, descripcion, comprobar in AFIRMACIONES:
        try:
            resultado[ident] = bool(comprobar(findings))
        except (KeyError, IndexError, TypeError, StopIteration) as exc:
            raise ErrorInforme(f"Afirmación «{ident}» no evaluable: {exc}") from exc
    falsas = [k for k, v in resultado.items() if not v]
    if falsas:
        raise ErrorInforme("Afirmaciones cualitativas falsas con estos datos (revisar plantillas): " + ", ".join(falsas))
    return resultado


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

    def afirmacion(ident: str) -> str:
        if not findings.get("afirmaciones", {}).get(ident):
            raise ErrorInforme(f"La plantilla usa la afirmación «{ident}», que no es verdadera o no existe.")
        return ""

    env.globals.update(cifra=cifra, valor=valor, afirmacion=afirmacion, f=findings)
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
