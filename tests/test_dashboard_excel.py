"""Verificación del dashboard de Excel (dashboard/Dashboard_Airbnb_Espana.xlsx).

- Estructura con openpyxl: hojas, Tabla de datos, validaciones, gráficos, sin datos personales.
- Ningún valor cacheado es un error de Excel.
- Valores cacheados = cálculo independiente en pandas (sobre el Parquet, sin usar src.dashboard_excel).
- Las fórmulas se RECALCULAN con un evaluador propio (traduce la sintaxis de Excel de este libro a numpy):
  con la selección por defecto deben dar el valor cacheado y, cambiando los selectores, lo que calcula pandas.
- Recálculo real con LibreOffice solo si ``soffice`` está instalado (si no, el test se omite y lo dice).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile

import numpy as np
import openpyxl
import pandas as pd
import pytest
from openpyxl.worksheet.formula import ArrayFormula

from src import config

RUTA = config.RUTA_DASHBOARD_XLSX
HOJAS = ["Inicio", "Dashboard", "Comparador", "Renta y precio", "Resumen ciudades", "Datos", "Cálculos"]
COLUMNAS_ESPERADAS = [
    "Ciudad", "Municipio (INE)", "Barrio o municipio (Inside Airbnb)", "Distrito (Inside Airbnb)",
    "Código de distrito censal (INE)", "Tipo de alojamiento", "Capacidad (personas)", "Tramo de capacidad",
    "Precio por noche (€)", "Tramo de precio", "Superhost", "Anfitrión con varios anuncios",
    f"Estancia mínima ≥ {config.MIN_NIGHTS_LARGA_ESTANCIA} noches", "Tipo de licencia", "Valoración (1-5)",
    "Nº de reseñas", "Renta neta por persona del distrito (€)", "Índice de Gini del distrito",
    "Renta neta por persona del municipio (€)",
]
ERRORES_EXCEL = {"#REF!", "#NAME?", "#DIV/0!", "#VALUE!", "#N/A", "#NUM!", "#NULL!"}
PROHIBIDAS = re.compile(r"\b(LET|LAMBDA|FILTER|UNIQUE|SORT|SORTBY|XLOOKUP|SEQUENCE)\(|_xlfn|_xlws")
TOL = 1e-6
# Celdas de las tarjetas KPI del Dashboard y de los resultados del Comparador.
KPI = {"n": "B8", "mediana": "C8", "media": "D8", "p25": "E8", "p75": "F8", "pct_superhost": "G8",
       "pct_multianuncio": "H8", "renta_media": "I8"}
CMP = {"n": "C9", "mediana": "C10", "p25": "C11", "p75": "C12", "media": "C13"}
COMBINACIONES = [("Madrid", "Alojamiento entero"), ("Sevilla", "Habitación privada"),
                 ("Todas", "Habitación privada"), ("Barcelona", "Todos"), ("Menorca", "Habitación de hotel")]

pytestmark = pytest.mark.skipif(not RUTA.exists(), reason="falta el dashboard: ejecuta python run_pipeline.py")


# --------------------------------------------------------------------------
# Cálculo independiente en pandas (no importa src.dashboard_excel)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pandas_ref() -> pd.DataFrame:
    df = pd.read_parquet(config.RUTA_PARQUET)
    cp = df[df["price"].notna() & (df["price"] > 0)].copy()
    cp["nombre"] = cp["ciudad"].map({k: c.nombre for k, c in config.CIUDADES_ESPANA.items()})
    cap = pd.Series("", index=cp.index)
    for etq, lo, hi in config.TRAMOS_CAPACIDAD_EXCEL:
        cap[(cp["accommodates"] >= lo) & ((cp["accommodates"] <= hi) if hi else True)] = etq
    cp["tramo_cap"] = cap
    return cp


def esperado(cp: pd.DataFrame, ciudad: str, tipo: str, cap: str | None = None) -> dict:
    m = pd.Series(True, index=cp.index)
    if ciudad != "Todas":
        m &= cp["nombre"] == ciudad
    if tipo != "Todos":
        m &= cp["tipo_habitacion"] == tipo
    if cap is not None:
        m &= cp["tramo_cap"] == cap
    s = cp[m]
    if s.empty:
        return {"n": 0, **{k: "" for k in KPI if k != "n"}}
    return {"n": len(s), "mediana": s["price"].median(), "media": s["price"].mean(),
            "p25": s["price"].quantile(0.25), "p75": s["price"].quantile(0.75),
            "pct_superhost": s["host_is_superhost"].mean(), "pct_multianuncio": s["es_multianuncio"].mean(),
            "renta_media": s[config.RENTA_PRINCIPAL].mean()}


def igual(a, b) -> bool:
    vacio = lambda x: x is None or x == "" or (isinstance(x, float) and np.isnan(x))  # noqa: E731
    if vacio(a) or vacio(b):
        return vacio(a) and vacio(b)
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))


# --------------------------------------------------------------------------
# Evaluador de las fórmulas del libro (subconjunto de Excel que usa el dashboard)
# --------------------------------------------------------------------------
OPERADORES_CRITERIO = (">=", "<=", "<>", ">", "<", "=")


def _comodines(texto: str) -> re.Pattern:
    """Patrón de un criterio de texto de Excel: ``*`` y ``?`` son comodines y ``~`` escapa el siguiente carácter."""
    partes, i = [], 0
    while i < len(texto):
        ch = texto[i]
        if ch == "~" and i + 1 < len(texto):
            partes.append(re.escape(texto[i + 1]))
            i += 2
            continue
        partes.append(".*" if ch == "*" else "." if ch == "?" else re.escape(ch))
        i += 1
    return re.compile("".join(partes), re.IGNORECASE | re.DOTALL)


def _a_numero(texto: str) -> float | None:
    try:
        return float(texto.strip())
    except ValueError:
        return None


def _criterio(arr: np.ndarray, crit) -> np.ndarray:
    """Celdas de ``arr`` que cumplen ``crit`` con la semántica de COUNTIFS/AVERAGEIFS de Excel.

    Un criterio de texto que empieza por ``>=``, ``<=``, ``<>``, ``>``, ``<`` o ``=`` es una COMPARACIÓN (el
    resto es el operando), no un texto literal: «< 50 €» significa «menor que " 50 €"» y no cuenta las celdas
    que contienen «< 50 €». Sin operador, es igualdad sin distinguir mayúsculas y con comodines ``*``, ``?`` y ``~``.
    Los números se comparan solo con números y el texto solo con texto (las celdas vacías no cumplen).
    """
    if not isinstance(crit, str):
        return np.array([isinstance(x, (int, float)) and not isinstance(x, bool) and x == crit for x in arr])
    op = next((o for o in OPERADORES_CRITERIO if crit.startswith(o)), "")
    operando = crit[len(op):]
    num = _a_numero(operando)
    es_texto = lambda x: isinstance(x, str)  # noqa: E731
    es_num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and not np.isnan(x)  # noqa: E731
    if op in ("", "="):
        if num is not None:
            # Excel también convierte el texto con aspecto de número (p. ej. el código «0801902»).
            return np.array([(es_num(x) and x == num) or (es_texto(x) and _a_numero(x) == num) for x in arr])
        patron = _comodines(operando)
        return np.array([es_texto(x) and patron.fullmatch(x) is not None for x in arr])
    if op == "<>":
        if num is not None:
            return np.array([not (es_num(x) and x == num) for x in arr])
        patron = _comodines(operando)
        return np.array([not (es_texto(x) and patron.fullmatch(x) is not None) for x in arr])
    comparar = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal}[op]
    if num is not None:
        return np.array([es_num(x) and bool(comparar(x, num)) for x in arr])
    o = operando.lower()
    return np.array([es_texto(x) and bool(comparar(x.lower(), o)) for x in arr])


def _mascara(pares) -> np.ndarray:
    m = None
    for rango, crit in zip(pares[::2], pares[1::2]):
        c = _criterio(rango, crit)
        m = c if m is None else m & c
    return m


def _validos(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[~np.isnan(x)]


FUNCIONES = {
    "F_COUNTIFS": lambda *a: np.float64(_mascara(a).sum()),
    "F_AVERAGEIFS": lambda avg, *a: (lambda v: np.float64(v.mean()) if v.size else np.float64("nan"))(
        _validos(np.asarray(avg, dtype=float)[_mascara(a)])),
    "F_MEDIAN": lambda x: np.float64(np.median(v)) if (v := _validos(x)).size else np.float64("nan"),
    "F_PERCENTILE": lambda x, q: np.float64(np.percentile(v, q * 100)) if (v := _validos(x)).size
    else np.float64("nan"),
    "F_IF": lambda c, a, b=False: (np.where(np.asarray(c).astype(bool), a, np.nan) if np.ndim(c)
                                   else (a if c else b)),
    "F_IFERROR": lambda x, alt: alt if isinstance(x, float) and not np.isfinite(x) else x,
    "F_CONCATENATE": lambda *a: "".join(str(v) for v in a),
    "F_LEN": lambda x: len(str(x)),
}


def traducir(formula: str) -> str:
    """Traduce una fórmula de Excel de este libro a una expresión de Python."""
    cadenas: list[str] = []

    def guardar(m):
        cadenas.append(m.group(1))
        return f"__S{len(cadenas) - 1}__"

    f = re.sub(r'"([^"]*)"', guardar, formula.lstrip("="))
    f = f.replace("<>", "!=")
    f = re.sub(r"(?<![<>!=])=(?!=)", "==", f)
    f = re.sub(r"\b((?:d|sel|crit|cmp|lista)[A-Z]\w*)\b", r'_N("\1")', f)
    f = re.sub(r"\$?\b([A-Z]{1,3})\$?([0-9]+)\b", r'_C("\1\2")', f)
    f = re.sub(r"\b(COUNTIFS|AVERAGEIFS|MEDIAN|PERCENTILE|IFERROR|IF|CONCATENATE|LEN)\(", r"F_\1(", f)
    if re.search(r"\b[A-Z]{2,}\(", f):
        raise AssertionError(f"función no soportada por el evaluador: {formula}")
    return re.sub(r"__S(\d+)__", lambda m: repr(cadenas[int(m.group(1))]), f)


class Evaluador:
    def __init__(self, wb, sobrescribir: dict | None = None):
        self.wb, self.sob, self._rangos = wb, sobrescribir or {}, {}

    def formula(self, hoja: str, coord: str) -> str | None:
        v = self.wb[hoja][coord].value
        if isinstance(v, ArrayFormula):
            v = v.text
        return v if isinstance(v, str) and v.startswith("=") else None

    def nombre(self, n: str):
        if n in self.sob:
            return self.sob[n]
        hoja, ref = next(iter(self.wb.defined_names[n].destinations))
        if ":" in ref:
            if n not in self._rangos:
                vals = [fila[0].value for fila in self.wb[hoja][ref.replace("$", "")]]
                numerico = all(v is None or isinstance(v, (int, float)) for v in vals)
                self._rangos[n] = (np.array([np.nan if v is None else v for v in vals], dtype=float) if numerico
                                   else np.array(vals, dtype=object))
            return self._rangos[n]
        return self.celda(hoja, ref.replace("$", ""))

    def celda(self, hoja: str, coord: str):
        f = self.formula(hoja, coord)
        return self.evaluar(f, hoja) if f else self.wb[hoja][coord].value

    def evaluar(self, formula: str, hoja: str):
        entorno = {"_N": self.nombre, "_C": lambda c: self.celda(hoja, c), **FUNCIONES}
        with np.errstate(all="ignore"):
            return eval(traducir(formula), entorno)  # noqa: S307 (expresión generada por traducir)


@pytest.fixture(scope="module")
def wb():
    return openpyxl.load_workbook(RUTA)


@pytest.fixture(scope="module")
def cache() -> dict:
    wbv = openpyxl.load_workbook(RUTA, data_only=True, read_only=True)
    salida = {}
    for ws in wbv.worksheets:
        for fila in ws.iter_rows():
            for c in fila:
                if getattr(c, "value", None) is not None:
                    salida[(ws.title, c.coordinate)] = c.value
    wbv.close()
    return salida


@pytest.fixture(scope="module")
def findings() -> dict:
    return json.loads(config.RUTA_FINDINGS.read_text(encoding="utf-8"))


def celdas_formula(wb):
    for ws in wb.worksheets:
        if ws.title == "Datos":
            continue
        for fila in ws.iter_rows():
            for c in fila:
                v = c.value.text if isinstance(c.value, ArrayFormula) else c.value
                if isinstance(v, str) and v.startswith("="):
                    yield ws.title, c.coordinate, v


# --------------------------------------------------------------------------
# (a) Estructura
# --------------------------------------------------------------------------
def test_hojas_tabla_y_columnas(wb, pandas_ref):
    assert wb.sheetnames == HOJAS
    tabla = wb["Datos"].tables["TablaDatos"]
    assert [c.name for c in tabla.tableColumns] == COLUMNAS_ESPERADAS
    assert tabla.ref == f"A1:S{len(pandas_ref) + 1}"
    assert tabla.autoFilter is not None
    cab = " ".join(COLUMNAS_ESPERADAS).lower()
    for prohibida in ("host_id", "host_name", "url", "foto", "picture", "latitud", "longitud", "nombre del anfitri"):
        assert prohibida not in cab
    # Ninguna celda de Datos parece una URL (sin datos personales ni enlaces a perfiles)
    assert not any(isinstance(v, str) and "http" in v for fila in wb["Datos"].iter_rows(max_row=2000, values_only=True)
                   for v in fila)


def test_validaciones_de_los_selectores(wb):
    def validaciones(hoja):
        return {str(dv.sqref): dv for dv in wb[hoja].data_validations.dataValidation}
    d = validaciones("Dashboard")
    assert set(d) == {"C4", "C5"}
    assert d["C4"].type == "list" and d["C4"].formula1 == "listaCiudades"
    assert d["C5"].formula1 == "listaTipos"
    c = validaciones("Comparador")
    assert set(c) == {"C4", "C5", "C6"} and c["C6"].formula1 == "listaCapacidades"
    ev = Evaluador(wb)
    assert list(ev.nombre("listaCiudades")) == ["Todas", *sorted(config.CIUDAD_POR_NOMBRE,
                                                                  key=lambda s: s.replace("á", "a"))]
    assert list(ev.nombre("listaTipos")) == ["Todos", *config.TRADUCCION_TIPO_HABITACION.values()]


def test_graficos_nativos_con_titulo_y_texto_alternativo(wb):
    assert len(wb["Dashboard"]._charts) == 5
    with zipfile.ZipFile(RUTA) as z:
        charts = [n for n in z.namelist() if re.match(r"xl/charts/chart\d+\.xml$", n)]
        assert len(charts) == 5
        for n in charts:
            xml = z.read(n).decode("utf-8")
            assert "<c:title>" in xml and "<c:valAx>" in xml
        dibujo = next(z.read(n).decode("utf-8") for n in z.namelist() if re.match(r"xl/drawings/drawing\d+\.xml$", n))
        assert dibujo.count('descr="') == 5
        libro = z.read("xl/workbook.xml").decode("utf-8")
        assert 'fullCalcOnLoad="1"' in libro
        assert not any(n.endswith(".bin") or "vbaProject" in n for n in z.namelist())  # sin macros


def test_evaluador_usa_la_semantica_de_criterios_de_excel():
    """El evaluador interpreta «<», «>», «=» iniciales como operadores (así se habría detectado H1)."""
    textos = np.array(["< 50 €", "Menos de 50 €", "50-100 €", None], dtype=object)
    assert _criterio(textos, "< 50 €").sum() == 0                      # «<» es un operador, no texto literal
    assert _criterio(textos, "Menos de 50 €").tolist() == [False, True, False, False]
    assert _criterio(textos, "=< 50 €").tolist() == [True, False, False, False]
    assert _criterio(textos, "*").tolist() == [True, True, True, False]
    assert _criterio(textos, "menos*").tolist() == [False, True, False, False]
    assert _criterio(textos, "<>50-100 €").tolist() == [True, True, False, True]
    numeros = np.array([10.0, 50.0, 120.0, np.nan])
    assert _criterio(numeros, "<50").tolist() == [True, False, False, False]
    assert _criterio(numeros, ">=50").tolist() == [False, True, True, False]
    assert _criterio(np.array(["0801902", "0801903"], dtype=object), "0801902").tolist() == [True, False]


def _llamadas(formula: str, funcion: str) -> list[list[str]]:
    """Argumentos (texto) de cada llamada a ``funcion`` en ``formula``, respetando paréntesis y comillas."""
    salida = []
    for m in re.finditer(rf"\b{funcion}\(", formula):
        i, nivel, actual, args, en_cadena = m.end(), 1, "", [], False
        while i < len(formula) and nivel:
            ch = formula[i]
            if ch == '"':
                en_cadena = not en_cadena
            elif not en_cadena and ch == "(":
                nivel += 1
            elif not en_cadena and ch == ")":
                nivel -= 1
                if nivel == 0:
                    break
            if not en_cadena and ch == "," and nivel == 1:
                args.append(actual)
                actual = ""
            else:
                actual += ch
            i += 1
        salida.append([*args, actual])
    return salida


def _empieza_por_operador(v) -> bool:
    return isinstance(v, str) and v[:1] in ("<", ">", "=")


def test_ningun_criterio_empieza_por_operador(wb):
    """Ninguna celda o valor usado como criterio de COUNTIFS/AVERAGEIFS (ni como etiqueta comparada en
    MEDIAN(IF())/PERCENTILE(IF())) empieza por «<», «>» o «=» (H1: «< 50 €» contaba 0 en Excel)."""
    ev = Evaluador(wb)
    revisados = 0
    for hoja, coord, f in celdas_formula(wb):
        criterios = [a for args in _llamadas(f, "COUNTIFS") for a in args[1::2]]
        criterios += [a for args in _llamadas(f, "AVERAGEIFS") for a in args[2::2]]
        criterios += re.findall(r"\(d\w+=(\$?[A-Z]{1,3}\$?\d+|[a-z]+[A-Z]\w*)\)", f)
        for crit in criterios:
            valor = ev.evaluar("=" + crit.strip(), hoja)
            assert not _empieza_por_operador(valor), (hoja, coord, crit, valor)
            revisados += 1
    assert revisados > 300
    # Los criterios que dependen de los desplegables toman cualquier valor de sus listas.
    for lista in ("listaCiudades", "listaTipos", "listaCapacidades"):
        assert not any(_empieza_por_operador(v) for v in ev.nombre(lista)), lista
    # Y ninguna etiqueta de las columnas de texto usadas como rango de criterio.
    for nombre in ("dTramoPrecio", "dTramoCap", "dCiudad", "dTipo"):
        assert not any(_empieza_por_operador(v) for v in set(ev.nombre(nombre))), nombre


def test_tramo_mas_barato_cuenta_lo_mismo_que_pandas(wb, pandas_ref):
    """La primera barra del gráfico de tramos de precio (H1) con criterios de Excel frente a pandas."""
    r = _bloque(wb, "Anuncios por tramo de precio")
    assert wb["Cálculos"].cell(r, 1).value == f"Menos de {config.LIMITES_TRAMOS_PRECIO_EXCEL[0]} €"
    for ciudad, tipo in [("Todas", "Todos"), *COMBINACIONES]:
        ev = Evaluador(wb, {"selCiudad": ciudad, "selTipo": tipo})
        m = pandas_ref["price"] < config.LIMITES_TRAMOS_PRECIO_EXCEL[0]
        if ciudad != "Todas":
            m &= pandas_ref["nombre"] == ciudad
        if tipo != "Todos":
            m &= pandas_ref["tipo_habitacion"] == tipo
        assert ev.celda("Cálculos", f"B{r}") == int(m.sum()), (ciudad, tipo)
    assert Evaluador(wb).celda("Cálculos", f"B{r}") > 0


def test_formulas_compatibles_excel_2016(wb):
    formulas = list(celdas_formula(wb))
    assert len(formulas) > 300
    for hoja, coord, f in formulas:
        assert not PROHIBIDAS.search(f), (hoja, coord, f)


def test_tamano_menor_25_mb():
    assert RUTA.stat().st_size < config.LIMITE_SUBIDA_BYTES
    assert RUTA.stat().st_size < 10_000_000


# --------------------------------------------------------------------------
# (b) Ningún valor cacheado es un error
# --------------------------------------------------------------------------
def test_sin_errores_cacheados(cache):
    errores = {k: v for k, v in cache.items() if isinstance(v, str) and v.strip() in ERRORES_EXCEL}
    assert not errores


# --------------------------------------------------------------------------
# (c) Valores cacheados y fórmulas frente a pandas
# --------------------------------------------------------------------------
def test_kpis_cacheados_estado_por_defecto(cache, pandas_ref):
    assert cache[("Dashboard", "C4")] == "Todas" and cache[("Dashboard", "C5")] == "Todos"
    esp = esperado(pandas_ref, "Todas", "Todos")
    for k, coord in KPI.items():
        assert igual(cache[("Dashboard", coord)], esp[k]), (k, cache[("Dashboard", coord)], esp[k])


def test_cada_formula_recalculada_coincide_con_su_cache(wb, cache):
    ev = Evaluador(wb)
    n = 0
    for hoja, coord, f in celdas_formula(wb):
        valor = ev.evaluar(f, hoja)
        assert igual(valor, cache.get((hoja, coord), "")), (hoja, coord, f, valor, cache.get((hoja, coord)))
        n += 1
    assert n > 300


@pytest.mark.parametrize("ciudad,tipo", COMBINACIONES)
def test_kpis_responden_a_los_selectores(wb, pandas_ref, ciudad, tipo):
    ev = Evaluador(wb, {"selCiudad": ciudad, "selTipo": tipo})
    esp = esperado(pandas_ref, ciudad, tipo)
    for k, coord in KPI.items():
        assert igual(ev.celda("Dashboard", coord), esp[k]), (ciudad, tipo, k)


def _bloque(wb, titulo: str) -> int:
    """Primera fila de datos de la tabla auxiliar cuyo título empieza por ``titulo``."""
    for fila in wb["Cálculos"].iter_rows(min_col=1, max_col=1):
        if isinstance(fila[0].value, str) and fila[0].value.startswith(titulo):
            return fila[0].row + 2
    raise AssertionError(titulo)


@pytest.mark.parametrize("ciudad,tipo", [("Todas", "Todos"), *COMBINACIONES[:3]])
def test_tablas_auxiliares_frente_a_pandas(wb, pandas_ref, ciudad, tipo):
    ev = Evaluador(wb, {"selCiudad": ciudad, "selTipo": tipo})
    ws = wb["Cálculos"]
    r = _bloque(wb, "Mediana del precio por ciudad")
    ciudades = [ws.cell(r + i, 1).value for i in range(len(config.CIUDADES_ESPANA))]
    assert sorted(ciudades) == sorted(config.CIUDAD_POR_NOMBRE)
    for i, c in enumerate(ciudades):
        esp = esperado(pandas_ref, c, tipo)
        assert igual(ev.celda("Cálculos", f"B{r + i}"), esp["n"])
        assert igual(ev.celda("Cálculos", f"C{r + i}"), esp["mediana"])
    r = _bloque(wb, "Mediana del precio por tipo")
    for i, t in enumerate(config.TRADUCCION_TIPO_HABITACION.values()):
        esp = esperado(pandas_ref, ciudad, t)
        assert ws.cell(r + i, 1).value == t
        assert igual(ev.celda("Cálculos", f"B{r + i}"), esp["n"])
        assert igual(ev.celda("Cálculos", f"C{r + i}"), esp["mediana"])
    r = _bloque(wb, "Anuncios y mediana por capacidad")
    for i, (etq, _, _) in enumerate(config.TRAMOS_CAPACIDAD_EXCEL):
        esp = esperado(pandas_ref, ciudad, tipo, etq)
        assert igual(ev.celda("Cálculos", f"B{r + i}"), esp["n"])
        assert igual(ev.celda("Cálculos", f"C{r + i}"), esp["mediana"])
    r = _bloque(wb, "Anuncios por tramo de precio")
    bordes = [-np.inf, *config.LIMITES_TRAMOS_PRECIO_EXCEL, np.inf]
    base = esperado(pandas_ref, ciudad, tipo)["n"]
    total = 0
    for i in range(len(bordes) - 1):
        m = pandas_ref["price"].ge(bordes[i]) & pandas_ref["price"].lt(bordes[i + 1])
        if ciudad != "Todas":
            m &= pandas_ref["nombre"] == ciudad
        if tipo != "Todos":
            m &= pandas_ref["tipo_habitacion"] == tipo
        v = ev.celda("Cálculos", f"B{r + i}")
        assert igual(v, int(m.sum()))
        total += v
    assert total == base


def test_distritos_de_la_dispersion(wb, cache, pandas_ref, findings):
    ws = wb["Cálculos"]
    r = _bloque(wb, "Distritos censales con al menos")
    filas = []
    while ws.cell(r, 1).value:
        filas.append((ws.cell(r, 1).value, ws.cell(r, 2).value, cache[("Cálculos", f"D{r}")],
                      cache[("Cálculos", f"E{r}")], cache[("Cálculos", f"F{r}")]))
        r += 1
    assert len(filas) == findings["estadistica"]["renta_precio"]["ecologico_global"]["n_zonas"]
    g = (pandas_ref.dropna(subset=["codigo_distrito_ine", config.RENTA_PRINCIPAL])
         .groupby(["nombre", "codigo_distrito_ine"])
         .agg(n=("price", "size"), renta=(config.RENTA_PRINCIPAL, "mean"), mediana=("price", "median")))
    g = g[g["n"] >= config.MIN_N_GRUPO]
    assert len(g) == len(filas)
    for ciudad, codigo, n, renta, mediana in filas:
        e = g.loc[(ciudad, codigo)]
        assert n == e["n"] and igual(renta, e["renta"]) and igual(mediana, e["mediana"])


def test_comparador_por_defecto_y_combinaciones(wb, cache, pandas_ref):
    c, t, cap = cache[("Comparador", "C4")], cache[("Comparador", "C5")], cache[("Comparador", "C6")]
    esp = esperado(pandas_ref, c, t, cap)
    for k, coord in CMP.items():
        assert igual(cache[("Comparador", coord)], esp[k]), k
    pequena = None
    for (ciudad, tipo), etq in [(x, e) for x in COMBINACIONES for e, _, _ in config.TRAMOS_CAPACIDAD_EXCEL]:
        ev = Evaluador(wb, {"cmpCiudad": ciudad, "cmpTipo": tipo, "cmpCap": etq})
        esp = esperado(pandas_ref, ciudad, tipo, etq)
        for k, coord in CMP.items():
            assert igual(ev.celda("Comparador", coord), esp[k]), (ciudad, tipo, etq, k)
        aviso = ev.celda("Comparador", "B15")
        if esp["n"] == 0:
            assert aviso.startswith("Sin anuncios")
        elif esp["n"] < config.MIN_N_GRUPO:
            assert "cautela" in aviso
            pequena = (ciudad, tipo, etq)
        else:
            assert aviso == ""
    assert pequena is not None, "ninguna combinación probada tiene n < 30: el aviso no se ha comprobado"


def test_resumen_y_renta_leen_findings(cache, findings):
    por_ciudad = findings["descriptivo"]["por_ciudad"]
    for i, c in enumerate(por_ciudad):
        fila = 5 + i
        assert cache[("Resumen ciudades", f"B{fila}")] == c["grupo"]
        assert cache[("Resumen ciudades", f"D{fila}")] == c["n_anuncios"]
        assert igual(cache[("Resumen ciudades", f"H{fila}")], c["mediana"])
    rp = findings["estadistica"]["renta_precio"]
    assert igual(cache[("Renta y precio", "D6")], rp["global_distrito"]["rho"])
    # España no tiene Holm: la columna da la lectura prudente del p bootstrap (H2), nunca un «Sí» suelto.
    gz = rp["global_distrito"]["cluster_zona"]
    assert cache[("Renta y precio", "H6")] == "No aplica"
    assert cache[("Renta y precio", "I6")] == gz["lectura"].capitalize()
    if gz["en_el_limite"]:
        assert cache[("Renta y precio", "I6")] == "En el límite"
    for i, c in enumerate(rp["por_ciudad"]):
        assert cache[("Renta y precio", f"B{7 + i}")] == c["ciudad"]
        assert igual(cache[("Renta y precio", f"D{7 + i}")], c["distrito"]["rho"])
        assert cache[("Renta y precio", f"I{7 + i}")] == ("Sí" if c["distrito"]["holm"]["significativo_holm"] else "No")


def test_construir_con_otra_seleccion_cachea_lo_que_calcula_pandas(tmp_path, findings):
    """El constructor cachea la selección que se le pide (submuestra para que sea rápido)."""
    from src import dashboard_excel as de

    df = pd.read_parquet(config.RUTA_PARQUET).sample(n=4000, random_state=1)
    datos = de.tabla_datos(df)
    sel = de.Seleccion(ciudad="Sevilla", tipo="Alojamiento entero", cmp_ciudad="Todas", cmp_tipo="Todos",
                       cmp_capacidad="2 personas")
    ruta = tmp_path / "d.xlsx"
    de.construir_libro(datos, findings, ruta, sel)
    w = openpyxl.load_workbook(ruta)
    wv = openpyxl.load_workbook(ruta, data_only=True)
    cp = df[df["price"].notna() & (df["price"] > 0)].copy()
    cp["nombre"] = cp["ciudad"].map({k: c.nombre for k, c in config.CIUDADES_ESPANA.items()})
    cp["tramo_cap"] = np.where(cp["accommodates"] == 2, "2 personas", "otro")
    esp = esperado(cp, "Sevilla", "Alojamiento entero")
    ev = Evaluador(w)
    for k, coord in KPI.items():
        assert igual(wv["Dashboard"][coord].value, esp[k]), k
        assert igual(ev.celda("Dashboard", coord), esp[k]), k
    esp = esperado(cp, "Todas", "Todos", "2 personas")
    for k, coord in CMP.items():
        assert igual(wv["Comparador"][coord].value, esp[k]), k


# --------------------------------------------------------------------------
# (e) Recálculo real con LibreOffice (solo si está instalado)
# --------------------------------------------------------------------------
def _soffice() -> str | None:
    for cand in ("soffice", "soffice.exe", r"C:\Program Files\LibreOffice\program\soffice.exe"):
        ruta = shutil.which(cand) or (cand if cand.endswith(".exe") and __import__("os").path.exists(cand) else None)
        if ruta:
            return ruta
    return None


@pytest.mark.skipif(_soffice() is None, reason="LibreOffice (soffice) no está instalado: no se ha recalculado el "
                                               "libro en un motor de hojas de cálculo real")
def test_recalculo_con_libreoffice(tmp_path, cache):
    copia = tmp_path / "copia.xlsx"
    shutil.copy(RUTA, copia)
    salida = tmp_path / "out"
    subprocess.run([_soffice(), "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(salida), str(copia)],
                   check=True, timeout=600)
    rec = openpyxl.load_workbook(salida / "copia.xlsx", data_only=True)
    for hoja in ("Dashboard", "Comparador"):
        for coord in (KPI if hoja == "Dashboard" else CMP).values():
            assert igual(rec[hoja][coord].value, cache[(hoja, coord)]), (hoja, coord)
