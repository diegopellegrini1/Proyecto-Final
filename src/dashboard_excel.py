"""Dashboard operativo en Excel (``dashboard/Dashboard_Airbnb_Espana.xlsx``) y su vista previa en PNG.

El libro se construye con xlsxwriter a partir del conjunto final (``data/processed``) y de
``reports/findings.json``; no contiene ninguna cifra escrita a mano. Los KPI, el comparador y las tablas
que alimentan los gráficos son FÓRMULAS NATIVAS de Excel (COUNTIFS, AVERAGEIFS y fórmulas matriciales
clásicas MEDIAN(IF(...)) / PERCENTILE(IF(...))) sobre la hoja «Datos», compatibles con Excel 2016 o
posterior y sin macros. Cada fórmula lleva su valor cacheado, calculado aquí con pandas para la selección
por defecto, y el libro pide a Excel un recálculo completo al abrirse (``fullCalcOnLoad``).

Las fórmulas se escriben en sintaxis inglesa (comas y nombres de función en inglés), que es como se
guardan en el formato xlsx; Excel en español las muestra traducidas (CONTAR.SI.CONJUNTO, MEDIANA, SI...).

Uso independiente (sin repetir el pipeline):

    python -m src.dashboard_excel
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src import config, formato

logger = logging.getLogger(__name__)

TODAS = "Todas"
TODOS = "Todos"
HOJA_INICIO, HOJA_DASHBOARD, HOJA_COMPARADOR = "Inicio", "Dashboard", "Comparador"
HOJA_RENTA, HOJA_RESUMEN, HOJA_DATOS, HOJA_CALCULOS = "Renta y precio", "Resumen ciudades", "Datos", "Cálculos"
HOJAS: tuple[str, ...] = (HOJA_INICIO, HOJA_DASHBOARD, HOJA_COMPARADOR, HOJA_RENTA, HOJA_RESUMEN, HOJA_DATOS,
                          HOJA_CALCULOS)
NOMBRE_TABLA_DATOS = "TablaDatos"
TITULO_VISTA_PREVIA = "Vista previa generada con Python de los gráficos del dashboard (no es una captura de Excel)"
AVISO_MUESTRA = "Muestra pequeña: interpretar con cautela"
AVISO_SIN_DATOS = "Sin anuncios con precio para esta combinación"
SIN_DATO = ""                     # valor que muestran las fórmulas cuando no hay anuncios (IFERROR)

# Paleta sobria y apta para daltónicos (Okabe-Ito) y colores neutros.
AZUL, AZUL_OSCURO, AZUL_CLARO = "#0072B2", "#1F3B57", "#E8F1F8"
GRIS, GRIS_CLARO, BORDE = "#595959", "#F2F2F2", "#BFBFBF"
AMARILLO_SUAVE, MARRON_AVISO = "#FFF2CC", "#7F4F00"
OKABE_ITO: tuple[str, ...] = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000",
                              "#999999", "#F0E442")
MARCADORES: tuple[str, ...] = ("circle", "square", "diamond", "triangle", "x", "circle", "square", "diamond",
                               "triangle")
FMT_EUR = '#,##0.00 "€"'
FMT_EUR0 = '#,##0 "€"'
FMT_PCT = "0.0%"
FMT_MILES = "#,##0"
FMT_DEC3 = "0.000"
FMT_PCT_SIGNO = '+0.0 "%";-0.0 "%"'


# --------------------------------------------------------------------------
# Hoja «Datos»: columnas (cabecera legible, nombre definido para las fórmulas, formato)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnaDatos:
    """Columna de la hoja «Datos»: cabecera en español, nombre definido (o None) y formato numérico."""

    cabecera: str
    nombre: str | None
    formato: str | None
    ancho: float


COLUMNAS_DATOS: tuple[ColumnaDatos, ...] = (
    ColumnaDatos("Ciudad", "dCiudad", None, 12),
    ColumnaDatos("Municipio (INE)", None, None, 22),
    ColumnaDatos("Barrio o municipio (Inside Airbnb)", None, None, 26),
    ColumnaDatos("Distrito (Inside Airbnb)", None, None, 20),
    ColumnaDatos("Código de distrito censal (INE)", "dDistrito", None, 14),
    ColumnaDatos("Tipo de alojamiento", "dTipo", None, 20),
    ColumnaDatos("Capacidad (personas)", None, "0", 11),
    ColumnaDatos("Tramo de capacidad", "dTramoCap", None, 15),
    ColumnaDatos("Precio por noche (€)", "dPrecio", FMT_EUR, 13),
    ColumnaDatos("Tramo de precio", "dTramoPrecio", None, 12),
    ColumnaDatos("Superhost", "dSuperhost", None, 10),
    ColumnaDatos("Anfitrión con varios anuncios", "dMulti", None, 13),
    ColumnaDatos(f"Estancia mínima ≥ {config.MIN_NIGHTS_LARGA_ESTANCIA} noches", None, None, 13),
    ColumnaDatos("Tipo de licencia", None, None, 17),
    ColumnaDatos("Valoración (1-5)", None, "0.00", 10),
    ColumnaDatos("Nº de reseñas", None, FMT_MILES, 10),
    ColumnaDatos("Renta neta por persona del distrito (€)", "dRenta", FMT_EUR0, 15),
    ColumnaDatos("Índice de Gini del distrito", None, "0.0", 10),
    ColumnaDatos("Renta neta por persona del municipio (€)", None, FMT_EUR0, 15),
)
CABECERAS_DATOS: tuple[str, ...] = tuple(c.cabecera for c in COLUMNAS_DATOS)
SI, NO = "Sí", "No"


def etiquetas_tramos_precio() -> list[str]:
    """Etiquetas de los tramos de precio [a, b): «Menos de 50 €», «50-100 €», …, «500 € o más».

    Ninguna empieza por «<», «>» o «=»: en un criterio de COUNTIFS Excel leería ese primer carácter como
    un operador de comparación y el tramo contaría 0 anuncios.
    """
    lim = config.LIMITES_TRAMOS_PRECIO_EXCEL
    return ([f"Menos de {lim[0]} €"] + [f"{a}-{b} €" for a, b in zip(lim, lim[1:])] + [f"{lim[-1]} € o más"])


def tramo_precio(precio: pd.Series) -> pd.Series:
    """Tramo de precio de cada anuncio (intervalos cerrados por la izquierda)."""
    bordes = [-np.inf, *config.LIMITES_TRAMOS_PRECIO_EXCEL, np.inf]
    return pd.cut(precio, bordes, right=False, labels=etiquetas_tramos_precio()).astype(str)


def etiquetas_tramos_capacidad() -> list[str]:
    """Etiquetas de los tramos de capacidad del dashboard."""
    return [t[0] for t in config.TRAMOS_CAPACIDAD_EXCEL]


def tramo_capacidad(capacidad: pd.Series) -> pd.Series:
    """Tramo de capacidad (1, 2, 3-4, 5-6, 7 o más personas)."""
    salida = pd.Series(pd.NA, index=capacidad.index, dtype="object")
    for etiqueta, minimo, maximo in config.TRAMOS_CAPACIDAD_EXCEL:
        m = capacidad.ge(minimo) & (capacidad.le(maximo) if maximo is not None else True)
        salida[m] = etiqueta
    return salida


def nombres_tipos() -> list[str]:
    """Tipos de alojamiento en el orden de ``config.TRADUCCION_TIPO_HABITACION``."""
    return list(config.TRADUCCION_TIPO_HABITACION.values())


def _clave_alfabetica(texto: str) -> str:
    """Clave de orden sin tildes (Málaga entre Madrid y Mallorca)."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def nombres_ciudades() -> list[str]:
    """Nombres de ciudad en orden alfabético español."""
    return sorted(config.CIUDAD_POR_NOMBRE, key=_clave_alfabetica)


def tabla_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Anuncios CON PRECIO y solo las columnas del dashboard, con cabeceras legibles y sin datos personales.

    No incluye ``host_id``, nombres, URLs, fotos, descripciones ni coordenadas.
    """
    cp = df[df["precio_valido"]]
    si_no = lambda s: np.where(s.astype(bool), SI, NO)  # noqa: E731
    valores = [
        cp["ciudad"].map(lambda c: config.CIUDADES_ESPANA[c].nombre), cp["municipio_ine"],
        cp["neighbourhood_cleansed"], cp["neighbourhood_group_cleansed"], cp["codigo_distrito_ine"],
        cp["tipo_habitacion"], cp["accommodates"], tramo_capacidad(cp["accommodates"]), cp["price"],
        tramo_precio(cp["price"]), si_no(cp["host_is_superhost"]), si_no(cp["es_multianuncio"]),
        si_no(cp["alquiler_larga_estancia"]), cp["licencia_tipo"], cp["review_scores_rating"],
        cp["number_of_reviews"], cp[config.RENTA_PRINCIPAL], cp["ine_dis_gini"], cp[config.RENTA_MUNICIPAL],
    ]
    datos = pd.DataFrame({c: np.asarray(v, dtype=object) for c, v in zip(CABECERAS_DATOS, valores)})
    datos["_ciudad_orden"] = datos["Ciudad"].map(_clave_alfabetica)
    datos["_id"] = cp["id"].to_numpy()
    datos = datos.sort_values(["_ciudad_orden", "_id"], kind="mergesort").drop(columns=["_ciudad_orden", "_id"])
    datos = datos.reset_index(drop=True)
    for c in COLUMNAS_DATOS:
        if c.formato is not None:
            datos[c.cabecera] = pd.to_numeric(datos[c.cabecera], errors="coerce").astype("float64")
        else:
            datos[c.cabecera] = datos[c.cabecera].astype(object).where(datos[c.cabecera].notna(), None)
    return datos


# --------------------------------------------------------------------------
# Cálculos en pandas (valores cacheados de las fórmulas y vista previa)
# --------------------------------------------------------------------------
def mascara(datos: pd.DataFrame, ciudad: str = TODAS, tipo: str = TODOS, capacidad: str | None = None) -> np.ndarray:
    """Filas de ``datos`` que cumplen la selección (Todas/Todos = sin filtro)."""
    m = np.ones(len(datos), dtype=bool)
    if ciudad != TODAS:
        m &= datos["Ciudad"].to_numpy() == ciudad
    if tipo != TODOS:
        m &= datos["Tipo de alojamiento"].to_numpy() == tipo
    if capacidad is not None:
        m &= datos["Tramo de capacidad"].to_numpy() == capacidad
    return m


def _o_vacio(x: float) -> float | str:
    """El valor, o ``SIN_DATO`` si es NaN (lo que devuelve IFERROR en Excel)."""
    return SIN_DATO if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def resumen_precio(datos: pd.DataFrame, m: np.ndarray) -> dict[str, Any]:
    """n, mediana, media, P25, P75, % superhost, % multi-anuncio y renta media de las filas ``m``."""
    precio = datos["Precio por noche (€)"].to_numpy(dtype=float)[m]
    n = int(m.sum())
    renta = datos["Renta neta por persona del distrito (€)"].to_numpy(dtype=float)[m]
    renta = renta[~np.isnan(renta)]
    if n == 0:
        return {"n": 0, **{k: SIN_DATO for k in ("mediana", "media", "p25", "p75", "pct_superhost",
                                                   "pct_multianuncio", "renta_media")}}
    return {
        "n": n, "mediana": float(np.median(precio)), "media": float(precio.mean()),
        "p25": float(np.percentile(precio, 25)), "p75": float(np.percentile(precio, 75)),
        "pct_superhost": float((datos["Superhost"].to_numpy()[m] == SI).mean()),
        "pct_multianuncio": float((datos["Anfitrión con varios anuncios"].to_numpy()[m] == SI).mean()),
        "renta_media": _o_vacio(float(renta.mean())) if renta.size else SIN_DATO,
    }


def distritos_dispersion(datos: pd.DataFrame) -> pd.DataFrame:
    """Distritos censales con al menos ``MIN_N_GRUPO`` anuncios con precio y renta del INE (ciudad × distrito)."""
    d = datos.dropna(subset=["Código de distrito censal (INE)", "Renta neta por persona del distrito (€)"])
    g = d.groupby(["Ciudad", "Código de distrito censal (INE)"], sort=False).agg(
        municipio=("Municipio (INE)", "first"), n=("Precio por noche (€)", "size"),
        renta=("Renta neta por persona del distrito (€)", "mean"), mediana=("Precio por noche (€)", "median"))
    g = g[g["n"] >= config.MIN_N_GRUPO].reset_index()
    g["_orden"] = g["Ciudad"].map(_clave_alfabetica)
    return g.sort_values(["_orden", "Código de distrito censal (INE)"]).drop(columns="_orden").reset_index(drop=True)


@dataclass
class Seleccion:
    """Valores por defecto de los selectores (los que se guardan en el libro y se cachean)."""

    ciudad: str = TODAS
    tipo: str = TODOS
    cmp_ciudad: str = config.CIUDAD_REFERENCIA
    cmp_tipo: str = "Alojamiento entero"
    cmp_capacidad: str = config.TRAMOS_CAPACIDAD_EXCEL[2][0]


def tablas_graficos(datos: pd.DataFrame, findings: dict[str, Any], sel: Seleccion) -> dict[str, pd.DataFrame]:
    """Tablas auxiliares de los gráficos calculadas en pandas para la selección ``sel``."""
    orden_ciudades = [c["grupo"] for c in findings["descriptivo"]["por_ciudad"]]
    precio = datos["Precio por noche (€)"].to_numpy(dtype=float)

    def med(m: np.ndarray) -> float | str:
        return float(np.median(precio[m])) if m.any() else SIN_DATO

    ciudad = pd.DataFrame([{"etiqueta": c, "n": int(mascara(datos, c, sel.tipo).sum()),
                            "mediana": med(mascara(datos, c, sel.tipo))} for c in orden_ciudades])
    tipo = pd.DataFrame([{"etiqueta": t, "n": int(mascara(datos, sel.ciudad, t).sum()),
                          "mediana": med(mascara(datos, sel.ciudad, t))} for t in nombres_tipos()])
    base = mascara(datos, sel.ciudad, sel.tipo)
    tramos = datos["Tramo de precio"].to_numpy()
    precio_t = pd.DataFrame([{"etiqueta": t, "n": int((base & (tramos == t)).sum())}
                             for t in etiquetas_tramos_precio()])
    cap = pd.DataFrame([{"etiqueta": t, "n": int(mascara(datos, sel.ciudad, sel.tipo, t).sum()),
                         "mediana": med(mascara(datos, sel.ciudad, sel.tipo, t))}
                        for t in etiquetas_tramos_capacidad()])
    return {"ciudad": ciudad, "tipo": tipo, "tramo_precio": precio_t, "capacidad": cap,
            "distritos": distritos_dispersion(datos)}


# --------------------------------------------------------------------------
# Fórmulas (sintaxis inglesa de Excel; nombres definidos para que sean legibles)
# --------------------------------------------------------------------------
def _cond_sel(nombre_rango: str, celda: str, todos: str) -> str:
    """Condición matricial «celda = Todos/Todas o rango = celda» (1/0)."""
    return f'(({celda}="{todos}")+({nombre_rango}={celda}))'


COND_DASHBOARD = f'{_cond_sel("dCiudad", "selCiudad", TODAS)}*{_cond_sel("dTipo", "selTipo", TODOS)}'
COND_COMPARADOR = (f'{_cond_sel("dCiudad", "cmpCiudad", TODAS)}*{_cond_sel("dTipo", "cmpTipo", TODOS)}'
                   '*(dTramoCap=cmpCap)')
CRIT_DASHBOARD = "dCiudad,critCiudad,dTipo,critTipo"
CRIT_COMPARADOR = "dCiudad,critCmpCiudad,dTipo,critCmpTipo,dTramoCap,cmpCap"


def f_n(crit: str) -> str:
    return f"=COUNTIFS({crit})"


def f_mediana(cond: str) -> str:
    return f'{{=IFERROR(MEDIAN(IF({cond},dPrecio)),"{SIN_DATO}")}}'


def f_percentil(cond: str, q: float) -> str:
    return f'{{=IFERROR(PERCENTILE(IF({cond},dPrecio),{q}),"{SIN_DATO}")}}'


def f_media(crit: str, rango: str = "dPrecio") -> str:
    return f'=IFERROR(AVERAGEIFS({rango},{crit}),"{SIN_DATO}")'


def f_proporcion(crit: str, rango: str) -> str:
    return f'=IFERROR(COUNTIFS({crit},{rango},"{SI}")/COUNTIFS({crit}),"{SIN_DATO}")'


# --------------------------------------------------------------------------
# Construcción del libro
# --------------------------------------------------------------------------
class _Libro:
    """Envoltorio de xlsxwriter con los formatos del proyecto."""

    def __init__(self, ruta: Path, findings: dict[str, Any]) -> None:
        import xlsxwriter
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.wb = xlsxwriter.Workbook(str(ruta), {"strings_to_numbers": False, "strings_to_urls": False})
        fecha = dt.datetime.fromisoformat(str(findings["meta"]["fecha_referencia"])[:10])
        self.wb.set_properties({
            "title": "Dashboard operativo: Airbnb en España y renta de la zona",
            "subject": "Proyecto Final Máster Data Analytics", "author": findings["meta"]["autor"],
            "keywords": "Airbnb, INE, renta, precio, España", "created": fecha,  # fecha fija: archivo reproducible
            "comments": "Generado por src/dashboard_excel.py a partir de reports/findings.json y data/processed.",
        })
        self.wb.set_calc_mode("auto")  # xlsxwriter escribe además fullCalcOnLoad="1": recálculo al abrir
        self.wb.set_size(1600, 1000)
        base = {"font_name": "Calibri", "font_size": 11, "valign": "vcenter"}
        F = lambda **k: self.wb.add_format({**base, **k})  # noqa: E731
        self.f = {
            "titulo": F(bold=True, font_size=18, font_color=AZUL_OSCURO),
            "subtitulo": F(italic=True, font_color=GRIS, text_wrap=True, valign="top"),
            "h2": F(bold=True, font_size=13, font_color=AZUL_OSCURO, bottom=1, bottom_color=BORDE),
            "texto": F(text_wrap=True, valign="top"),
            "nota": F(italic=True, font_size=9, font_color=GRIS, text_wrap=True, valign="top"),
            "negrita": F(bold=True),
            "enlace": F(font_color=AZUL, underline=1),
            "cabecera": F(bold=True, font_color="#FFFFFF", bg_color=AZUL_OSCURO, text_wrap=True, border=1,
                          border_color=BORDE, valign="top"),
            "celda": F(border=1, border_color=BORDE),
            "eur": F(num_format=FMT_EUR, border=1, border_color=BORDE),
            "eur0": F(num_format=FMT_EUR0, border=1, border_color=BORDE),
            "pct": F(num_format=FMT_PCT, border=1, border_color=BORDE),
            "pct100": F(num_format='0.0 "%"', border=1, border_color=BORDE),
            "pct_signo": F(num_format=FMT_PCT_SIGNO, border=1, border_color=BORDE),
            "miles": F(num_format=FMT_MILES, border=1, border_color=BORDE),
            "dec3": F(num_format=FMT_DEC3, border=1, border_color=BORDE),
            "selector_lbl": F(bold=True, font_color=AZUL_OSCURO, align="right"),
            "selector": F(bold=True, bg_color=AMARILLO_SUAVE, border=2, border_color=AZUL, font_color=AZUL_OSCURO),
            "card_lbl": F(bold=True, font_size=9, font_color=GRIS, bg_color=AZUL_CLARO, align="center",
                          text_wrap=True, top=2, top_color=AZUL, left=1, right=1, left_color=BORDE,
                          right_color=BORDE),
            "aviso": F(bold=True, font_color=MARRON_AVISO, text_wrap=True),
            "total": F(bold=True, bg_color=GRIS_CLARO, border=1, border_color=BORDE),
        }
        for clave, nf in (("card_n", FMT_MILES), ("card_eur", FMT_EUR), ("card_pct", FMT_PCT)):
            self.f[clave] = F(bold=True, font_size=16, font_color=AZUL_OSCURO, bg_color=AZUL_CLARO, align="center",
                              num_format=nf, bottom=1, left=1, right=1, bottom_color=BORDE, left_color=BORDE,
                              right_color=BORDE)
        for clave, nf in (("total_eur", FMT_EUR), ("total_pct", FMT_PCT), ("total_miles", FMT_MILES),
                          ("total_pct100", '0.0 "%"'), ("total_eur0", FMT_EUR0), ("total_dec3", FMT_DEC3)):
            self.f[clave] = F(bold=True, bg_color=GRIS_CLARO, border=1, border_color=BORDE, num_format=nf)
        self.hojas = {nombre: self.wb.add_worksheet(nombre) for nombre in HOJAS}
        for h in self.hojas.values():
            h.hide_gridlines(2)
        self.hojas[HOJA_DATOS].hide_gridlines(0)

    def nombre(self, nombre: str, hoja: str, rango: str) -> None:
        """Define un nombre de libro (rango con nombre)."""
        self.wb.define_name(nombre, f"='{hoja}'!{rango}")


def _celda(fila: int, col: int, absoluta: bool = True) -> str:
    from xlsxwriter.utility import xl_rowcol_to_cell
    return xl_rowcol_to_cell(fila, col, row_abs=absoluta, col_abs=absoluta)


def _formula(ws: Any, fila: int, col: int, formula: str, fmt: Any, valor: Any) -> None:
    """Escribe una fórmula normal o matricial (``{=...}``) con su valor cacheado."""
    valor = SIN_DATO if valor is None else valor
    if formula.startswith("{"):
        ws.write_array_formula(fila, col, fila, col, formula, fmt, valor)
    else:
        ws.write_formula(fila, col, formula, fmt, valor)


def _hoja_datos(lb: _Libro, datos: pd.DataFrame) -> int:
    """Escribe la Tabla de Excel con los anuncios con precio y define los rangos con nombre. Devuelve n filas."""
    ws = lb.hojas[HOJA_DATOS]
    n = len(datos)
    filas = datos.astype(object).where(datos.notna(), None).to_numpy().tolist()
    columnas = [{"header": c.cabecera, **({"format": lb.wb.add_format({"num_format": c.formato})}
                                          if c.formato else {})} for c in COLUMNAS_DATOS]
    ws.add_table(0, 0, n, len(COLUMNAS_DATOS) - 1, {
        "name": NOMBRE_TABLA_DATOS, "data": filas, "columns": columnas, "style": "Table Style Light 9",
        "autofilter": True})
    for j, c in enumerate(COLUMNAS_DATOS):
        ws.set_column(j, j, c.ancho)
        if c.nombre:
            lb.nombre(c.nombre, HOJA_DATOS, f"{_celda(1, j)}:{_celda(n, j)}")
    ws.set_row(0, 45)
    ws.freeze_panes(1, 1)
    return n


def _hoja_calculos(lb: _Libro, tablas: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Listas de los desplegables, criterios y tablas auxiliares de los gráficos (con fórmulas)."""
    ws, f = lb.hojas[HOJA_CALCULOS], lb.f
    ws.set_column(0, 0, 30)
    ws.set_column(1, 2, 16)
    ws.set_column(3, 6, 16)
    ws.write(0, 0, "Cálculos auxiliares del dashboard", f["titulo"])
    ws.write(1, 0, "Hoja de apoyo: listas de los desplegables, criterios de filtro y tablas que alimentan los "
                   "gráficos. Todo son fórmulas sobre la hoja Datos; no hace falta editar nada aquí.", f["nota"])
    pos: dict[str, Any] = {}
    # Listas de los desplegables (columnas A-C, filas 4..)
    listas = {"ciudades": [TODAS, *nombres_ciudades()], "tipos": [TODOS, *nombres_tipos()],
              "capacidades": etiquetas_tramos_capacidad()}
    for j, (clave, titulo) in enumerate((("ciudades", "Lista: ciudades"), ("tipos", "Lista: tipos"),
                                         ("capacidades", "Lista: capacidad"))):
        ws.write(3, j, titulo, f["cabecera"])
        ws.write_column(4, j, listas[clave], f["celda"])
        lb.nombre(f"lista{clave.capitalize()}", HOJA_CALCULOS,
                  f"{_celda(4, j)}:{_celda(4 + len(listas[clave]) - 1, j)}")
    # Criterios para COUNTIFS/AVERAGEIFS («*» = cualquier texto cuando el selector es Todas/Todos)
    ws.write(3, 4, "Criterio de filtro", f["cabecera"])
    ws.write(3, 5, "Valor", f["cabecera"])
    pos["criterios"] = {}
    for i, (nombre, selector, todos, desc) in enumerate((
            ("critCiudad", "selCiudad", TODAS, "Ciudad (Dashboard)"), ("critTipo", "selTipo", TODOS, "Tipo (Dashboard)"),
            ("critCmpCiudad", "cmpCiudad", TODAS, "Ciudad (Comparador)"),
            ("critCmpTipo", "cmpTipo", TODOS, "Tipo (Comparador)"))):
        ws.write(4 + i, 4, desc, f["celda"])
        lb.nombre(nombre, HOJA_CALCULOS, _celda(4 + i, 5))
        pos["criterios"][nombre] = (selector, todos, 4 + i)
    fila = 4 + max(len(v) for v in listas.values()) + 2

    def bloque(clave: str, titulo: str, cabeceras: list[str]) -> int:
        nonlocal fila
        ws.write(fila, 0, titulo, f["h2"])
        ws.write_row(fila + 1, 0, cabeceras, f["cabecera"])
        inicio = fila + 2
        fila = inicio + len(tablas[clave]) + 2
        return inicio

    pos["ciudad"] = bloque("ciudad", "Mediana del precio por ciudad (filtro: tipo de alojamiento)",
                           ["Ciudad", "Anuncios con precio", "Mediana (€)"])
    pos["tramo_precio"] = bloque("tramo_precio", "Anuncios por tramo de precio (filtros: ciudad y tipo)",
                                 ["Tramo de precio", "Anuncios con precio"])
    pos["tipo"] = bloque("tipo", "Mediana del precio por tipo de alojamiento (filtro: ciudad)",
                         ["Tipo de alojamiento", "Anuncios con precio", "Mediana (€)"])
    pos["capacidad"] = bloque("capacidad", "Anuncios y mediana por capacidad (filtros: ciudad y tipo)",
                              ["Tramo de capacidad", "Anuncios con precio", "Mediana (€)"])
    pos["titulos"] = fila
    fila += 7
    pos["distritos"] = bloque(
        "distritos", f"Distritos censales con al menos {config.MIN_N_GRUPO} anuncios con precio (no depende de "
                     "los selectores)",
        ["Ciudad", "Código de distrito censal (INE)", "Municipio (INE)", "Anuncios con precio",
         "Renta neta por persona del distrito (€)", "Mediana del precio (€)"])
    return pos


def _rellenar_calculos(lb: _Libro, pos: dict[str, Any], tablas: dict[str, pd.DataFrame], sel: Seleccion) -> None:
    """Escribe las fórmulas de la hoja de cálculos con sus valores cacheados."""
    ws, f = lb.hojas[HOJA_CALCULOS], lb.f
    valores_sel = {"selCiudad": sel.ciudad, "selTipo": sel.tipo, "cmpCiudad": sel.cmp_ciudad, "cmpTipo": sel.cmp_tipo}
    for nombre, (selector, todos, fila) in pos["criterios"].items():
        v = valores_sel[selector]
        _formula(ws, fila, 5, f'=IF({selector}="{todos}","*",{selector})', f["celda"], "*" if v == todos else v)
    sel_tipo = _cond_sel("dTipo", "selTipo", TODOS)
    sel_ciudad = _cond_sel("dCiudad", "selCiudad", TODAS)
    for i, r in tablas["ciudad"].iterrows():
        fi = pos["ciudad"] + i
        a = f"A{fi + 1}"
        ws.write(fi, 0, r["etiqueta"], f["celda"])
        _formula(ws, fi, 1, f"=COUNTIFS(dCiudad,{a},dTipo,critTipo)", f["miles"], r["n"])
        _formula(ws, fi, 2, f'{{=IFERROR(MEDIAN(IF((dCiudad={a})*{sel_tipo},dPrecio)),"{SIN_DATO}")}}', f["eur"],
                 r["mediana"])
    for i, r in tablas["tramo_precio"].iterrows():
        fi = pos["tramo_precio"] + i
        a = f"A{fi + 1}"
        ws.write(fi, 0, r["etiqueta"], f["celda"])
        _formula(ws, fi, 1, f"=COUNTIFS(dTramoPrecio,{a},{CRIT_DASHBOARD})", f["miles"], r["n"])
    for i, r in tablas["tipo"].iterrows():
        fi = pos["tipo"] + i
        a = f"A{fi + 1}"
        ws.write(fi, 0, r["etiqueta"], f["celda"])
        _formula(ws, fi, 1, f"=COUNTIFS(dTipo,{a},dCiudad,critCiudad)", f["miles"], r["n"])
        _formula(ws, fi, 2, f'{{=IFERROR(MEDIAN(IF((dTipo={a})*{sel_ciudad},dPrecio)),"{SIN_DATO}")}}', f["eur"],
                 r["mediana"])
    for i, r in tablas["capacidad"].iterrows():
        fi = pos["capacidad"] + i
        a = f"A{fi + 1}"
        ws.write(fi, 0, r["etiqueta"], f["celda"])
        _formula(ws, fi, 1, f"=COUNTIFS(dTramoCap,{a},{CRIT_DASHBOARD})", f["miles"], r["n"])
        _formula(ws, fi, 2, f'{{=IFERROR(MEDIAN(IF((dTramoCap={a})*{COND_DASHBOARD},dPrecio)),"{SIN_DATO}")}}',
                 f["eur"], r["mediana"])
    # Títulos dinámicos de los gráficos que dependen de los selectores
    ft = pos["titulos"]
    ws.write(ft, 0, "Títulos dinámicos de los gráficos", f["h2"])
    titulos = [
        ("Mediana del precio por ciudad · ", "selTipo", None, sel.tipo),
        ("Anuncios por tramo de precio · ", "selCiudad", "selTipo", f"{sel.ciudad} · {sel.tipo}"),
        ("Mediana del precio por tipo de alojamiento · ", "selCiudad", None, sel.ciudad),
        ("Anuncios por capacidad · ", "selCiudad", "selTipo", f"{sel.ciudad} · {sel.tipo}"),
    ]
    pos["celdas_titulo"] = []
    for i, (texto, s1, s2, sufijo) in enumerate(titulos):
        partes = f'"{texto}",{s1}' + (f'," · ",{s2}' if s2 else "")
        _formula(ws, ft + 1 + i, 0, f"=CONCATENATE({partes})", f["celda"], texto + sufijo)
        pos["celdas_titulo"].append(ft + 1 + i)
    for i, r in tablas["distritos"].iterrows():
        fi = pos["distritos"] + i
        a, b = f"A{fi + 1}", f"B{fi + 1}"
        ws.write(fi, 0, r["Ciudad"], f["celda"])
        ws.write_string(fi, 1, r["Código de distrito censal (INE)"], f["celda"])
        ws.write(fi, 2, r["municipio"] if isinstance(r["municipio"], str) else "", f["celda"])
        _formula(ws, fi, 3, f"=COUNTIFS(dCiudad,{a},dDistrito,{b})", f["miles"], int(r["n"]))
        _formula(ws, fi, 4, f'=IFERROR(AVERAGEIFS(dRenta,dCiudad,{a},dDistrito,{b}),"{SIN_DATO}")', f["eur0"],
                 float(r["renta"]))
        _formula(ws, fi, 5, f'{{=IFERROR(MEDIAN(IF((dCiudad={a})*(dDistrito={b}),dPrecio)),"{SIN_DATO}")}}',
                 f["eur"], float(r["mediana"]))


def _grafico_barras(lb: _Libro, tipo: str, hoja: str, fila0: int, n: int, col_val: int, titulo: list | str,
                    eje_val: str, eje_cat: str, fmt: str, descripcion: str) -> Any:
    """Gráfico de barras/columnas de una serie sobre la hoja de cálculos."""
    ch = lb.wb.add_chart({"type": tipo})
    ch.add_series({
        "name": eje_val, "categories": [hoja, fila0, 0, fila0 + n - 1, 0],
        "values": [hoja, fila0, col_val, fila0 + n - 1, col_val],
        "fill": {"color": AZUL}, "border": {"none": True}, "gap": 60,
        "data_labels": {"value": True, "num_format": fmt, "font": {"size": 8, "color": GRIS}},
    })
    ch.set_title({"name": titulo, "name_font": {"size": 11, "bold": True, "color": AZUL_OSCURO}})
    eje_v = {"name": eje_val, "num_format": fmt, "name_font": {"size": 9, "bold": False},
             "num_font": {"size": 8}, "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}}
    eje_c = {"name": eje_cat, "name_font": {"size": 9, "bold": False}, "num_font": {"size": 8}}
    if tipo == "bar":
        ch.set_x_axis(eje_v)
        ch.set_y_axis({**eje_c, "reverse": True})
    else:
        ch.set_y_axis(eje_v)
        ch.set_x_axis(eje_c)
    ch.set_legend({"none": True})
    ch.set_size({"width": 520, "height": 320})
    ch.set_chartarea({"border": {"color": BORDE}})
    return ch, descripcion


def _hoja_dashboard(lb: _Libro, pos: dict[str, Any], tablas: dict[str, pd.DataFrame], kpi: dict[str, Any],
                    findings: dict[str, Any], sel: Seleccion) -> int:
    """Selectores, tarjetas KPI y 5 gráficos nativos. Devuelve el número de gráficos insertados."""
    ws, f, wb = lb.hojas[HOJA_DASHBOARD], lb.f, lb.wb
    ws.set_column(0, 0, 2)
    ws.set_column(1, 8, 17)
    ws.set_column(9, 9, 2)
    ws.write(0, 1, "Dashboard operativo · Airbnb en España y renta de la zona", f["titulo"])
    ws.set_row(0, 30)
    ws.merge_range(1, 1, 1, 8, "Qué explica el precio por noche de un alojamiento turístico y cómo se relaciona con "
                               "la renta de la zona (anuncios con precio).", f["subtitulo"])
    ws.write(3, 1, "Ciudad", f["selector_lbl"])
    ws.write(3, 2, sel.ciudad, f["selector"])
    ws.write(4, 1, "Tipo de alojamiento", f["selector_lbl"])
    ws.write(4, 2, sel.tipo, f["selector"])
    lb.nombre("selCiudad", HOJA_DASHBOARD, "$C$4")
    lb.nombre("selTipo", HOJA_DASHBOARD, "$C$5")
    ws.data_validation(3, 2, 3, 2, {"validate": "list", "source": "=listaCiudades", "input_title": "Ciudad",
                                    "input_message": f"Elige una ciudad o «{TODAS}».", "error_title": "Valor no válido",
                                    "error_message": "Elige un valor de la lista."})
    ws.data_validation(4, 2, 4, 2, {"validate": "list", "source": "=listaTipos", "input_title": "Tipo",
                                    "input_message": f"Elige un tipo de alojamiento o «{TODOS}».",
                                    "error_title": "Valor no válido", "error_message": "Elige un valor de la lista."})
    ws.merge_range(3, 3, 4, 8, "◀ Cambia los desplegables amarillos: las tarjetas y los cuatro primeros gráficos se "
                               "recalculan solos. El gráfico de dispersión muestra todos los distritos.", f["nota"])
    tarjetas = [
        ("Anuncios con precio", f_n(CRIT_DASHBOARD), "card_n", kpi["n"]),
        ("Mediana del precio por noche", f_mediana(COND_DASHBOARD), "card_eur", kpi["mediana"]),
        ("Media del precio por noche", f_media(CRIT_DASHBOARD), "card_eur", kpi["media"]),
        ("Percentil 25 (P25)", f_percentil(COND_DASHBOARD, 0.25), "card_eur", kpi["p25"]),
        ("Percentil 75 (P75)", f_percentil(COND_DASHBOARD, 0.75), "card_eur", kpi["p75"]),
        ("% anuncios de superhost", f_proporcion(CRIT_DASHBOARD, "dSuperhost"), "card_pct", kpi["pct_superhost"]),
        ("% anuncios de anfitriones con varios anuncios", f_proporcion(CRIT_DASHBOARD, "dMulti"), "card_pct",
         kpi["pct_multianuncio"]),
        ("Renta neta media por persona de la zona (INE)", f_media(CRIT_DASHBOARD, "dRenta"), "card_eur",
         kpi["renta_media"]),
    ]
    ws.set_row(6, 34)
    ws.set_row(7, 34)
    for j, (lbl, formula, fmt, valor) in enumerate(tarjetas):
        ws.write(6, 1 + j, lbl, f["card_lbl"])
        _formula(ws, 7, 1 + j, formula, f[fmt], valor)
    ws.merge_range(8, 1, 8, 8, "Precio = cotización por noche de Airbnb con los descuentos aplicados y sin tasas (no "
                               "es el precio pagado). Renta: INE, distrito censal, año " + str(findings["meta"]["anio_ine"])
                   + ". Todas las tarjetas, incluidos los % de superhost y de anfitriones con varios anuncios, se "
                     "calculan sobre los anuncios CON PRECIO (el README y el informe dan esos % sobre todos los "
                     "anuncios, por eso difieren un poco). Las relaciones son asociaciones, no efectos causales.",
                   f["nota"])
    ws.set_row(8, 42)
    ws.freeze_panes(9, 0)
    hc = HOJA_CALCULOS
    ct = pos["celdas_titulo"]
    graficos = [
        _grafico_barras(lb, "bar", hc, pos["ciudad"], len(tablas["ciudad"]), 2, [hc, ct[0], 0],
                        "Mediana del precio por noche (€)", "Ciudad o región", FMT_EUR0,
                        "Barras horizontales con la mediana del precio por noche de cada ciudad para el tipo de "
                        "alojamiento elegido."),
        _grafico_barras(lb, "column", hc, pos["tramo_precio"], len(tablas["tramo_precio"]), 1, [hc, ct[1], 0],
                        "Anuncios con precio", "Tramo de precio por noche (€)", FMT_MILES,
                        "Columnas con el número de anuncios en cada tramo de precio para la ciudad y el tipo elegidos."),
        _grafico_barras(lb, "bar", hc, pos["tipo"], len(tablas["tipo"]), 2, [hc, ct[2], 0],
                        "Mediana del precio por noche (€)", "Tipo de alojamiento", FMT_EUR0,
                        "Barras horizontales con la mediana del precio por tipo de alojamiento en la ciudad elegida."),
        _grafico_barras(lb, "column", hc, pos["capacidad"], len(tablas["capacidad"]), 1, [hc, ct[3], 0],
                        "Anuncios con precio", "Capacidad (personas)", FMT_MILES,
                        "Columnas con el número de anuncios por tramo de capacidad para la ciudad y el tipo elegidos."),
    ]
    disp = wb.add_chart({"type": "scatter"})
    dist = tablas["distritos"]
    f0 = pos["distritos"]
    for k, ciudad in enumerate(dist["Ciudad"].drop_duplicates()):
        idx = dist.index[dist["Ciudad"] == ciudad]
        a, b = f0 + int(idx.min()), f0 + int(idx.max())
        color = OKABE_ITO[k % len(OKABE_ITO)]
        disp.add_series({"name": ciudad, "categories": [hc, a, 4, b, 4], "values": [hc, a, 5, b, 5],
                         "marker": {"type": MARCADORES[k % len(MARCADORES)], "size": 5,
                                    "fill": {"color": color}, "border": {"color": color}}})
    disp.set_title({"name": f"Renta de la zona y mediana del precio por distrito censal (distritos con al menos "
                            f"{config.MIN_N_GRUPO} anuncios)", "name_font": {"size": 11, "bold": True,
                                                                             "color": AZUL_OSCURO}})
    disp.set_x_axis({"name": f"Renta neta media por persona del distrito (€, INE {findings['meta']['anio_ine']})",
                     "num_format": FMT_EUR0, "name_font": {"size": 9, "bold": False}, "num_font": {"size": 8},
                     "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
    disp.set_y_axis({"name": "Mediana del precio por noche (€)", "num_format": FMT_EUR0, "log_base": 10,
                     "name_font": {"size": 9, "bold": False}, "num_font": {"size": 8},
                     "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
    disp.set_legend({"position": "right", "font": {"size": 8}})
    disp.set_size({"width": 1060, "height": 380})
    disp.set_chartarea({"border": {"color": BORDE}})
    graficos.append((disp, "Dispersión: cada punto es un distrito censal; eje X renta neta media por persona del "
                           "INE, eje Y mediana del precio por noche (escala logarítmica); un color y un marcador "
                           "por ciudad."))
    anclas = [(10, 1), (10, 5), (27, 1), (27, 5), (44, 1)]
    for (ch, descripcion), (r, c) in zip(graficos, anclas):
        ws.insert_chart(r, c, ch, {"x_offset": 4, "y_offset": 4, "description": descripcion, "object_position": 3})
    ws.write(64, 1, "Fuente: Inside Airbnb (CC BY 4.0) e INE (ADRH). " + findings["dataset"]["fuente_ine"]["cita"]
             + ".", f["nota"])
    ws.set_landscape()
    ws.set_paper(9)  # A4
    ws.fit_to_pages(1, 0)
    ws.print_area(0, 0, 65, 9)
    ws.center_horizontally()
    ws.set_margins(left=0.4, right=0.4, top=0.5, bottom=0.5)
    ws.set_header("&L&8Dashboard Airbnb España&R&8&D")
    ws.set_footer("&C&8Página &P de &N")
    ws.activate()
    return len(anclas)


def _hoja_comparador(lb: _Libro, datos: pd.DataFrame, sel: Seleccion) -> None:
    """Comparador ciudad × tipo × capacidad con aviso si la muestra es pequeña."""
    ws, f = lb.hojas[HOJA_COMPARADOR], lb.f
    ws.set_column(0, 0, 2)
    ws.set_column(1, 1, 34)
    ws.set_column(2, 2, 24)
    ws.set_column(3, 3, 60)
    ws.write(0, 1, "Comparador: ¿cuánto cuesta un alojamiento como el mío?", f["titulo"])
    ws.set_row(0, 30)
    ws.write(1, 1, "Elige ciudad, tipo y capacidad: el resultado se calcula con los anuncios con precio de la hoja "
                   "Datos.", f["subtitulo"])
    for i, (lbl, val, nombre, lista) in enumerate((
            ("Ciudad", sel.cmp_ciudad, "cmpCiudad", "=listaCiudades"),
            ("Tipo de alojamiento", sel.cmp_tipo, "cmpTipo", "=listaTipos"),
            ("Capacidad (personas)", sel.cmp_capacidad, "cmpCap", "=listaCapacidades"))):
        ws.write(3 + i, 1, lbl, f["selector_lbl"])
        ws.write(3 + i, 2, val, f["selector"])
        lb.nombre(nombre, HOJA_COMPARADOR, _celda(3 + i, 2))
        ws.data_validation(3 + i, 2, 3 + i, 2, {"validate": "list", "source": lista, "error_title": "Valor no válido",
                                                "error_message": "Elige un valor de la lista."})
    m = mascara(datos, sel.cmp_ciudad, sel.cmp_tipo, sel.cmp_capacidad)
    r = resumen_precio(datos, m)
    ws.write(7, 1, "Resultado", f["cabecera"])
    ws.write(7, 2, "Valor", f["cabecera"])
    filas = [
        ("Anuncios con precio (n)", f_n(CRIT_COMPARADOR), "miles", r["n"]),
        ("Mediana del precio por noche", f_mediana(COND_COMPARADOR), "eur", r["mediana"]),
        ("Percentil 25 (P25)", f_percentil(COND_COMPARADOR, 0.25), "eur", r["p25"]),
        ("Percentil 75 (P75)", f_percentil(COND_COMPARADOR, 0.75), "eur", r["p75"]),
        ("Media del precio por noche", f_media(CRIT_COMPARADOR), "eur", r["media"]),
    ]
    for i, (lbl, formula, fmt, valor) in enumerate(filas):
        ws.write(8 + i, 1, lbl, f["celda"])
        _formula(ws, 8 + i, 2, formula, f[fmt], valor)
    n_min = config.MIN_N_GRUPO
    aviso = AVISO_SIN_DATOS if r["n"] == 0 else (f"{AVISO_MUESTRA} (n < {n_min})" if r["n"] < n_min else "")
    ws.merge_range(14, 1, 14, 3, "", f["aviso"])
    _formula(ws, 14, 1, f'=IF(C9=0,"{AVISO_SIN_DATOS}",IF(C9<{n_min},"{AVISO_MUESTRA} (n < {n_min})",""))',
             f["aviso"], aviso)
    ws.conditional_format(14, 1, 14, 3, {"type": "formula", "criteria": '=LEN($B$15)>0',
                                         "format": lb.wb.add_format({"bg_color": "#FFEB9C"})})
    ws.write(8, 3, "La mediana y los percentiles resisten mejor los precios extremos que la media.", f["nota"])
    ws.write(9, 3, "P25-P75: la mitad central de los anuncios comparables está en ese rango.", f["nota"])
    ws.set_row(16, 30)
    ws.merge_range(16, 1, 16, 3, "Precio = cotización por noche con los descuentos aplicados y sin tasas; no es el "
                                 "precio pagado. Comparar con anuncios del mismo tipo, capacidad y ciudad es una "
                                 "referencia, no una tasación.", f["nota"])


def _hoja_renta(lb: _Libro, findings: dict[str, Any]) -> None:
    """ρ de Spearman renta-precio por ciudad y elasticidad del OLS, leídos de findings.json."""
    ws, f = lb.hojas[HOJA_RENTA], lb.f
    rp, o, meta = findings["estadistica"]["renta_precio"], findings["estadistica"]["ols"], findings["meta"]
    af = findings["afirmaciones"]
    ws.set_column(0, 0, 2)
    ws.set_column(1, 1, 40)
    ws.set_column(2, 8, 14)
    ws.write(0, 1, "Renta de la zona y precio", f["titulo"])
    ws.set_row(0, 30)
    ws.merge_range(1, 1, 1, 8, "Resultados del análisis estadístico (Python, reports/findings.json). No se calculan en "
                               "Excel: se copian del análisis para consultarlos junto al dashboard.", f["subtitulo"])
    ws.write(3, 1, "Correlación de Spearman entre la renta neta por persona del distrito y el precio, anuncio a "
                   "anuncio", f["h2"])
    cab = ["Ámbito", "Anuncios", "ρ de Spearman", "IC 95 % inferior", "IC 95 % superior", "p bootstrap",
           "p ajustado (Holm)", "¿Significativa tras Holm?"]
    ws.write_row(4, 1, cab, f["cabecera"])
    ws.set_row(4, 32)
    g = rp["global_distrito"]
    fila = 5
    ws.write(fila, 1, "España (todas las zonas mezcladas)", f["total"])
    ws.write_number(fila, 2, g["n"], f["total_miles"])
    for j, v in enumerate((g["rho"], g["cluster_zona"]["ic_inf"], g["cluster_zona"]["ic_sup"],
                           g["cluster_zona"]["p_boot"])):
        ws.write_number(fila, 3 + j, v, f["total_dec3"])
    ws.write(fila, 7, "No aplica", f["total"])
    # Sin Holm: se muestra la lectura prudente del p bootstrap (p. ej. «En el límite»), no un Sí/No.
    ws.write(fila, 8, str(g["cluster_zona"]["lectura"]).capitalize(), f["total"])
    for c in rp["por_ciudad"]:
        fila += 1
        d = c["distrito"]
        ws.write(fila, 1, c["ciudad"], f["celda"])
        ws.write_number(fila, 2, d["n"], f["miles"])
        for j, v in enumerate((d["rho"], d["cluster_zona"]["ic_inf"], d["cluster_zona"]["ic_sup"],
                               d["cluster_zona"]["p_boot"], d["holm"]["p_holm"])):
            ws.write_number(fila, 3 + j, v, f["dec3"])
        ws.write(fila, 8, SI if d["holm"]["significativo_holm"] else NO, f["celda"])
    ws.conditional_format(6, 3, fila, 3, {"type": "data_bar", "bar_color": "#9DC3E6", "bar_solid": True,
                                          "min_type": "num", "min_value": 0})
    fila += 1
    ws.merge_range(fila, 1, fila, 8, "IC 95 % y p por bootstrap remuestreando distritos censales (la renta solo varía "
                                     "entre distritos). En España (global) no hay Holm: el p sale de "
                                     f"{formato.n0(g['cluster_zona']['n_boot'])} réplicas y se lee con prudencia "
                                     f"(«En el límite» si p está cerca de {formato.n2(meta['alfa'])} o el IC roza el 0)."
                                     " El recuento por "
                                     "ciudad tras Holm es orientativo: hay ciudades con muy pocos distritos.",
                   f["nota"])
    ws.set_row(fila, 30)
    fila += 2
    ws.write(fila, 1, "Regresión OLS sobre ln(precio) con efectos fijos de ciudad (elasticidad)", f["h2"])
    fila += 1
    ws.write_row(fila, 1, ["Indicador", "Valor", "IC 95 % inferior", "IC 95 % superior"], f["cabecera"])
    cr = o["coef_renta"]
    urb = o["sensibilidades"]["solo_urbanas_con_distancia"]["renta"]
    filas = [
        ("Variación del precio asociada a una renta un 10 % mayor (errores agrupados por distrito)",
         cr["pct_por_10pct_renta"], cr["ic_inf_10pct_distrito"], cr["ic_sup_10pct_distrito"], "pct_signo"),
        ("Lo mismo solo en ciudades urbanas, controlando la distancia al centro", urb["pct_por_10pct_renta"],
         urb["ic_inf_10pct"], urb["ic_sup_10pct"], "pct_signo"),
        ("R² ajustado del modelo con las variables del INE", o["r2_ajustado"], None, None, "dec3"),
        ("R² ajustado del modelo sin las variables del INE", o["sin_ine"]["r2_ajustado"], None, None, "dec3"),
        ("Aporte del INE al R² ajustado (puntos porcentuales)", o["delta_r2_ajustado_ine_pp"], None, None, "pct100"),
    ]
    for lbl, v, lo, hi, fmt in filas:
        fila += 1
        ws.write(fila, 1, lbl, f["texto"])
        ws.write_number(fila, 2, v, f[fmt])
        for j, x in enumerate((lo, hi)):
            if x is not None:
                ws.write_number(fila, 3 + j, x, f[fmt])
        ws.set_row(fila, 30)
    fila += 2
    ws.write(fila, 1, "Cómo leerlo", f["h2"])
    eco = rp["ecologico_global"]
    gz = g["cluster_zona"]
    lecturas = {
        "en el límite": (f"Está en el límite de la significación (p bootstrap = {formato.n3(gz['p_boot'])} con "
                         f"{formato.n0(gz['n_boot'])} réplicas, p ≈ {formato.n2(meta['alfa'])}; el IC roza el 0): "
                         "no se apoya ninguna conclusión en ella."),
        "significativa": "Es significativa remuestreando distritos, pero débil.",
        "no significativa": "No es significativa remuestreando distritos.",
    }
    textos = [
        f"Mezclando todas las zonas, la correlación anuncio a anuncio es débil (ρ = {formato.n3(g['rho'])}). "
        + lecturas[gz["lectura"]],
        f"Por ciudad, la estimación puntual es positiva en {rp['n_ciudades_positivas']} de "
        f"{rp['n_ciudades_evaluables']} y modesta (de {formato.n2(rp['rho_ciudad_min']['distrito']['rho'])} a "
        f"{formato.n2(rp['rho_ciudad_max']['distrito']['rho'])}); tras Holm es significativa en "
        f"{rp['n_ciudades_positivas_holm']}. Es un recuento orientativo: las ciudades urbanas tienen solo "
        f"{rp['n_distritos_urbanas_min']}-{rp['n_distritos_urbanas_max']} distritos y el bootstrap por distritos "
        "es inestable con tan pocos.",
        f"Entre distritos de toda España (cada distrito con al menos {config.MIN_N_GRUPO} anuncios cuenta una vez), "
        f"ρ = {formato.n3(eco['rho'])} ({formato.p_texto(eco['p'])}): "
        + ("sin relación significativa." if not eco["significativo"] else "relación significativa."),
        f"Con efectos fijos de ciudad y el resto de controles, una renta un 10 % mayor se asocia con un precio "
        f"{formato.signo_pct(cr['pct_por_10pct_renta'])}, pero las variables del INE solo añaden "
        f"{formato.n1(o['delta_r2_ajustado_ine_pp'])} puntos de R² ajustado"
        + (": la renta explica poco del precio." if af["renta_aporta_poco_r2"] else "."),
        f"Es una asociación, no un efecto causal. La renta es de {meta['anio_ine']} (INE) y los anuncios, de los "
        f"snapshots del {formato.fecha(meta['snapshot_min'])} al {formato.fecha(meta['snapshot_max'])}.",
    ]
    for t in textos:
        fila += 1
        ws.merge_range(fila, 1, fila, 8, "• " + t, f["texto"])
        ws.set_row(fila, 32)
    ws.freeze_panes(5, 0)


def _hoja_resumen(lb: _Libro, findings: dict[str, Any], datos: pd.DataFrame) -> None:
    """Tabla por ciudad (findings.json) con la renta media de la zona calculada con fórmula."""
    ws, f = lb.hojas[HOJA_RESUMEN], lb.f
    d = findings["descriptivo"]
    ws.set_column(0, 0, 2)
    ws.set_column(1, 1, 22)
    ws.set_column(2, 14, 13)
    ws.write(0, 1, "Resumen por ciudad", f["titulo"])
    ws.set_row(0, 30)
    ws.merge_range(1, 1, 1, 14, "Columnas de anuncios, porcentajes y renta mediana: findings.json (todos los anuncios "
                                "salvo donde se indica). Última columna: fórmula sobre la hoja Datos (anuncios con "
                                "precio).", f["subtitulo"])
    ws.set_row(1, 30)
    cab = ["Ciudad o región", "Tipo", "Anuncios (total)", "Anuncios con precio", "% con precio",
           "% emparejado con el INE", "Mediana del precio (€)", "Media del precio (€)", "P25 (€)", "P75 (€)",
           "% anuncios de superhost", "% anuncios de anfitriones con varios anuncios",
           "Renta neta por persona del distrito: mediana de los anuncios (€)",
           "Renta neta media por persona de la zona (anuncios con precio, €)"]
    ws.write_row(3, 1, cab, f["cabecera"])
    ws.set_row(3, 62)
    renta = datos["Renta neta por persona del distrito (€)"].to_numpy(dtype=float)
    ciudades = datos["Ciudad"].to_numpy()
    fila = 3
    for c in d["por_ciudad"]:
        fila += 1
        r = renta[(ciudades == c["grupo"]) & ~np.isnan(renta)]
        ws.write(fila, 1, c["grupo"], f["celda"])
        ws.write(fila, 2, "Urbana" if c["tipo"] == config.TIPO_URBANA else "Región", f["celda"])
        for j, (k, fmt) in enumerate((("n_anuncios", "miles"), ("n", "miles"), ("pct_con_precio", "pct100"),
                                      ("pct_emparejado_ine", "pct100"), ("mediana", "eur"), ("media", "eur"),
                                      ("q1", "eur"), ("q3", "eur"), ("pct_superhost", "pct100"),
                                      ("pct_multianuncio", "pct100"), ("renta_dis_mediana", "eur0"))):
            ws.write_number(fila, 3 + j, c[k], f[fmt])
        _formula(ws, fila, 14, f'=IFERROR(AVERAGEIFS(dRenta,dCiudad,B{fila + 1}),"{SIN_DATO}")', f["eur0"],
                 float(r.mean()) if r.size else SIN_DATO)
    fila += 1
    tot, p, h = findings["dataset"]["union"]["total"], d["precio"], d["hosts"]
    r = renta[~np.isnan(renta)]
    ws.write(fila, 1, "España (todas)", f["total"])
    ws.write(fila, 2, "", f["total"])
    valores = [(tot["anuncios"], "total_miles"), (p["n"], "total_miles"),
               (p["n"] / tot["anuncios"] * 100, "total_pct100"), (tot["pct_distrito"], "total_pct100"),
               (p["mediana"], "total_eur"), (p["media"], "total_eur"), (p["q1"], "total_eur"), (p["q3"], "total_eur"),
               (h["pct_anuncios_superhost"], "total_pct100"), (h["pct_anuncios_multianuncio"], "total_pct100"),
               (d["renta_dis"]["mediana"], "total_eur0")]
    for j, (v, fmt) in enumerate(valores):
        ws.write_number(fila, 3 + j, v, f[fmt])
    _formula(ws, fila, 14, f'=IFERROR(AVERAGEIFS(dRenta,dCiudad,"*"),"{SIN_DATO}")', f["total_eur0"],
             float(r.mean()) if r.size else SIN_DATO)
    ultima = fila - 1
    ws.conditional_format(4, 7, ultima, 7, {"type": "data_bar", "bar_color": "#9DC3E6", "bar_solid": True})
    for col in (11, 12):
        ws.conditional_format(4, col, ultima, col, {"type": "2_color_scale", "min_color": "#FFFFFF",
                                                    "max_color": "#BDD7EE"})
    ws.write(fila + 2, 1, "Ordenado de mayor a menor mediana del precio. Fuente: reports/findings.json "
                          "(descriptivo.por_ciudad).", f["nota"])
    ws.freeze_panes(4, 2)


def _hoja_inicio(lb: _Libro, findings: dict[str, Any], n_datos: int) -> None:
    """Portada: pregunta, cómo usar el libro, índice con hipervínculos, fuentes y limitaciones."""
    ws, f = lb.hojas[HOJA_INICIO], lb.f
    meta, ine = findings["meta"], findings["dataset"]["fuente_ine"]
    lic = findings["dataset"]["fuente_airbnb"]["licencia"]
    ws.set_column(0, 0, 2)
    ws.set_column(1, 1, 26)
    ws.set_column(2, 2, 100)
    fila = 0
    ws.write(fila, 1, "Airbnb en España y renta de la zona · Dashboard operativo", f["titulo"])
    ws.set_row(fila, 30)
    fila += 1
    ws.write(fila, 1, f"Proyecto Final del Máster en Data Analytics · Autor: {meta['autor']}", f["subtitulo"])
    fila += 2

    def seccion(titulo: str) -> None:
        nonlocal fila
        ws.write(fila, 1, titulo, f["h2"])
        ws.write(fila, 2, "", f["h2"])
        fila += 1

    def par(etiqueta: str, texto: str, alto: int | None = None) -> None:
        nonlocal fila
        ws.write(fila, 1, etiqueta, f["negrita"])
        ws.write(fila, 2, texto, f["texto"])
        if alto:
            ws.set_row(fila, alto)
        fila += 1

    seccion("Pregunta de negocio")
    par("Pregunta", "Qué explica el precio por noche de un alojamiento turístico de Airbnb en España y cómo se "
                    "relaciona con la renta de la zona (distrito censal del INE).", 30)
    seccion("Cómo usar el libro")
    pasos = [
        ("1. Dashboard", f"Elige Ciudad (o «{TODAS}») y Tipo de alojamiento (o «{TODOS}») en las celdas amarillas. "
                         "Las tarjetas y los gráficos de barras se recalculan con fórmulas de Excel."),
        ("2. Comparador", "Elige ciudad, tipo y capacidad para ver el precio típico de anuncios comparables, con un "
                          f"aviso si hay menos de {config.MIN_N_GRUPO} anuncios."),
        ("3. Renta y precio", "Consulta la correlación renta-precio por ciudad y la elasticidad de la regresión."),
        ("4. Datos", f"Filtra los {formato.n0(n_datos)} anuncios con precio con los botones de la cabecera."),
        ("Recálculo", "El libro no tiene macros. Si Excel abre en modo de cálculo manual, pulsa F9."),
    ]
    for etq, texto in pasos:
        par(etq, texto, 30)
    seccion("Índice")
    descripciones = {
        HOJA_DASHBOARD: "Selectores, indicadores (KPI) y gráficos.",
        HOJA_COMPARADOR: "Precio típico de anuncios comparables (ciudad, tipo y capacidad).",
        HOJA_RENTA: "Correlación de Spearman, IC 95 %, Holm y elasticidad del OLS (de findings.json).",
        HOJA_RESUMEN: "Tabla por ciudad: oferta, precio, anfitriones y renta.",
        HOJA_DATOS: "Tabla de Excel con los anuncios con precio (sin datos personales).",
        HOJA_CALCULOS: "Tablas auxiliares con las fórmulas que alimentan los gráficos.",
    }
    for hoja, desc in descripciones.items():
        ws.write_url(fila, 1, f"internal:'{hoja}'!A1", f["enlace"], string=hoja, tip=f"Ir a la hoja {hoja}")
        ws.write(fila, 2, desc, f["texto"])
        fila += 1
    seccion("Fuentes")
    par("Inside Airbnb", f"insideairbnb.com («Get the Data»), {findings['descriptivo']['n_ciudades']} ciudades y "
                         f"regiones de España. Licencia {lic['nombre']}. Datos limpiados y transformados para este "
                         "proyecto.", 30)
    par("INE", f"Atlas de Distribución de Renta de los Hogares (ADRH), año {meta['anio_ine']}, por distrito censal. "
               f"«{ine['cita']}». Fecha de actualización del ADRH: {formato.fecha(ine['actualizacion']['fecha'])}.", 30)
    ws.write(fila, 1, "Snapshots por ciudad", f["negrita"])
    fila += 1
    ws.write_row(fila, 1, ["Ciudad o región", "Fecha del snapshot de Inside Airbnb"], f["cabecera"])
    for c in sorted(meta["ciudades"].values(), key=lambda c: _clave_alfabetica(c["nombre"])):
        fila += 1
        ws.write(fila, 1, c["nombre"], f["celda"])
        ws.write(fila, 2, formato.fecha(c["fecha_snapshot"]), f["celda"])
    fila += 2
    seccion("Limitaciones clave")
    limitaciones = [
        ("Precio", "Es la cotización por noche de Airbnb para unas fechas concretas, con los descuentos aplicados y "
                   "sin tasas. No es el precio pagado por un huésped."),
        ("Asociación", "Los datos son observacionales: las relaciones son asociaciones, no efectos causales."),
        ("Desfase temporal", f"La renta del INE es de {meta['anio_ine']} y los anuncios, de snapshots del "
                             f"{formato.fecha(meta['snapshot_min'])} al {formato.fecha(meta['snapshot_max'])}."),
        ("Renta y precio", "La renta de la zona explica poco del precio: pesan mucho más el tipo de alojamiento, la "
                           "capacidad y la ciudad (ver la hoja Renta y precio)."),
        ("Muestra", "Solo anuncios con precio publicado; los que no lo tienen no son aleatorios. Coordenadas "
                    "anonimizadas por Airbnb."),
    ]
    for etq, texto in limitaciones:
        par(etq, texto, 30)
    fila += 1
    ws.write(fila, 1, "Generado automáticamente por src/dashboard_excel.py (python run_pipeline.py) desde "
                      "data/processed y reports/findings.json: ninguna cifra está escrita a mano.", f["nota"])
    ws.set_first_sheet()


def construir_libro(datos: pd.DataFrame, findings: dict[str, Any], ruta: Path,
                    sel: Seleccion | None = None) -> dict[str, Any]:
    """Construye el libro de Excel en ``ruta``. Devuelve los valores cacheados de los KPI y las tablas."""
    sel = sel or Seleccion()
    lb = _Libro(ruta, findings)
    n = _hoja_datos(lb, datos)
    tablas = tablas_graficos(datos, findings, sel)
    pos = _hoja_calculos(lb, tablas)
    _rellenar_calculos(lb, pos, tablas, sel)
    kpi = resumen_precio(datos, mascara(datos, sel.ciudad, sel.tipo))
    n_graficos = _hoja_dashboard(lb, pos, tablas, kpi, findings, sel)
    _hoja_comparador(lb, datos, sel)
    _hoja_renta(lb, findings)
    _hoja_resumen(lb, findings, datos)
    _hoja_inicio(lb, findings, n)
    lb.wb.close()
    tam = ruta.stat().st_size
    if tam >= config.LIMITE_SUBIDA_BYTES:
        from src.errores import ErrorPipeline
        raise ErrorPipeline(f"{ruta.name} pesa {tam} bytes (límite {config.LIMITE_SUBIDA_BYTES}).")
    return {"kpi": kpi, "tablas": tablas, "n_datos": n, "n_graficos": n_graficos, "bytes": tam}


# --------------------------------------------------------------------------
# Vista previa con matplotlib (no es una captura de Excel)
# --------------------------------------------------------------------------
def vista_previa(resultado: dict[str, Any], findings: dict[str, Any], ruta: Path, sel: Seleccion | None = None) -> None:
    """PNG con los mismos gráficos del dashboard para la selección por defecto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Estilo por defecto de matplotlib: la imagen no depende del estilo que hayan fijado otras etapas.
    with plt.style.context("default"):
        _dibujar_vista_previa(resultado, findings, ruta, sel or Seleccion(), plt, FuncFormatter)


def _dibujar_vista_previa(resultado: dict[str, Any], findings: dict[str, Any], ruta: Path, sel: Seleccion,
                          plt: Any, FuncFormatter: Any) -> None:  # noqa: N803
    """Dibuja y guarda la vista previa (llamada dentro de un contexto de estilo fijo)."""
    t, kpi = resultado["tablas"], resultado["kpi"]
    eur = FuncFormatter(lambda x, _: formato.eur0(x))
    miles = FuncFormatter(lambda x, _: formato.n0(x))
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), dpi=100)
    fig.suptitle(TITULO_VISTA_PREVIA, fontsize=13, color=AZUL_OSCURO, fontweight="bold")

    def barras_h(a: Any, tabla: pd.DataFrame, titulo: str) -> None:
        y = pd.to_numeric(tabla["mediana"], errors="coerce").fillna(0)
        a.barh(tabla["etiqueta"], y, color=AZUL)
        a.invert_yaxis()
        a.set_title(titulo, fontsize=10, color=AZUL_OSCURO)
        a.xaxis.set_major_formatter(eur)
        a.set_xlabel("Mediana del precio por noche (€)", fontsize=9)
        for i, v in enumerate(y):
            a.text(v, i, " " + formato.eur0(v), va="center", fontsize=7, color=GRIS)

    def columnas(a: Any, tabla: pd.DataFrame, titulo: str, xlabel: str) -> None:
        a.bar(tabla["etiqueta"], tabla["n"], color=AZUL)
        a.set_title(titulo, fontsize=10, color=AZUL_OSCURO)
        a.yaxis.set_major_formatter(miles)
        a.set_xlabel(xlabel, fontsize=9)
        a.set_ylabel("Anuncios con precio", fontsize=9)
        a.tick_params(axis="x", labelsize=7, rotation=20)

    barras_h(ax[0, 0], t["ciudad"], f"Mediana del precio por ciudad · {sel.tipo}")
    columnas(ax[0, 1], t["tramo_precio"], f"Anuncios por tramo de precio · {sel.ciudad} · {sel.tipo}",
             "Tramo de precio por noche (€)")
    barras_h(ax[0, 2], t["tipo"], f"Mediana del precio por tipo de alojamiento · {sel.ciudad}")
    columnas(ax[1, 0], t["capacidad"], f"Anuncios por capacidad · {sel.ciudad} · {sel.tipo}", "Capacidad (personas)")
    dist = t["distritos"]
    a = ax[1, 1]
    for k, ciudad in enumerate(dist["Ciudad"].drop_duplicates()):
        s = dist[dist["Ciudad"] == ciudad]
        a.scatter(s["renta"], s["mediana"], s=12, color=OKABE_ITO[k % len(OKABE_ITO)], label=ciudad,
                  marker="osD^xosD^"[k % 9])
    a.set_yscale("log")
    a.set_title(f"Renta y mediana del precio por distrito censal (n ≥ {config.MIN_N_GRUPO})", fontsize=10,
                color=AZUL_OSCURO)
    a.set_xlabel(f"Renta neta media por persona (€, INE {findings['meta']['anio_ine']})", fontsize=9)
    a.set_ylabel("Mediana del precio (€)", fontsize=9)
    a.xaxis.set_major_formatter(eur)
    a.yaxis.set_major_formatter(eur)
    a.legend(fontsize=7, ncol=2)
    a = ax[1, 2]
    a.axis("off")
    lineas = [f"KPI · {sel.ciudad} · {sel.tipo}", "",
              f"Anuncios con precio: {formato.n0(kpi['n'])}", f"Mediana: {formato.eur2(kpi['mediana'])}",
              f"Media: {formato.eur2(kpi['media'])}",
              f"P25-P75: {formato.eur2(kpi['p25'])} – {formato.eur2(kpi['p75'])}",
              f"% superhost: {formato.pct1(kpi['pct_superhost'] * 100)}",
              f"% anfitriones con varios anuncios: {formato.pct1(kpi['pct_multianuncio'] * 100)}",
              f"Renta neta media de la zona: {formato.eur0(kpi['renta_media'])}"]
    a.text(0.02, 0.95, "\n".join(lineas), va="top", fontsize=11, color=AZUL_OSCURO, family="sans-serif")
    for a in ax.flat:
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=100, metadata={"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# README del dashboard
# --------------------------------------------------------------------------
def texto_readme(resultado: dict[str, Any]) -> str:
    """Documentación del libro (sin cifras de resultados; solo parámetros de config)."""
    cap = ", ".join(etiquetas_tramos_capacidad())
    precio = ", ".join(etiquetas_tramos_precio())
    nombre = config.RUTA_DASHBOARD_XLSX.name
    return f"""# Dashboard operativo en Excel

[{nombre}]({nombre}) es el dashboard del proyecto. Lo genera `src/dashboard_excel.py` (etapa del pipeline
`python run_pipeline.py`) a partir de `data/processed/airbnb_espana_ine.parquet` y `reports/findings.json`.
Ninguna cifra está escrita a mano. No tiene macros y funciona sin Python.

![{TITULO_VISTA_PREVIA}](../reports/figures/{config.NOMBRE_VISTA_PREVIA_DASHBOARD})

*{TITULO_VISTA_PREVIA}.*

## Hojas

| Hoja | Qué contiene |
|---|---|
| {HOJA_INICIO} | Pregunta de negocio, cómo usar el libro, índice con hipervínculos, fuentes (Inside Airbnb e INE con su cita y fecha de actualización), fechas de los snapshots y limitaciones. |
| {HOJA_DASHBOARD} | Desplegables de Ciudad (incluye «{TODAS}») y Tipo de alojamiento (incluye «{TODOS}»), tarjetas KPI y {resultado['n_graficos']} gráficos nativos de Excel. |
| {HOJA_COMPARADOR} | Ciudad + tipo + capacidad → anuncios, mediana, P25, P75 y media, con aviso si hay menos de {config.MIN_N_GRUPO} anuncios. |
| {HOJA_RENTA} | ρ de Spearman renta-precio por ciudad con IC 95 %, p y Holm, y la elasticidad del OLS. Se leen de `findings.json` (no se calculan en Excel). |
| {HOJA_RESUMEN} | Una fila por ciudad: anuncios, con precio, % emparejado con el INE, mediana, media, % superhost, % multi-anuncio y renta. |
| {HOJA_DATOS} | Tabla de Excel `{NOMBRE_TABLA_DATOS}` con los anuncios **con precio** y {len(COLUMNAS_DATOS)} columnas, con filtros. |
| {HOJA_CALCULOS} | Listas de los desplegables, criterios de filtro y tablas auxiliares (fórmulas) que alimentan los gráficos. |

## Cómo usar los selectores

- **Dashboard:** las celdas amarillas `C4` (Ciudad) y `C5` (Tipo de alojamiento) son listas desplegables. Al cambiarlas se
  recalculan las tarjetas (anuncios con precio, mediana, media, P25, P75, % superhost, % de anuncios de anfitriones con
  varios anuncios y renta neta media por persona de la zona) y cuatro gráficos: mediana por ciudad (responde al tipo),
  anuncios por tramo de precio y por capacidad (responden a los dos) y mediana por tipo (responde a la ciudad). La
  dispersión renta-precio por distrito censal muestra siempre todos los distritos con al menos {config.MIN_N_GRUPO}
  anuncios con precio.
- **Base de los porcentajes:** las tarjetas de % superhost y % de anuncios de anfitriones con varios anuncios se
  calculan sobre los anuncios **con precio** (la hoja {HOJA_DATOS}). El README y el informe dan esos porcentajes sobre
  **todos** los anuncios, así que difieren ligeramente; no es una incoherencia.
- **Comparador:** elige ciudad, tipo y capacidad ({cap}).
- Tramos de precio del gráfico de columnas (intervalos cerrados por la izquierda): {precio}.

## Fórmulas y compatibilidad

- Excel 2016 o posterior y Microsoft 365 (Windows y Mac). Solo se usan COUNTIFS, AVERAGEIFS, IF, IFERROR,
  CONCATENATE y fórmulas matriciales clásicas `{{=MEDIAN(IF(...))}}` y `{{=PERCENTILE(IF(...))}}`. No se usan LET,
  LAMBDA, FILTER, UNIQUE, SORT ni matrices dinámicas. En Excel en español se muestran traducidas
  (CONTAR.SI.CONJUNTO, PROMEDIO.SI.CONJUNTO, MEDIANA, PERCENTIL, SI.ERROR...).
- Las fórmulas usan rangos con nombre legibles (`dPrecio`, `dCiudad`, `dTipo`, `selCiudad`, `selTipo`...;
  ver *Fórmulas > Administrador de nombres*). Los rangos cubren las filas actuales de la tabla: si se añaden anuncios
  a mano, hay que ampliar los nombres o regenerar el libro.
- El libro guarda los valores calculados para la selección por defecto y pide un recálculo completo al abrirse.
  Si Excel está en cálculo manual, pulsa F9. Con {formato.n0(resultado['n_datos'])} filas, el recálculo de las
  fórmulas matriciales puede tardar unos segundos.
- Los formatos numéricos usan el separador del sistema: en una configuración regional española se ve 1.234,56 €.
- **LibreOffice Calc:** abre el libro, los desplegables y los gráficos; las fórmulas son estándar y deberían
  recalcularse, pero no se ha comprobado en este proyecto (LibreOffice no estaba instalado).
- **Google Sheets:** al importar el .xlsx convierte las fórmulas matriciales en `ARRAYFORMULA` y mantiene las listas
  desplegables; algunos formatos de gráfico (escala logarítmica, marcadores) pueden cambiar. No comprobado.

## Cómo regenerarlo

```bash
python run_pipeline.py --sin-descarga    # todo el pipeline (incluye el dashboard)
python -m src.dashboard_excel            # solo el dashboard, la vista previa y este README
python -B -m pytest -q tests/test_dashboard_excel.py
```

## Limitaciones

- Precio = cotización por noche con los descuentos aplicados y sin tasas; no es el precio pagado.
- Asociaciones, no efectos causales. La renta del INE es anterior a los anuncios (desfase temporal).
- La renta de la zona explica poco del precio: pesan mucho más el tipo, la capacidad y la ciudad.
- Solo anuncios con precio. Sin datos personales: no incluye nombres, URLs, fotos, coordenadas ni `host_id`.
- La apertura en Excel de escritorio no se ha comprobado de forma automática: los tests leen el archivo con
  openpyxl y recalculan las fórmulas con un evaluador propio en Python.
"""


def construir(df: pd.DataFrame, findings: dict[str, Any], ruta_xlsx: Path = config.RUTA_DASHBOARD_XLSX,
              ruta_png: Path = config.RUTA_VISTA_PREVIA_DASHBOARD,
              ruta_readme: Path = config.RUTA_DASHBOARD_README) -> dict[str, Any]:
    """Genera el libro, la vista previa y el README del dashboard."""
    datos = tabla_datos(df)
    resultado = construir_libro(datos, findings, ruta_xlsx)
    vista_previa(resultado, findings, ruta_png)
    with open(ruta_readme, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(texto_readme(resultado))
    logger.info("Dashboard: %s (%s bytes, %s filas, %s gráficos).", ruta_xlsx.name, f"{resultado['bytes']:,}",
                resultado["n_datos"], resultado["n_graficos"])
    return resultado


def main() -> int:
    """Regenera el dashboard desde los artefactos ya generados por el pipeline."""
    sys.dont_write_bytecode = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    findings = json.loads(config.RUTA_FINDINGS.read_text(encoding="utf-8"))
    construir(pd.read_parquet(config.RUTA_PARQUET), findings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
