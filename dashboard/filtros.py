"""Funciones puras del dashboard (sin Streamlit): filtros, KPIs, comparador, límites de precio y exportación CSV."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import config

CLAVE_DISTRITO = "filtro_distrito"
CLAVE_BARRIO = "filtro_barrio"
CLAVE_TIPO = "filtro_tipo"
CLAVE_PRECIO_MIN = "filtro_precio_min"
CLAVE_PRECIO_MAX = "filtro_precio_max"
CLAVE_CAPACIDAD = "filtro_capacidad"
CLAVE_SUPERHOST = "filtro_superhost"
CLAVE_EXCLUIR_LARGAS = "filtro_excluir_largas"
CLAVE_TEXTO = "filtro_texto"
CLAVE_VISTA_MAPA = "vista_mapa"
CLAVE_COMP_TIPO = "comparador_tipo"
CLAVE_COMP_CAPACIDAD = "comparador_capacidad"
CLAVE_COMP_NIVEL = "comparador_nivel"
CLAVE_COMP_ZONA = "comparador_zona"
MAX_LONGITUD_TEXTO = 100
PASO_PRECIO = 5.0
NOCHES_LARGA_ESTANCIA = config.MIN_NIGHTS_LARGA_ESTANCIA
SUPERHOST_OPCIONES: dict[str, bool | None] = {"Todos": None, "Solo superhosts": True, "Solo no superhosts": False}
SEPARADOR_CSV = ";"
DECIMAL_CSV = ","

ETIQUETAS_KPI: dict[str, str] = {
    "anuncios": "Anuncios",
    "precio_mediano": "Precio mediano",
    "precio_medio": "Precio medio",
    "no_disponible_media": "% días no disponibles (próx. 365)",
    "pct_superhost": "% superhost",
    "pct_multianuncio": "% multi-anuncio",
    "valoracion_media": "Valoración media",
}


def es_estancia_larga(df: pd.DataFrame) -> pd.Series:
    """True si la estancia mínima es de 30 noches o más o la cotización usa tarifa mensual."""
    larga = df["minimum_nights"].ge(NOCHES_LARGA_ESTANCIA).fillna(False).astype(bool)
    if "es_tarifa_mensual" in df:
        larga |= df["es_tarifa_mensual"].fillna(False).astype(bool)
    return larga


def aplicar_filtros(df: pd.DataFrame, distritos: Sequence[str], barrios: Sequence[str], tipos: Sequence[str],
                    rango_precio: tuple[float, float] | None, texto: str = "", *,
                    capacidad: tuple[int, int] | None = None, superhost: bool | None = None,
                    excluir_largas: bool = False) -> pd.DataFrame:
    """Filtra por distrito, barrio, tipo, rango de precio (inclusivo), texto, capacidad, superhost y estancia.

    Listas vacías o ``None`` no filtran. La búsqueda de texto es literal (``regex=False``),
    sin distinguir mayúsculas, recortada a ``MAX_LONGITUD_TEXTO`` caracteres e ignorada si está vacía.
    ``excluir_largas`` quita las estancias mínimas de 30 noches o más y las tarifas mensuales.
    """
    mascara = pd.Series(True, index=df.index)
    if distritos:
        mascara &= df["neighbourhood_group_cleansed"].isin(list(distritos))
    if barrios:
        mascara &= df["neighbourhood_cleansed"].isin(list(barrios))
    if tipos:
        mascara &= df["tipo_habitacion"].isin(list(tipos))
    if rango_precio is not None:
        lo, hi = sorted(rango_precio)
        mascara &= df["price"].between(lo, hi)
    if capacidad is not None:
        mascara &= df["accommodates"].between(*sorted(capacidad))
    if superhost is not None:
        mascara &= df["host_is_superhost"].astype(bool) == superhost
    if excluir_largas:
        mascara &= ~es_estancia_larga(df)
    termino = (texto or "").strip()[:MAX_LONGITUD_TEXTO]
    if termino:
        mascara &= df["name"].astype("string").str.contains(termino, case=False, regex=False, na=False)
    return df[mascara.fillna(False).astype(bool)]


def opciones_barrio(df: pd.DataFrame, distritos: Sequence[str]) -> list[str]:
    """Barrios disponibles (en cascada según los distritos elegidos), ordenados."""
    sub = df[df["neighbourhood_group_cleansed"].isin(list(distritos))] if distritos else df
    return sorted(sub["neighbourhood_cleansed"].dropna().astype(str).unique())


def seleccion_invalidada(seleccion: Sequence[str] | None, opciones: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separa una selección previa en (valores que siguen siendo válidos, valores que la cascada invalida)."""
    previa = list(seleccion or [])
    validas = set(opciones)
    return [v for v in previa if v in validas], [v for v in previa if v not in validas]


def limites_precio(df: pd.DataFrame) -> tuple[float, float, tuple[float, float]]:
    """Límites de precio (N2): floor(min/5)·5 y ceil(max/5)·5; valor por defecto hasta ceil(P99/10)·10."""
    precios = df["price"].dropna()
    if precios.empty:
        return 0.0, PASO_PRECIO, (0.0, PASO_PRECIO)
    lo = math.floor(precios.min() / PASO_PRECIO) * PASO_PRECIO
    hi = math.ceil(precios.max() / PASO_PRECIO) * PASO_PRECIO
    defecto_hi = min(hi, math.ceil(precios.quantile(0.99) / 10) * 10)
    return float(lo), float(hi), (float(lo), float(defecto_hi))


def limites_capacidad(df: pd.DataFrame) -> tuple[int, int]:
    """Capacidad mínima y máxima (personas) del conjunto."""
    cap = df["accommodates"].dropna()
    if cap.empty:
        return 1, 1
    return int(cap.min()), int(cap.max())


def calcular_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    """KPIs del subconjunto filtrado. Con 0 filas, todo es ``None`` salvo ``anuncios`` = 0."""
    if df.empty:
        return {k: (0 if k == "anuncios" else None) for k in ETIQUETAS_KPI}
    activos = df[df["es_activo_probable"]]
    return {
        "anuncios": int(len(df)),
        "precio_mediano": float(df["price"].median()),
        "precio_medio": float(df["price"].mean()),
        "no_disponible_media": float(activos["no_disponible_365"].mean() * 100) if len(activos) else None,
        "pct_superhost": float(df["host_is_superhost"].mean() * 100),
        "pct_multianuncio": float(df["es_multianuncio"].mean() * 100),
        "valoracion_media": float(df["review_scores_rating"].mean()) if df["review_scores_rating"].notna().any() else None,
    }


def comparables(df: pd.DataFrame, tipo: str | None, capacidad: int | None, zona_columna: str | None,
                zona: str | None, min_n: int) -> dict[str, Any]:
    """Anuncios comparables (mismo tipo, misma capacidad y misma zona): n, mediana y P25-P75.

    ``aviso`` es True si hay menos de ``min_n`` comparables (la mediana es poco fiable).
    """
    mascara = pd.Series(True, index=df.index)
    if tipo:
        mascara &= df["tipo_habitacion"] == tipo
    if capacidad is not None:
        mascara &= df["accommodates"] == capacidad
    if zona_columna and zona:
        mascara &= df[zona_columna].astype(str) == zona
    precios = df.loc[mascara, "price"].dropna()
    n = int(len(precios))
    if n == 0:
        return {"n": 0, "mediana": None, "p25": None, "p75": None, "aviso": True}
    return {"n": n, "mediana": float(precios.median()), "p25": float(precios.quantile(0.25)),
            "p75": float(precios.quantile(0.75)), "aviso": n < min_n}


def muestrear(df: pd.DataFrame, maximo: int, semilla: int) -> tuple[pd.DataFrame, bool]:
    """Muestra aleatoria reproducible si ``df`` supera ``maximo`` filas. Devuelve (datos, se_muestreó)."""
    if len(df) <= maximo:
        return df, False
    return df.sample(n=maximo, random_state=semilla), True


def neutralizar_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Antepone ``'`` a celdas de texto que empiezan por =, +, - o @ (inyección de fórmulas, N14)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            s = out[col].astype("string")
            peligrosas = s.str.match(r"^[=+\-@]", na=False)
            out[col] = s.where(~peligrosas, "'" + s)
    return out


def con_etiquetas(df: pd.DataFrame, etiquetas: dict[str, str]) -> pd.DataFrame:
    """Renombra columnas a etiquetas legibles en español (las que no tengan etiqueta se dejan igual)."""
    return df.rename(columns={c: etiquetas.get(c, c) for c in df.columns})


def a_csv(df: pd.DataFrame) -> bytes:
    """CSV para Excel en español: separador ``;``, coma decimal, UTF-8 con BOM y fórmulas neutralizadas."""
    return neutralizar_formulas(df).to_csv(index=False, sep=SEPARADOR_CSV, decimal=DECIMAL_CSV).encode("utf-8-sig")


def resumen_kpis_texto(kpis: dict[str, Any]) -> dict[str, str]:
    """Textos de los KPIs en formato es-ES (– si no hay dato)."""
    from src import formato as fmt
    return {
        "anuncios": fmt.n0(kpis["anuncios"]),
        "precio_mediano": fmt.eur2(kpis["precio_mediano"]),
        "precio_medio": fmt.eur2(kpis["precio_medio"]),
        "no_disponible_media": fmt.pct1(kpis["no_disponible_media"]),
        "pct_superhost": fmt.pct1(kpis["pct_superhost"]),
        "pct_multianuncio": fmt.pct1(kpis["pct_multianuncio"]),
        "valoracion_media": fmt.n2(kpis["valoracion_media"]),
    }


def mediana_por(df: pd.DataFrame, grupo: str, min_n: int = 1) -> pd.DataFrame:
    """Mediana de precio y n por ``grupo`` (n ≥ min_n), ordenada de mayor a menor mediana."""
    if df.empty:
        return pd.DataFrame(columns=[grupo, "mediana", "n"])
    g = df.groupby(grupo, observed=True)["price"].agg(mediana="median", n="size").reset_index()
    g = g[g["n"] >= min_n]
    return g.sort_values(["mediana", grupo], ascending=[False, True]).reset_index(drop=True)


def rango_color(serie: pd.Series) -> list[float]:
    """Rango de color P5-P95 (solo visual) para mapas."""
    s = serie.dropna()
    if s.empty:
        return [0.0, 1.0]
    lo, hi = np.quantile(s, [0.05, 0.95])
    return [float(lo), float(max(hi, lo + 1))]
