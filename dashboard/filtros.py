"""Funciones puras del dashboard (sin Streamlit): filtros, KPIs, límites del slider y exportación CSV."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

CLAVE_DISTRITO = "filtro_distrito"
CLAVE_BARRIO = "filtro_barrio"
CLAVE_TIPO = "filtro_tipo"
CLAVE_PRECIO = "filtro_precio"
CLAVE_TEXTO = "filtro_texto"
CLAVE_VISTA_MAPA = "vista_mapa"
MAX_LONGITUD_TEXTO = 100
PASO_PRECIO = 5.0

ETIQUETAS_KPI: dict[str, str] = {
    "anuncios": "Anuncios",
    "precio_mediano": "Precio mediano",
    "precio_medio": "Precio medio",
    "ocupacion_media": "Ocupación estimada media",
    "pct_superhost": "% superhost",
    "pct_multianuncio": "% multi-anuncio",
    "valoracion_media": "Valoración media",
}


def aplicar_filtros(df: pd.DataFrame, distritos: Sequence[str], barrios: Sequence[str], tipos: Sequence[str],
                    rango_precio: tuple[float, float] | None, texto: str = "") -> pd.DataFrame:
    """Filtra por distrito, barrio, tipo, rango de precio (inclusivo) y texto en el nombre.

    Listas vacías o ``None`` no filtran. La búsqueda de texto es literal (``regex=False``),
    sin distinguir mayúsculas, recortada a ``MAX_LONGITUD_TEXTO`` caracteres e ignorada si está vacía.
    """
    mascara = pd.Series(True, index=df.index)
    if distritos:
        mascara &= df["neighbourhood_group_cleansed"].isin(list(distritos))
    if barrios:
        mascara &= df["neighbourhood_cleansed"].isin(list(barrios))
    if tipos:
        mascara &= df["tipo_habitacion"].isin(list(tipos))
    if rango_precio is not None:
        lo, hi = rango_precio
        mascara &= df["price"].between(lo, hi)
    termino = (texto or "").strip()[:MAX_LONGITUD_TEXTO]
    if termino:
        mascara &= df["name"].astype("string").str.contains(termino, case=False, regex=False, na=False)
    return df[mascara.fillna(False).astype(bool)]


def opciones_barrio(df: pd.DataFrame, distritos: Sequence[str]) -> list[str]:
    """Barrios disponibles (en cascada según los distritos elegidos), ordenados."""
    sub = df[df["neighbourhood_group_cleansed"].isin(list(distritos))] if distritos else df
    return sorted(sub["neighbourhood_cleansed"].dropna().astype(str).unique())


def limites_precio(df: pd.DataFrame) -> tuple[float, float, tuple[float, float]]:
    """Límites del slider (N2): floor(min/5)·5 y ceil(max/5)·5; valor por defecto hasta ceil(P99/10)·10."""
    precios = df["price"].dropna()
    if precios.empty:
        return 0.0, PASO_PRECIO, (0.0, PASO_PRECIO)
    lo = math.floor(precios.min() / PASO_PRECIO) * PASO_PRECIO
    hi = math.ceil(precios.max() / PASO_PRECIO) * PASO_PRECIO
    defecto_hi = min(hi, math.ceil(precios.quantile(0.99) / 10) * 10)
    return float(lo), float(hi), (float(lo), float(defecto_hi))


def calcular_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    """KPIs del subconjunto filtrado. Con 0 filas, todo es ``None`` salvo ``anuncios`` = 0."""
    if df.empty:
        return {k: (0 if k == "anuncios" else None) for k in ETIQUETAS_KPI}
    activos = df[df["es_activo_probable"]]
    return {
        "anuncios": int(len(df)),
        "precio_mediano": float(df["price"].median()),
        "precio_medio": float(df["price"].mean()),
        "ocupacion_media": float(activos["ocupacion_estimada"].mean() * 100) if len(activos) else None,
        "pct_superhost": float(df["host_is_superhost"].mean() * 100),
        "pct_multianuncio": float(df["es_multianuncio"].mean() * 100),
        "valoracion_media": float(df["review_scores_rating"].mean()) if df["review_scores_rating"].notna().any() else None,
    }


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


def a_csv(df: pd.DataFrame) -> bytes:
    """CSV UTF-8 con BOM (Excel en español) y celdas de texto neutralizadas."""
    return neutralizar_formulas(df).to_csv(index=False).encode("utf-8-sig")


def resumen_kpis_texto(kpis: dict[str, Any]) -> dict[str, str]:
    """Textos de los KPIs en formato es-ES (– si no hay dato)."""
    from src import formato as fmt
    return {
        "anuncios": fmt.n0(kpis["anuncios"]),
        "precio_mediano": fmt.eur2(kpis["precio_mediano"]),
        "precio_medio": fmt.eur2(kpis["precio_medio"]),
        "ocupacion_media": fmt.pct1(kpis["ocupacion_media"]),
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
