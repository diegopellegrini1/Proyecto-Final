"""Carga cacheada de los datos procesados para el dashboard (no limpia nada)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import config


@st.cache_data(show_spinner="Cargando datos…")
def cargar_datos(ruta: str, mtime: float) -> pd.DataFrame:
    """Lee las columnas del dashboard del Parquet procesado; ``mtime`` invalida la caché si cambia."""
    return pd.read_parquet(ruta, columns=list(config.COLUMNAS_DASHBOARD))


@st.cache_data(show_spinner=False)
def cargar_geojson(ruta: str, mtime: float) -> dict[str, Any]:
    """Lee el GeoJSON de barrios."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def columnas_disponibles(ruta: Path) -> set[str]:
    """Columnas del Parquet sin leer los datos."""
    import pyarrow.parquet as pq
    return set(pq.read_schema(ruta).names)
