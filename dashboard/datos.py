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


def validar_geojson(ruta: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Devuelve (GeoJSON, None) si es utilizable o (None, mensaje accionable) si falta o está dañado."""
    if not ruta.exists():
        return None, (f"No se encuentra {ruta.name} en data/raw: la vista de barrios queda desactivada. "
                      "Ejecuta python run_pipeline.py para descargarlo.")
    try:
        geojson = cargar_geojson(str(ruta), ruta.stat().st_mtime)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{ruta.name} no se puede leer ({type(exc).__name__}): la vista de barrios queda desactivada."
    feats = geojson.get("features") or []
    if not feats or any("neighbourhood" not in (ft.get("properties") or {}) for ft in feats):
        return None, f"{ruta.name} no tiene barrios con la propiedad «neighbourhood»: la vista de barrios queda desactivada."
    return geojson, None


def columnas_disponibles(ruta: Path) -> set[str]:
    """Columnas del Parquet sin leer los datos."""
    import pyarrow.parquet as pq
    return set(pq.read_schema(ruta).names)
