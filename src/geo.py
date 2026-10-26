"""Utilidades geográficas: distancia haversine y comprobación de bounding box."""

from __future__ import annotations

import numpy as np
import pandas as pd

RADIO_TIERRA_KM: float = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> float | np.ndarray:
    """Distancia de círculo máximo en km entre puntos dados en grados.

    Acepta escalares, arrays de numpy o Series de pandas (vectorizado).
    Devuelve un float si todas las entradas son escalares.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = (np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    d = 2.0 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    if np.ndim(d) == 0:
        return float(d)
    return d


def dentro_bbox(lat: pd.Series, lon: pd.Series, bbox: tuple[float, float, float, float]) -> pd.Series:
    """Devuelve una máscara booleana: coordenadas no nulas y dentro de (lat_min, lat_max, lon_min, lon_max)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    mascara = lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
    return mascara.fillna(False).astype(bool)
