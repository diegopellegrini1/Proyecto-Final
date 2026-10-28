"""Utilidades comunes de adquisición y lectura de ficheros crudos.

La descarga de las dos fuentes (Inside Airbnb e INE), sus manifiestos con hash SHA-256 y la unión están
en ``src/fuentes.py``; aquí quedan las piezas compartidas: excepciones de reintento, hash, sesión HTTP,
detección de licencia, validación de contenido y lectura de ``listings.csv.gz`` y de los GeoJSON.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src import config
from src.errores import ErrorDatos, ErrorDescarga

logger = logging.getLogger(__name__)

_LICENCIA_CC_BY = {
    "nombre": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "url": "https://creativecommons.org/licenses/by/4.0/",
}


class _NoEncontrado(Exception):
    """HTTP 404/403/410: no tiene sentido reintentar la misma URL."""


class _Recuperable(Exception):
    """Fallo transitorio o fichero corrupto: se puede reintentar."""


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def ahora_utc() -> str:
    """Fecha y hora UTC actual en ISO 8601 (segundos)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_fichero(ruta: Path, bloque: int = 1 << 20) -> str:
    """SHA-256 hexadecimal de un fichero leído por bloques."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while trozo := f.read(bloque):
            h.update(trozo)
    return h.hexdigest()


def nueva_sesion() -> requests.Session:
    """Sesión HTTP con User-Agent descriptivo y sin credenciales."""
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def detectar_licencia(html: str) -> dict[str, Any]:
    """Comprueba si la página enlaza la licencia CC BY 4.0 y devuelve lo verificado."""
    if "creativecommons.org/licenses/by/4.0" in html:
        return {**_LICENCIA_CC_BY, "verificada_en_pagina": True, "fecha_verificacion": ahora_utc()[:10]}
    logger.warning("No se encontró el enlace a la licencia CC BY 4.0 en la página de Inside Airbnb.")
    return {"nombre": "no verificada", "url": config.URL_PAGINA_DATOS,
            "verificada_en_pagina": False, "fecha_verificacion": ahora_utc()[:10]}


# --------------------------------------------------------------------------
# Validación de contenido
# --------------------------------------------------------------------------
def validar_contenido(ruta: Path, nombre: str, max_descomprimido: int = config.MAX_DESCOMPRIMIDO_BYTES) -> None:
    """Valida un fichero descargado por completo. Lanza ``_Recuperable`` si está corrupto.

    Para los ``.gz`` cuenta los bytes descomprimidos y aborta con ``ErrorDescarga`` (sin reintentar)
    si superan ``max_descomprimido``, para no gastar CPU con un fichero que se expande sin medida.
    """
    try:
        if ruta.stat().st_size == 0:
            raise _Recuperable("fichero vacío")
        if nombre.endswith(".gz"):
            with gzip.open(ruta, "rb") as f:
                cabecera = f.read(1 << 16)
                total = len(cabecera)
                while trozo := f.read(1 << 20):
                    total += len(trozo)
                    if total > max_descomprimido:
                        raise ErrorDescarga(f"{nombre} supera el tope de {max_descomprimido} bytes descomprimidos.")
            if nombre.endswith(".csv.gz"):
                pd.read_csv(io.BytesIO(cabecera), nrows=5)
        elif nombre.endswith(".geojson"):
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
            if not datos.get("features"):
                raise _Recuperable("GeoJSON sin 'features'")
        elif nombre.endswith(".csv"):
            pd.read_csv(ruta, nrows=5)
    except _Recuperable:
        raise
    except (OSError, EOFError, ValueError, pd.errors.ParserError, json.JSONDecodeError) as exc:
        raise _Recuperable(f"fichero corrupto ({type(exc).__name__}: {exc})") from exc


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------
def leer_listings(ruta: Path) -> pd.DataFrame:
    """Lee un ``listings.csv.gz`` sin conversiones (se hacen en ``limpieza``) y valida columnas obligatorias."""
    try:
        df = pd.read_csv(ruta, low_memory=False)
    except (OSError, EOFError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ErrorDatos(f"No se pudo leer {ruta.name}: {type(exc).__name__}: {exc}") from exc
    faltan = [c for c in config.COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ErrorDatos(f"Faltan columnas obligatorias en {ruta.name}: {', '.join(faltan)}")
    if df.empty:
        raise ErrorDatos(f"{ruta.name} no contiene filas.")
    return df


def leer_geojson(ruta: Path) -> dict[str, Any]:
    """Lee el GeoJSON de barrios y valida que todas las features tengan ``neighbourhood``."""
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ErrorDatos(f"No se pudo leer {ruta.name}: {exc}") from exc
    feats = datos.get("features") or []
    if not feats or any("neighbourhood" not in (ft.get("properties") or {}) for ft in feats):
        raise ErrorDatos(f"{ruta.name} no tiene features con la propiedad 'neighbourhood'.")
    return datos
