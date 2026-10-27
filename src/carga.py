"""Adquisición de datos de Inside Airbnb: descubrimiento, descarga verificada y lectura.

Garantías:
- Solo HTTPS y dominios de Inside Airbnb; las redirecciones se siguen a mano y se validan.
- Cada fichero se descarga a ``.part``, se valida por completo (tamaño, gzip íntegro con tope de
  bytes descomprimidos, CSV/GeoJSON parseable) y solo entonces se renombra.
- Los tres ficheros obligatorios se descargan como conjunto en un directorio temporal; solo si
  todos pasan y sus URLs tienen la misma fecha de snapshot se sustituyen los de ``data/raw``.
- El manifiesto (URL, fecha efectiva, bytes y SHA-256 por fichero) se escribe al final.
- Si ``data/raw`` ya está completo y su hash coincide con el manifiesto, no se usa la red.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from src import config
from src.config import Ciudad
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


@dataclass
class UrlsSnapshot:
    """Resultado del descubrimiento de URLs para una ciudad."""

    fecha: str
    urls: dict[str, str]
    licencia: dict[str, Any]
    origen: str  # "pagina" o "url_directa"
    extras: dict[str, str] = field(default_factory=dict)


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


def validar_url(url: str) -> None:
    """Lanza ``ErrorDescarga`` si la URL no es HTTPS o su host no está permitido."""
    partes = urlparse(url)
    if partes.scheme != "https" or partes.hostname not in config.DOMINIOS_PERMITIDOS:
        raise ErrorDescarga(f"URL no permitida por seguridad (solo HTTPS de Inside Airbnb): {url}")


def nueva_sesion() -> requests.Session:
    """Sesión HTTP con User-Agent descriptivo y sin credenciales."""
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def _get(sesion: requests.Session, url: str, *, stream: bool) -> requests.Response:
    """GET siguiendo redirecciones a mano; cada salto se valida antes de pedirlo."""
    actual = url
    for _ in range(config.MAX_REDIRECCIONES + 1):
        validar_url(actual)
        try:
            resp = sesion.get(
                actual,
                stream=stream,
                allow_redirects=False,
                timeout=(config.TIMEOUT_CONEXION_S, config.TIMEOUT_LECTURA_S),
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise _Recuperable(f"error de red en {actual}: {exc}") from exc
        if resp.status_code in (301, 302, 303, 307, 308):
            destino = resp.headers.get("Location", "")
            resp.close()
            actual = urljoin(actual, destino)
            continue
        if resp.status_code in (403, 404, 410):
            resp.close()
            raise _NoEncontrado(f"HTTP {resp.status_code} en {actual}")
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.close()
            raise _Recuperable(f"HTTP {resp.status_code} en {actual}")
        if resp.status_code != 200:
            resp.close()
            raise ErrorDescarga(f"Respuesta inesperada HTTP {resp.status_code} en {actual}")
        return resp
    raise ErrorDescarga(f"Demasiadas redirecciones al pedir {url}")


# --------------------------------------------------------------------------
# Descubrimiento
# --------------------------------------------------------------------------
def extraer_urls(html: str, ciudad: Ciudad) -> dict[str, dict[str, str]]:
    """Extrae del HTML las URLs de la ciudad agrupadas por fecha de snapshot."""
    patron = re.compile(
        r"https://data\.insideairbnb\.com/" + re.escape(ciudad.slug_ruta)
        + r"/(\d{4}-\d{2}-\d{2})/(data|visualisations)/"
        r"(listings\.csv\.gz|neighbourhoods\.csv|neighbourhoods\.geojson|reviews\.csv\.gz|calendar\.csv\.gz)"
    )
    por_fecha: dict[str, dict[str, str]] = {}
    for m in patron.finditer(html):
        fecha, carpeta, nombre = m.groups()
        clave_ok = config.FICHEROS_OBLIGATORIOS.get(nombre) or config.FICHEROS_EXTRA.get(nombre)
        if clave_ok == carpeta:
            por_fecha.setdefault(fecha, {})[nombre] = m.group(0)
    return por_fecha


def detectar_licencia(html: str) -> dict[str, Any]:
    """Comprueba si la página enlaza la licencia CC BY 4.0 y devuelve lo verificado."""
    if "creativecommons.org/licenses/by/4.0" in html:
        return {**_LICENCIA_CC_BY, "verificada_en_pagina": True, "fecha_verificacion": ahora_utc()[:10]}
    logger.warning("No se encontró el enlace a la licencia CC BY 4.0 en la página de Inside Airbnb.")
    return {"nombre": "no verificada", "url": config.URL_PAGINA_DATOS,
            "verificada_en_pagina": False, "fecha_verificacion": ahora_utc()[:10]}


def descubrir_urls(ciudad: Ciudad, sesion: requests.Session, fecha_fija: str | None = None) -> UrlsSnapshot:
    """Localiza en la página de Inside Airbnb el snapshot más reciente con los ficheros obligatorios.

    Si ``fecha_fija`` se indica, se exige ese snapshot. Si la página no responde,
    se recurre a ``ciudad.url_directa_conocida`` (licencia no verificada).
    """
    html = None
    for intento in range(1, config.REINTENTOS + 1):
        try:
            resp = _get(sesion, config.URL_PAGINA_DATOS, stream=False)
            html = resp.text
            break
        except (_Recuperable, _NoEncontrado) as exc:
            logger.warning("Intento %d/%d de leer la página de datos falló: %s", intento, config.REINTENTOS, exc)
            if intento < config.REINTENTOS:
                time.sleep(2 ** intento)
    if html is None:
        if not ciudad.url_directa_conocida:
            raise ErrorDescarga("No se pudo leer la página de Inside Airbnb y no hay URL directa conocida.")
        logger.warning("Se usan las URLs directas conocidas: snapshot no verificado en la página.")
        fecha = _fecha_de_url(next(iter(ciudad.url_directa_conocida.values())))
        return UrlsSnapshot(fecha=fecha, urls=dict(ciudad.url_directa_conocida),
                            licencia={**_LICENCIA_CC_BY, "verificada_en_pagina": False,
                                      "fecha_verificacion": ahora_utc()[:10]},
                            origen="url_directa")
    licencia = detectar_licencia(html)
    por_fecha = extraer_urls(html, ciudad)
    completas = sorted(f for f, d in por_fecha.items() if set(config.FICHEROS_OBLIGATORIOS) <= set(d))
    if fecha_fija is not None:
        if fecha_fija not in por_fecha:
            logger.warning("La página ya no lista el snapshot %s; se construyen las URLs con esa fecha.", fecha_fija)
        return UrlsSnapshot(fecha=fecha_fija, urls={}, licencia=licencia, origen="pagina",
                            extras=por_fecha.get(fecha_fija, {}))
    if not completas:
        if not ciudad.url_directa_conocida:
            raise ErrorDescarga(f"La página no lista un snapshot completo para {ciudad.nombre}.")
        logger.warning("La página no lista un snapshot completo; se usan las URLs directas conocidas.")
        fecha = _fecha_de_url(next(iter(ciudad.url_directa_conocida.values())))
        return UrlsSnapshot(fecha=fecha, urls=dict(ciudad.url_directa_conocida), licencia=licencia, origen="url_directa")
    fecha = completas[-1]
    urls = {k: v for k, v in por_fecha[fecha].items() if k in config.FICHEROS_OBLIGATORIOS}
    extras = {k: v for k, v in por_fecha[fecha].items() if k in config.FICHEROS_EXTRA}
    return UrlsSnapshot(fecha=fecha, urls=urls, licencia=licencia, origen="pagina", extras=extras)


def _fecha_de_url(url: str) -> str:
    """Extrae la fecha YYYY-MM-DD de una URL de Inside Airbnb."""
    m = re.search(r"/(\d{4}-\d{2}-\d{2})/", url)
    if not m:
        raise ErrorDescarga(f"La URL no contiene fecha de snapshot: {url}")
    return m.group(1)


# --------------------------------------------------------------------------
# Descarga y validación
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


def _descargar_una_vez(url: str, destino: Path, sesion: requests.Session) -> None:
    """Un intento de descarga a ``.part`` con validación y renombrado atómico."""
    parte = destino.with_name(destino.name + ".part")
    resp = _get(sesion, url, stream=True)
    esperado = resp.headers.get("Content-Length")
    escritos = 0
    try:
        with open(parte, "wb") as f:
            for trozo in resp.iter_content(chunk_size=1 << 20):
                escritos += len(trozo)
                if escritos > config.MAX_DESCARGA_BYTES:
                    raise ErrorDescarga(f"La descarga supera el tope de {config.MAX_DESCARGA_BYTES} bytes: {url}")
                f.write(trozo)
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
        parte.unlink(missing_ok=True)
        raise _Recuperable(f"descarga interrumpida: {exc}") from exc
    except BaseException:
        parte.unlink(missing_ok=True)
        raise
    finally:
        resp.close()
    try:
        if esperado is not None and escritos != int(esperado):
            raise _Recuperable(f"tamaño incompleto ({escritos} de {esperado} bytes)")
        validar_contenido(parte, destino.name)
    except BaseException:
        parte.unlink(missing_ok=True)
        raise
    os.replace(parte, destino)


def descargar_fichero(url: str, destino: Path, sesion: requests.Session,
                      reintentos: int = config.REINTENTOS, espera_base: float = 2.0) -> Path:
    """Descarga ``url`` en ``destino`` con reintentos y backoff exponencial (2, 4, 8 s).

    Lanza ``_NoEncontrado`` ante 403/404/410 y ``ErrorDescarga`` si se agotan los reintentos.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(1, reintentos + 1):
        try:
            _descargar_una_vez(url, destino, sesion)
            logger.info("Descargado %s (%s bytes)", destino.name, f"{destino.stat().st_size:,}")
            return destino
        except _Recuperable as exc:
            logger.warning("Intento %d/%d de %s falló: %s", intento, reintentos, destino.name, exc)
            if intento < reintentos:
                time.sleep(espera_base * 2 ** (intento - 1))
    raise ErrorDescarga(f"No se pudo descargar {url} tras {reintentos} intentos.")


# --------------------------------------------------------------------------
# Manifiesto
# --------------------------------------------------------------------------
def leer_manifiesto(dir_raw: Path = config.DIR_RAW) -> dict[str, Any] | None:
    """Lee ``MANIFEST.json`` si existe."""
    ruta = dir_raw / "MANIFEST.json"
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def escribir_manifiesto(manifiesto: dict[str, Any], dir_raw: Path = config.DIR_RAW) -> None:
    """Escribe ``MANIFEST.json`` con sangría y UTF-8."""
    with open(dir_raw / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)
        f.write("\n")


def verificar_manifiesto(dir_raw: Path = config.DIR_RAW) -> dict[str, Any]:
    """Recalcula el SHA-256 de los ficheros obligatorios y lo compara con el manifiesto.

    Lanza ``ErrorDatos`` si falta algo o algún hash no coincide.
    """
    manifiesto = leer_manifiesto(dir_raw)
    if manifiesto is None:
        raise ErrorDatos("No existe data/raw/MANIFEST.json.")
    registrados = {f["archivo"]: f for f in manifiesto.get("ficheros", [])}
    for nombre in config.FICHEROS_OBLIGATORIOS:
        ruta = dir_raw / nombre
        if nombre not in registrados or not ruta.exists():
            raise ErrorDatos(f"Falta {nombre} en data/raw o en el manifiesto.")
        if sha256_fichero(ruta) != registrados[nombre]["sha256"]:
            raise ErrorDatos(f"data/raw/{nombre} ha sido modificado (hash distinto); usa --forzar-descarga.")
    return manifiesto


def _raw_completo(dir_raw: Path) -> bool:
    """True si existen el manifiesto y los tres ficheros obligatorios."""
    return (dir_raw / "MANIFEST.json").exists() and all((dir_raw / n).exists() for n in config.FICHEROS_OBLIGATORIOS)


def descargar_snapshot(ciudad: Ciudad, snapshot: UrlsSnapshot, dir_raw: Path,
                       sesion: requests.Session) -> list[dict[str, Any]]:
    """Descarga los ficheros obligatorios como un conjunto atómico y devuelve sus entradas de manifiesto.

    1. Los tres ficheros se descargan y validan en un directorio temporal junto a ``dir_raw``.
    2. Si alguno falla, se reintenta solo ese con la URL directa conocida (fallback).
    3. Se exige que las fechas de snapshot de TODAS las URLs efectivas coincidan; si no, ``ErrorDescarga``
       y ``data/raw`` queda intacto (no se mezclan ficheros de dos fechas).
    4. Solo entonces se sustituyen los ficheros de ``dir_raw`` (``os.replace`` por fichero).
    """
    dir_raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dir_raw.parent, prefix=".descarga_") as tmp:
        tmp_dir = Path(tmp)
        ficheros: list[dict[str, Any]] = []
        for nombre in config.FICHEROS_OBLIGATORIOS:
            url = snapshot.urls[nombre]
            try:
                descargar_fichero(url, tmp_dir / nombre, sesion)
            except (_NoEncontrado, ErrorDescarga) as exc:
                directa = ciudad.url_directa_conocida.get(nombre)
                if not directa or directa == url:
                    raise ErrorDescarga(str(exc)) from exc
                logger.warning("%s; se reintenta con la URL directa conocida.", exc)
                url = directa
                descargar_fichero(url, tmp_dir / nombre, sesion)
            ruta = tmp_dir / nombre
            ficheros.append({"archivo": nombre, "url": url, "fecha_snapshot": _fecha_de_url(url),
                             "bytes": ruta.stat().st_size, "sha256": sha256_fichero(ruta),
                             "descargado_utc": ahora_utc()})
        fechas = sorted({f["fecha_snapshot"] for f in ficheros})
        if len(fechas) != 1:
            detalle = ", ".join(f"{f['archivo']}={f['fecha_snapshot']}" for f in ficheros)
            raise ErrorDescarga(f"Los ficheros descargados pertenecen a snapshots distintos ({detalle}); "
                                "no se mezclan fechas y data/raw no se modifica.")
        for f in ficheros:
            os.replace(tmp_dir / f["archivo"], dir_raw / f["archivo"])
    return ficheros


def asegurar_datos_raw(ciudad: Ciudad, *, forzar: bool = False, sin_descarga: bool = False,
                       dir_raw: Path = config.DIR_RAW, sesion: requests.Session | None = None) -> dict[str, Any]:
    """Garantiza que ``data/raw`` contiene los ficheros obligatorios íntegros y devuelve el manifiesto.

    Modo offline (sin red) si ya están completos y el hash coincide, o si ``sin_descarga``.
    """
    if sin_descarga or (not forzar and _raw_completo(dir_raw)):
        manifiesto = verificar_manifiesto(dir_raw)
        logger.info("data/raw verificado por hash (modo offline, sin peticiones de red).")
        return manifiesto

    sesion = sesion or nueva_sesion()
    try:
        snapshot = descubrir_urls(ciudad, sesion)
        logger.info("Snapshot %s de %s (origen: %s)", snapshot.fecha, ciudad.nombre, snapshot.origen)
        ficheros = descargar_snapshot(ciudad, snapshot, dir_raw, sesion)
    except (ErrorDescarga, _NoEncontrado) as exc:
        if _raw_completo(dir_raw):
            logger.warning("Descarga fallida (%s); se usan los datos válidos ya presentes en data/raw.", exc)
            return verificar_manifiesto(dir_raw)
        raise ErrorDescarga(
            f"No se pudieron descargar los datos de {ciudad.nombre}: {exc}. "
            "No se generan datos sintéticos. Como plan B manual se podría adaptar el proyecto a "
            "Barcelona (spain/catalonia/barcelona), documentando el cambio."
        ) from exc

    manifiesto = {
        "ciudad": ciudad.nombre,
        "fecha_snapshot": ficheros[0]["fecha_snapshot"],
        "origen_urls": snapshot.origen,
        "pagina_fuente": config.URL_PAGINA_DATOS,
        "licencia": snapshot.licencia,
        "ficheros": ficheros,
        "extras": [],
    }
    escribir_manifiesto(manifiesto, dir_raw)
    return manifiesto


def asegurar_extras(ciudad: Ciudad, manifiesto: dict[str, Any], incluir: Sequence[str], *,
                    sin_descarga: bool = False, dir_extra: Path = config.DIR_EXTRA, dir_raw: Path = config.DIR_RAW,
                    sesion: requests.Session | None = None) -> dict[str, Any]:
    """Descarga ficheros extra (reviews/calendar) a ``data/extra`` fijando el snapshot del manifiesto (M2).

    Primero comprueba la caché local contra el manifiesto; solo si falta algo usa la red. En modo
    offline (``sin_descarga``) nunca hace peticiones: si falta un extra, ``ErrorDescarga`` con el motivo.
    Ningún análisis los usa; quedan disponibles para ampliaciones. Devuelve el manifiesto actualizado.
    """
    if not incluir:
        return manifiesto
    fecha = manifiesto["fecha_snapshot"]
    registrados = {e["archivo"]: e for e in manifiesto.get("extras", [])}
    pendientes = []
    for clave in incluir:
        nombre = f"{clave}.csv.gz"
        destino = dir_extra / nombre
        previo = registrados.get(nombre)
        if destino.exists() and previo and sha256_fichero(destino) == previo["sha256"]:
            logger.info("%s ya presente y verificado en data/extra.", nombre)
        else:
            pendientes.append(nombre)
    if not pendientes:
        return manifiesto
    if sin_descarga:
        raise ErrorDescarga(f"Modo sin descarga: faltan en data/extra (o no coinciden con el manifiesto) "
                            f"{', '.join(pendientes)}. Ejecuta sin --sin-descarga o quita --incluir-reviews/"
                            "--incluir-calendar.")
    sesion = sesion or nueva_sesion()
    snapshot = descubrir_urls(ciudad, sesion, fecha_fija=fecha)
    for nombre in pendientes:
        destino = dir_extra / nombre
        url = snapshot.extras.get(nombre) or (
            f"https://data.insideairbnb.com/{ciudad.slug_ruta}/{fecha}/{config.FICHEROS_EXTRA[nombre]}/{nombre}")
        logger.warning("Descargando %s (puede pesar más de 100 MB; no se sube a GitHub).", nombre)
        try:
            descargar_fichero(url, destino, sesion)
        except _NoEncontrado as exc:
            raise ErrorDescarga(f"No disponible {nombre} para el snapshot {fecha}: {exc}") from exc
        registrados[nombre] = {"archivo": nombre, "url": url, "bytes": destino.stat().st_size,
                               "sha256": sha256_fichero(destino), "descargado_utc": ahora_utc(),
                               "destino": "data/extra"}
    manifiesto = {**manifiesto, "extras": list(registrados.values())}
    escribir_manifiesto(manifiesto, dir_raw)
    return manifiesto


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------
def leer_listings(ruta: Path = config.RUTA_LISTINGS) -> pd.DataFrame:
    """Lee ``listings.csv.gz`` sin conversiones (se hacen en ``limpieza``) y valida columnas obligatorias."""
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


def leer_barrios(ruta: Path = config.RUTA_BARRIOS) -> pd.DataFrame:
    """Lee ``neighbourhoods.csv`` (columnas ``neighbourhood_group`` y ``neighbourhood``)."""
    try:
        df = pd.read_csv(ruta)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ErrorDatos(f"No se pudo leer {ruta.name}: {exc}") from exc
    if not {"neighbourhood_group", "neighbourhood"} <= set(df.columns):
        raise ErrorDatos(f"{ruta.name} no tiene las columnas neighbourhood_group y neighbourhood.")
    return df


def leer_geojson(ruta: Path = config.RUTA_GEOJSON) -> dict[str, Any]:
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
