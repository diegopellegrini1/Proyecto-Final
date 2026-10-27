"""Dos fuentes de datos para la ampliación a España y su unión.

Fuente 1: Inside Airbnb, ``listings.csv.gz`` + ``neighbourhoods.csv/.geojson`` de todas las ciudades y
regiones de España publicadas en https://insideairbnb.com/get-the-data/ (licencia CC BY 4.0).

Fuente 2: INE, Atlas de Distribución de Renta de los Hogares (ADRH): renta, Gini, umbrales de renta e
indicadores demográficos por municipio, distrito y sección censal, descargados de INEbase (jaxiT3) sin
cuenta. Los CSV del INE pesan hasta ~200 MB, así que se guardan comprimidos con gzip (sin pérdida, mtime=0):
``gzip -d`` devuelve los bytes exactos servidos por el INE, cuyo SHA-256 consta en el manifiesto.

Unión: cada anuncio se asigna a su sección censal por punto en polígono (coordenadas del anuncio dentro del
seccionado oficial del INE a 1-1-2024). De la sección salen, sin emparejar nombres, el distrito censal
(7 dígitos) y el municipio (5 dígitos), que son las claves de las tablas del ADRH. Los anuncios sin
sección o sin dato del INE se conservan con NaN y se avisa.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html as html_lib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests

from src import carga, config
from src.config import CiudadEspana
from src.errores import ErrorDatos, ErrorDescarga, ErrorPipeline

logger = logging.getLogger(__name__)

NOMBRE_MANIFIESTO = "MANIFEST.json"
COLUMNAS_BASE_ADRH: tuple[str, ...] = ("Municipios", "Distritos", "Secciones")
_VALORES_AUSENTES_INE = frozenset({"", ".", "..", "-"})
_LONGITUD_CODIGO = {"municipio": 5, "distrito": 7, "seccion": 10}

LICENCIA_INE: dict[str, str] = {
    "nombre": "Aviso legal del INE: reutilización autorizada citando la fuente",
    "url": ("https://www.ine.es/ss/Satellite?L=es_ES&c=Page&cid=1254735849170&p=1254735849170"
            "&pagename=Ayuda%2FINELayout"),
    "condiciones": ("Reutilización comercial o no comercial de la información cuya fuente original es el INE; "
                    "no desnaturalizar su sentido; citar «Elaboración propia con datos extraídos del sitio web "
                    "del INE: www.ine.es» si se tratan los datos; indicar la fecha de última actualización; "
                    "no sugerir que el INE patrocina el uso."),
}


# --------------------------------------------------------------------------
# URLs: tildes y doble codificación
# --------------------------------------------------------------------------
def reparar_mojibake(texto: str) -> str:
    """Deshace la doble codificación UTF-8 leída como Latin-1 (``'andalucÃ\\xada'`` -> ``'andalucía'``).

    Es idempotente: si el texto ya es correcto (o es ASCII) se devuelve igual.
    """
    for _ in range(3):  # la página puede llegar doble o triplemente codificada
        try:
            reparado = texto.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texto
        if reparado == texto:
            return texto
        texto = reparado
    return texto


def codificar_url(url: str) -> str:
    """Repara la tilde y codifica la ruta en porcentaje (UTF-8); no recodifica los ``%XX`` existentes."""
    url = reparar_mojibake(html_lib.unescape(url))
    partes = urlparse(url)
    return partes._replace(path=quote(partes.path, safe="/%-._~")).geturl()


_PATRON_SPAIN = re.compile(r"https://data\.insideairbnb\.com/spain/[^\"'<>\s]+")
_PATRON_PARTES = re.compile(
    r"^https://data\.insideairbnb\.com/(spain/.+?)/(\d{4}-\d{2}-\d{2})/(data|visualisations)/([^/]+)$")


def descubrir_urls_espana(html: str, ciudades: dict[str, CiudadEspana] = config.CIUDADES_ESPANA
                          ) -> dict[str, dict[str, Any]]:
    """Extrae de la página de Inside Airbnb el snapshot más reciente y completo de cada ciudad española.

    Devuelve ``{clave: {"fecha_snapshot": "AAAA-MM-DD", "urls": {nombre_fichero: url_codificada}}}``.
    Las ciudades españolas de la página que no estén en ``ciudades`` se avisan y se ignoran.
    """
    slug_a_clave = {c.slug_ruta: c.clave for c in ciudades.values()}
    por_ciudad: dict[str, dict[str, dict[str, str]]] = {}
    desconocidas: set[str] = set()
    for bruta in _PATRON_SPAIN.findall(html):
        url = reparar_mojibake(html_lib.unescape(bruta))
        m = _PATRON_PARTES.match(url)
        if not m:
            continue
        slug, fecha, carpeta, nombre = m.groups()
        if config.FICHEROS_OBLIGATORIOS.get(nombre) != carpeta:
            continue
        clave = slug_a_clave.get(slug)
        if clave is None:
            desconocidas.add(slug)
            continue
        por_ciudad.setdefault(clave, {}).setdefault(fecha, {})[nombre] = codificar_url(url)
    for slug in sorted(desconocidas):
        logger.warning("La página lista una ciudad española no configurada: %s (se ignora).", slug)
    resultado: dict[str, dict[str, Any]] = {}
    for clave, por_fecha in por_ciudad.items():
        completas = sorted(f for f, d in por_fecha.items() if set(config.FICHEROS_OBLIGATORIOS) <= set(d))
        if completas:
            resultado[clave] = {"fecha_snapshot": completas[-1], "urls": dict(sorted(por_fecha[completas[-1]].items()))}
    for clave in sorted(set(ciudades) - set(resultado)):
        logger.warning("La página no lista un snapshot completo para %s.", ciudades[clave].nombre)
    return dict(sorted(resultado.items()))


# --------------------------------------------------------------------------
# Descarga verificada genérica
# --------------------------------------------------------------------------
def validar_url(url: str, dominios: Iterable[str]) -> None:
    """Lanza ``ErrorDescarga`` si la URL no es HTTPS o su host no está en ``dominios``."""
    partes = urlparse(url)
    if partes.scheme != "https" or partes.hostname not in set(dominios):
        raise ErrorDescarga(f"URL no permitida por seguridad (solo HTTPS de {sorted(dominios)}): {url}")


def _get(sesion: requests.Session, url: str, dominios: Iterable[str], *, stream: bool) -> requests.Response:
    """GET siguiendo redirecciones a mano; cada salto se valida contra ``dominios``."""
    actual = url
    for _ in range(config.MAX_REDIRECCIONES + 1):
        validar_url(actual, dominios)
        try:
            resp = sesion.get(actual, stream=stream, allow_redirects=False,
                              timeout=(config.TIMEOUT_CONEXION_S, config.TIMEOUT_LECTURA_S))
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise carga._Recuperable(f"error de red en {actual}: {exc}") from exc
        if resp.status_code in (301, 302, 303, 307, 308):
            destino = resp.headers.get("Location", "")
            resp.close()
            actual = urljoin(actual, destino)
            continue
        if resp.status_code in (403, 404, 410):
            resp.close()
            raise carga._NoEncontrado(f"HTTP {resp.status_code} en {actual}")
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.close()
            raise carga._Recuperable(f"HTTP {resp.status_code} en {actual}")
        if resp.status_code != 200:
            resp.close()
            raise ErrorDescarga(f"Respuesta inesperada HTTP {resp.status_code} en {actual}")
        return resp
    raise ErrorDescarga(f"Demasiadas redirecciones al pedir {url}")


def leer_texto(url: str, sesion: requests.Session, dominios: Iterable[str]) -> str:
    """Descarga una página HTML con reintentos y devuelve su texto."""
    for intento in range(1, config.REINTENTOS + 1):
        try:
            resp = _get(sesion, url, dominios, stream=False)
            if "charset" not in resp.headers.get("Content-Type", "").lower():
                resp.encoding = "utf-8"
            return resp.text
        except carga._Recuperable as exc:
            logger.warning("Intento %d/%d de leer %s falló: %s", intento, config.REINTENTOS, url, exc)
            if intento < config.REINTENTOS:
                time.sleep(2 ** intento)
        except carga._NoEncontrado as exc:
            raise ErrorDescarga(str(exc)) from exc
    raise ErrorDescarga(f"No se pudo leer {url} tras {config.REINTENTOS} intentos.")


def _comprimir_determinista(origen: Path, destino: Path) -> None:
    """gzip nivel 9 sin nombre ni fecha en la cabecera: mismo contenido -> mismos bytes."""
    with open(origen, "rb") as fin, open(destino, "wb") as fbruto:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fbruto, compresslevel=9, mtime=0) as fgz:
            shutil.copyfileobj(fin, fgz, length=1 << 20)


def _descargar_una_vez(url: str, destino: Path, sesion: requests.Session, dominios: Iterable[str],
                       validar: Callable[[Path], None], comprimir: bool) -> dict[str, Any]:
    """Un intento: descarga a ``.part``, comprueba tamaño, valida, comprime si procede y renombra."""
    parte = destino.with_name(destino.name + ".part")
    parte_gz = destino.with_name(destino.name + ".gz.part")
    resp = _get(sesion, url, dominios, stream=True)
    esperado = resp.headers.get("Content-Length")
    h = hashlib.sha256()
    escritos = 0
    try:
        try:
            with open(parte, "wb") as f:
                for trozo in resp.iter_content(chunk_size=1 << 20):
                    escritos += len(trozo)
                    if escritos > config.MAX_DESCARGA_BYTES:
                        raise ErrorDescarga(f"La descarga supera el tope de {config.MAX_DESCARGA_BYTES} bytes: {url}")
                    h.update(trozo)
                    f.write(trozo)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            raise carga._Recuperable(f"descarga interrumpida: {exc}") from exc
        finally:
            resp.close()
        if esperado is not None and escritos != int(esperado):
            raise carga._Recuperable(f"tamaño incompleto ({escritos} de {esperado} bytes)")
        validar(parte)
        entrada: dict[str, Any] = {"url": url, "bytes_original": escritos, "sha256_original": h.hexdigest()}
        if comprimir:
            _comprimir_determinista(parte, parte_gz)
            os.replace(parte_gz, destino)
            parte.unlink()
        else:
            os.replace(parte, destino)
        entrada.update({"bytes": destino.stat().st_size, "sha256": carga.sha256_fichero(destino),
                        "descargado_utc": carga.ahora_utc()})
        return entrada
    finally:
        parte.unlink(missing_ok=True)
        parte_gz.unlink(missing_ok=True)


def descargar_verificado(url: str, destino: Path, sesion: requests.Session, dominios: Iterable[str], *,
                         validar: Callable[[Path], None], comprimir: bool = False,
                         reintentos: int = config.REINTENTOS, espera_base: float = 2.0) -> dict[str, Any]:
    """Descarga ``url`` en ``destino`` con validación completa, reintentos y backoff exponencial.

    ``validar(ruta)`` debe lanzar ``carga._Recuperable`` si el fichero está corrupto. Devuelve la entrada
    de manifiesto (URL, bytes y SHA-256 del original y del fichero guardado, fecha de descarga).
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(1, reintentos + 1):
        try:
            entrada = _descargar_una_vez(url, destino, sesion, dominios, validar, comprimir)
            logger.info("Descargado %s (%s bytes)", destino.name, f"{entrada['bytes']:,}")
            return entrada
        except carga._Recuperable as exc:
            logger.warning("Intento %d/%d de %s falló: %s", intento, reintentos, destino.name, exc)
            if intento < reintentos:
                time.sleep(espera_base * 2 ** (intento - 1))
        except carga._NoEncontrado as exc:
            raise ErrorDescarga(str(exc)) from exc
    raise ErrorDescarga(f"No se pudo descargar {url} tras {reintentos} intentos.")


# --------------------------------------------------------------------------
# Manifiestos
# --------------------------------------------------------------------------
def leer_manifiesto(directorio: Path) -> dict[str, Any] | None:
    """Lee ``MANIFEST.json`` de ``directorio`` si existe."""
    ruta = directorio / NOMBRE_MANIFIESTO
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def escribir_manifiesto(manifiesto: dict[str, Any], directorio: Path) -> None:
    """Escribe ``MANIFEST.json`` en UTF-8 con sangría y fin de línea LF."""
    directorio.mkdir(parents=True, exist_ok=True)
    with open(directorio / NOMBRE_MANIFIESTO, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)
        f.write("\n")


def verificar_manifiesto(directorio: Path, base: Path | None = None) -> dict[str, Any]:
    """Recalcula el SHA-256 de cada fichero del manifiesto. ``ErrorDatos`` si falta o no coincide."""
    manifiesto = leer_manifiesto(directorio)
    if manifiesto is None:
        raise ErrorDatos(f"No existe {directorio / NOMBRE_MANIFIESTO}.")
    base = base or directorio
    for entrada in manifiesto.get("ficheros", []):
        ruta = base / entrada["archivo"]
        if not ruta.exists():
            raise ErrorDatos(f"Falta {ruta} (registrado en el manifiesto).")
        if carga.sha256_fichero(ruta) != entrada["sha256"]:
            raise ErrorDatos(f"{ruta} no coincide con el SHA-256 del manifiesto.")
    return manifiesto


def _entrada_vigente(manifiesto: dict[str, Any] | None, archivo: str, base: Path) -> dict[str, Any] | None:
    """Entrada previa del manifiesto si el fichero existe y su hash coincide (para no volver a descargar)."""
    if not manifiesto:
        return None
    for entrada in manifiesto.get("ficheros", []):
        if entrada["archivo"] == archivo:
            ruta = base / archivo
            if ruta.exists() and carga.sha256_fichero(ruta) == entrada["sha256"]:
                return entrada
    return None


# --------------------------------------------------------------------------
# Fuente 1: Inside Airbnb (España)
# --------------------------------------------------------------------------
def _validador_airbnb(nombre: str) -> Callable[[Path], None]:
    return lambda ruta: carga.validar_contenido(ruta, nombre)


def descargar_inside_airbnb_espana(dir_base: Path = config.DIR_RAW_AIRBNB, *, forzar: bool = False,
                                   sesion: requests.Session | None = None) -> dict[str, Any]:
    """Descubre y descarga los ficheros de todas las ciudades españolas a ``dir_base/<ciudad>/``.

    Reutiliza los ficheros ya presentes cuyo hash coincide con el manifiesto. Cada fichero debe pesar
    menos de ``config.LIMITE_SUBIDA_BYTES``.
    """
    sesion = sesion or carga.nueva_sesion()
    html = leer_texto(config.URL_PAGINA_DATOS, sesion, config.DOMINIOS_PERMITIDOS)
    licencia = carga.detectar_licencia(html)
    snapshots = descubrir_urls_espana(html)
    if not snapshots:
        raise ErrorDescarga("La página de Inside Airbnb no lista ninguna ciudad española configurada.")
    previo = None if forzar else leer_manifiesto(dir_base)
    ficheros: list[dict[str, Any]] = []
    for clave, snap in snapshots.items():
        for nombre, url in snap["urls"].items():
            archivo = f"{clave}/{nombre}"
            vigente = _entrada_vigente(previo, archivo, dir_base)
            if vigente and vigente.get("url") == url:
                ficheros.append(vigente)
                continue
            entrada = descargar_verificado(url, dir_base / clave / nombre, sesion, config.DOMINIOS_PERMITIDOS,
                                           validar=_validador_airbnb(nombre))
            ficheros.append({"archivo": archivo, "ciudad": clave, "fecha_snapshot": snap["fecha_snapshot"],
                             "url": url, "bytes": entrada["bytes"], "sha256": entrada["sha256"],
                             "descargado_utc": entrada["descargado_utc"]})
    _comprobar_limite_subida(dir_base, ficheros)
    manifiesto = {
        "fuente": "Inside Airbnb",
        "pagina_fuente": config.URL_PAGINA_DATOS,
        "licencia": licencia,
        "ciudades": {c: {"nombre": config.CIUDADES_ESPANA[c].nombre, "fecha_snapshot": s["fecha_snapshot"]}
                     for c, s in snapshots.items()},
        "ficheros": ficheros,
    }
    escribir_manifiesto(manifiesto, dir_base)
    return manifiesto


def _comprobar_limite_subida(base: Path, ficheros: list[dict[str, Any]]) -> None:
    grandes = [f["archivo"] for f in ficheros if (base / f["archivo"]).stat().st_size >= config.LIMITE_SUBIDA_BYTES]
    if grandes:
        raise ErrorDatos(f"Ficheros por encima del límite de subida ({config.LIMITE_SUBIDA_BYTES} bytes): {grandes}")


def leer_listings_espana(dir_base: Path = config.DIR_RAW_AIRBNB) -> pd.DataFrame:
    """Lee y concatena los ``listings.csv.gz`` del manifiesto, con las columnas ``ciudad`` y ``fecha_snapshot``.

    Verifica antes los hashes. Lanza ``ErrorDatos`` si un ``id`` se repite dentro de una ciudad.
    """
    manifiesto = verificar_manifiesto(dir_base)
    partes = []
    for entrada in manifiesto["ficheros"]:
        if not entrada["archivo"].endswith("listings.csv.gz"):
            continue
        df = carga.leer_listings(dir_base / entrada["archivo"])
        if df["id"].duplicated().any():
            raise ErrorDatos(f"Hay ids repetidos en {entrada['archivo']}.")
        df.insert(0, "ciudad", entrada["ciudad"])
        df.insert(1, "fecha_snapshot", entrada["fecha_snapshot"])
        partes.append(df)
    if not partes:
        raise ErrorDatos(f"El manifiesto de {dir_base} no registra ningún listings.csv.gz.")
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------
# Fuente 2: INE, Atlas de Distribución de Renta de los Hogares
# --------------------------------------------------------------------------
_PATRON_TABLA_INE = re.compile(r'href="[^"]*?Tabla\.htm\?t=(\d+)[^"]*"[^>]*>(.*?)</a>', re.S)


def descubrir_tablas_adrh(html: str) -> dict[str, str]:
    """Extrae de la página provincial del ADRH ``{título exacto: identificador de tabla}``."""
    tablas: dict[str, str] = {}
    for m in _PATRON_TABLA_INE.finditer(html):
        titulo = html_lib.unescape(re.sub(r"<[^>]+>|\s+", " ", m.group(2))).strip()
        tablas.setdefault(titulo, m.group(1))
    return tablas


def validar_csv_ine(ruta: Path) -> None:
    """Valida un CSV del ADRH (plano o gzip): cabecera esperada y filas legibles. ``_Recuperable`` si no."""
    abrir = gzip.open if ruta.name.endswith(".gz") or _es_gzip(ruta) else open
    try:
        with abrir(ruta, "rb") as f:
            # El trozo puede cortar un carácter multibyte al final: se sustituye (la última línea no se valida).
            cabecera = f.read(1 << 16).decode("utf-8-sig", errors="replace").splitlines()
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise carga._Recuperable(f"CSV del INE ilegible: {exc}") from exc
    if len(cabecera) < 2:
        raise carga._Recuperable("CSV del INE vacío")
    columnas = cabecera[0].split(";")
    if tuple(columnas[:3]) != COLUMNAS_BASE_ADRH or columnas[-2:] != ["Periodo", "Total"]:
        raise carga._Recuperable(f"Cabecera inesperada en el CSV del INE: {cabecera[0][:200]}")
    if any(len(linea.split(";")) != len(columnas) for linea in cabecera[1:-1]):
        raise carga._Recuperable("Filas con número de campos distinto a la cabecera")


def _es_gzip(ruta: Path) -> bool:
    with open(ruta, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def validar_zip_seccionado(ruta: Path) -> None:
    """Valida el ZIP de la cartografía: íntegro y con .shp/.shx/.dbf."""
    try:
        with zipfile.ZipFile(ruta) as z:
            if z.testzip() is not None:
                raise carga._Recuperable("ZIP con miembros corruptos")
            extensiones = {Path(n).suffix.lower() for n in z.namelist()}
    except zipfile.BadZipFile as exc:
        raise carga._Recuperable(f"ZIP corrupto: {exc}") from exc
    if not {".shp", ".shx", ".dbf"} <= extensiones:
        raise carga._Recuperable("El ZIP no contiene .shp, .shx y .dbf")


def descargar_ine(dir_ine: Path = config.DIR_RAW_INE, dir_extra: Path = config.DIR_EXTRA_INE, *,
                  provincias: Iterable[str] | None = None, forzar: bool = False,
                  sesion: requests.Session | None = None) -> dict[str, Any]:
    """Descarga las tablas del ADRH por provincia (``dir_ine``, gzip) y el seccionado (``dir_extra``).

    Los identificadores de tabla se descubren en la página provincial del ADRH por su título exacto
    (``config.TABLAS_ADRH``). El seccionado (~65 MB) no se versiona: queda en ``data/extra``.
    """
    sesion = sesion or carga.nueva_sesion()
    provincias = sorted(set(provincias or {p for c in config.CIUDADES_ESPANA.values() for p in c.provincias}))
    previo = None if forzar else leer_manifiesto(dir_ine)
    ficheros: list[dict[str, Any]] = []
    for cod in provincias:
        prov = config.PROVINCIAS_ADRH[cod]
        url_pagina = f"{config.URL_PAGINA_ADRH}&capsel={prov.capsel}"
        tablas = descubrir_tablas_adrh(leer_texto(url_pagina, sesion, config.DOMINIOS_PERMITIDOS_INE))
        for clave, titulo in config.TABLAS_ADRH.items():
            if titulo not in tablas:
                raise ErrorDescarga(f"La página del ADRH de {prov.nombre} no lista la tabla «{titulo}».")
            tabla = tablas[titulo]
            archivo = f"adrh_{cod}_{clave}_{tabla}.csv.gz"
            url = config.URL_TABLA_INE_CSV.format(tabla=tabla)
            vigente = _entrada_vigente(previo, archivo, dir_ine)
            if vigente is None:
                entrada = descargar_verificado(url, dir_ine / archivo, sesion, config.DOMINIOS_PERMITIDOS_INE,
                                               validar=validar_csv_ine, comprimir=True)
                vigente = {"archivo": archivo, "provincia": cod, "nombre_provincia": prov.nombre,
                           "tabla": clave, "id_tabla_ine": tabla, "titulo": titulo,
                           "url_tabla": f"https://www.ine.es/jaxiT3/Tabla.htm?t={tabla}", "url": url,
                           "compresion": "gzip nivel 9, mtime=0, sin nombre (los bytes originales se recuperan con gzip -d)",
                           **{k: entrada[k] for k in ("bytes_original", "sha256_original", "bytes", "sha256",
                                                      "descargado_utc")}}
            ficheros.append(vigente)
    _comprobar_limite_subida(dir_ine, ficheros)
    cartografia = _asegurar_seccionado(dir_extra, previo, sesion)
    manifiesto = {
        "fuente": "INE - Atlas de Distribución de Renta de los Hogares (ADRH)",
        "pagina_fuente": config.URL_PAGINA_ADRH,
        "licencia": {**LICENCIA_INE, "fecha_verificacion": carga.ahora_utc()[:10]},
        "ficheros": ficheros,
        "cartografia": cartografia,
    }
    escribir_manifiesto(manifiesto, dir_ine)
    return manifiesto


def _asegurar_seccionado(dir_extra: Path, previo: dict[str, Any] | None,
                         sesion: requests.Session) -> dict[str, Any]:
    """Descarga (si falta o cambió) el seccionado censal del INE a ``data/extra/ine``."""
    nombre = Path(urlparse(config.URL_SECCIONADO_INE).path).name
    ruta = dir_extra / nombre
    anterior = (previo or {}).get("cartografia")
    if anterior and ruta.exists() and carga.sha256_fichero(ruta) == anterior["sha256"]:
        return anterior
    entrada = descargar_verificado(config.URL_SECCIONADO_INE, ruta, sesion, config.DOMINIOS_PERMITIDOS_INE,
                                   validar=validar_zip_seccionado)
    return {"archivo": nombre, "destino": "data/extra/ine (no se versiona por tamaño)",
            "descripcion": "Contornos de secciones censales del INE a 1-1-2024 (ETRS89 / UTM 30N)",
            "url": config.URL_SECCIONADO_INE, "bytes": entrada["bytes"], "sha256": entrada["sha256"],
            "descargado_utc": entrada["descargado_utc"]}


def parsear_numero_ine(serie: pd.Series) -> pd.Series:
    """Convierte cifras del INE (``'17.216'``, ``'39,9'``, ``'.'`` = sin dato) a float.

    Lanza ``ErrorDatos`` si algún valor no vacío no es numérico.
    """
    texto = serie.astype("string").str.strip()
    ausente = texto.isna() | texto.isin(_VALORES_AUSENTES_INE)
    limpio = texto.where(~ausente).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    valores = pd.to_numeric(limpio, errors="coerce")
    malos = valores.isna() & ~ausente
    if malos.any():
        raise ErrorDatos(f"Valores no numéricos en una tabla del INE: {texto[malos].unique()[:5].tolist()}")
    return valores.astype("float64")


def leer_tabla_adrh(ruta: Path) -> pd.DataFrame:
    """Lee un CSV del ADRH y lo devuelve en formato largo.

    Columnas: ``nivel`` (municipio/distrito/seccion), ``codigo``, ``nombre``, ``indicador``, ``periodo``
    (int) y ``valor`` (float). Si la tabla desglosa por sexo se conserva solo ``Sexo == 'Total'``.
    """
    try:
        df = pd.read_csv(ruta, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False,
                         compression="gzip" if _es_gzip(ruta) else None)
    except (OSError, EOFError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ErrorDatos(f"No se pudo leer {ruta.name}: {type(exc).__name__}: {exc}") from exc
    base = {*COLUMNAS_BASE_ADRH, "Sexo", "Periodo", "Total"}
    faltan = [c for c in (*COLUMNAS_BASE_ADRH, "Periodo", "Total") if c not in df.columns]
    otras = [c for c in df.columns if c not in base]
    if faltan or len(otras) != 1:
        raise ErrorDatos(f"Estructura inesperada en {ruta.name}: faltan {faltan}, columnas de indicador {otras}.")
    if "Sexo" in df.columns:
        df = df[df["Sexo"] == "Total"]
    mun, dis, sec = (df[c].str.strip() for c in COLUMNAS_BASE_ADRH)
    etiqueta = sec.where(sec != "", dis.where(dis != "", mun))
    partes = etiqueta.str.extract(r"^(\d+)\s+(.*)$")
    # El nivel se deduce de la longitud del código (5/7/10 dígitos) y no de la columna: el INE publica
    # alguna fila suelta en la columna equivocada (p. ej. una sección de Alicante en «Municipios» de Baleares).
    por_longitud = {v: k for k, v in _LONGITUD_CODIGO.items()}
    nivel = partes[0].str.len().map(por_longitud)
    nivel_columna = np.select([sec != "", dis != ""], ["seccion", "distrito"], default="municipio")
    descolocadas = nivel.notna() & (nivel != nivel_columna)
    if descolocadas.any():
        logger.warning("%s: %d filas con el código en otra columna (%s); se clasifican por longitud del código.",
                       ruta.name, int(descolocadas.sum()), partes.loc[descolocadas, 0].unique()[:3].tolist())
    salida = pd.DataFrame({
        "nivel": nivel,
        "codigo": partes[0],
        "nombre": partes[1],
        "indicador": df[otras[0]].str.strip(),
        "periodo": pd.to_numeric(df["Periodo"], errors="coerce"),
        "valor": parsear_numero_ine(df["Total"]),
    }, index=df.index)
    sin_codigo = salida["codigo"].isna()
    if sin_codigo.any():
        logger.warning("%s: %d filas sin código territorial (agregados); se descartan.", ruta.name, int(sin_codigo.sum()))
        salida = salida[~sin_codigo]
    if salida["nivel"].isna().any():
        raise ErrorDatos(f"{ruta.name}: códigos territoriales con longitud inesperada "
                         f"({salida.loc[salida['nivel'].isna(), 'codigo'].unique()[:5].tolist()}).")
    if salida["periodo"].isna().any():
        raise ErrorDatos(f"{ruta.name}: periodos no numéricos.")
    salida["periodo"] = salida["periodo"].astype("int64")
    return salida.reset_index(drop=True)


def leer_adrh(dir_ine: Path = config.DIR_RAW_INE) -> pd.DataFrame:
    """Lee todas las tablas del manifiesto del INE (verificando hashes) en un único formato largo."""
    manifiesto = verificar_manifiesto(dir_ine)
    partes = [leer_tabla_adrh(dir_ine / e["archivo"]) for e in manifiesto["ficheros"]]
    if not partes:
        raise ErrorDatos("El manifiesto del INE no registra tablas.")
    return pd.concat(partes, ignore_index=True)


def elegir_anio_adrh(largo: pd.DataFrame, cobertura_min: float = 0.5) -> int:
    """Año más reciente en el que todos los indicadores de ``config.INDICADORES_ADRH`` tienen dato en al
    menos ``cobertura_min`` de los distritos censales. ``ErrorDatos`` si ningún año lo cumple."""
    dis = largo[(largo["nivel"] == "distrito") & largo["indicador"].isin(config.INDICADORES_ADRH)]
    cobertura = dis.groupby(["periodo", "indicador"])["valor"].apply(lambda s: s.notna().mean()).unstack()
    cobertura = cobertura.reindex(columns=list(config.INDICADORES_ADRH))
    validos = cobertura.index[(cobertura.fillna(0) >= cobertura_min).all(axis=1)]
    if len(validos) == 0:
        raise ErrorDatos("Ningún año del ADRH tiene todos los indicadores con cobertura suficiente.")
    return int(max(validos))


def indicadores_adrh(largo: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Tabla ancha ``(nivel, codigo) -> indicadores`` del año ``anio`` (municipio y distrito)."""
    sel = largo[largo["nivel"].isin(["municipio", "distrito"]) & (largo["periodo"] == anio)
                & largo["indicador"].isin(config.INDICADORES_ADRH)]
    if sel.duplicated(["nivel", "codigo", "indicador"]).any():
        raise ErrorDatos("El ADRH tiene filas duplicadas por territorio e indicador.")
    ancho = sel.pivot(index=["nivel", "codigo"], columns="indicador", values="valor")
    ancho = ancho.reindex(columns=list(config.INDICADORES_ADRH)).rename(columns=config.INDICADORES_ADRH)
    ancho.columns.name = None
    nombres = sel.drop_duplicates(["nivel", "codigo"]).set_index(["nivel", "codigo"])["nombre"]
    ancho.insert(0, "nombre", nombres)
    return ancho.reset_index()


# --------------------------------------------------------------------------
# Unión: punto en polígono con el seccionado del INE
# --------------------------------------------------------------------------
def cargar_secciones(ruta_zip: Path, provincias: Iterable[str]) -> tuple[list[Any], np.ndarray, pd.DataFrame]:
    """Lee del seccionado del INE las secciones de ``provincias``.

    Devuelve (geometrías shapely, códigos CUSEC, tabla de atributos CUSEC/CUDIS/CUMUN/NMUN/CPRO).
    """
    import shapefile  # pyshp
    import shapely.geometry

    provincias = set(provincias)
    with zipfile.ZipFile(ruta_zip) as z:
        nombres = {Path(n).suffix.lower(): n for n in z.namelist()}
        lector = shapefile.Reader(shp=io.BytesIO(z.read(nombres[".shp"])), shx=io.BytesIO(z.read(nombres[".shx"])),
                                  dbf=io.BytesIO(z.read(nombres[".dbf"])), encoding="utf-8")
        campos = [f[0] for f in lector.fields[1:]]
        registros = pd.DataFrame(lector.records(), columns=campos)
        indices = registros.index[registros["CPRO"].isin(provincias)].tolist()
        geometrias = [shapely.geometry.shape(lector.shape(i).__geo_interface__) for i in indices]
    atributos = registros.loc[indices, ["CUSEC", "CUDIS", "CUMUN", "NMUN", "CPRO"]].reset_index(drop=True)
    if atributos.empty:
        raise ErrorDatos(f"El seccionado no contiene secciones de las provincias {sorted(provincias)}.")
    return geometrias, atributos["CUSEC"].to_numpy(), atributos


def asignar_seccion(lat: pd.Series, lon: pd.Series, geometrias: list[Any], codigos: np.ndarray) -> pd.Series:
    """Código de sección censal (CUSEC) que contiene cada punto (lat/lon WGS84); NaN si ninguna.

    Los puntos se proyectan a ETRS89 / UTM 30N (CRS del seccionado). Si un punto cae en el borde de
    dos secciones se toma la primera; se registra cuántos casos hay.
    """
    import shapely
    from pyproj import Transformer

    trans = Transformer.from_crs(config.CRS_LISTINGS, config.CRS_SECCIONADO, always_xy=True)
    x, y = trans.transform(lon.to_numpy(dtype="float64"), lat.to_numpy(dtype="float64"))
    puntos = shapely.points(x, y)
    arbol = shapely.STRtree(geometrias)
    idx_punto, idx_geom = arbol.query(puntos, predicate="intersects")
    pares = pd.DataFrame({"p": idx_punto, "g": idx_geom}).sort_values(["p", "g"])
    multiples = int(pares["p"].duplicated().sum())
    if multiples:
        logger.info("%d puntos caen en el borde de varias secciones; se toma la primera.", multiples)
    pares = pares.drop_duplicates("p")
    resultado = pd.Series(pd.NA, index=lat.index, dtype="string")
    resultado.iloc[pares["p"].to_numpy()] = codigos[pares["g"].to_numpy()]
    return resultado


def calcular_asignacion(listings: pd.DataFrame, ruta_zip: Path) -> pd.DataFrame:
    """Tabla ``ciudad, id, latitude, longitude, codigo_seccion_ine`` por punto en polígono."""
    provincias = sorted({p for c in listings["ciudad"].unique() for p in config.CIUDADES_ESPANA[c].provincias})
    geometrias, codigos, _ = cargar_secciones(ruta_zip, provincias)
    asignacion = listings[["ciudad", "id", "latitude", "longitude"]].copy()
    asignacion["codigo_seccion_ine"] = asignar_seccion(asignacion["latitude"], asignacion["longitude"],
                                                       geometrias, codigos)
    return asignacion


def asegurar_asignacion(listings: pd.DataFrame, *, ruta: Path = config.RUTA_ASIGNACION_SECCIONES,
                        dir_extra: Path = config.DIR_EXTRA_INE, forzar: bool = False) -> pd.DataFrame:
    """Devuelve la asignación anuncio -> sección. Usa la caché versionada si coincide con los anuncios
    (mismos ``ciudad, id, latitude, longitude``); si no, la recalcula con el seccionado de ``data/extra``."""
    clave = ["ciudad", "id", "latitude", "longitude"]
    if ruta.exists() and not forzar:
        cache = pd.read_csv(ruta, dtype={"codigo_seccion_ine": "string", "ciudad": "string"},
                            float_precision="round_trip")
        actual = listings[clave].reset_index(drop=True)
        if len(cache) == len(actual) and all(
                np.array_equal(cache[c].astype(str).to_numpy(), actual[c].astype(str).to_numpy()) for c in clave):
            return cache
        logger.warning("La caché de asignación no coincide con los anuncios actuales; se recalcula.")
    manifiesto = leer_manifiesto(config.DIR_RAW_INE) or {}
    carto = manifiesto.get("cartografia")
    ruta_zip = dir_extra / (carto["archivo"] if carto else Path(urlparse(config.URL_SECCIONADO_INE).path).name)
    if not ruta_zip.exists():
        raise ErrorDatos(f"Falta {ruta_zip}: ejecuta «python -m src.fuentes» con red para descargarlo.")
    if carto and carga.sha256_fichero(ruta_zip) != carto["sha256"]:
        raise ErrorDatos(f"{ruta_zip} no coincide con el SHA-256 del manifiesto del INE.")
    asignacion = calcular_asignacion(listings, ruta_zip)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as fbruto, gzip.GzipFile(filename="", mode="wb", fileobj=fbruto, mtime=0) as fgz:
        fgz.write(asignacion.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return asignacion


def unir_fuentes(listings: pd.DataFrame, asignacion: pd.DataFrame, indicadores: pd.DataFrame,
                 anio: int) -> pd.DataFrame:
    """Une anuncios e indicadores del ADRH por distrito censal y por municipio (left join).

    Añade ``codigo_seccion_ine``, ``codigo_distrito_ine``, ``codigo_municipio_ine``, ``municipio_ine``,
    ``ine_anio_referencia`` y, por cada indicador, ``ine_dis_<indicador>`` y ``ine_mun_<indicador>``.
    Nunca descarta anuncios: los no emparejados quedan con NaN (se avisa).
    """
    base = listings.merge(asignacion[["ciudad", "id", "codigo_seccion_ine"]], on=["ciudad", "id"],
                          how="left", validate="one_to_one")
    base["codigo_seccion_ine"] = base["codigo_seccion_ine"].astype("string")
    base["codigo_distrito_ine"] = base["codigo_seccion_ine"].str[:7]
    base["codigo_municipio_ine"] = base["codigo_seccion_ine"].str[:5]
    valores = [c for c in indicadores.columns if c not in ("nivel", "codigo", "nombre")]
    dis = indicadores[indicadores["nivel"] == "distrito"].set_index("codigo")[valores].add_prefix("ine_dis_")
    mun = indicadores[indicadores["nivel"] == "municipio"].set_index("codigo")[["nombre", *valores]]
    mun = mun.rename(columns={"nombre": "municipio_ine"}).rename(columns={v: f"ine_mun_{v}" for v in valores})
    unido = base.merge(dis, left_on="codigo_distrito_ine", right_index=True, how="left", validate="many_to_one")
    unido = unido.merge(mun, left_on="codigo_municipio_ine", right_index=True, how="left", validate="many_to_one")
    unido["ine_anio_referencia"] = anio
    if len(unido) != len(listings):
        raise ErrorDatos("La unión ha cambiado el número de anuncios.")
    tasas = tasas_emparejamiento(unido)
    for ciudad, fila in tasas.iterrows():
        if fila["pct_distrito"] < 100:
            logger.warning("%s: %d anuncios sin dato del INE por distrito (quedan con NaN).",
                           ciudad, int(fila["anuncios"] - fila["con_distrito"]))
    return unido


def tasas_emparejamiento(unido: pd.DataFrame) -> pd.DataFrame:
    """Anuncios y % emparejados por ciudad (sección hallada, dato de distrito y de municipio) y total."""
    marcas = pd.DataFrame({
        "ciudad": unido["ciudad"],
        "con_seccion": unido["codigo_seccion_ine"].notna(),
        "con_distrito": unido["ine_dis_renta_neta_persona"].notna(),
        "con_municipio": unido["ine_mun_renta_neta_persona"].notna(),
    })
    tabla = marcas.groupby("ciudad").agg(anuncios=("ciudad", "size"), con_seccion=("con_seccion", "sum"),
                                         con_distrito=("con_distrito", "sum"), con_municipio=("con_municipio", "sum"))
    tabla.loc["TOTAL"] = tabla.sum()
    for col in ("seccion", "distrito", "municipio"):
        tabla[f"pct_{col}"] = (100 * tabla[f"con_{col}"] / tabla["anuncios"]).round(2)
    return tabla


def comprobar_requisitos_union(unido: pd.DataFrame, min_filas: int = config.MIN_FILAS_UNION,
                               min_columnas: int = config.MIN_COLUMNAS_UNION) -> None:
    """``ErrorDatos`` si el conjunto unido no tiene ``min_filas`` filas, ``min_columnas`` columnas o
    ninguna columna con datos de la segunda fuente."""
    filas, columnas = unido.shape
    if filas < min_filas:
        raise ErrorDatos(f"El conjunto unido tiene {filas} filas; el enunciado exige al menos {min_filas}.")
    if columnas < min_columnas:
        raise ErrorDatos(f"El conjunto unido tiene {columnas} columnas; el enunciado exige al menos {min_columnas}.")
    cols_ine = [c for c in unido.columns if c.startswith(("ine_dis_", "ine_mun_"))]
    if not cols_ine or unido[cols_ine].notna().any(axis=1).sum() == 0:
        raise ErrorDatos("Ningún anuncio tiene datos del INE: la unión de las dos fuentes no ha funcionado.")


def construir_conjunto_unido(*, sin_descarga: bool = False, forzar: bool = False) -> tuple[pd.DataFrame, int]:
    """Asegura las dos fuentes, une y comprueba requisitos. Devuelve (conjunto unido, año del ADRH)."""
    if not sin_descarga:
        descargar_inside_airbnb_espana(forzar=forzar)
        descargar_ine(forzar=forzar)
    listings = leer_listings_espana()
    largo = leer_adrh()
    anio = elegir_anio_adrh(largo)
    unido = unir_fuentes(listings, asegurar_asignacion(listings, forzar=forzar), indicadores_adrh(largo, anio), anio)
    comprobar_requisitos_union(unido)
    return unido, anio


# --------------------------------------------------------------------------
# Validación de la unión (normalización determinista de nombres, sin similitud difusa)
# --------------------------------------------------------------------------
def normalizar_nombre(texto: Any) -> str:
    """Minúsculas, sin tildes ni signos, con el artículo pospuesto delante (``'Palmas, Las'`` -> ``'las palmas'``)."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)) or texto is pd.NA:
        return ""
    t = str(texto).strip()
    m = re.match(r"^(.*),\s*(el|la|los|las|l'|els|les|es|sa|ses|s')$", t, flags=re.I)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def variantes_nombre(texto: Any) -> set[str]:
    """Nombre normalizado y cada variante bilingüe (``'Donostia/San Sebastián'`` -> 3 variantes)."""
    texto = "" if texto is None or texto is pd.NA else str(texto)
    partes = [texto, *re.split(r"\s*/\s*|\s+-\s+", texto)] if texto else []
    return {n for n in (normalizar_nombre(p) for p in partes) if n}


def coincide_nombre(a: Any, b: Any) -> bool:
    """True si alguna variante normalizada de ``a`` es idéntica a alguna de ``b`` (igualdad exacta)."""
    return bool(variantes_nombre(a) & variantes_nombre(b))


def concordancia(unido: pd.DataFrame) -> pd.DataFrame:
    """Contrasta, anuncio a anuncio, la unidad asignada por punto en polígono con la que declara Inside Airbnb.

    - Madrid, Barcelona, Valencia y Sevilla: distrito de Inside Airbnb -> CUDIS oficial
      (``config.DISTRITOS_OFICIALES``) frente a ``codigo_distrito_ine``.
    - Resto: municipio de Inside Airbnb (Málaga: la propia ciudad) frente a ``municipio_ine``, por igualdad
      exacta de nombres normalizados más la tabla revisada ``config.ALIAS_MUNICIPIOS_INE``.
    ``coincide`` es NA si el anuncio no tiene unidad asignada o no hay referencia con la que compararlo.
    """
    partes = []
    for ciudad, s in unido.groupby("ciudad", sort=True):
        if ciudad in config.DISTRITOS_OFICIALES:
            declarado = s["neighbourhood_group_cleansed"].astype("string")
            esperado = declarado.map(config.DISTRITOS_OFICIALES[ciudad]).astype("string")
            asignado = s["codigo_distrito_ine"].astype("string")
            coincide = (esperado == asignado).astype("boolean")
            metodo = "distrito oficial"
        else:
            if ciudad == "malaga":
                declarado = pd.Series("Málaga", index=s.index, dtype="string")
            else:
                declarado = s["neighbourhood_cleansed"].map(reparar_mojibake).astype("string")
            esperado = declarado.map(lambda n: config.ALIAS_MUNICIPIOS_INE.get(n, n)).astype("string")
            asignado = s["municipio_ine"].astype("string")
            coincide = pd.Series([coincide_nombre(e, a) for e, a in zip(esperado, asignado)],
                                 index=s.index, dtype="boolean")
            metodo = "municipio (nombre normalizado + alias)"
        coincide = coincide.mask(esperado.isna() | asignado.isna())
        partes.append(pd.DataFrame({
            "ciudad": ciudad, "id": s["id"], "latitude": s["latitude"], "longitude": s["longitude"],
            "declarado_airbnb": declarado, "esperado_ine": esperado, "asignado_ine": asignado,
            "codigo_seccion_ine": s["codigo_seccion_ine"], "metodo": metodo, "coincide": coincide}))
    return pd.concat(partes, ignore_index=True)


def resumen_concordancia(tabla: pd.DataFrame) -> pd.DataFrame:
    """% de anuncios cuya unidad asignada coincide con la declarada, por ciudad y total."""
    validados = tabla[tabla["coincide"].notna()]
    res = validados.groupby("ciudad").agg(validados=("coincide", "size"), coinciden=("coincide", "sum"))
    res.loc["TOTAL"] = res.sum()
    res["pct_coinciden"] = (100 * res["coinciden"] / res["validados"]).round(2)
    return res


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """``python -m src.fuentes [--sin-descarga] [--forzar]``: descarga, une y muestra las tasas."""
    parser = argparse.ArgumentParser(description="Descarga Inside Airbnb (España) + INE ADRH y prueba la unión.")
    parser.add_argument("--sin-descarga", action="store_true", help="usar solo los ficheros locales")
    parser.add_argument("--forzar", action="store_true", help="volver a descargar y recalcular la asignación")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        unido, anio = construir_conjunto_unido(sin_descarga=args.sin_descarga, forzar=args.forzar)
    except ErrorPipeline as exc:
        logger.error("%s", exc)
        return exc.codigo_salida
    print(f"Conjunto unido: {unido.shape[0]:,} filas x {unido.shape[1]} columnas; ADRH {anio}.")
    print(tasas_emparejamiento(unido).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
