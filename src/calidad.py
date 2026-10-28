"""Perfilado de calidad de datos (antes/después) y comprobación de requisitos mínimos."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.errores import ErrorDatos

TEXTO_LIBRE: frozenset[str] = frozenset({
    "name", "description", "host_about", "license", "neighborhood_overview", "amenities",
    "price_quote_raw", "host_location", "bathrooms_text",
})
GEOESPACIALES: frozenset[str] = frozenset({"latitude", "longitude"})
_PATRON_FECHA = r"^\d{4}-\d{2}-\d{2}$"
_PATRON_NUMERO_TEXTO = r"^\$?-?[\d,]+(\.\d+)?%?$"
MAX_CARDINALIDAD_CATEGORICA = 60


def _tipo_logico(nombre: str, s: pd.Series) -> str:
    """Asigna un tipo lógico a una columna (ver ``clasificar_columnas``)."""
    tipos = pd.api.types
    if nombre in GEOESPACIALES:
        return "geoespacial"
    if nombre == "id" or nombre.endswith("_id") or nombre.endswith("_url"):
        return "identificador"
    if tipos.is_bool_dtype(s):
        return "booleano"
    if tipos.is_datetime64_any_dtype(s):
        return "fecha"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorico"
    if tipos.is_numeric_dtype(s):
        return "numerico"
    no_nulos = s.dropna().astype("string")
    if no_nulos.empty:
        return "vacia"
    if no_nulos.isin(["t", "f"]).all():
        return "booleano"
    if no_nulos.str.match(_PATRON_FECHA).all():
        return "fecha"
    if nombre in TEXTO_LIBRE:
        return "texto"
    if no_nulos.str.match(_PATRON_NUMERO_TEXTO).all():
        return "numerico"
    if no_nulos.nunique() <= MAX_CARDINALIDAD_CATEGORICA:
        return "categorico"
    return "texto"


def clasificar_columnas(df: pd.DataFrame) -> dict[str, str]:
    """Tipo lógico por columna: numerico, categorico, booleano, fecha, texto, geoespacial, identificador o vacia."""
    return {c: _tipo_logico(c, df[c]) for c in df.columns}


def contar_tipos(clasificacion: dict[str, str]) -> dict[str, int]:
    """Cuenta columnas por tipo lógico (orden alfabético)."""
    conteo: dict[str, int] = {}
    for t in clasificacion.values():
        conteo[t] = conteo.get(t, 0) + 1
    return dict(sorted(conteo.items()))


def perfilar(df: pd.DataFrame, top_n_nulos: int = 15) -> dict[str, Any]:
    """Resumen de calidad: filas, columnas, nulos (total y top N), tipos y duplicados."""
    nulos = df.isna().sum()
    total_celdas = df.shape[0] * df.shape[1]
    top = nulos[nulos > 0].sort_values(ascending=False, kind="mergesort").head(top_n_nulos)
    dtypes = df.dtypes.astype(str).value_counts().sort_index()
    clasif = clasificar_columnas(df)
    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nulos_total": int(nulos.sum()),
        "pct_nulos_total": float(nulos.sum() / total_celdas * 100) if total_celdas else 0.0,
        "columnas_con_nulos": int((nulos > 0).sum()),
        "columnas_100_nulas": sorted(nulos.index[nulos == len(df)].tolist()) if len(df) else [],
        "top_nulos": [{"columna": c, "n": int(n), "pct": float(n / len(df) * 100)} for c, n in top.items()],
        "nulos_por_columna_pct": {c: float(n / len(df) * 100) for c, n in nulos.items()},
        "dtypes": {k: int(v) for k, v in dtypes.items()},
        "tipos_logicos": contar_tipos(clasif),
        "duplicados_id": int(df["id"].duplicated().sum()) if "id" in df else 0,
        "duplicados_filas": int(df.duplicated().sum()),
        "memoria_mb": float(df.memory_usage(deep=True).sum() / 1e6),
    }


def comprobar_requisitos(df_final: pd.DataFrame, perfil_union: dict[str, Any]) -> dict[str, Any]:
    """Requisitos del enunciado sobre el conjunto FINAL procesado (≥ 50.000 filas y ≥ 20 columnas).

    Lanza ``ErrorDatos`` si no se cumplen (no se publica un análisis sobre un conjunto que no cumple).
    También informa de los tipos de variable presentes en el conjunto unido.
    """
    filas, columnas = df_final.shape
    if filas < config.MIN_FILAS_UNION or columnas < config.MIN_COLUMNAS_UNION:
        raise ErrorDatos(f"El conjunto final tiene {filas} filas y {columnas} columnas; el enunciado exige al "
                         f"menos {config.MIN_FILAS_UNION} filas y {config.MIN_COLUMNAS_UNION} columnas.")
    tipos = perfil_union["tipos_logicos"]
    presentes = {
        "numerico": tipos.get("numerico", 0) > 0,
        "categorico": tipos.get("categorico", 0) + tipos.get("booleano", 0) > 0,
        "fecha": tipos.get("fecha", 0) > 0,
        "texto": tipos.get("texto", 0) > 0,
        "geoespacial": tipos.get("geoespacial", 0) > 0,
    }
    return {
        "min_filas": config.MIN_FILAS_UNION, "min_columnas": config.MIN_COLUMNAS_UNION,
        "filas_final": int(filas), "columnas_final": int(columnas),
        "cumple_filas": bool(filas >= config.MIN_FILAS_UNION),
        "cumple_columnas": bool(columnas >= config.MIN_COLUMNAS_UNION),
        "tipos_presentes": presentes, "cumple_tipos": all(presentes.values()),
    }


def perfilar_por_ciudad(df: pd.DataFrame, columnas_clave: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Filas, % de celdas nulas y % de nulos de ``columnas_clave`` por ciudad."""
    salida: dict[str, dict[str, Any]] = {}
    for clave, s in df.groupby("ciudad", sort=True):
        celdas = s.shape[0] * s.shape[1]
        salida[str(clave)] = {
            "filas": int(len(s)), "columnas": int(s.shape[1]),
            "pct_nulos_total": float(s.isna().sum().sum() / celdas * 100) if celdas else 0.0,
            "duplicados_id": int(s["id"].duplicated().sum()) if "id" in s else 0,
            "nulos_pct": {c: float(s[c].isna().mean() * 100) for c in columnas_clave if c in s},
        }
    return salida


def construir_informe_calidad(antes: dict[str, Any], despues: dict[str, Any], registro: dict[str, Any],
                              por_ciudad_antes: dict[str, Any], por_ciudad_despues: dict[str, Any]) -> dict[str, Any]:
    """Ensambla ``calidad_datos.json``: perfiles global y por ciudad (antes/después) y registro de limpieza."""
    claves = ("conversiones", "reglas", "columnas_eliminadas", "columnas_reconstruidas",
              "politica_nulos", "contabilidad_filas", "duplicados", "imputados", "mojibake_reparado")
    ciudades = sorted(set(por_ciudad_antes) | set(por_ciudad_despues))
    return {
        "antes": {k: v for k, v in antes.items() if k != "nulos_por_columna_pct"},
        "despues": {k: v for k, v in despues.items() if k != "nulos_por_columna_pct"},
        "por_ciudad": {c: {"antes": por_ciudad_antes.get(c), "despues": por_ciudad_despues.get(c),
                           "reglas": registro["por_ciudad"].get(c)} for c in ciudades},
        **{k: registro.get(k) for k in claves},
    }
