"""Variables derivadas (ingeniería de features) sobre el DataFrame limpio."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.config import Ciudad
from src.geo import haversine_km
from src.limpieza import limpiar_html

logger = logging.getLogger(__name__)

ETIQUETAS_CATEGORIA_PRECIO: tuple[str, ...] = ("Económico", "Medio-bajo", "Medio-alto", "Alto")
CORTES_CAPACIDAD: tuple[float, ...] = (0, 2, 4, 6, np.inf)
ETIQUETAS_CAPACIDAD: tuple[str, ...] = ("1-2 personas", "3-4 personas", "5-6 personas", "7 o más")
CORTES_SEGMENTO_HOST: tuple[float, ...] = (0, 1, 5, np.inf)
ETIQUETAS_SEGMENTO_HOST: tuple[str, ...] = ("1 anuncio", "2-5 anuncios", "6 o más anuncios")
LICENCIA_SIN = "Sin licencia"
LICENCIA_EXENTA = "Exenta"
LICENCIA_REGISTRO = "Con nº de registro"
LICENCIA_OTRO = "Otro / en proceso"


def agrupar_raras(serie: pd.Series, validas: pd.Series, min_n: int, etiqueta: str) -> tuple[pd.Series, dict[str, Any]]:
    """Agrupa en ``etiqueta`` las categorías con menos de ``min_n`` filas **válidas** (con precio).

    Regla única del proyecto (M4). Devuelve la serie agrupada y la descripción de la agrupación.
    """
    conteo = serie[validas.astype(bool)].value_counts()
    raras = sorted(str(c) for c in conteo.index[conteo < min_n])
    raras += sorted(str(c) for c in set(serie.dropna().unique()) - set(conteo.index))
    agrupada = serie.astype("string").where(~serie.astype("string").isin(raras), etiqueta)
    info = {"min_n_con_precio": min_n, "agrupadas": raras, "etiqueta": etiqueta,
            "n_con_precio_por_categoria": {str(k): int(v) for k, v in conteo.sort_index().items()}}
    return agrupada, info


def tipo_habitacion(room_type: pd.Series) -> pd.Series:
    """Traduce ``room_type`` al español."""
    return room_type.map(config.TRADUCCION_TIPO_HABITACION).fillna(room_type).astype("string")


def precio_por_persona(price: pd.Series, accommodates: pd.Series) -> pd.Series:
    """``price / accommodates``; capacidad ≤ 0 o precio no positivo → NaN."""
    capacidad = accommodates.where(accommodates > 0)
    return (price.where(price > 0) / capacidad).astype("float64")


def antiguedad_host(df: pd.DataFrame, fecha_referencia: pd.Timestamp) -> tuple[pd.Series, str]:
    """Antigüedad del host en años.

    Usa ``host_since`` si existe; si falta, ``hosts_time_as_host_years + months/12`` (coalesce fila a fila).
    Devuelve la serie y la fuente utilizada.
    """
    desde_fecha = pd.Series(np.nan, index=df.index, dtype="float64")
    if "host_since" in df and df["host_since"].notna().any():
        desde_fecha = ((fecha_referencia - df["host_since"]).dt.days / 365.25).astype("float64")
    alternativa = pd.Series(np.nan, index=df.index, dtype="float64")
    if {"hosts_time_as_host_years", "hosts_time_as_host_months"} <= set(df.columns):
        alternativa = (df["hosts_time_as_host_years"] + df["hosts_time_as_host_months"] / 12).astype("float64")
    resultado = desde_fecha.fillna(alternativa)
    if desde_fecha.notna().any():
        fuente = "host_since"
    elif alternativa.notna().any():
        fuente = "hosts_time_as_host_years + hosts_time_as_host_months / 12 (host_since 100 % nulo)"
    else:
        fuente = "no disponible"
    return resultado, fuente


def no_disponible_365(availability_365: pd.Series) -> pd.Series:
    """Fracción de días NO disponibles en los próximos 365: ``(365 − availability_365) / 365`` en [0, 1].

    No es ocupación: mezcla reservas y bloqueos del anfitrión, y mira hacia el futuro (no a los
    últimos 12 meses, como ``estimated_occupancy_l365d``).
    """
    return ((365 - availability_365) / 365).clip(0, 1).astype("float64")


def mes_cotizacion(checkin: pd.Series) -> pd.Series:
    """Mes (``AAAA-MM``) de la fecha de entrada de la cotización de precio; ``<NA>`` si no hay cotización."""
    return checkin.dt.strftime("%Y-%m").astype("string")


def licencia_tipo(license_: pd.Series) -> pd.Series:
    """Clasifica el texto de licencia (N9): Sin licencia / Exenta / Con nº de registro / Otro / en proceso."""
    texto = limpiar_html(license_).str.lower()
    res = pd.Series(LICENCIA_OTRO, index=license_.index, dtype="string")
    res = res.mask(texto.str.contains("registration number", na=False), LICENCIA_REGISTRO)
    res = res.mask(texto.str.contains("exempt", na=False), LICENCIA_EXENTA)
    res = res.mask(license_.isna() | texto.fillna("").eq(""), LICENCIA_SIN)
    return res


def categoria_precio(price: pd.Series, validas: pd.Series) -> tuple[pd.Series, list[float]]:
    """Cuartiles de precio (sobre precios válidos) con etiquetas en español. Devuelve serie y cortes en €."""
    _, cortes = pd.qcut(price[validas], 4, retbins=True)
    categorias = pd.cut(price.where(validas), bins=cortes, labels=list(ETIQUETAS_CATEGORIA_PRECIO),
                        include_lowest=True, ordered=True)
    return categorias, [float(c) for c in cortes]


def agregar_features(df: pd.DataFrame, ciudad: Ciudad, fecha_referencia: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Añade todas las variables derivadas. Devuelve (DataFrame nuevo, metadatos de features)."""
    out = df.copy()
    meta: dict[str, Any] = {}
    out["tipo_habitacion"] = tipo_habitacion(out["room_type"])
    out["tipo_habitacion_agrupado"], meta["agrupacion_tipo"] = agrupar_raras(
        out["tipo_habitacion"], out["precio_valido"], config.MIN_N_CATEGORIA_RARA, config.ETIQUETA_OTRAS_TIPO)
    out["log_precio"] = np.log(out["price"].where(out["precio_valido"])).astype("float64")
    out["precio_por_persona"] = precio_por_persona(out["price"], out["accommodates"])
    out["antiguedad_host_anios"], meta["antiguedad_fuente"] = antiguedad_host(out, fecha_referencia)
    out["no_disponible_365"] = no_disponible_365(out["availability_365"])
    out["ocupacion_inside_airbnb"] = (out["estimated_occupancy_l365d"] / 365).clip(0, 1).astype("float64")
    out["es_activo_probable"] = ~((out["availability_365"] == 0) & (out["number_of_reviews_ltm"] == 0))
    # Proxy hipotético: precio de UNA cotización × días no disponibles futuros. No acota nada.
    out["ingreso_hipotetico_anual"] = (out["price"].where(out["precio_valido"])
                                       * (365 - out["availability_365"])).astype("float64")
    out["mes_cotizacion"] = mes_cotizacion(out["price_quote_checkin_date"])
    out["mes_cotizacion_agrupado"], meta["agrupacion_mes"] = agrupar_raras(
        out["mes_cotizacion"], out["precio_valido"], config.MIN_N_CATEGORIA_RARA, config.ETIQUETA_OTROS_MESES)
    distancia = haversine_km(out["latitude"], out["longitude"], *ciudad.centro)
    out["distancia_centro_km"] = pd.Series(distancia, index=out.index).where(out["coords_validas"]).astype("float64")
    out["categoria_precio"], meta["cortes_categoria_precio"] = categoria_precio(out["price"], out["precio_valido"])
    out["categoria_capacidad"] = pd.cut(out["accommodates"], bins=list(CORTES_CAPACIDAD),
                                        labels=list(ETIQUETAS_CAPACIDAD), ordered=True)
    out["es_multianuncio"] = (out["calculated_host_listings_count"] > 1).astype(bool)
    out["segmento_host"] = pd.cut(out["calculated_host_listings_count"], bins=list(CORTES_SEGMENTO_HOST),
                                  labels=list(ETIQUETAS_SEGMENTO_HOST), ordered=True)
    if "license" in out:
        out["licencia_tipo"] = licencia_tipo(out["license"])
    else:  # Serie explícita (no escalar) para conservar el dtype string
        out["licencia_tipo"] = pd.Series(LICENCIA_SIN, index=out.index, dtype="string")
    out["tiene_resenas"] = (out["number_of_reviews"] > 0).astype(bool)
    out["dias_desde_ultima_resena"] = (fecha_referencia - out["last_review"]).dt.days.astype("float64")
    meta["fecha_referencia"] = fecha_referencia
    meta["licencia_tipo_conteo"] = {str(k): int(v) for k, v in out["licencia_tipo"].value_counts().sort_index().items()}
    return out, meta
