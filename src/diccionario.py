"""Diccionario de datos del Parquet procesado: columna → (tipo lógico, descripción, origen).

``generar_markdown`` produce ``reports/diccionario_datos.md``. Las columnas que no
estén aquí se marcan «sin documentar» (aviso, no error) para tolerar snapshots futuros.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src import config

O = "original"
C = "convertida"
D = "derivada"

DICCIONARIO: dict[str, tuple[str, str, str]] = {
    "id": ("identificador", "Identificador único del anuncio", O),
    "last_scraped": ("fecha", "Fecha en que se scrapeó el anuncio", C),
    "name": ("texto", "Título del anuncio", O),
    "host_id": ("identificador", "Identificador numérico público del anfitrión en Airbnb; se conserva solo para "
                                 "agrupar anuncios por anfitrión (no se cruza con otras fuentes)", O),
    "hosts_time_as_user_years": ("numerico", "Años como usuario de Airbnb (parte entera)", O),
    "hosts_time_as_user_months": ("numerico", "Meses adicionales como usuario (resto, 0-11)", O),
    "hosts_time_as_host_years": ("numerico", "Años como anfitrión (parte entera)", O),
    "hosts_time_as_host_months": ("numerico", "Meses adicionales como anfitrión (resto, 0-11)", O),
    "host_is_superhost": ("booleano", "El anfitrión es superhost", C),
    "host_listings_count": ("numerico", "Nº de anuncios del anfitrión según Airbnb", O),
    "host_has_profile_pic": ("booleano", "El anfitrión tiene foto de perfil", C),
    "host_identity_verified": ("booleano", "Identidad del anfitrión verificada", C),
    "neighbourhood_cleansed": ("categorico", "Barrio (128 barrios oficiales de Madrid)", O),
    "neighbourhood_group_cleansed": ("categorico", "Distrito (21 distritos de Madrid)", O),
    "latitude": ("geoespacial", "Latitud (anonimizada por Airbnb, error de unos cientos de metros)", O),
    "longitude": ("geoespacial", "Longitud (anonimizada por Airbnb)", O),
    "property_type": ("categorico", "Tipo de propiedad declarado", O),
    "room_type": ("categorico", "Tipo de habitación original (inglés)", O),
    "accommodates": ("numerico", "Capacidad máxima de huéspedes", O),
    "bathrooms": ("numerico", "Nº de baños reconstruido desde bathrooms_text", D),
    "bedrooms": ("numerico", "Nº de dormitorios (imputado si faltaba, ver bedrooms_imputado)", C),
    "beds": ("numerico", "Nº de camas (imputado si faltaba, ver beds_imputado)", C),
    "price": ("numerico", "Precio por noche en euros (cotización para una ventana de fechas)", C),
    "price_quote_checkin_date": ("fecha", "Fecha de entrada de la cotización de precio", C),
    "price_quote_checkout_date": ("fecha", "Fecha de salida de la cotización de precio", C),
    "price_quote_total_price": ("numerico", "Precio total de la cotización", O),
    "price_quote_price_per_night": ("numerico", "Precio por noche de la cotización", O),
    "minimum_nights": ("numerico", "Estancia mínima en noches", O),
    "maximum_nights": ("numerico", "Estancia máxima en noches", O),
    "minimum_minimum_nights": ("numerico", "Mínimo de la estancia mínima en el calendario", O),
    "maximum_minimum_nights": ("numerico", "Máximo de la estancia mínima en el calendario", O),
    "minimum_maximum_nights": ("numerico", "Mínimo de la estancia máxima en el calendario", O),
    "maximum_maximum_nights": ("numerico", "Máximo de la estancia máxima en el calendario", O),
    "minimum_nights_avg_ntm": ("numerico", "Media de estancia mínima en los próximos 365 días", O),
    "maximum_nights_avg_ntm": ("numerico", "Media de estancia máxima en los próximos 365 días", O),
    "has_availability": ("booleano", "Tiene disponibilidad (nullable)", C),
    "availability_30": ("numerico", "Días disponibles en los próximos 30", O),
    "availability_60": ("numerico", "Días disponibles en los próximos 60", O),
    "availability_90": ("numerico", "Días disponibles en los próximos 90", O),
    "availability_365": ("numerico", "Días disponibles en los próximos 365", O),
    "calendar_last_scraped": ("fecha", "Fecha de scrape del calendario", C),
    "number_of_reviews": ("numerico", "Nº total de reseñas", O),
    "number_of_reviews_ltm": ("numerico", "Reseñas en los últimos 12 meses", O),
    "number_of_reviews_l30d": ("numerico", "Reseñas en los últimos 30 días", O),
    "availability_eoy": ("numerico", "Días disponibles hasta fin de año", O),
    "number_of_reviews_ly": ("numerico", "Reseñas del último año natural", O),
    "estimated_occupancy_l365d": ("numerico", "Ocupación estimada por Inside Airbnb (noches, 12 meses)", O),
    "estimated_revenue_l365d": ("numerico", "Ingresos estimados por Inside Airbnb (12 meses)", O),
    "first_review": ("fecha", "Fecha de la primera reseña", C),
    "last_review": ("fecha", "Fecha de la última reseña", C),
    "review_scores_rating": ("numerico", "Puntuación global (1-5)", O),
    "review_scores_accuracy": ("numerico", "Puntuación de veracidad (1-5)", O),
    "review_scores_cleanliness": ("numerico", "Puntuación de limpieza (1-5)", O),
    "review_scores_checkin": ("numerico", "Puntuación de llegada (1-5)", O),
    "review_scores_communication": ("numerico", "Puntuación de comunicación (1-5)", O),
    "review_scores_location": ("numerico", "Puntuación de ubicación (1-5)", O),
    "review_scores_value": ("numerico", "Puntuación calidad-precio (1-5)", O),
    "license": ("texto", "Texto de licencia turística (con HTML)", O),
    "calculated_host_listings_count": ("numerico", "Anuncios del anfitrión en este snapshot", O),
    "calculated_host_listings_count_entire_homes": ("numerico", "Alojamientos enteros del anfitrión", O),
    "calculated_host_listings_count_private_rooms": ("numerico", "Habitaciones privadas del anfitrión", O),
    "calculated_host_listings_count_shared_rooms": ("numerico", "Habitaciones compartidas del anfitrión", O),
    "reviews_per_month": ("numerico", "Reseñas por mes", O),
    "bano_compartido": ("booleano", "El baño es compartido (desde bathrooms_text)", D),
    "n_amenities": ("numerico", "Nº de amenities del anuncio", D),
    "tiene_wifi": ("booleano", "Amenity: wifi", D),
    "tiene_aire_acondicionado": ("booleano", "Amenity: aire acondicionado", D),
    "tiene_cocina": ("booleano", "Amenity: cocina", D),
    "tiene_calefaccion": ("booleano", "Amenity: calefacción", D),
    "tiene_lavadora": ("booleano", "Amenity: lavadora", D),
    "tiene_ascensor": ("booleano", "Amenity: ascensor", D),
    "tiene_terraza_balcon": ("booleano", "Amenity: terraza, patio o balcón", D),
    "tiene_parking_cualquiera": ("booleano", "Amenity: cualquier tipo de parking (gratuito o de pago)", D),
    "tiene_piscina": ("booleano", "Amenity: piscina", D),
    "tiene_tv": ("booleano", "Amenity: televisión", D),
    "tiene_espacio_trabajo": ("booleano", "Amenity: espacio de trabajo", D),
    "admite_mascotas": ("booleano", "Amenity: admite mascotas", D),
    "llegada_autonoma": ("booleano", "Amenity: llegada autónoma (self check-in)", D),
    "nombre_longitud": ("numerico", "Longitud del título sin HTML (caracteres)", D),
    "descripcion_longitud": ("numerico", "Longitud de la descripción sin HTML (caracteres)", D),
    "descripcion_palabras": ("numerico", "Palabras de la descripción sin HTML", D),
    "host_tiene_descripcion": ("booleano", "El anfitrión tiene texto «sobre mí»", D),
    "precio_valido": ("booleano", "Precio publicado y mayor que 0", D),
    "es_outlier_bajo": ("booleano", "Precio por debajo de la valla inferior (IQR en log)", D),
    "es_outlier_alto": ("booleano", "Precio por encima de la valla superior (IQR en log)", D),
    "es_outlier_precio": ("booleano", "Precio atípico por cualquiera de las dos vallas", D),
    "minimum_nights_extremo": ("booleano", "Estancia mínima mayor de 365 noches", D),
    "alquiler_larga_estancia": ("booleano", "Estancia mínima de 30 noches o más", D),
    "coords_validas": ("booleano", "Coordenadas dentro del bounding box de Madrid", D),
    "bedrooms_imputado": ("booleano", "bedrooms se imputó", D),
    "beds_imputado": ("booleano", "beds se imputó", D),
    "tipo_habitacion": ("categorico", "Tipo de habitación en español", D),
    "tipo_habitacion_agrupado": ("categorico", "Tipo de habitación con categorías raras (< 200 con precio) agrupadas", D),
    "log_precio": ("numerico", "Logaritmo natural del precio", D),
    "precio_por_persona": ("numerico", "Precio dividido por la capacidad (€)", D),
    "antiguedad_host_anios": ("numerico", "Antigüedad del anfitrión en años", D),
    "no_disponible_365": ("numerico", "Fracción de días NO disponibles en los próximos 365: (365 − availability_365) / 365. "
                          "Mezcla reservas y bloqueos; no es ocupación real", D),
    "ocupacion_inside_airbnb": ("numerico", "estimated_occupancy_l365d / 365 (estimación de Inside Airbnb, últimos 12 meses)", D),
    "es_activo_probable": ("booleano", "No (availability_365 = 0 y sin reseñas en 12 meses)", D),
    "ingreso_hipotetico_anual": ("numerico", "Proxy hipotético: price de una cotización × días no disponibles futuros (€). "
                                 "No es un ingreso observado ni una cota", D),
    "mes_cotizacion": ("categorico", "Mes (AAAA-MM) de la fecha de entrada de la cotización de precio", D),
    "mes_cotizacion_agrupado": ("categorico", "Mes de la cotización con meses raros (< 200 con precio) agrupados", D),
    "noches_cotizadas": ("numerico", "Noches de la estancia cotizada (salida − entrada de la cotización)", D),
    "es_tarifa_mensual": ("booleano", "La cotización usa tarifa mensual (is_monthly_priced en price_quote_raw)", D),
    "distancia_centro_km": ("numerico", "Distancia haversine a la Puerta del Sol (km)", D),
    "categoria_precio": ("categorico", "Cuartil de precio (Económico … Alto)", D),
    "categoria_capacidad": ("categorico", "Tramo de capacidad (1-2, 3-4, 5-6, 7 o más)", D),
    "es_multianuncio": ("booleano", "El anfitrión tiene más de un anuncio", D),
    "segmento_host": ("categorico", "Segmento del anfitrión por nº de anuncios", D),
    "licencia_tipo": ("categorico", "Sin licencia / Exenta / Con nº de registro / Otro o en proceso", D),
    "tiene_resenas": ("booleano", "Tiene al menos una reseña", D),
    "dias_desde_ultima_resena": ("numerico", "Días desde la última reseña hasta la fecha de referencia", D),
}

SIN_DOCUMENTAR = ("sin documentar", "Columna nueva en el snapshot, no documentada", "original")


def documentadas() -> set[str]:
    """Conjunto de columnas documentadas."""
    return set(DICCIONARIO)


def tabla(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Filas del diccionario para las columnas de ``df`` (con dtype real y % de nulos)."""
    filas = []
    for col in df.columns:
        tipo, desc, origen = DICCIONARIO.get(col, SIN_DOCUMENTAR)
        filas.append({"columna": col, "tipo_logico": tipo, "dtype": str(df[col].dtype),
                      "origen": origen, "descripcion": desc,
                      "pct_nulos": float(df[col].isna().mean() * 100)})
    return filas


def generar_markdown(filas: list[dict[str, Any]], formatear_pct) -> str:
    """Genera el Markdown del diccionario a partir de ``tabla`` (sin cifras escritas a mano)."""
    lineas = [
        "# Diccionario de datos: `data/processed/listings_limpio.parquet`",
        "",
        "Generado automáticamente por `src/diccionario.py` al ejecutar `python run_pipeline.py`.",
        "Origen: *original* (columna de Inside Airbnb sin cambios), *convertida* (tipo corregido) "
        "o *derivada* (creada en este proyecto).",
        "",
        "| Columna | Tipo lógico | dtype | Origen | % nulos | Descripción |",
        "|---|---|---|---|---|---|",
    ]
    for f in filas:
        lineas.append(f"| `{f['columna']}` | {f['tipo_logico']} | `{f['dtype']}` | {f['origen']} | "
                      f"{formatear_pct(f['pct_nulos'])} | {f['descripcion']} |")
    return "\n".join(lineas) + "\n"


def columnas_requeridas_documentadas() -> list[str]:
    """Columnas del dashboard/tabla/ML que no están documentadas (debe ser vacía)."""
    usadas = set(config.COLUMNAS_DASHBOARD) | set(config.COLUMNAS_TABLA) | set(config.FEATURES_ML)
    return sorted(usadas - set(DICCIONARIO))
