"""Diccionario de datos del conjunto final procesado: columna → (tipo lógico, descripción, fuente, origen).

``generar_markdown`` produce ``reports/diccionario_datos.md``. Toda columna del conjunto final debe estar
documentada (el pipeline se detiene si no lo está).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src import config

# Fuente de cada columna.
AIRBNB = "Inside Airbnb"
INE = "INE (ADRH)"
UNION = "Unión (INE)"
PROPIA = "Derivada"
# Origen de la transformación.
O = "original"
C = "convertida"
D = "derivada"

_INDICADORES: dict[str, str] = {
    "renta_neta_persona": "Renta neta media por persona (€/año)",
    "renta_mediana_uc": "Mediana de la renta por unidad de consumo (€/año)",
    "gini": "Índice de Gini de la renta (0-100)",
    "edad_media": "Edad media de la población (años)",
    "pct_hogares_unipersonales": "% de hogares unipersonales",
    "poblacion": "Población (personas)",
    "pct_bajo_60_mediana": "% de población con ingresos por unidad de consumo por debajo del 60 % de la mediana",
}

DICCIONARIO: dict[str, tuple[str, str, str, str]] = {
    "ciudad": ("categorico", "Clave de la ciudad o región de Inside Airbnb (9 en España)", AIRBNB, D),
    "id": ("identificador", "Identificador único del anuncio", AIRBNB, O),
    "fecha_snapshot": ("fecha", "Fecha del snapshot de Inside Airbnb (de la URL de descarga)", AIRBNB, D),
    "latitude": ("geoespacial", "Latitud (anonimizada por Airbnb, error de unos cientos de metros)", AIRBNB, O),
    "longitude": ("geoespacial", "Longitud (anonimizada por Airbnb)", AIRBNB, O),
    "neighbourhood_cleansed": ("categorico", "Barrio (ciudades) o municipio (regiones), con la doble "
                               "codificación de Euskadi reparada", AIRBNB, C),
    "neighbourhood_group_cleansed": ("categorico", "Distrito (Madrid, Barcelona, Valencia, Sevilla) o provincia "
                                     "(Euskadi); vacío en el resto", AIRBNB, C),
    "coords_validas": ("booleano", "Coordenadas dentro de la envolvente (bbox) de su ciudad", PROPIA, D),
    "distancia_centro_km": ("numerico", "Distancia haversine al centro de su ciudad (km); solo ciudades urbanas, "
                            "NaN en las regiones", PROPIA, D),
    "codigo_distrito_ine": ("identificador", "Distrito censal del INE (7 dígitos) asignado por punto en polígono",
                            UNION, D),
    "codigo_municipio_ine": ("identificador", "Municipio del INE (5 dígitos) asignado por punto en polígono",
                             UNION, D),
    "municipio_ine": ("categorico", "Nombre del municipio según el INE", UNION, D),
    **{f"ine_dis_{k}": ("numerico", f"{v} del distrito censal (ADRH, año de referencia)", INE, O)
       for k, v in _INDICADORES.items()},
    **{f"ine_mun_{k}": ("numerico", f"{v} del municipio (ADRH, año de referencia)", INE, O)
       for k, v in _INDICADORES.items()},
    "tipo_habitacion": ("categorico", "Tipo de habitación en español", AIRBNB, C),
    "tipo_habitacion_agrupado": ("categorico", "Tipo de habitación con categorías raras (< 200 con precio) agrupadas",
                                 PROPIA, D),
    "accommodates": ("numerico", "Capacidad máxima de huéspedes", AIRBNB, O),
    "categoria_capacidad": ("categorico", "Tramo de capacidad (1-2, 3-4, 5-6, 7 o más)", PROPIA, D),
    "bedrooms": ("numerico", "Nº de dormitorios (imputado si faltaba, ver bedrooms_imputado)", AIRBNB, C),
    "beds": ("numerico", "Nº de camas (imputado si faltaba, ver beds_imputado)", AIRBNB, C),
    "bathrooms": ("numerico", "Nº de baños reconstruido desde bathrooms_text", AIRBNB, D),
    "bano_compartido": ("booleano", "El baño es compartido (desde bathrooms_text)", PROPIA, D),
    "bedrooms_imputado": ("booleano", "bedrooms se imputó (mediana por tipo y capacidad)", PROPIA, D),
    "beds_imputado": ("booleano", "beds se imputó (mediana por tipo y capacidad)", PROPIA, D),
    "n_amenities": ("numerico", "Nº de amenities del anuncio", PROPIA, D),
    "tiene_wifi": ("booleano", "Amenity: wifi", PROPIA, D),
    "tiene_aire_acondicionado": ("booleano", "Amenity: aire acondicionado", PROPIA, D),
    "tiene_cocina": ("booleano", "Amenity: cocina", PROPIA, D),
    "tiene_calefaccion": ("booleano", "Amenity: calefacción", PROPIA, D),
    "tiene_lavadora": ("booleano", "Amenity: lavadora", PROPIA, D),
    "tiene_ascensor": ("booleano", "Amenity: ascensor", PROPIA, D),
    "tiene_terraza_balcon": ("booleano", "Amenity: terraza, patio o balcón", PROPIA, D),
    "tiene_parking_cualquiera": ("booleano", "Amenity: cualquier tipo de parking (gratuito o de pago)", PROPIA, D),
    "tiene_piscina": ("booleano", "Amenity: piscina", PROPIA, D),
    "tiene_tv": ("booleano", "Amenity: televisión", PROPIA, D),
    "tiene_espacio_trabajo": ("booleano", "Amenity: espacio de trabajo", PROPIA, D),
    "admite_mascotas": ("booleano", "Amenity: admite mascotas", PROPIA, D),
    "llegada_autonoma": ("booleano", "Amenity: llegada autónoma (self check-in)", PROPIA, D),
    "price": ("numerico", "Precio por noche en euros de una cotización fechada (descuentos aplicados, sin tasas)",
              AIRBNB, C),
    "precio_valido": ("booleano", "Precio publicado y mayor que 0", PROPIA, D),
    "log_precio": ("numerico", "Logaritmo natural del precio", PROPIA, D),
    "es_outlier_precio": ("booleano", "Precio fuera de las vallas de Tukey sobre ln(precio) del conjunto", PROPIA, D),
    "es_tarifa_mensual": ("booleano", "La cotización usa tarifa mensual (is_monthly_priced en price_quote_raw)",
                          PROPIA, D),
    "minimum_nights": ("numerico", "Estancia mínima en noches (si faltaba, minimum_nights_avg_ntm; ver "
                                   "minimum_nights_imputado)", AIRBNB, C),
    "minimum_nights_imputado": ("booleano", "minimum_nights faltaba y se rellenó con minimum_nights_avg_ntm "
                                            "(media de los próximos 30 días)", PROPIA, D),
    "alquiler_larga_estancia": ("booleano", "Estancia mínima de 30 noches o más", PROPIA, D),
    "mes_cotizacion": ("categorico", "Mes (AAAA-MM) de la fecha de entrada de la cotización de precio", PROPIA, D),
    "mes_cotizacion_agrupado": ("categorico", "Mes de la cotización con meses raros (< 200 con precio) agrupados",
                                PROPIA, D),
    "host_id": ("identificador", "Identificador numérico público del anfitrión en Airbnb; solo para agrupar "
                                 "anuncios por anfitrión (no se cruza con otras fuentes)", AIRBNB, O),
    "host_is_superhost": ("booleano", "El anfitrión es superhost", AIRBNB, C),
    "calculated_host_listings_count": ("numerico", "Anuncios del anfitrión en el snapshot de su ciudad", AIRBNB, O),
    "es_multianuncio": ("booleano", "El anfitrión tiene más de un anuncio en su ciudad", PROPIA, D),
    "segmento_host": ("categorico", "Segmento del anfitrión por nº de anuncios (1, 2-5, 6 o más)", PROPIA, D),
    "antiguedad_host_anios": ("numerico", "Antigüedad como anfitrión en años (hosts_time_as_host_*)", PROPIA, D),
    "licencia_tipo": ("categorico", "Sin licencia / Exenta / Con nº de registro / Otro o en proceso", PROPIA, D),
    "licencia_registro_nacional": ("booleano", "El texto de licencia incluye el número del registro único nacional",
                                   PROPIA, D),
    "availability_365": ("numerico", "Días disponibles en los próximos 365", AIRBNB, O),
    "no_disponible_365": ("numerico", "Proxy de ocupación: % de días NO disponibles en los próximos 365, "
                          "(365 − availability_365) / 365; mezcla reservas y bloqueos", PROPIA, D),
    "es_activo_probable": ("booleano", "No (availability_365 = 0 y sin reseñas en 12 meses)", PROPIA, D),
    "number_of_reviews": ("numerico", "Nº total de reseñas", AIRBNB, O),
    "review_scores_rating": ("numerico", "Puntuación global (1-5)", AIRBNB, O),
    "estimated_occupancy_l365d": ("numerico", "Noches ocupadas estimadas por Inside Airbnb (últimos 12 meses)",
                                  AIRBNB, O),
}


def documentadas() -> set[str]:
    """Conjunto de columnas documentadas."""
    return set(DICCIONARIO)


def tabla(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Filas del diccionario para las columnas de ``df`` (con dtype real y % de nulos)."""
    filas = []
    for col in df.columns:
        tipo, desc, fuente, origen = DICCIONARIO[col]
        filas.append({"columna": col, "tipo_logico": tipo, "dtype": str(df[col].dtype), "fuente": fuente,
                      "origen": origen, "descripcion": desc, "pct_nulos": float(df[col].isna().mean() * 100)})
    return filas


def generar_markdown(filas: list[dict[str, Any]], formatear_pct, formatear_n) -> str:
    """Genera el Markdown del diccionario a partir de ``tabla`` (sin cifras escritas a mano)."""
    por_fuente: dict[str, int] = {}
    for f in filas:
        por_fuente[f["fuente"]] = por_fuente.get(f["fuente"], 0) + 1
    resumen = ", ".join(f"{formatear_n(n)} de {fuente}" for fuente, n in sorted(por_fuente.items()))
    lineas = [
        f"# Diccionario de datos: `{config.RUTA_PARQUET.relative_to(config.RAIZ).as_posix()}`",
        "",
        "Generado automáticamente por `src/diccionario.py` al ejecutar `python run_pipeline.py`.",
        f"Columnas: {formatear_n(len(filas))} ({resumen}).",
        "",
        "Fuente: *Inside Airbnb* (fuente 1), *INE (ADRH)* (fuente 2, indicadores de la zona), *Unión (INE)* "
        "(claves asignadas por punto en polígono con el seccionado censal del INE) o *Derivada* (creada en este "
        "proyecto). Origen: *original* (sin cambios), *convertida* (tipo o texto corregido) o *derivada*.",
        "",
        "| Columna | Fuente | Tipo lógico | dtype | Origen | % nulos | Descripción |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in filas:
        lineas.append(f"| `{f['columna']}` | {f['fuente']} | {f['tipo_logico']} | `{f['dtype']}` | {f['origen']} | "
                      f"{formatear_pct(f['pct_nulos'])} | {f['descripcion']} |")
    return "\n".join(lineas) + "\n"


def columnas_requeridas_documentadas() -> list[str]:
    """Columnas finales o del ML que no están documentadas (debe ser vacía)."""
    usadas = set(config.COLUMNAS_FINALES) | set(config.FEATURES_ML)
    return sorted(usadas - set(DICCIONARIO))
