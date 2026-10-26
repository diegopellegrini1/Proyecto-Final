"""Configuración central: rutas, constantes, ciudad y listas de columnas.

Todas las constantes del proyecto están nombradas aquí; ninguna función usa
valores mágicos. Las rutas son relativas a la raíz del repositorio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ: Path = Path(__file__).resolve().parents[1]
DIR_DATA: Path = RAIZ / "data"
DIR_RAW: Path = DIR_DATA / "raw"
DIR_PROCESADO: Path = DIR_DATA / "processed"
DIR_EXTRA: Path = DIR_DATA / "extra"
DIR_REPORTS: Path = RAIZ / "reports"
DIR_FIGURAS: Path = DIR_REPORTS / "figures"
DIR_PLANTILLAS: Path = RAIZ / "src" / "plantillas"
DIR_NOTEBOOKS: Path = RAIZ / "notebooks"

RUTA_MANIFIESTO: Path = DIR_RAW / "MANIFEST.json"
RUTA_LISTINGS: Path = DIR_RAW / "listings.csv.gz"
RUTA_BARRIOS: Path = DIR_RAW / "neighbourhoods.csv"
RUTA_GEOJSON: Path = DIR_RAW / "neighbourhoods.geojson"
RUTA_PARQUET: Path = DIR_PROCESADO / "listings_limpio.parquet"
RUTA_MUESTRA_CSV: Path = DIR_PROCESADO / "listings_muestra.csv"
RUTA_FINDINGS: Path = DIR_REPORTS / "findings.json"
RUTA_CALIDAD: Path = DIR_REPORTS / "calidad_datos.json"
RUTA_INFORME: Path = DIR_REPORTS / "informe.md"
RUTA_DICCIONARIO: Path = DIR_REPORTS / "diccionario_datos.md"
RUTA_README: Path = RAIZ / "README.md"

# --------------------------------------------------------------------------
# Parámetros de análisis
# --------------------------------------------------------------------------
SEMILLA: int = 42
ALFA: float = 0.05
N_BOOT: int = 2000
N_BOOT_MIN: int = 100
N_BOOT_MAX: int = 20_000
N_SHAPIRO: int = 5000
MIN_N_GRUPO: int = 30            # mínimo de anuncios con precio para tests y rankings
MIN_N_CATEGORIA_RARA: int = 200  # por debajo (en filas con precio), la categoría se agrupa
MIN_N_BARRIO_MAPA: int = 10
TOP_N_DISTRITOS: int = 10
IQR_K: float = 1.5               # vallas de Tukey sobre ln(precio)
MIN_NIGHTS_EXTREMO: int = 365
MIN_NIGHTS_LARGA_ESTANCIA: int = 30
TAM_MUESTRA_CSV: int = 500
LIMITE_SUBIDA_BYTES: int = 25_000_000
MAX_DESCARGA_BYTES: int = 300_000_000
UMBRAL_FILAS: int = 1000
UMBRAL_COLUMNAS: int = 20

# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------
URL_PAGINA_DATOS: str = "https://insideairbnb.com/get-the-data/"
DOMINIOS_PERMITIDOS: frozenset[str] = frozenset(
    {"insideairbnb.com", "www.insideairbnb.com", "data.insideairbnb.com"}
)
USER_AGENT: str = "Proyecto-Final-Airbnb-Madrid (+https://github.com/diegopellegrini1; uso academico)"
TIMEOUT_CONEXION_S: int = 15
TIMEOUT_LECTURA_S: int = 120
REINTENTOS: int = 3
MAX_REDIRECCIONES: int = 3

# Ficheros obligatorios y la subcarpeta de la URL en la que se publican.
FICHEROS_OBLIGATORIOS: dict[str, str] = {
    "listings.csv.gz": "data",
    "neighbourhoods.csv": "visualisations",
    "neighbourhoods.geojson": "visualisations",
}
FICHEROS_EXTRA: dict[str, str] = {
    "reviews.csv.gz": "data",
    "calendar.csv.gz": "data",
}


@dataclass(frozen=True)
class Ciudad:
    """Parámetros geográficos y de origen de datos de una ciudad."""

    clave: str
    slug_ruta: str
    nombre: str
    centro: tuple[float, float]
    centro_nombre: str
    bbox: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    distrito_referencia: str
    url_directa_conocida: dict[str, str] = field(default_factory=dict)


_BASE_MADRID = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2026-09-17"

# Solo Madrid está validada. Barcelona se menciona como plan B en el mensaje de ErrorDescarga.
CIUDADES: dict[str, Ciudad] = {
    "madrid": Ciudad(
        clave="madrid",
        slug_ruta="spain/comunidad-de-madrid/madrid",
        nombre="Madrid",
        centro=(40.4169, -3.7035),
        centro_nombre="Puerta del Sol",
        bbox=(40.31, 40.65, -3.90, -3.50),
        distrito_referencia="Centro",
        # Último snapshot verificado; se usa solo si falla la página de descubrimiento.
        url_directa_conocida={
            "listings.csv.gz": f"{_BASE_MADRID}/data/listings.csv.gz",
            "neighbourhoods.csv": f"{_BASE_MADRID}/visualisations/neighbourhoods.csv",
            "neighbourhoods.geojson": f"{_BASE_MADRID}/visualisations/neighbourhoods.geojson",
        },
    ),
}
CIUDAD_POR_DEFECTO: str = "madrid"

# --------------------------------------------------------------------------
# Columnas
# --------------------------------------------------------------------------
COLUMNAS_OBLIGATORIAS: tuple[str, ...] = (
    "id", "price", "room_type", "accommodates", "latitude", "longitude",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "host_id",
    "host_is_superhost", "availability_365", "amenities", "minimum_nights",
    "number_of_reviews", "last_scraped",
)

TRADUCCION_TIPO_HABITACION: dict[str, str] = {
    "Entire home/apt": "Alojamiento entero",
    "Private room": "Habitación privada",
    "Shared room": "Habitación compartida",
    "Hotel room": "Habitación de hotel",
}
ETIQUETA_OTRAS_TIPO: str = "Otras (compartida + hotel)"

# Patrones (regex, sin distinguir mayúsculas) sobre el texto crudo de amenities.
PATRONES_AMENITIES: dict[str, str] = {
    "tiene_wifi": r"wifi",
    "tiene_aire_acondicionado": r"air conditioning",
    "tiene_cocina": r"kitchen",
    "tiene_calefaccion": r"heating",
    "tiene_lavadora": r"washer",
    "tiene_ascensor": r"elevator",
    "tiene_terraza_balcon": r"balcony|patio|terrace",
    "tiene_parking_cualquiera": r"parking",
    "tiene_piscina": r"pool",
    "tiene_tv": r"\btv\b",
    "tiene_espacio_trabajo": r"workspace",
    "admite_mascotas": r"pets allowed",
    "llegada_autonoma": r"self check-in",
}

# Nombres legibles de los flags de amenities.
NOMBRES_AMENITIES: dict[str, str] = {
    "tiene_wifi": "Wifi", "tiene_aire_acondicionado": "Aire acondicionado", "tiene_cocina": "Cocina",
    "tiene_calefaccion": "Calefacción", "tiene_lavadora": "Lavadora", "tiene_ascensor": "Ascensor",
    "tiene_terraza_balcon": "Terraza/balcón", "tiene_parking_cualquiera": "Parking (cualquiera)",
    "tiene_piscina": "Piscina", "tiene_tv": "TV", "tiene_espacio_trabajo": "Espacio de trabajo",
    "admite_mascotas": "Admite mascotas", "llegada_autonoma": "Llegada autónoma",
}

# Lista blanca de variables del modelo ML (M5).
FEATURES_ML: tuple[str, ...] = (
    "accommodates", "bedrooms", "beds", "bathrooms", "bano_compartido", "n_amenities",
    *PATRONES_AMENITIES.keys(),
    "distancia_centro_km", "tipo_habitacion", "neighbourhood_group_cleansed",
    "host_is_superhost", "es_multianuncio", "licencia_tipo", "number_of_reviews",
    "review_scores_rating", "review_scores_location", "review_scores_cleanliness",
    "minimum_nights", "availability_365", "antiguedad_host_anios",
)
FEATURES_ML_CATEGORICAS: tuple[str, ...] = ("tipo_habitacion", "neighbourhood_group_cleansed", "licencia_tipo")

# Lista negra: segunda barrera contra la fuga del objetivo (precio).
COLUMNAS_PROHIBIDAS_FUGA: frozenset[str] = frozenset({
    "price", "log_precio", "precio_por_persona", "ingresos_estimados_anuales",
    "estimated_revenue_l365d", "categoria_precio", "precio_valido",
    "es_outlier_precio", "es_outlier_alto", "es_outlier_bajo",
})
PREFIJOS_PROHIBIDOS_FUGA: tuple[str, ...] = ("price_", "precio_", "ingresos_", "es_outlier")

# Columnas que lee el dashboard (subconjunto del Parquet) y las de la tabla.
COLUMNAS_DASHBOARD: tuple[str, ...] = (
    "id", "name", "host_id", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "tipo_habitacion", "room_type", "price", "precio_valido", "precio_por_persona",
    "accommodates", "bedrooms", "latitude", "longitude", "coords_validas",
    "distancia_centro_km", "host_is_superhost", "es_multianuncio", "segmento_host",
    "calculated_host_listings_count", "ocupacion_estimada", "es_activo_probable",
    "number_of_reviews", "review_scores_rating", "n_amenities", "minimum_nights",
    "categoria_precio", *PATRONES_AMENITIES.keys(),
)
COLUMNAS_TABLA: tuple[str, ...] = (
    "id", "name", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
    "tipo_habitacion", "price", "accommodates", "bedrooms", "distancia_centro_km",
    "host_is_superhost", "number_of_reviews", "review_scores_rating", "minimum_nights",
)
MAX_PUNTOS_MAPA: int = 10_000
