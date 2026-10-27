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
MAX_DESCOMPRIMIDO_BYTES: int = 2_000_000_000   # tope de bytes descomprimidos al validar un .gz
N_FOLDS_ML: int = 5                            # GroupKFold por anfitrión
TEST_SIZE_ALEATORIO: float = 0.2               # partición aleatoria (solo contraste optimista)
MIN_N_CATEGORIA_RARA_ML: int = 30              # agrupación de categorías raras dentro del Pipeline ML
N_REPETICIONES_PERMUTACION: int = 3
ETIQUETA_OTROS_MESES: str = "Otros meses"
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
# Contrato de columnas: todas las que el pipeline lee sin guarda. Si falta alguna, ErrorDatos
# accionable al leer (nunca KeyError en una etapa posterior). Las demás se usan con «if col in df».
COLUMNAS_OBLIGATORIAS: tuple[str, ...] = (
    "id", "name", "price", "room_type", "accommodates", "latitude", "longitude",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "host_id",
    "host_is_superhost", "availability_365", "amenities", "minimum_nights",
    "number_of_reviews", "number_of_reviews_ltm", "last_scraped", "last_review",
    "calculated_host_listings_count", "estimated_revenue_l365d", "estimated_occupancy_l365d",
    "review_scores_rating", "price_quote_checkin_date", "price_quote_checkout_date",
)

TRADUCCION_TIPO_HABITACION: dict[str, str] = {
    "Entire home/apt": "Alojamiento entero",
    "Private room": "Habitación privada",
    "Shared room": "Habitación compartida",
    "Hotel room": "Habitación de hotel",
}
ETIQUETA_OTRAS_TIPO: str = "Otras (compartida + hotel)"

# Patrones (regex) que se aplican a CADA elemento de la lista JSON de amenities, ya normalizado
# (minúsculas, espacios simples). Un anuncio tiene el flag si algún elemento coincide.
# Casos ambiguos resueltos (comprobados con cadenas reales del snapshot, ver tests):
# «HDTV» sí es TV; «Dishwasher» no es lavadora; «Pool table», «Pool toys» y «Pool view» no son
# piscina; «Pocket wifi» (router portátil) no es wifi del alojamiento; «Outdoor kitchen» no es cocina.
PATRONES_AMENITIES: dict[str, str] = {
    "tiene_wifi": r"^(?:fast )?wifi\b",
    "tiene_aire_acondicionado": r"\bair conditioning\b",
    "tiene_cocina": r"^kitchen(?:ette)?$",
    "tiene_calefaccion": r"\bheating\b",
    "tiene_lavadora": r"(?<!dish)\bwasher\b",
    "tiene_ascensor": r"\belevator\b",
    "tiene_terraza_balcon": r"\b(?:balcony|patio|terrace)\b",
    "tiene_parking_cualquiera": r"\b(?:parking|garage)\b",
    "tiene_piscina": r"\bpool\b(?! ?(?:table|toys|view))",
    "tiene_tv": r"\b(?:hd)?tv\b",
    "tiene_espacio_trabajo": r"\bworkspace\b",
    "admite_mascotas": r"^pets allowed$",
    "llegada_autonoma": r"^self check-in$",
}

# Cotas de prevalencia plausibles (% de anuncios) por patrón. Fuera de ellas el patrón
# probablemente ha dejado de funcionar (formato cambiado o falso positivo masivo): ErrorDatos.
# Solo se aplican con al menos UMBRAL_FILAS anuncios (con pocos datos no son informativas).
COTAS_PREVALENCIA_AMENITIES: dict[str, tuple[float, float]] = {
    "tiene_wifi": (50, 100), "tiene_aire_acondicionado": (20, 98), "tiene_cocina": (50, 100),
    "tiene_calefaccion": (30, 100), "tiene_lavadora": (30, 100), "tiene_ascensor": (10, 95),
    "tiene_terraza_balcon": (2, 70), "tiene_parking_cualquiera": (2, 80), "tiene_piscina": (0.1, 30),
    "tiene_tv": (30, 100), "tiene_espacio_trabajo": (10, 95), "admite_mascotas": (1, 60),
    "llegada_autonoma": (5, 90),
}

# Nombres legibles de los flags de amenities.
NOMBRES_AMENITIES: dict[str, str] = {
    "tiene_wifi": "Wifi", "tiene_aire_acondicionado": "Aire acondicionado", "tiene_cocina": "Cocina",
    "tiene_calefaccion": "Calefacción", "tiene_lavadora": "Lavadora", "tiene_ascensor": "Ascensor",
    "tiene_terraza_balcon": "Terraza/balcón", "tiene_parking_cualquiera": "Parking (cualquiera)",
    "tiene_piscina": "Piscina", "tiene_tv": "TV", "tiene_espacio_trabajo": "Espacio de trabajo",
    "admite_mascotas": "Admite mascotas", "llegada_autonoma": "Llegada autónoma",
}

# Lista blanca de variables del modelo ML (M5). ``mes_cotizacion`` controla la temporada de la
# cotización (mes de entrada); no deriva del precio sino de la primera fecha disponible.
FEATURES_ML: tuple[str, ...] = (
    "accommodates", "bedrooms", "beds", "bathrooms", "bano_compartido", "n_amenities",
    *PATRONES_AMENITIES.keys(),
    "distancia_centro_km", "tipo_habitacion", "neighbourhood_group_cleansed",
    "host_is_superhost", "es_multianuncio", "licencia_tipo", "number_of_reviews",
    "review_scores_rating", "review_scores_location", "review_scores_cleanliness",
    "minimum_nights", "availability_365", "antiguedad_host_anios", "mes_cotizacion",
)
FEATURES_ML_CATEGORICAS: tuple[str, ...] = (
    "tipo_habitacion", "neighbourhood_group_cleansed", "licencia_tipo", "mes_cotizacion")
# Variables con imputación aprendida dentro del Pipeline (mediana por tipo y capacidad del train).
FEATURES_ML_IMPUTADAS: tuple[str, ...] = ("bedrooms", "beds")

# Lista negra: segunda barrera contra la fuga del objetivo (precio).
COLUMNAS_PROHIBIDAS_FUGA: frozenset[str] = frozenset({
    "price", "log_precio", "precio_por_persona", "ingreso_hipotetico_anual",
    "estimated_revenue_l365d", "categoria_precio", "precio_valido",
    "es_outlier_precio", "es_outlier_alto", "es_outlier_bajo",
})
PREFIJOS_PROHIBIDOS_FUGA: tuple[str, ...] = ("price_", "precio_", "ingreso_", "ingresos_", "es_outlier")

# Columnas que lee el dashboard (solo las que usa de verdad) y las de la tabla.
COLUMNAS_DASHBOARD: tuple[str, ...] = (
    "id", "name", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "tipo_habitacion", "price", "precio_valido", "accommodates", "bedrooms",
    "latitude", "longitude", "coords_validas", "distancia_centro_km", "host_is_superhost",
    "es_multianuncio", "segmento_host", "no_disponible_365", "es_activo_probable",
    "number_of_reviews", "review_scores_rating", "minimum_nights", "es_tarifa_mensual",
    *PATRONES_AMENITIES.keys(),
)
COLUMNAS_TABLA: tuple[str, ...] = (
    "id", "name", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
    "tipo_habitacion", "price", "accommodates", "bedrooms", "distancia_centro_km",
    "host_is_superhost", "number_of_reviews", "review_scores_rating", "minimum_nights",
)
# Etiquetas legibles en español (tabla y CSV del dashboard, figura 13).
ETIQUETAS_COLUMNAS: dict[str, str] = {
    "id": "ID del anuncio", "name": "Nombre del anuncio", "neighbourhood_group_cleansed": "Distrito",
    "neighbourhood_cleansed": "Barrio", "tipo_habitacion": "Tipo de habitación", "price": "Precio por noche (€)",
    "accommodates": "Capacidad (personas)", "bedrooms": "Dormitorios", "beds": "Camas", "bathrooms": "Baños",
    "bano_compartido": "Baño compartido", "n_amenities": "Nº de amenities",
    "distancia_centro_km": "Distancia al centro (km)", "host_is_superhost": "Superhost",
    "es_multianuncio": "Anfitrión con varios anuncios", "licencia_tipo": "Tipo de licencia",
    "number_of_reviews": "Nº de reseñas", "review_scores_rating": "Valoración (1-5)",
    "review_scores_location": "Valoración de la ubicación", "review_scores_cleanliness": "Valoración de la limpieza",
    "minimum_nights": "Estancia mínima (noches)", "availability_365": "Días disponibles (próx. 365)",
    "antiguedad_host_anios": "Antigüedad del anfitrión (años)", "mes_cotizacion": "Mes de la cotización",
    **NOMBRES_AMENITIES,
}
MAX_PUNTOS_MAPA: int = 10_000
MIN_N_COMPARADOR: int = 30   # por debajo, el comparador del dashboard avisa de muestra pequeña
PREFIJO_GRAFICOS_DASHBOARD: str = "dashboard_"
INDICE_GRAFICOS_DASHBOARD: str = "dashboard_graficos.json"
