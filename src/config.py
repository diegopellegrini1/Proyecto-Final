"""Configuración central: rutas, constantes, ciudades de España y listas de columnas.

Todas las constantes del proyecto están nombradas aquí; ninguna función usa
valores mágicos. Las rutas son relativas a la raíz del repositorio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ: Path = Path(__file__).resolve().parents[1]
DIR_DATA: Path = RAIZ / "data"
DIR_RAW: Path = DIR_DATA / "raw"
DIR_RAW_AIRBNB: Path = DIR_RAW / "inside_airbnb"
DIR_RAW_INE: Path = DIR_RAW / "ine"
DIR_INTERIM: Path = DIR_DATA / "interim"
DIR_PROCESADO: Path = DIR_DATA / "processed"
DIR_EXTRA: Path = DIR_DATA / "extra"
DIR_EXTRA_INE: Path = DIR_EXTRA / "ine"          # cartografía pesada (no se versiona)
DIR_REPORTS: Path = RAIZ / "reports"
DIR_FIGURAS: Path = DIR_REPORTS / "figures"
DIR_PLANTILLAS: Path = RAIZ / "src" / "plantillas"
DIR_NOTEBOOKS: Path = RAIZ / "notebooks"

RUTA_ASIGNACION_SECCIONES: Path = DIR_INTERIM / "asignacion_secciones.csv.gz"
RUTA_PARQUET: Path = DIR_PROCESADO / "airbnb_espana_ine.parquet"
RUTA_MUESTRA_CSV: Path = DIR_PROCESADO / "airbnb_espana_ine_muestra.csv"
RUTA_FINDINGS: Path = DIR_REPORTS / "findings.json"
RUTA_CALIDAD: Path = DIR_REPORTS / "calidad_datos.json"
RUTA_INFORME: Path = DIR_REPORTS / "informe.md"
RUTA_DICCIONARIO: Path = DIR_REPORTS / "diccionario_datos.md"
RUTA_README: Path = RAIZ / "README.md"

# Dashboard operativo en Excel (se genera con src/dashboard_excel.py desde el Parquet y findings.json).
DIR_DASHBOARD: Path = RAIZ / "dashboard"
RUTA_DASHBOARD_XLSX: Path = DIR_DASHBOARD / "Dashboard_Airbnb_Espana.xlsx"
RUTA_DASHBOARD_README: Path = DIR_DASHBOARD / "README_dashboard.md"
NOMBRE_VISTA_PREVIA_DASHBOARD: str = "dashboard_vista_previa.png"
RUTA_VISTA_PREVIA_DASHBOARD: Path = DIR_FIGURAS / NOMBRE_VISTA_PREVIA_DASHBOARD
# Tramos de precio del dashboard: límites inferiores cerrados [a, b) en euros; el último tramo es abierto.
LIMITES_TRAMOS_PRECIO_EXCEL: tuple[int, ...] = (50, 100, 150, 200, 300, 500)
# Tramos de capacidad del dashboard: (etiqueta, mínimo, máximo); None = sin tope. Etiquetas con texto para que
# Excel no las convierta en números o fechas al elegirlas en un desplegable.
TRAMOS_CAPACIDAD_EXCEL: tuple[tuple[str, int, int | None], ...] = (
    ("1 persona", 1, 1), ("2 personas", 2, 2), ("3-4 personas", 3, 4), ("5-6 personas", 5, 6),
    ("7 o más personas", 7, None),
)

# --------------------------------------------------------------------------
# Parámetros de análisis
# --------------------------------------------------------------------------
SEMILLA: int = 42
ALFA: float = 0.05
N_BOOT: int = 2000
N_BOOT_MIN: int = 100
N_BOOT_MAX: int = 20_000
N_BOOT_CORRELACION: int = 1000   # tope de réplicas del bootstrap por clusters de la ρ de Spearman (coste)
# ρ global renta-precio: su p queda cerca de α, así que se recalcula con más réplicas (con N_BOOT por defecto,
# 10.000) y un generador propio (semilla, SUBSEMILLA_RHO_GLOBAL) que no altera el resto del análisis.
N_BOOT_RHO_GLOBAL: int = 10_000
FACTOR_N_BOOT_RHO_GLOBAL: int = 5
SUBSEMILLA_RHO_GLOBAL: int = 1
# Lectura prudente de un p bootstrap: «en el límite» si p está en esta banda o el IC 95 % roza el 0 (su extremo
# más cercano a 0 está a menos de MARGEN_IC_LIMITE), sin depender de que p caiga a un lado u otro de α.
BANDA_P_LIMITE: tuple[float, float] = (0.03, 0.07)
MARGEN_IC_LIMITE: float = 0.01
# Año del ADRH: el más reciente en el que cada indicador tiene dato en al menos esta fracción de distritos.
COBERTURA_MIN_ADRH: float = 0.5
MIN_DISTRITOS_ECOLOGICO: int = 5 # mínimo de distritos con n ≥ MIN_N_GRUPO para la correlación por distrito
N_SHAPIRO: int = 5000
MIN_N_GRUPO: int = 30            # mínimo de anuncios con precio para tests y rankings
MIN_N_CATEGORIA_RARA: int = 200  # por debajo (en filas con precio), la categoría se agrupa
MIN_N_ZONA_RANKING: int = 100    # mínimo de anuncios con precio para que una zona entre en un ranking
TOP_N_ZONAS: int = 10
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
# Requisitos del enunciado sobre el conjunto final tras la unión.
MIN_FILAS_UNION: int = 50_000
MIN_COLUMNAS_UNION: int = 20
LIMITE_PARQUET_BYTES: int = 25_000_000
LOTE_SUBIDA_ARCHIVOS: int = 100                # máximo de archivos por lote al subir a GitHub arrastrando
LOTE_SUBIDA_BYTES: int = 40_000_000            # peso orientativo máximo por lote (la subida web es lenta con más)

# --------------------------------------------------------------------------
# Descarga: fuente 1 (Inside Airbnb)
# --------------------------------------------------------------------------
URL_PAGINA_DATOS: str = "https://insideairbnb.com/get-the-data/"
DOMINIOS_PERMITIDOS: frozenset[str] = frozenset(
    {"insideairbnb.com", "www.insideairbnb.com", "data.insideairbnb.com"}
)
USER_AGENT: str = "Proyecto-Final-Airbnb-Espana (+https://github.com/diegopellegrini1; uso academico)"
TIMEOUT_CONEXION_S: int = 15
TIMEOUT_LECTURA_S: int = 120
REINTENTOS: int = 3
MAX_REDIRECCIONES: int = 3

# Ficheros obligatorios por ciudad y la subcarpeta de la URL en la que se publican.
FICHEROS_OBLIGATORIOS: dict[str, str] = {
    "listings.csv.gz": "data",
    "neighbourhoods.csv": "visualisations",
    "neighbourhoods.geojson": "visualisations",
}

# --------------------------------------------------------------------------
# Ciudades y regiones de España (Inside Airbnb) con su geografía
# --------------------------------------------------------------------------
TIPO_URBANA = "urbana"
TIPO_REGION = "region"


@dataclass(frozen=True)
class CiudadEspana:
    """Ciudad o región de Inside Airbnb en España, sus provincias INE y su geografía.

    ``bbox`` (lat_min, lat_max, lon_min, lon_max) es la envolvente del GeoJSON oficial de barrios de
    Inside Airbnb con un margen de 0,05 grados. ``centro`` solo existe en las ciudades urbanas, donde
    «distancia al centro» tiene sentido; en las regiones (varios municipios dispersos, costa e interior)
    no hay un centro que lo justifique y la distancia queda en NaN.
    """

    clave: str
    nombre: str
    slug_ruta: str          # ruta en data.insideairbnb.com, ya con la tilde correcta (sin codificar)
    provincias: tuple[str, ...]
    tipo: str
    bbox: tuple[float, float, float, float]
    centro: tuple[float, float] | None = None
    centro_nombre: str | None = None


@dataclass(frozen=True)
class ProvinciaIne:
    """Provincia del ADRH: código INE de 2 dígitos y parámetro ``capsel`` de su página en INEbase."""

    codigo: str
    nombre: str
    capsel: int


CIUDADES_ESPANA: dict[str, CiudadEspana] = {c.clave: c for c in (
    CiudadEspana("madrid", "Madrid", "spain/comunidad-de-madrid/madrid", ("28",), TIPO_URBANA,
                 (40.31, 40.65, -3.90, -3.50), (40.4169, -3.7035), "Puerta del Sol"),
    CiudadEspana("barcelona", "Barcelona", "spain/catalonia/barcelona", ("08",), TIPO_URBANA,
                 (41.26, 41.52, 2.00, 2.28), (41.3870, 2.1700), "Plaça de Catalunya"),
    CiudadEspana("valencia", "Valencia", "spain/vc/valencia", ("46",), TIPO_URBANA,
                 (39.23, 39.62, -0.48, -0.22), (39.4699, -0.3763), "Plaza del Ayuntamiento"),
    CiudadEspana("sevilla", "Sevilla", "spain/andalucía/sevilla", ("41",), TIPO_URBANA,
                 (37.25, 37.50, -6.08, -5.77), (37.3862, -5.9926), "Catedral de Sevilla"),
    CiudadEspana("malaga", "Málaga", "spain/andalucía/malaga", ("29",), TIPO_URBANA,
                 (36.58, 36.95, -4.64, -4.21), (36.7213, -4.4214), "Plaza de la Constitución"),
    CiudadEspana("girona", "Girona", "spain/catalonia/girona", ("17", "08"), TIPO_REGION,
                 (41.60, 42.55, 1.67, 3.38)),
    CiudadEspana("mallorca", "Mallorca", "spain/islas-baleares/mallorca", ("07",), TIPO_REGION,
                 (39.21, 40.01, 2.25, 3.53)),
    CiudadEspana("menorca", "Menorca", "spain/islas-baleares/menorca", ("07",), TIPO_REGION,
                 (39.75, 40.15, 3.74, 4.38)),
    CiudadEspana("euskadi", "Euskadi", "spain/pv/euskadi", ("01", "20", "48"), TIPO_REGION,
                 (42.42, 43.51, -3.50, -1.67)),
)}
CIUDAD_POR_NOMBRE: dict[str, CiudadEspana] = {c.nombre: c for c in CIUDADES_ESPANA.values()}
NOMBRES_CIUDADES: tuple[str, ...] = tuple(sorted(CIUDAD_POR_NOMBRE))
CIUDADES_URBANAS: tuple[str, ...] = tuple(sorted(c.nombre for c in CIUDADES_ESPANA.values()
                                                 if c.tipo == TIPO_URBANA))
CIUDAD_REFERENCIA: str = "Madrid"   # categoría de referencia de los efectos fijos de ciudad

# --------------------------------------------------------------------------
# Descarga: fuente 2 (INE, Atlas de Distribución de Renta de los Hogares)
# --------------------------------------------------------------------------
URL_PAGINA_ADRH: str = "https://www.ine.es/dynt3/inebase/es/index.htm?padre=7132"
URL_TABLA_INE_CSV: str = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/{tabla}.csv"
URL_SECCIONADO_INE: str = "https://www.ine.es/prodyser/cartografia/seccionado_2024.zip"
URL_AVISO_LEGAL_INE: str = "https://www.ine.es/aviso_legal"
DOMINIOS_PERMITIDOS_INE: frozenset[str] = frozenset({"www.ine.es", "ine.es"})
CRS_SECCIONADO: str = "EPSG:25830"   # ETRS89 / UTM 30N (según el .prj del seccionado)
CRS_LISTINGS: str = "EPSG:4326"      # WGS84 (lat/lon de Inside Airbnb)
CITA_INE: str = "Elaboración propia con datos extraídos del sitio web del INE: www.ine.es"
# Fecha de la última actualización del ADRH (publicación de los datos de 2024), tal como figura en la
# página de resultados de la operación en INEbase («Publicado: 19/10/2026»), consultada el 2026-10-27.
ACTUALIZACION_ADRH: dict[str, str] = {
    "fecha": "2026-10-19",
    "url": ("https://ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736177088"
            "&idp=1254735976608&menu=resultados"),
}

# Tablas del ADRH que se descargan por provincia, identificadas por su título exacto en INEbase
# (los identificadores numéricos cambian entre provincias; se descubren en la página del ADRH).
TABLAS_ADRH: dict[str, str] = {
    "renta": "Indicadores de renta media y mediana",
    "gini": "Índice de Gini y Distribución de la renta P80/P20",
    "demografia": "Indicadores demográficos",
    "umbrales": ("Porcentaje de población con ingresos por unidad de consumo por debajo/encima de "
                 "determinados umbrales relativos por sexo"),
}

# Indicadores del ADRH que pasan al conjunto unido (texto exacto del INE -> nombre de columna).
INDICADORES_ADRH: dict[str, str] = {
    "Renta neta media por persona": "renta_neta_persona",
    "Renta neta media por hogar": "renta_neta_hogar",
    "Media de la renta por unidad de consumo": "renta_media_uc",
    "Mediana de la renta por unidad de consumo": "renta_mediana_uc",
    "Renta bruta media por persona": "renta_bruta_persona",
    "Renta bruta media por hogar": "renta_bruta_hogar",
    "Índice de Gini": "gini",
    "Distribución de la renta P80/P20": "p80_p20",
    "Edad media de la población": "edad_media",
    "Porcentaje de población menor de 18 años": "pct_menores_18",
    "Porcentaje de población de 65 y más años": "pct_65_mas",
    "Tamaño medio del hogar": "tamano_medio_hogar",
    "Porcentaje de hogares unipersonales": "pct_hogares_unipersonales",
    "Población": "poblacion",
    "Porcentaje de población española": "pct_poblacion_espanola",
    "Población con ingresos por unidad de consumo por debajo 60% de la mediana": "pct_bajo_60_mediana",
}
# Indicadores que se conservan en el conjunto final procesado (por distrito y por municipio).
INDICADORES_FINALES: tuple[str, ...] = (
    "renta_neta_persona", "renta_mediana_uc", "gini", "edad_media", "pct_hogares_unipersonales",
    "poblacion", "pct_bajo_60_mediana",
)

NOMBRES_INDICADORES: dict[str, str] = {
    "renta_neta_persona": "Renta neta media por persona (€)",
    "renta_mediana_uc": "Mediana de la renta por unidad de consumo (€)",
    "gini": "Índice de Gini",
    "edad_media": "Edad media (años)",
    "pct_hogares_unipersonales": "% hogares unipersonales",
    "poblacion": "Población",
    "pct_bajo_60_mediana": "% población bajo el 60 % de la mediana",
}
# Indicador de renta principal del análisis renta-precio (y su versión municipal).
RENTA_PRINCIPAL: str = "ine_dis_renta_neta_persona"
RENTA_MUNICIPAL: str = "ine_mun_renta_neta_persona"

PROVINCIAS_ADRH: dict[str, ProvinciaIne] = {p.codigo: p for p in (
    ProvinciaIne("01", "Araba/Álava", 5654),
    ProvinciaIne("07", "Balears, Illes", 5658),
    ProvinciaIne("08", "Barcelona", 5659),
    ProvinciaIne("17", "Girona", 5681),
    ProvinciaIne("20", "Gipuzkoa", 5680),
    ProvinciaIne("28", "Madrid", 5690),
    ProvinciaIne("29", "Málaga", 5691),
    ProvinciaIne("41", "Sevilla", 5702),
    ProvinciaIne("46", "Valencia/València", 5707),
    ProvinciaIne("48", "Bizkaia", 5670),
)}

# Validación de la unión (no interviene en ella: la unión es por punto en polígono).
# 1) Ciudades cuyos distritos municipales coinciden con los distritos censales del INE: distrito de
#    Inside Airbnb (neighbourhood_group_cleansed) -> código CUDIS según la numeración oficial municipal.
#    Málaga no figura: sus distritos censales no coinciden con los municipales.
DISTRITOS_OFICIALES: dict[str, dict[str, str]] = {
    "madrid": {n: f"28079{i:02d}" for i, n in enumerate((
        "Centro", "Arganzuela", "Retiro", "Salamanca", "Chamartín", "Tetuán", "Chamberí",
        "Fuencarral - El Pardo", "Moncloa - Aravaca", "Latina", "Carabanchel", "Usera", "Puente de Vallecas",
        "Moratalaz", "Ciudad Lineal", "Hortaleza", "Villaverde", "Villa de Vallecas", "Vicálvaro",
        "San Blas - Canillejas", "Barajas"), start=1)},
    "barcelona": {n: f"08019{i:02d}" for i, n in enumerate((
        "Ciutat Vella", "Eixample", "Sants-Montjuïc", "Les Corts", "Sarrià-Sant Gervasi", "Gràcia",
        "Horta-Guinardó", "Nou Barris", "Sant Andreu", "Sant Martí"), start=1)},
    "valencia": {n: f"46250{i:02d}" for i, n in enumerate((
        "CIUTAT VELLA", "L'EIXAMPLE", "EXTRAMURS", "CAMPANAR", "LA SAIDIA", "EL PLA DEL REAL", "L'OLIVERETA",
        "PATRAIX", "JESUS", "QUATRE CARRERES", "POBLATS MARITIMS", "CAMINS AL GRAU", "ALGIROS", "BENIMACLET",
        "RASCANYA", "BENICALAP", "POBLATS DEL NORD", "POBLATS DE L'OEST", "POBLATS DEL SUD"), start=1)},
    "sevilla": {n: f"41091{i:02d}" for i, n in enumerate((
        "Casco Antiguo", "Macarena", "Nervión", "Cerro - Amate", "Sur", "Triana", "Macarena - Norte",
        "San Pablo - Santa Justa", "Este - Alcosa - Torreblanca", "Palmera - Bellavista", "Los Remedios"),
        start=1)},
}
# 2) Regiones donde neighbourhood_cleansed es el municipio: nombre de Inside Airbnb -> nombre del INE cuando
#    difieren más allá de tildes, mayúsculas, signos o el artículo pospuesto (revisado uno a uno).
ALIAS_MUNICIPIOS_INE: dict[str, str] = {
    "Palma de Mallorca": "Palma",
    "Deyá": "Deià",
    "Vilafranc de Bonany": "Vilafranca de Bonany",
    "Mahón": "Maó",
    "Castell-Platja d'Aro": "Castell d'Aro, Platja d'Aro i s'Agaró",
    "Calonge": "Calonge i Sant Antoni",
    "Saus": "Saus, Camallera i Llampaies",
    "Boadella d'Empordà": "Boadella i les Escaules",
    "Brunyola": "Brunyola i Sant Martí Sapresa",
    "Masarac": "Masarac i Vilarnadal",
    "Sopelana": "Sopela",
    "Mungai": "Mungia",
    "Legutiano": "Legutio",
    "Arrankudiaga": "Arrankudiaga-Zollo",
    "Bidegoian": "Bidania-Goiatz",
    "Oyón": "Oyón-Oion",
    "Etxebarri, Anteiglesia de San Esteban-Etxebarri Doneztebeko Elizatea": "Etxebarri",
    "Garay": "Garai",
    "Arrazua-Ubarrundia": "Arratzua-Ubarrundia",
}
CIUDADES_VALIDACION_MUNICIPIO: tuple[str, ...] = ("girona", "mallorca", "menorca", "euskadi", "malaga")

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
# Aire acondicionado: «AC - split type ductless system» y «Window AC unit» también lo son, pero su gemela
# «Heating - split type ductless system» es calefacción (se excluye todo lo que empieza por «heating»).
# Calefacción: «Portable heater» y «Heat pump» cuentan; «fast-heating» dentro de la descripción de un horno, no.
PATRONES_AMENITIES: dict[str, str] = {
    "tiene_wifi": r"^(?:fast )?wifi\b",
    "tiene_aire_acondicionado": r"^(?!heating\b).*\b(?:air ?conditioning|airconditioner|ac|aire acondicionado)\b",
    "tiene_cocina": r"^kitchen(?:ette)?$",
    "tiene_calefaccion": r"^(?:central |radiant )?heating\b|^portable heaters?$|^heat pump$",
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
    "tiene_terraza_balcon": (2, 70), "tiene_parking_cualquiera": (2, 80), "tiene_piscina": (0.1, 40),
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

# Columnas del conjunto final procesado (Parquet): solo las que usan el análisis, los notebooks y el
# diccionario. El conjunto unido completo tiene 129 columnas y 73 MB en Parquet; este, unas 70 y < 25 MB.
COLUMNAS_INE_FINALES: tuple[str, ...] = tuple(
    f"{nivel}{ind}" for nivel in ("ine_dis_", "ine_mun_") for ind in INDICADORES_FINALES)
COLUMNAS_FINALES: tuple[str, ...] = (
    # Identificación y geografía
    "ciudad", "id", "fecha_snapshot", "latitude", "longitude", "neighbourhood_cleansed",
    "neighbourhood_group_cleansed", "coords_validas", "distancia_centro_km",
    # Fuente 2 (INE): claves de la unión e indicadores por distrito censal y por municipio
    "codigo_distrito_ine", "codigo_municipio_ine", "municipio_ine", *COLUMNAS_INE_FINALES,
    # Alojamiento
    "tipo_habitacion", "tipo_habitacion_agrupado", "accommodates", "categoria_capacidad", "bedrooms", "beds",
    "bathrooms", "bano_compartido", "bedrooms_imputado", "beds_imputado", "n_amenities",
    *PATRONES_AMENITIES.keys(),
    # Precio (cotización fechada), estancias y outliers
    "price", "precio_valido", "log_precio", "es_outlier_precio", "es_tarifa_mensual", "minimum_nights",
    "minimum_nights_imputado", "alquiler_larga_estancia", "mes_cotizacion", "mes_cotizacion_agrupado",
    # Anfitrión y licencia
    "host_id", "host_is_superhost", "calculated_host_listings_count", "es_multianuncio", "segmento_host",
    "antiguedad_host_anios", "licencia_tipo", "licencia_registro_nacional",
    # Demanda y reseñas
    "availability_365", "no_disponible_365", "es_activo_probable", "number_of_reviews",
    "review_scores_rating", "estimated_occupancy_l365d",
)

# Lista blanca de variables del modelo ML. ``mes_cotizacion`` controla la temporada de la cotización
# (mes de entrada). ``distancia_centro_km`` es NaN en las regiones (HistGradientBoosting admite NaN).
VARIABLES_INE_ML: tuple[str, ...] = (
    "ine_dis_renta_neta_persona", "ine_dis_gini", "ine_dis_edad_media", "ine_dis_pct_hogares_unipersonales",
    "ine_dis_poblacion", "ine_dis_pct_bajo_60_mediana",
)
FEATURES_ML: tuple[str, ...] = (
    "accommodates", "bedrooms", "beds", "bathrooms", "bano_compartido", "n_amenities",
    *PATRONES_AMENITIES.keys(),
    "distancia_centro_km", "tipo_habitacion", "ciudad",
    "host_is_superhost", "es_multianuncio", "licencia_tipo", "number_of_reviews",
    "review_scores_rating", "minimum_nights", "availability_365", "antiguedad_host_anios", "mes_cotizacion",
    *VARIABLES_INE_ML,
)
FEATURES_ML_CATEGORICAS: tuple[str, ...] = ("tipo_habitacion", "ciudad", "licencia_tipo", "mes_cotizacion")
# Variables con imputación aprendida dentro del Pipeline (mediana por tipo y capacidad del train).
FEATURES_ML_IMPUTADAS: tuple[str, ...] = ("bedrooms", "beds")

# Lista negra: segunda barrera contra la fuga del objetivo (precio).
COLUMNAS_PROHIBIDAS_FUGA: frozenset[str] = frozenset({
    "price", "log_precio", "precio_por_persona", "ingreso_hipotetico_anual",
    "estimated_revenue_l365d", "categoria_precio", "precio_valido",
    "es_outlier_precio", "es_outlier_alto", "es_outlier_bajo",
})
PREFIJOS_PROHIBIDOS_FUGA: tuple[str, ...] = ("price_", "precio_", "ingreso_", "ingresos_", "es_outlier")

# Etiquetas legibles en español (figuras y tablas).
ETIQUETAS_COLUMNAS: dict[str, str] = {
    "id": "ID del anuncio", "ciudad": "Ciudad", "neighbourhood_cleansed": "Barrio o municipio",
    "tipo_habitacion": "Tipo de habitación", "price": "Precio por noche (€)",
    "accommodates": "Capacidad (personas)", "bedrooms": "Dormitorios", "beds": "Camas", "bathrooms": "Baños",
    "bano_compartido": "Baño compartido", "n_amenities": "Nº de amenities",
    "distancia_centro_km": "Distancia al centro (km)", "host_is_superhost": "Superhost",
    "es_multianuncio": "Anfitrión con varios anuncios", "licencia_tipo": "Tipo de licencia",
    "number_of_reviews": "Nº de reseñas", "review_scores_rating": "Valoración (1-5)",
    "minimum_nights": "Estancia mínima (noches)", "availability_365": "Días disponibles (próx. 365)",
    "antiguedad_host_anios": "Antigüedad del anfitrión (años)", "mes_cotizacion": "Mes de la cotización",
    "ine_dis_renta_neta_persona": "Renta neta por persona del distrito (€)",
    "ine_dis_gini": "Índice de Gini del distrito", "ine_dis_edad_media": "Edad media del distrito",
    "ine_dis_pct_hogares_unipersonales": "% hogares unipersonales del distrito",
    "ine_dis_poblacion": "Población del distrito", "ine_dis_pct_bajo_60_mediana": "% población bajo el 60 % de la mediana",
    **NOMBRES_AMENITIES,
}
