# Diccionario de datos: `data/processed/airbnb_espana_ine.parquet`

Generado automáticamente por `src/diccionario.py` al ejecutar `python run_pipeline.py`.
Columnas: 74 (36 de Derivada, 14 de INE (ADRH), 21 de Inside Airbnb, 3 de Unión (INE)).

Fuente: *Inside Airbnb* (fuente 1), *INE (ADRH)* (fuente 2, indicadores de la zona), *Unión (INE)* (claves asignadas por punto en polígono con el seccionado censal del INE) o *Derivada* (creada en este proyecto). Origen: *original* (sin cambios), *convertida* (tipo o texto corregido) o *derivada*.

| Columna | Fuente | Tipo lógico | dtype | Origen | % nulos | Descripción |
|---|---|---|---|---|---|---|
| `ciudad` | Inside Airbnb | categorico | `str` | derivada | 0,0 % | Clave de la ciudad o región de Inside Airbnb (9 en España) |
| `id` | Inside Airbnb | identificador | `int64` | original | 0,0 % | Identificador único del anuncio |
| `fecha_snapshot` | Inside Airbnb | fecha | `str` | derivada | 0,0 % | Fecha del snapshot de Inside Airbnb (de la URL de descarga) |
| `latitude` | Inside Airbnb | geoespacial | `float64` | original | 0,0 % | Latitud (anonimizada por Airbnb, error de unos cientos de metros) |
| `longitude` | Inside Airbnb | geoespacial | `float64` | original | 0,0 % | Longitud (anonimizada por Airbnb) |
| `neighbourhood_cleansed` | Inside Airbnb | categorico | `string` | convertida | 0,0 % | Barrio (ciudades) o municipio (regiones), con la doble codificación de Euskadi reparada |
| `neighbourhood_group_cleansed` | Inside Airbnb | categorico | `string` | convertida | 44,3 % | Distrito (Madrid, Barcelona, Valencia, Sevilla) o provincia (Euskadi); vacío en el resto |
| `coords_validas` | Derivada | booleano | `bool` | derivada | 0,0 % | Coordenadas dentro de la envolvente (bbox) de su ciudad |
| `distancia_centro_km` | Derivada | numerico | `float64` | derivada | 41,3 % | Distancia haversine al centro de su ciudad (km); solo ciudades urbanas, NaN en las regiones |
| `codigo_distrito_ine` | Unión (INE) | identificador | `str` | derivada | 0,3 % | Distrito censal del INE (7 dígitos) asignado por punto en polígono |
| `codigo_municipio_ine` | Unión (INE) | identificador | `str` | derivada | 0,3 % | Municipio del INE (5 dígitos) asignado por punto en polígono |
| `municipio_ine` | Unión (INE) | categorico | `str` | derivada | 0,3 % | Nombre del municipio según el INE |
| `ine_dis_renta_neta_persona` | INE (ADRH) | numerico | `float64` | original | 0,4 % | Renta neta media por persona (€/año) del distrito censal (ADRH, año de referencia) |
| `ine_dis_renta_mediana_uc` | INE (ADRH) | numerico | `float64` | original | 0,4 % | Mediana de la renta por unidad de consumo (€/año) del distrito censal (ADRH, año de referencia) |
| `ine_dis_gini` | INE (ADRH) | numerico | `float64` | original | 0,3 % | Índice de Gini de la renta (0-100) del distrito censal (ADRH, año de referencia) |
| `ine_dis_edad_media` | INE (ADRH) | numerico | `float64` | original | 0,3 % | Edad media de la población (años) del distrito censal (ADRH, año de referencia) |
| `ine_dis_pct_hogares_unipersonales` | INE (ADRH) | numerico | `float64` | original | 0,3 % | % de hogares unipersonales del distrito censal (ADRH, año de referencia) |
| `ine_dis_poblacion` | INE (ADRH) | numerico | `float64` | original | 0,3 % | Población (personas) del distrito censal (ADRH, año de referencia) |
| `ine_dis_pct_bajo_60_mediana` | INE (ADRH) | numerico | `float64` | original | 0,4 % | % de población con ingresos por unidad de consumo por debajo del 60 % de la mediana del distrito censal (ADRH, año de referencia) |
| `ine_mun_renta_neta_persona` | INE (ADRH) | numerico | `float64` | original | 0,4 % | Renta neta media por persona (€/año) del municipio (ADRH, año de referencia) |
| `ine_mun_renta_mediana_uc` | INE (ADRH) | numerico | `float64` | original | 0,4 % | Mediana de la renta por unidad de consumo (€/año) del municipio (ADRH, año de referencia) |
| `ine_mun_gini` | INE (ADRH) | numerico | `float64` | original | 0,3 % | Índice de Gini de la renta (0-100) del municipio (ADRH, año de referencia) |
| `ine_mun_edad_media` | INE (ADRH) | numerico | `float64` | original | 0,3 % | Edad media de la población (años) del municipio (ADRH, año de referencia) |
| `ine_mun_pct_hogares_unipersonales` | INE (ADRH) | numerico | `float64` | original | 0,3 % | % de hogares unipersonales del municipio (ADRH, año de referencia) |
| `ine_mun_poblacion` | INE (ADRH) | numerico | `float64` | original | 0,3 % | Población (personas) del municipio (ADRH, año de referencia) |
| `ine_mun_pct_bajo_60_mediana` | INE (ADRH) | numerico | `float64` | original | 0,4 % | % de población con ingresos por unidad de consumo por debajo del 60 % de la mediana del municipio (ADRH, año de referencia) |
| `tipo_habitacion` | Inside Airbnb | categorico | `string` | convertida | 0,0 % | Tipo de habitación en español |
| `tipo_habitacion_agrupado` | Derivada | categorico | `string` | derivada | 0,0 % | Tipo de habitación con categorías raras (< 200 con precio) agrupadas |
| `accommodates` | Inside Airbnb | numerico | `int64` | original | 0,0 % | Capacidad máxima de huéspedes |
| `categoria_capacidad` | Derivada | categorico | `category` | derivada | 0,0 % | Tramo de capacidad (1-2, 3-4, 5-6, 7 o más) |
| `bedrooms` | Inside Airbnb | numerico | `float64` | convertida | 0,0 % | Nº de dormitorios (imputado si faltaba, ver bedrooms_imputado) |
| `beds` | Inside Airbnb | numerico | `float64` | convertida | 0,0 % | Nº de camas (imputado si faltaba, ver beds_imputado) |
| `bathrooms` | Inside Airbnb | numerico | `float64` | derivada | 0,1 % | Nº de baños reconstruido desde bathrooms_text |
| `bano_compartido` | Derivada | booleano | `bool` | derivada | 0,0 % | El baño es compartido (desde bathrooms_text) |
| `bedrooms_imputado` | Derivada | booleano | `bool` | derivada | 0,0 % | bedrooms se imputó (mediana por tipo y capacidad) |
| `beds_imputado` | Derivada | booleano | `bool` | derivada | 0,0 % | beds se imputó (mediana por tipo y capacidad) |
| `n_amenities` | Derivada | numerico | `int64` | derivada | 0,0 % | Nº de amenities del anuncio |
| `tiene_wifi` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: wifi |
| `tiene_aire_acondicionado` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: aire acondicionado |
| `tiene_cocina` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: cocina |
| `tiene_calefaccion` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: calefacción |
| `tiene_lavadora` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: lavadora |
| `tiene_ascensor` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: ascensor |
| `tiene_terraza_balcon` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: terraza, patio o balcón |
| `tiene_parking_cualquiera` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: cualquier tipo de parking (gratuito o de pago) |
| `tiene_piscina` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: piscina |
| `tiene_tv` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: televisión |
| `tiene_espacio_trabajo` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: espacio de trabajo |
| `admite_mascotas` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: admite mascotas |
| `llegada_autonoma` | Derivada | booleano | `bool` | derivada | 0,0 % | Amenity: llegada autónoma (self check-in) |
| `price` | Inside Airbnb | numerico | `float64` | convertida | 18,1 % | Precio por noche en euros de una cotización fechada (descuentos aplicados, sin tasas) |
| `precio_valido` | Derivada | booleano | `bool` | derivada | 0,0 % | Precio publicado y mayor que 0 |
| `log_precio` | Derivada | numerico | `float64` | derivada | 18,1 % | Logaritmo natural del precio |
| `es_outlier_precio` | Derivada | booleano | `bool` | derivada | 0,0 % | Precio fuera de las vallas de Tukey sobre ln(precio) del conjunto |
| `es_tarifa_mensual` | Derivada | booleano | `bool` | derivada | 0,0 % | La cotización usa tarifa mensual (is_monthly_priced en price_quote_raw) |
| `minimum_nights` | Inside Airbnb | numerico | `float64` | convertida | 0,0 % | Estancia mínima en noches (si faltaba, minimum_nights_avg_ntm; ver minimum_nights_imputado) |
| `minimum_nights_imputado` | Derivada | booleano | `bool` | derivada | 0,0 % | minimum_nights faltaba y se rellenó con minimum_nights_avg_ntm (media de los próximos 30 días) |
| `alquiler_larga_estancia` | Derivada | booleano | `bool` | derivada | 0,0 % | Estancia mínima de 30 noches o más |
| `mes_cotizacion` | Derivada | categorico | `string` | derivada | 11,2 % | Mes (AAAA-MM) de la fecha de entrada de la cotización de precio |
| `mes_cotizacion_agrupado` | Derivada | categorico | `string` | derivada | 11,2 % | Mes de la cotización con meses raros (< 200 con precio) agrupados |
| `host_id` | Inside Airbnb | identificador | `int64` | original | 0,0 % | Identificador numérico público del anfitrión en Airbnb; solo para agrupar anuncios por anfitrión (no se cruza con otras fuentes) |
| `host_is_superhost` | Inside Airbnb | booleano | `bool` | convertida | 0,0 % | El anfitrión es superhost |
| `calculated_host_listings_count` | Inside Airbnb | numerico | `int64` | original | 0,0 % | Anuncios del anfitrión en el snapshot de su ciudad |
| `es_multianuncio` | Derivada | booleano | `bool` | derivada | 0,0 % | El anfitrión tiene más de un anuncio en su ciudad |
| `segmento_host` | Derivada | categorico | `category` | derivada | 0,0 % | Segmento del anfitrión por nº de anuncios (1, 2-5, 6 o más) |
| `antiguedad_host_anios` | Derivada | numerico | `float64` | derivada | 0,0 % | Antigüedad como anfitrión en años (hosts_time_as_host_*) |
| `licencia_tipo` | Derivada | categorico | `string` | derivada | 0,0 % | Sin licencia / Exenta / Con nº de registro / Otro o en proceso |
| `licencia_registro_nacional` | Derivada | booleano | `bool` | derivada | 0,0 % | El texto de licencia incluye el número del registro único nacional |
| `availability_365` | Inside Airbnb | numerico | `int64` | original | 0,0 % | Días disponibles en los próximos 365 |
| `no_disponible_365` | Derivada | numerico | `float64` | derivada | 0,0 % | Proxy de ocupación: % de días NO disponibles en los próximos 365, (365 − availability_365) / 365; mezcla reservas y bloqueos |
| `es_activo_probable` | Derivada | booleano | `bool` | derivada | 0,0 % | No (availability_365 = 0 y sin reseñas en 12 meses) |
| `number_of_reviews` | Inside Airbnb | numerico | `int64` | original | 0,0 % | Nº total de reseñas |
| `review_scores_rating` | Inside Airbnb | numerico | `float64` | original | 14,5 % | Puntuación global (1-5) |
| `estimated_occupancy_l365d` | Inside Airbnb | numerico | `int64` | original | 0,0 % | Noches ocupadas estimadas por Inside Airbnb (últimos 12 meses) |
