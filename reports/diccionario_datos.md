# Diccionario de datos: `data/processed/listings_limpio.parquet`

Generado automáticamente por `src/diccionario.py` al ejecutar `python run_pipeline.py`.
Origen: *original* (columna de Inside Airbnb sin cambios), *convertida* (tipo corregido) o *derivada* (creada en este proyecto).

| Columna | Tipo lógico | dtype | Origen | % nulos | Descripción |
|---|---|---|---|---|---|
| `id` | identificador | `int64` | original | 0,0 % | Identificador único del anuncio |
| `last_scraped` | fecha | `datetime64[us]` | convertida | 0,0 % | Fecha en que se scrapeó el anuncio |
| `name` | texto | `str` | original | 0,0 % | Título del anuncio |
| `host_id` | identificador | `int64` | original | 0,0 % | Identificador anonimizado del anfitrión |
| `hosts_time_as_user_years` | numerico | `int64` | original | 0,0 % | Años como usuario de Airbnb (parte entera) |
| `hosts_time_as_user_months` | numerico | `int64` | original | 0,0 % | Meses adicionales como usuario (resto, 0-11) |
| `hosts_time_as_host_years` | numerico | `int64` | original | 0,0 % | Años como anfitrión (parte entera) |
| `hosts_time_as_host_months` | numerico | `int64` | original | 0,0 % | Meses adicionales como anfitrión (resto, 0-11) |
| `host_location` | texto | `str` | original | 34,0 % | Ubicación declarada por el anfitrión |
| `host_is_superhost` | booleano | `bool` | convertida | 0,0 % | El anfitrión es superhost |
| `host_listings_count` | numerico | `int64` | original | 0,0 % | Nº de anuncios del anfitrión según Airbnb |
| `host_has_profile_pic` | booleano | `bool` | convertida | 0,0 % | El anfitrión tiene foto de perfil |
| `host_identity_verified` | booleano | `bool` | convertida | 0,0 % | Identidad del anfitrión verificada |
| `neighbourhood_cleansed` | categorico | `str` | original | 0,0 % | Barrio (128 barrios oficiales de Madrid) |
| `neighbourhood_group_cleansed` | categorico | `str` | original | 0,0 % | Distrito (21 distritos de Madrid) |
| `latitude` | geoespacial | `float64` | original | 0,0 % | Latitud (anonimizada por Airbnb, error de unos cientos de metros) |
| `longitude` | geoespacial | `float64` | original | 0,0 % | Longitud (anonimizada por Airbnb) |
| `property_type` | categorico | `str` | original | 0,0 % | Tipo de propiedad declarado |
| `room_type` | categorico | `str` | original | 0,0 % | Tipo de habitación original (inglés) |
| `accommodates` | numerico | `int64` | original | 0,0 % | Capacidad máxima de huéspedes |
| `bathrooms` | numerico | `float64` | derivada | 0,1 % | Nº de baños reconstruido desde bathrooms_text |
| `bedrooms` | numerico | `float64` | convertida | 0,0 % | Nº de dormitorios (imputado si faltaba, ver bedrooms_imputado) |
| `beds` | numerico | `float64` | convertida | 0,0 % | Nº de camas (imputado si faltaba, ver beds_imputado) |
| `price` | numerico | `float64` | convertida | 18,7 % | Precio por noche en euros (cotización para una ventana de fechas) |
| `price_quote_checkin_date` | fecha | `datetime64[us]` | convertida | 15,4 % | Fecha de entrada de la cotización de precio |
| `price_quote_checkout_date` | fecha | `datetime64[us]` | convertida | 15,4 % | Fecha de salida de la cotización de precio |
| `price_quote_total_price` | numerico | `float64` | original | 21,4 % | Precio total de la cotización |
| `price_quote_price_per_night` | numerico | `float64` | original | 21,4 % | Precio por noche de la cotización |
| `minimum_nights` | numerico | `float64` | original | 7,5 % | Estancia mínima en noches |
| `maximum_nights` | numerico | `float64` | original | 7,5 % | Estancia máxima en noches |
| `minimum_minimum_nights` | numerico | `float64` | original | 7,5 % | Mínimo de la estancia mínima en el calendario |
| `maximum_minimum_nights` | numerico | `float64` | original | 7,5 % | Máximo de la estancia mínima en el calendario |
| `minimum_maximum_nights` | numerico | `float64` | original | 7,5 % | Mínimo de la estancia máxima en el calendario |
| `maximum_maximum_nights` | numerico | `float64` | original | 7,5 % | Máximo de la estancia máxima en el calendario |
| `minimum_nights_avg_ntm` | numerico | `float64` | original | 0,0 % | Media de estancia mínima en los próximos 365 días |
| `maximum_nights_avg_ntm` | numerico | `float64` | original | 0,0 % | Media de estancia máxima en los próximos 365 días |
| `has_availability` | booleano | `boolean` | convertida | 0,4 % | Tiene disponibilidad (nullable) |
| `availability_30` | numerico | `int64` | original | 0,0 % | Días disponibles en los próximos 30 |
| `availability_60` | numerico | `int64` | original | 0,0 % | Días disponibles en los próximos 60 |
| `availability_90` | numerico | `int64` | original | 0,0 % | Días disponibles en los próximos 90 |
| `availability_365` | numerico | `int64` | original | 0,0 % | Días disponibles en los próximos 365 |
| `calendar_last_scraped` | fecha | `datetime64[us]` | convertida | 0,0 % | Fecha de scrape del calendario |
| `number_of_reviews` | numerico | `int64` | original | 0,0 % | Nº total de reseñas |
| `number_of_reviews_ltm` | numerico | `int64` | original | 0,0 % | Reseñas en los últimos 12 meses |
| `number_of_reviews_l30d` | numerico | `int64` | original | 0,0 % | Reseñas en los últimos 30 días |
| `availability_eoy` | numerico | `int64` | original | 0,0 % | Días disponibles hasta fin de año |
| `number_of_reviews_ly` | numerico | `int64` | original | 0,0 % | Reseñas del último año natural |
| `estimated_occupancy_l365d` | numerico | `int64` | original | 0,0 % | Ocupación estimada por Inside Airbnb (noches, 12 meses) |
| `estimated_revenue_l365d` | numerico | `float64` | original | 18,7 % | Ingresos estimados por Inside Airbnb (12 meses) |
| `first_review` | fecha | `datetime64[us]` | convertida | 16,5 % | Fecha de la primera reseña |
| `last_review` | fecha | `datetime64[us]` | convertida | 16,5 % | Fecha de la última reseña |
| `review_scores_rating` | numerico | `float64` | original | 16,5 % | Puntuación global (1-5) |
| `review_scores_accuracy` | numerico | `float64` | original | 16,5 % | Puntuación de veracidad (1-5) |
| `review_scores_cleanliness` | numerico | `float64` | original | 16,5 % | Puntuación de limpieza (1-5) |
| `review_scores_checkin` | numerico | `float64` | original | 16,5 % | Puntuación de llegada (1-5) |
| `review_scores_communication` | numerico | `float64` | original | 16,5 % | Puntuación de comunicación (1-5) |
| `review_scores_location` | numerico | `float64` | original | 16,5 % | Puntuación de ubicación (1-5) |
| `review_scores_value` | numerico | `float64` | original | 16,5 % | Puntuación calidad-precio (1-5) |
| `license` | texto | `str` | original | 47,2 % | Texto de licencia turística (con HTML) |
| `calculated_host_listings_count` | numerico | `int64` | original | 0,0 % | Anuncios del anfitrión en este snapshot |
| `calculated_host_listings_count_entire_homes` | numerico | `int64` | original | 0,0 % | Alojamientos enteros del anfitrión |
| `calculated_host_listings_count_private_rooms` | numerico | `int64` | original | 0,0 % | Habitaciones privadas del anfitrión |
| `calculated_host_listings_count_shared_rooms` | numerico | `int64` | original | 0,0 % | Habitaciones compartidas del anfitrión |
| `reviews_per_month` | numerico | `float64` | original | 16,5 % | Reseñas por mes |
| `bano_compartido` | booleano | `bool` | derivada | 0,0 % | El baño es compartido (desde bathrooms_text) |
| `n_amenities` | numerico | `int64` | derivada | 0,0 % | Nº de amenities del anuncio |
| `tiene_wifi` | booleano | `bool` | derivada | 0,0 % | Amenity: wifi |
| `tiene_aire_acondicionado` | booleano | `bool` | derivada | 0,0 % | Amenity: aire acondicionado |
| `tiene_cocina` | booleano | `bool` | derivada | 0,0 % | Amenity: cocina |
| `tiene_calefaccion` | booleano | `bool` | derivada | 0,0 % | Amenity: calefacción |
| `tiene_lavadora` | booleano | `bool` | derivada | 0,0 % | Amenity: lavadora |
| `tiene_ascensor` | booleano | `bool` | derivada | 0,0 % | Amenity: ascensor |
| `tiene_terraza_balcon` | booleano | `bool` | derivada | 0,0 % | Amenity: terraza, patio o balcón |
| `tiene_parking_cualquiera` | booleano | `bool` | derivada | 0,0 % | Amenity: cualquier tipo de parking (gratuito o de pago) |
| `tiene_piscina` | booleano | `bool` | derivada | 0,0 % | Amenity: piscina |
| `tiene_tv` | booleano | `bool` | derivada | 0,0 % | Amenity: televisión |
| `tiene_espacio_trabajo` | booleano | `bool` | derivada | 0,0 % | Amenity: espacio de trabajo |
| `admite_mascotas` | booleano | `bool` | derivada | 0,0 % | Amenity: admite mascotas |
| `llegada_autonoma` | booleano | `bool` | derivada | 0,0 % | Amenity: llegada autónoma (self check-in) |
| `nombre_longitud` | numerico | `int64` | derivada | 0,0 % | Longitud del título sin HTML (caracteres) |
| `descripcion_longitud` | numerico | `int64` | derivada | 0,0 % | Longitud de la descripción sin HTML (caracteres) |
| `descripcion_palabras` | numerico | `int64` | derivada | 0,0 % | Palabras de la descripción sin HTML |
| `host_tiene_descripcion` | booleano | `bool` | derivada | 0,0 % | El anfitrión tiene texto «sobre mí» |
| `precio_valido` | booleano | `bool` | derivada | 0,0 % | Precio publicado y mayor que 0 |
| `es_outlier_bajo` | booleano | `bool` | derivada | 0,0 % | Precio por debajo de la valla inferior (IQR en log) |
| `es_outlier_alto` | booleano | `bool` | derivada | 0,0 % | Precio por encima de la valla superior (IQR en log) |
| `es_outlier_precio` | booleano | `bool` | derivada | 0,0 % | Precio atípico por cualquiera de las dos vallas |
| `minimum_nights_extremo` | booleano | `bool` | derivada | 0,0 % | Estancia mínima mayor de 365 noches |
| `alquiler_larga_estancia` | booleano | `bool` | derivada | 0,0 % | Estancia mínima de 30 noches o más |
| `coords_validas` | booleano | `bool` | derivada | 0,0 % | Coordenadas dentro del bounding box de Madrid |
| `bedrooms_imputado` | booleano | `bool` | derivada | 0,0 % | bedrooms se imputó |
| `beds_imputado` | booleano | `bool` | derivada | 0,0 % | beds se imputó |
| `tipo_habitacion` | categorico | `string` | derivada | 0,0 % | Tipo de habitación en español |
| `tipo_habitacion_agrupado` | categorico | `string` | derivada | 0,0 % | Tipo de habitación con categorías raras (< 200 con precio) agrupadas |
| `log_precio` | numerico | `float64` | derivada | 18,7 % | Logaritmo natural del precio |
| `precio_por_persona` | numerico | `float64` | derivada | 18,7 % | Precio dividido por la capacidad (€) |
| `antiguedad_host_anios` | numerico | `float64` | derivada | 0,0 % | Antigüedad del anfitrión en años |
| `ocupacion_estimada` | numerico | `float64` | derivada | 0,0 % | (365 − availability_365) / 365; no es ocupación real |
| `ocupacion_inside_airbnb` | numerico | `float64` | derivada | 0,0 % | estimated_occupancy_l365d / 365 (referencia externa) |
| `es_activo_probable` | booleano | `bool` | derivada | 0,0 % | No (availability_365 = 0 y sin reseñas en 12 meses) |
| `ingresos_estimados_anuales` | numerico | `float64` | derivada | 18,7 % | price × (365 − availability_365), cota muy incierta (€) |
| `distancia_centro_km` | numerico | `float64` | derivada | 0,0 % | Distancia haversine a la Puerta del Sol (km) |
| `categoria_precio` | categorico | `category` | derivada | 18,7 % | Cuartil de precio (Económico … Alto) |
| `categoria_capacidad` | categorico | `category` | derivada | 0,0 % | Tramo de capacidad (1-2, 3-4, 5-6, 7 o más) |
| `es_multianuncio` | booleano | `bool` | derivada | 0,0 % | El anfitrión tiene más de un anuncio |
| `segmento_host` | categorico | `category` | derivada | 0,0 % | Segmento del anfitrión por nº de anuncios |
| `licencia_tipo` | categorico | `string` | derivada | 0,0 % | Sin licencia / Exenta / Con nº de registro / Otro o en proceso |
| `tiene_resenas` | booleano | `bool` | derivada | 0,0 % | Tiene al menos una reseña |
| `dias_desde_ultima_resena` | numerico | `float64` | derivada | 16,5 % | Días desde la última reseña hasta la fecha de referencia |
