# Airbnb en Madrid: EDA, análisis estadístico y dashboard

Proyecto final del Máster en Data Analytics. Análisis exploratorio, limpieza, análisis estadístico inferencial y un dashboard operativo en Streamlit sobre los anuncios públicos de **Inside Airbnb** para Madrid (snapshot <!--f:meta.fecha_snapshot_url:fecha-->2026-09-17<!--/f-->).

- **Autor:** Diego Hernández Pellegrini
- **GitHub:** [diegopellegrini1](https://github.com/diegopellegrini1)
- **Informe completo:** [reports/informe.md](reports/informe.md)

## Objetivo

Entender qué explica el precio por noche de los anuncios de Airbnb en Madrid (tipo de alojamiento, ubicación, capacidad, anfitrión, amenities) y cómo está estructurada la oferta, y ofrecer un dashboard con filtros para explorar el mercado por distrito, barrio, tipo de habitación y precio. Todo el flujo es reproducible con un solo comando.

## Dataset y requisitos mínimos

| Requisito (criterio propio) | Valor en el dataset | ¿Cumple? |
|---|---|---|
| Filas (al menos <!--f:meta.parametros.umbral_filas:n0-->1.000<!--/f-->) | <!--f:dataset.filas_raw:n0-->23.109<!--/f--> anuncios | Sí |
| Columnas (al menos <!--f:meta.parametros.umbral_columnas:n0-->20<!--/f-->) | <!--f:dataset.columnas_raw:n0-->90<!--/f--> columnas | Sí |
| Tipos de variable | <!--f:dataset.tipos_raw.numerico:n0-->54<!--/f--> numéricas, <!--f:dataset.tipos_raw.categorico:n0-->4<!--/f--> categóricas, <!--f:dataset.tipos_raw.booleano:n0-->4<!--/f--> booleanas, <!--f:dataset.tipos_raw.fecha:n0-->6<!--/f--> de fecha, <!--f:dataset.tipos_raw.texto:n0-->10<!--/f--> de texto libre, <!--f:dataset.tipos_raw.geoespacial:n0-->2<!--/f--> geoespaciales | Sí |

Fuente: `listings.csv.gz`, `neighbourhoods.csv` y `neighbourhoods.geojson` de [Inside Airbnb](https://insideairbnb.com/get-the-data/). La fecha del snapshot (<!--f:meta.fecha_snapshot_url:fecha-->2026-09-17<!--/f-->) es la que figura en la URL de descarga; los anuncios se scrapearon entre el <!--f:meta.last_scraped_min:fecha-->2026-09-19<!--/f--> y el <!--f:meta.last_scraped_max:fecha-->2026-09-22<!--/f-->. La clasificación por tipos y el cumplimiento se calculan por código (`src/calidad.py`). URLs, tamaños y hashes SHA-256 de los ficheros descargados: [data/raw/MANIFEST.json](data/raw/MANIFEST.json).

## Resumen de hallazgos

- **Precio muy asimétrico:** mediana <!--f:descriptivo.precio.mediana:eur2-->151,00 €<!--/f--> y media <!--f:descriptivo.precio.media:eur2-->208,63 €<!--/f--> por noche en <!--f:descriptivo.precio.n:n0-->18.785<!--/f--> anuncios con precio (asimetría <!--f:descriptivo.precio.asimetria:n2-->31,54<!--/f-->). El <!--f:limpieza.precio.pct_nulo:pct1-->18,7 %<!--/f--> de los anuncios no publica precio, sobre todo los que no tienen fechas disponibles.
- **El tipo de habitación es lo que más pesa:** Alojamiento entero <!--f:estadistica.bootstrap.tipo.0.mediana:eur2-->173,63 €<!--/f--> (IC 95 % <!--f:estadistica.bootstrap.tipo.0.ic_inf:eur2-->172,00 €<!--/f--> – <!--f:estadistica.bootstrap.tipo.0.ic_sup:eur2-->175,36 €<!--/f-->) frente a habitación privada <!--f:estadistica.bootstrap.tipo.1.mediana:eur2-->74,00 €<!--/f--> (<!--f:estadistica.bootstrap.tipo.1.ic_inf:eur2-->71,10 €<!--/f--> – <!--f:estadistica.bootstrap.tipo.1.ic_sup:eur2-->75,60 €<!--/f-->); Kruskal-Wallis con ε² = <!--f:estadistica.kruskal_tipo.tamano_efecto.valor:n3-->0,234<!--/f--> (efecto grande).
- **Ubicación:** Salamanca (<!--f:descriptivo.por_distrito.0.mediana:eur2-->188,34 €<!--/f-->) y Centro (<!--f:descriptivo.por_distrito.1.mediana:eur2-->180,00 €<!--/f-->) son los distritos con mayor mediana y Moratalaz (<!--f:descriptivo.distrito_mas_barato.mediana:eur2-->58,00 €<!--/f-->) el de menor. En la regresión, cada km desde la Puerta del Sol se asocia con un precio <!--f:estadistica.ols.coef_distancia.pct:signo_pct-->-4,7 %<!--/f-->.
- **Superhost:** la diferencia bruta (mediana <!--f:estadistica.mann_whitney_superhost.extra.mediana_a:eur2-->163,50 €<!--/f--> frente a <!--f:estadistica.mann_whitney_superhost.extra.mediana_b:eur2-->147,00 €<!--/f-->) es estadísticamente significativa pero con efecto despreciable (r = <!--f:estadistica.mann_whitney_superhost.tamano_efecto.valor:n3-->0,100<!--/f-->), y al controlar en la regresión deja de ser significativa (<!--f:estadistica.ols.coef_superhost.p_cluster:p-->p = 0,107<!--/f-->).
- **Regresión OLS sobre ln(precio):** R² ajustado = <!--f:estadistica.ols.r2_ajustado:n3-->0,571<!--/f--> con n = <!--f:estadistica.ols.n_efectivo:n0-->18.159<!--/f-->; capacidad <!--f:estadistica.ols.coef_accommodates.pct:signo_pct-->+12,0 %<!--/f--> por persona, habitación privada <!--f:estadistica.ols.coef_privada.pct:signo_pct-->-22,8 %<!--/f-->, estancia mínima larga <!--f:estadistica.ols.coef_por_termino.alquiler_larga_estancia.pct:signo_pct-->-45,0 %<!--/f-->, aire acondicionado <!--f:estadistica.ols.coef_por_termino.tiene_aire_acondicionado.pct:signo_pct-->+15,0 %<!--/f--> (errores agrupados por anfitrión).
- **Modelo predictivo:** HistGradientBoosting con R² en test = <!--f:estadistica.ml.r2_test:n3-->0,740<!--/f--> y MAE ≈ <!--f:estadistica.ml.mae_eur_test:eur0-->60 €<!--/f--> por noche (línea base: <!--f:estadistica.ml.mae_eur_base:eur0-->103 €<!--/f-->).
- **Oferta profesionalizada:** el <!--f:descriptivo.hosts.pct_anuncios_multianuncio:pct1-->73,5 %<!--/f--> de los anuncios es de anfitriones con varios anuncios y el <!--f:descriptivo.hosts.pct_anuncios_hosts_6_o_mas:pct1-->50,0 %<!--/f--> de anfitriones con seis o más, aunque el <!--f:descriptivo.hosts.pct_hosts_un_anuncio:pct1-->70,3 %<!--/f--> de los anfitriones tiene uno solo.

Todas las cifras de este README y del informe se generan desde [reports/findings.json](reports/findings.json) y se comprueban con [scripts/verificar_consistencia.py](scripts/verificar_consistencia.py).

## Estructura del repositorio

```
├─ run_pipeline.py                 pipeline completo (un solo comando)
├─ requirements.txt                dependencias con versiones fijas
├─ data/
│  ├─ raw/                         datos originales de Inside Airbnb + MANIFEST.json (URL, tamaño, SHA-256)
│  └─ processed/                   listings_limpio.parquet + listings_muestra.csv (muestra legible)
├─ src/                            módulos reutilizables (type hints y docstrings)
│  ├─ carga.py  limpieza.py  calidad.py  features.py  diccionario.py
│  ├─ estadistica.py  analisis.py  visualizacion.py  informe.py
│  ├─ config.py  formato.py  geo.py  errores.py
│  └─ plantillas/                  plantillas Jinja2 de README e informe
├─ notebooks/                      01_eda.ipynb y 02_analisis_estadistico.ipynb (ejecutados, con salidas)
│  └─ fuentes_py/                  fuentes .py de los notebooks
├─ dashboard/                      app Streamlit (app.py, datos.py, filtros.py, graficos.py)
├─ reports/
│  ├─ informe.md                   informe explicativo
│  ├─ findings.json                todas las cifras calculadas
│  ├─ calidad_datos.json           calidad de datos antes/después
│  ├─ diccionario_datos.md         diccionario de columnas
│  └─ figures/                     figuras PNG generadas por script
├─ scripts/                        ejecución de notebooks y verificadores
└─ tests/                          tests de pytest
```

## Instalación (Windows / PowerShell)

Requiere Python 3.13.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` fija con `==` todas las dependencias directas; pip resuelve las transitivas.

## Ejecución

```powershell
python run_pipeline.py                 # descarga (o verifica) datos, limpia, analiza, genera figuras, documentos y notebooks
python -m pytest -q                    # tests
streamlit run dashboard/app.py         # dashboard en http://localhost:8501
```

Opciones útiles de `run_pipeline.py`: `--sin-descarga` (modo offline, exige `data/raw` íntegro), `--forzar-descarga`, `--semilla`, `--n-boot`, `--sin-ml`, `--sin-notebooks`, `--sin-verificacion`. Si `data/raw` ya contiene los ficheros y su hash coincide con el manifiesto, el pipeline no usa la red. Tiempo de cómputo medido en la última ejecución (descarga o verificación, limpieza, análisis y figuras): <!--f:rendimiento.total_computo_s:n1-->29,6<!--/f--> s.

Notebooks: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) y [notebooks/02_analisis_estadistico.ipynb](notebooks/02_analisis_estadistico.ipynb). Se regeneran y ejecutan con `python scripts/ejecutar_notebooks.py` (el pipeline lo hace al final) y usan los mismos módulos de `src/`, sin duplicar la limpieza.

**README e informe se generan desde plantillas** (`src/plantillas/`): para cambiar el texto se edita la plantilla y se vuelve a ejecutar el pipeline.

## Dashboard

`streamlit run dashboard/app.py` abre una app con barra lateral de filtros (distrito, barrio en cascada, tipo de habitación, rango de precio y búsqueda por nombre) y cinco pestañas:

1. **Resumen:** KPIs (anuncios, precio mediano y medio, ocupación estimada, % superhost, % multi-anuncio, valoración media), histograma de precio y mediana por distrito.
2. **Mapa:** puntos coloreados por precio (muestreo aleatorio con semilla si hay más de <!--f:meta.parametros.max_puntos_mapa:n0-->10.000<!--/f--> puntos) o coropleta de la mediana por barrio. Mapas base de CARTO sin API key (necesita internet en el navegador).
3. **Precios:** precio por tipo de habitación, frente a la distancia al centro y por barrio.
4. **Anfitriones y calidad:** segmentos de anfitrión, puntuaciones y amenities.
5. **Datos:** tabla filtrable y botón para descargar el CSV filtrado.

El dashboard lee `data/processed/` (no limpia datos al arrancar) y usa `st.cache_data`. Si los filtros dejan cero anuncios muestra un aviso, y si falta el Parquet indica que hay que ejecutar `python run_pipeline.py`. No se incluyen capturas de pantalla: generarlas de forma fiable exigiría un navegador automatizado, y se prefirió no añadir imágenes que no salgan del código.

## Mapeo de los criterios de evaluación

| # | Criterio | Dónde se cumple |
|---|---|---|
| 1 | Requisitos mínimos del conjunto de datos | [Dataset y requisitos mínimos](#dataset-y-requisitos-mínimos), [src/calidad.py](src/calidad.py), [reports/findings.json](reports/findings.json) |
| 2 | Uso eficiente de Python y Pandas | [src/](src/) (operaciones vectorizadas, sin bucles por fila, type hints), [run_pipeline.py](run_pipeline.py), [tests/](tests/) |
| 3 | Uso eficiente de herramienta de visualización | [src/visualizacion.py](src/visualizacion.py), [reports/figures/](reports/figures/), [dashboard/graficos.py](dashboard/graficos.py) (Plotly) |
| 4 | Transformación y limpieza de datos | [src/limpieza.py](src/limpieza.py), [src/features.py](src/features.py), [reports/calidad_datos.json](reports/calidad_datos.json), [reports/diccionario_datos.md](reports/diccionario_datos.md) |
| 5 | Análisis descriptivo | [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), [src/analisis.py](src/analisis.py), [informe, sección de hallazgos descriptivos](reports/informe.md) |
| 6 | Análisis estadístico | [notebooks/02_analisis_estadistico.ipynb](notebooks/02_analisis_estadistico.ipynb), [src/estadistica.py](src/estadistica.py) |
| 7 | Dashboard operativo | [dashboard/app.py](dashboard/app.py), [tests/test_dashboard.py](tests/test_dashboard.py), [scripts/verificar_dashboard.py](scripts/verificar_dashboard.py) |
| 8 | Informe explicativo | [reports/informe.md](reports/informe.md) |
| 9 | README del proyecto | este archivo, generado desde [src/plantillas/README.md.j2](src/plantillas/README.md.j2) |
| 10 | Correcta organización del repositorio GitHub | [estructura](#estructura-del-repositorio), [requirements.txt](requirements.txt), [.gitignore](.gitignore), [LICENSE](LICENSE), [scripts/verificar_subida.py](scripts/verificar_subida.py) |

## Limitaciones

- **Datos scrapeados** de la web pública de Airbnb: pueden tener errores y campos vacíos (<!--f:calidad.n_columnas_100_nulas:n0-->13<!--/f--> columnas vienen vacías en este snapshot).
- **Precio de anuncio ≠ precio pagado**, y además es una **cotización para unas fechas concretas** (entradas entre el <!--f:precio_naturaleza.checkin_min:fecha-->2026-09-19<!--/f--> y el <!--f:precio_naturaleza.checkin_max:fecha-->2027-09-18<!--/f-->), no una media anual. El símbolo `$` del fichero es un artefacto: las cotizaciones están en EUR.
- **Ocupación e ingresos estimados:** `365 − availability_365` no es ocupación real (incluye bloqueos del anfitrión); los ingresos derivados son cotas con mucha incertidumbre.
- **Sesgo de selección:** los análisis de precio solo incluyen anuncios que publican precio, y los que no lo hacen son sobre todo anuncios sin disponibilidad.
- **Snapshot único** (fecha de la URL frente a fechas de scrape, explicadas arriba): no hay estacionalidad ni tendencia.
- **Correlación ≠ causalidad**, anuncios de un mismo anfitrión no independientes (se usan errores agrupados) y coordenadas anonimizadas por Airbnb.

Detalle completo en el [informe](reports/informe.md).

## Datos adicionales (reviews y calendar)

El pipeline puede descargar, para el mismo snapshot, las reseñas y el calendario:

```powershell
python run_pipeline.py --incluir-reviews --incluir-calendar
```

Se guardan en `data/extra/` (ignorada por git) y no se usan en el análisis. Son ficheros grandes (las reseñas superan los 25 MB de subida web de GitHub), por eso no forman parte del repositorio.

## Uso de herramientas de IA

En este proyecto se utilizó un asistente de IA como apoyo para escribir y depurar código, revisar la consistencia de los datos y redactar la documentación. El autor revisó y validó el código, los datos y las conclusiones del análisis. Ninguna cifra se ha escrito a mano: todas proceden de ejecutar el código sobre los datos públicos (`reports/findings.json`) y se comprueban automáticamente con el script de verificación de consistencia.

## Fuente de datos, atribución y licencias

- **Datos:** [Inside Airbnb](https://insideairbnb.com/), «Get the Data», Madrid, snapshot <!--f:meta.fecha_snapshot_url:fecha-->2026-09-17<!--/f-->. Licencia: **Creative Commons Attribution 4.0 International (CC BY 4.0)** ([https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)), verificada en la página de descargas el <!--f:dataset.licencia.fecha_verificacion:fecha-->2026-10-25<!--/f-->. Los datos se han limpiado y transformado para este proyecto. No se usan datos privados ni de ninguna otra fuente, y no se intenta reidentificar a los anfitriones.
- **Código:** licencia MIT, ver [LICENSE](LICENSE).
