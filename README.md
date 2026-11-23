# Airbnb en España y renta de la zona: limpieza, análisis descriptivo y estadístico

Proyecto final del Máster en Data Analytics. Une dos fuentes públicas, los anuncios de **Inside Airbnb** de <!--f:descriptivo.n_ciudades:n0-->9<!--/f--> ciudades y regiones de España y el **Atlas de Distribución de Renta de los Hogares del INE**, para responder a una pregunta de negocio: **qué explica el precio de un alojamiento turístico en España y cómo se relaciona con la renta de la zona**.

- **Autor:** Diego Hernández Pellegrini
- **GitHub:** [diegopellegrini1](https://github.com/diegopellegrini1)
- **Informe completo:** [reports/informe.md](reports/informe.md)

## Requisitos del enunciado y dónde se cumplen

| Requisito | Cómo se cumple | Archivos |
|---|---|---|
| Dos conjuntos de datos en bruto de dos fuentes distintas, adjuntos | Inside Airbnb: <!--f:dataset.fuente_airbnb.n_ficheros:n0-->27<!--/f--> ficheros de <!--f:descriptivo.n_ciudades:n0-->9<!--/f--> ciudades (<!--f:dataset.fuente_airbnb.bytes:n0-->63.162.113<!--/f--> bytes). INE: <!--f:dataset.fuente_ine.n_tablas:n0-->40<!--/f--> tablas del ADRH de <!--f:dataset.fuente_ine.n_provincias:n0-->10<!--/f--> provincias (<!--f:dataset.fuente_ine.bytes:n0-->34.273.215<!--/f--> bytes, gzip sin pérdida). Cada fuente con su manifiesto (URL, fecha, tamaño y SHA-256) | [data/raw/inside_airbnb/](data/raw/inside_airbnb/), [data/raw/ine/](data/raw/ine/) |
| Conjunto final unido con al menos <!--f:meta.parametros.min_filas:n0-->50.000<!--/f--> filas y <!--f:meta.parametros.min_columnas:n0-->20<!--/f--> columnas | <!--f:dataset.filas_procesadas:n0-->109.437<!--/f--> filas × <!--f:dataset.columnas_procesadas:n0-->74<!--/f--> columnas (<!--f:dataset.n_columnas_ine_final:n0-->14<!--/f--> del INE). El <!--f:dataset.union.total.pct_distrito:pct2-->99,61 %<!--/f--> de los anuncios recibe datos del INE por cruce punto en polígono. El código lo comprueba y se detiene si no se cumple | [data/processed/airbnb_espana_ine.parquet](data/processed/airbnb_espana_ine.parquet), [muestra CSV](data/processed/airbnb_espana_ine_muestra.csv), [src/fuentes.py](src/fuentes.py), [src/calidad.py](src/calidad.py) |
| Transformación y limpieza profunda | Conversión de tipos, reparación de textos, nulos con política explícita, atípicos por IQR sobre ln(precio), estancias extremas, coordenadas por ciudad, imputación con indicador, <!--f:features.n_flags_amenities:n0-->13<!--/f--> amenities y variables derivadas; informe de calidad antes/después **por ciudad** | [src/limpieza.py](src/limpieza.py), [src/features.py](src/features.py), [reports/calidad_datos.json](reports/calidad_datos.json), [reports/diccionario_datos.md](reports/diccionario_datos.md) |
| Análisis descriptivo | Nacional y por ciudad: precios, tipos, capacidad, anfitriones, licencias, ocupación, mapas y correlaciones | [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), [src/analisis.py](src/analisis.py) |
| Análisis estadístico | Kruskal-Wallis y Dunn-Bonferroni, Mann-Whitney, chi-cuadrado, bootstrap por anfitrión y por distrito, Spearman renta-precio, OLS con efectos fijos de ciudad y errores agrupados, VIF/GVIF, Holm, modelo ML con validación agrupada y sensibilidades | [notebooks/02_analisis_estadistico.ipynb](notebooks/02_analisis_estadistico.ipynb), [src/estadistica.py](src/estadistica.py) |
| Visualización | <!--f:meta.parametros.n_figuras:n0-->20<!--/f--> figuras con matplotlib y seaborn | [reports/figures/](reports/figures/), [src/visualizacion.py](src/visualizacion.py) |
| Dashboard operativo (Excel) | Libro de Excel sin macros generado desde los datos finales: selectores de ciudad y tipo, KPI y gráficos nativos que se recalculan con fórmulas, comparador de anuncios, renta y precio por ciudad, resumen por ciudad y tabla de datos con filtros | [dashboard/Dashboard_Airbnb_Espana.xlsx](dashboard/Dashboard_Airbnb_Espana.xlsx), [dashboard/README_dashboard.md](dashboard/README_dashboard.md), [src/dashboard_excel.py](src/dashboard_excel.py) |
| Informe explicativo | Metodología, resultados, limitaciones y recomendaciones | [reports/informe.md](reports/informe.md) |
| README con pasos e informe | Este archivo | [README.md](README.md) |
| Organización de carpetas | Ver la estructura más abajo | [requirements.txt](requirements.txt), [.gitignore](.gitignore), [LICENSE](LICENSE) |

## Resultados clave

- **Precio.** Mediana <!--f:descriptivo.precio.mediana:eur2-->170,00 €<!--/f--> y media <!--f:descriptivo.precio.media:eur2-->263,99 €<!--/f--> por noche (<!--f:descriptivo.precio.n:n0-->89.614<!--/f--> anuncios con precio). Mallorca es la zona más cara (mediana <!--f:descriptivo.ciudad_mas_cara.mediana:eur0-->312 €<!--/f-->) y Valencia la más barata (<!--f:descriptivo.ciudad_mas_barata.mediana:eur0-->134 €<!--/f-->). El <!--f:limpieza.precio.pct_nulo:pct1-->18,1 %<!--/f--> de los anuncios no publica precio.
- **Qué se asocia con el precio.** Tipo de habitación (Kruskal-Wallis, ε² = <!--f:estadistica.kruskal_tipo.tamano_efecto.valor:n3-->0,161<!--/f-->), ciudad (ε² = <!--f:estadistica.kruskal_ciudad.tamano_efecto.valor:n3-->0,114<!--/f-->) y capacidad (ρ = <!--f:descriptivo.correlaciones.con_precio_por_variable.accommodates.spearman:n2-->0,61<!--/f-->). En la regresión, cada persona más de capacidad se asocia con un precio <!--f:estadistica.ols.coef_accommodates.pct:signo_pct-->+9,0 %<!--/f--> y una estancia mínima de <!--f:meta.parametros.min_nights_larga_estancia:n0-->30<!--/f--> noches o más con <!--f:estadistica.ols.coef_larga_estancia.pct:signo_pct-->-52,2 %<!--/f--> por noche.
- **Renta de la zona.** Anuncio a anuncio y mezclando ciudades, la correlación de Spearman entre la renta del distrito y el precio es débil (ρ = <!--f:estadistica.renta_precio.global_distrito.rho:n3-->0,073<!--/f-->, IC 95 % remuestreando distritos <!--f:estadistica.renta_precio.global_distrito.cluster_zona.ic_inf:n3-->0,002<!--/f--> a <!--f:estadistica.renta_precio.global_distrito.cluster_zona.ic_sup:n3-->0,144<!--/f-->; en el límite de la significación, p ≈ <!--f:meta.alfa:n2-->0,05<!--/f--> y un IC que roza el cero). Por ciudad es positiva en las <!--f:estadistica.renta_precio.n_ciudades_positivas:n0-->9<!--/f--> como estimación puntual y modesta (de <!--f:estadistica.renta_precio.rho_ciudad_min.distrito.rho:n2-->0,03<!--/f--> a <!--f:estadistica.renta_precio.rho_ciudad_max.distrito.rho:n2-->0,29<!--/f-->), significativa en algunas tras Holm (recuento orientativo: las ciudades urbanas tienen solo <!--f:estadistica.renta_precio.n_distritos_urbanas_min:n0-->11<!--/f-->-<!--f:estadistica.renta_precio.n_distritos_urbanas_max:n0-->21<!--/f--> distritos). En la regresión conjunta con efectos fijos de ciudad y el resto de controles, una renta un 10 % mayor se asocia con un precio <!--f:estadistica.ols.coef_renta.pct_por_10pct_renta:signo_pct-->+4,4 %<!--/f--> mayor, pero el INE solo añade <!--f:estadistica.ols.delta_r2_ajustado_ine_pp:n1-->2,1<!--/f--> puntos de R² ajustado. **La renta explica poco del precio**, y es una asociación, no un efecto causal.
- **Modelos.** OLS sobre ln(precio): R² ajustado = <!--f:estadistica.ols.r2_ajustado:n3-->0,613<!--/f--> (n = <!--f:estadistica.ols.n_efectivo:n0-->89.182<!--/f-->). HistGradientBoosting validado con particiones agrupadas por anfitrión: R² = <!--f:estadistica.ml.cv.r2_media:n3-->0,717<!--/f--> ± <!--f:estadistica.ml.cv.r2_std:n3-->0,014<!--/f--> (con una partición aleatoria saldría <!--f:estadistica.ml.aleatorio.r2:n3-->0,770<!--/f-->, una cifra optimista). Las dos cifras no son comparables entre sí: el R² del OLS es dentro de muestra y el del HGB es de validación cruzada con más variables. **Con las mismas filas, variables y particiones**, la regresión lineal da R² = <!--f:estadistica.ml.comparacion_misma_muestra.lineal_vars_ols.r2_media:n3-->0,605<!--/f--> y el HGB <!--f:estadistica.ml.comparacion_misma_muestra.hgb_vars_ols.r2_media:n3-->0,665<!--/f-->. El HGB usa además los días disponibles del calendario (`availability_365`), que en parte reflejan el propio precio y la ocupación: su importancia no es una causa del precio.
- **Anfitriones.** El <!--f:descriptivo.hosts.pct_anuncios_multianuncio:pct1-->75,0 %<!--/f--> de los anuncios (sobre todos los anuncios; el dashboard calcula sus porcentajes solo sobre los anuncios con precio, por eso difieren un poco) es de anfitriones con varios anuncios, aunque el <!--f:descriptivo.hosts.pct_hosts_un_anuncio:pct1-->70,2 %<!--/f--> de los anfitriones tiene uno solo. Ser superhost se asocia con un precio <!--f:estadistica.ols.coef_superhost.pct:signo_pct-->+2,2 %<!--/f--> al controlar, una diferencia pequeña que no sobrevive a la corrección de Holm.

**Qué es el precio:** el precio por noche de una cotización fechada de Airbnb (entradas entre el <!--f:precio_naturaleza.checkin_min:fecha-->2026-09-17<!--/f--> y el <!--f:precio_naturaleza.checkin_max:fecha-->2027-09-28<!--/f-->), **con los descuentos ya aplicados** (el <!--f:precio_naturaleza.pct_con_descuento:pct1-->19,9 %<!--/f--> de las cotizaciones tiene alguno) y sin tasas. No es el precio pagado por un huésped.

Todas las cifras de este README y del informe se generan desde [reports/findings.json](reports/findings.json). [scripts/verificar_consistencia.py](scripts/verificar_consistencia.py) comprueba que estos documentos **reproducen** findings.json (cada cifra marcada coincide, el texto se regenera idéntico, sin enlaces rotos ni números escritos a mano en las plantillas). Es una comprobación de transcripción: no comprueba que las interpretaciones sean ciertas; eso lo sostienen el análisis, sus supuestos y las sensibilidades del informe.

## Dashboard

[dashboard/Dashboard_Airbnb_Espana.xlsx](dashboard/Dashboard_Airbnb_Espana.xlsx) es un dashboard operativo en Excel (Excel 2016 o posterior y Microsoft 365, sin macros; no necesita Python). Se genera con el pipeline desde el conjunto final y `findings.json`, sin cifras escritas a mano. Contiene:

- **Dashboard:** desplegables de ciudad y tipo de alojamiento; tarjetas con anuncios con precio, mediana, media, P25 y P75, % superhost, % de anuncios de anfitriones con varios anuncios y renta neta media de la zona; gráficos de mediana por ciudad, anuncios por tramo de precio, mediana por tipo, anuncios por capacidad y renta frente a precio por distrito censal. Todo se recalcula con fórmulas nativas al cambiar los desplegables.
- **Comparador:** ciudad + tipo + capacidad → precio típico de anuncios comparables, con aviso si hay menos de <!--f:meta.parametros.min_n_grupo:n0-->30<!--/f--> anuncios.
- **Renta y precio**, **Resumen ciudades**, **Datos** (tabla de Excel con los <!--f:descriptivo.precio.n:n0-->89.614<!--/f--> anuncios con precio, sin datos personales) y **Cálculos** (tablas auxiliares de los gráficos).

Guía de uso, compatibilidad y limitaciones: [dashboard/README_dashboard.md](dashboard/README_dashboard.md).

![Vista previa generada con Python de los gráficos del dashboard (no es una captura de Excel)](reports/figures/dashboard_vista_previa.png)

*Vista previa generada con Python de los gráficos del dashboard (no es una captura de Excel).*

## Datos

**Fuente 1, Inside Airbnb.** `listings.csv.gz`, `neighbourhoods.csv` y `neighbourhoods.geojson` de Barcelona, Euskadi, Girona, Madrid, Málaga, Mallorca, Menorca, Sevilla, Valencia, snapshots del <!--f:meta.snapshot_min:fecha-->2026-09-17<!--/f--> al <!--f:meta.snapshot_max:fecha-->2026-09-28<!--/f-->. Manifiesto: [data/raw/inside_airbnb/MANIFEST.json](data/raw/inside_airbnb/MANIFEST.json).

**Fuente 2, INE.** Atlas de Distribución de Renta de los Hogares (ADRH): renta neta y bruta por persona y por hogar, mediana por unidad de consumo, Gini, P80/P20, edad media, tamaño del hogar, hogares unipersonales, población y riesgo de pobreza, por municipio y distrito censal. Año de referencia: <!--f:meta.anio_ine:texto-->2024<!--/f-->. Manifiesto: [data/raw/ine/MANIFEST.json](data/raw/ine/MANIFEST.json).

**Unión.** Cada anuncio se sitúa en su sección censal cruzando sus coordenadas con el seccionado oficial del INE (punto en polígono). La sección da el distrito censal y el municipio, que son las claves del ADRH. El seccionado pesa más de <!--f:meta.parametros.limite_subida_mb:n0-->25<!--/f--> MB y no se versiona (se descarga en `data/extra/`); el resultado del cruce sí está en [data/interim/asignacion_secciones.csv.gz](data/interim/asignacion_secciones.csv.gz), así que la unión se reproduce sin red. Emparejados: <!--f:dataset.union.total.con_distrito:n0-->109.010<!--/f--> de <!--f:dataset.union.total.anuncios:n0-->109.437<!--/f--> anuncios (<!--f:dataset.union.total.pct_distrito:pct2-->99,61 %<!--/f-->); los <!--f:dataset.union.n_sin_emparejar:n0-->427<!--/f--> restantes se conservan con valores vacíos. La unidad asignada coincide con la declarada por Inside Airbnb en el <!--f:dataset.union.concordancia_total.pct_coinciden:pct2-->99,58 %<!--/f--> de los anuncios validados.

## Estructura del repositorio

```
├─ run_pipeline.py                 pipeline completo (un solo comando)
├─ requirements.txt                dependencias con versiones fijas
├─ data/
│  ├─ raw/
│  │  ├─ inside_airbnb/<ciudad>/   fuente 1: listings.csv.gz, neighbourhoods.csv/.geojson + MANIFEST.json
│  │  └─ ine/                      fuente 2: tablas del ADRH (.csv.gz) + MANIFEST.json
│  ├─ interim/                     asignación anuncio → sección censal (resultado del cruce geográfico)
│  └─ processed/                   conjunto final unido (Parquet) + muestra CSV legible
├─ src/                            módulos reutilizables (type hints y docstrings)
│  ├─ fuentes.py                   descarga, lectura y unión de las dos fuentes
│  ├─ limpieza.py  features.py  calidad.py  diccionario.py
│  ├─ analisis.py  estadistica.py  visualizacion.py  informe.py  dashboard_excel.py
│  ├─ carga.py  config.py  formato.py  geo.py  errores.py
│  └─ plantillas/                  plantillas Jinja2 del README y del informe
├─ notebooks/                      01_eda.ipynb y 02_analisis_estadistico.ipynb (ejecutados, con salidas)
│  └─ fuentes_py/                  fuentes .py de los notebooks
├─ reports/
│  ├─ informe.md                   informe explicativo
│  ├─ findings.json                todas las cifras calculadas
│  ├─ calidad_datos.json           calidad antes/después, global y por ciudad
│  ├─ diccionario_datos.md         diccionario de columnas con su fuente
│  └─ figures/                     figuras PNG (incluida la vista previa del dashboard)
├─ dashboard/
│  ├─ Dashboard_Airbnb_Espana.xlsx dashboard operativo en Excel (generado por src/dashboard_excel.py)
│  └─ README_dashboard.md          guía del dashboard
├─ scripts/                        recuentos y verificaciones independientes, ejecución de notebooks
└─ tests/                          tests de pytest
```

## Instalación

Requiere Python 3.13 y Git.

```bash
git clone https://github.com/diegopellegrini1/Proyecto-Final.git
cd Proyecto-Final
```

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fija con `==` todas las dependencias directas; pip resuelve las transitivas.

## Ejecución

```bash
python run_pipeline.py --sin-descarga   # todo desde data/raw, sin red: unión, limpieza, análisis, figuras, dashboard, documentos y notebooks
python -B -m pytest -q                  # tests (-B: sin __pycache__)
```

- Etapas: unión de las dos fuentes, perfil de calidad antes, limpieza por ciudad, variables derivadas, conjunto final, perfil de calidad después, análisis descriptivo y estadístico, figuras, `findings.json`, **dashboard de Excel** (`src/dashboard_excel.py`: libro, vista previa y guía en `dashboard/`), documentos, notebooks y verificación de consistencia. Solo el dashboard: `python -m src.dashboard_excel`.
- Sin `--sin-descarga`, el pipeline usa `data/raw` si sus hashes coinciden con los manifiestos y solo descarga si falta algo. `--forzar-descarga` vuelve a buscar los snapshots más recientes (pueden cambiar las cifras).
- Otras opciones: `--semilla`, `--n-boot`, `--sin-ml`, `--sin-notebooks`, `--sin-verificacion`. Los tiempos de cada etapa quedan en `reports/findings.json`.
- Recuentos y comprobaciones independientes (no importan `src/`): `python scripts/contar_filas_espana.py` y `python scripts/recalcular_cifras.py`.
- Antes de publicar el repositorio: `python scripts/verificar_subida.py --limpiar` borra cachés y comprueba que ningún archivo supera <!--f:meta.parametros.limite_subida_mb:n0-->25<!--/f--> MB. `.gitignore` deja fuera `.venv/`, `data/extra/` y `.agents/`.

Notebooks: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) y [notebooks/02_analisis_estadistico.ipynb](notebooks/02_analisis_estadistico.ipynb). Se regeneran con `python scripts/ejecutar_notebooks.py` (el pipeline lo hace al final).

### Cómo editar un texto

README e informe se generan desde plantillas: para cambiar un texto se edita `src/plantillas/README.md.j2` o `src/plantillas/informe.md.j2` y se vuelve a ejecutar `python run_pipeline.py --sin-descarga`. Si se edita `README.md` o `reports/informe.md` a mano, el verificador lo detecta y la siguiente ejecución lo sobrescribe. Las cifras se escriben en la plantilla con `cifra("ruta.en.findings", "formato")`, nunca a mano.

## Limitaciones

- Datos **scrapeados** de la web pública de Airbnb, un snapshot por ciudad (del <!--f:meta.snapshot_min:fecha-->2026-09-17<!--/f--> al <!--f:meta.snapshot_max:fecha-->2026-09-28<!--/f-->): sin evolución temporal.
- **Precio de anuncio ≠ precio pagado**, y los anuncios sin precio no son aleatorios (sesgo de selección).
- **Desfase temporal:** renta de <!--f:meta.anio_ine:texto-->2024<!--/f--> frente a anuncios de los snapshots indicados arriba.
- **Málaga:** sus distritos censales no coinciden con los municipales. **Coordenadas anonimizadas** por Airbnb.
- **Ocupación** aproximada por los días no disponibles del calendario (reservas y bloqueos), no observada.
- **Observacional:** asociaciones, no efectos causales.

Detalle en el [informe](reports/informe.md).

## Uso de herramientas de IA

Este proyecto lo desarrollé con un asistente de IA. Primero escribí mi propia versión del código con su ayuda, como ejercicio de aprendizaje. Después, el asistente implementó la versión final de este repositorio siguiendo mis instrucciones, y yo la estudié y la comparé con la mía para entender y validar cada decisión. Definí el objetivo, elegí los datos y el enfoque, y decidí qué correcciones aplicar tras dos revisiones de código independientes. Ninguna cifra se ha escrito a mano: todas proceden de ejecutar el código sobre los datos públicos (`reports/findings.json`), y un script comprueba que los documentos las reproducen fielmente.

## Datos personales y privacidad

- Los `listings.csv.gz` de `data/raw/inside_airbnb/` son los ficheros de Inside Airbnb **sin modificar** (se conservan para reproducir el análisis sin conexión y verificar su hash) de las <!--f:descriptivo.n_ciudades:n0-->9<!--/f--> ciudades y regiones. Contienen datos **públicos** de los anfitriones tal como aparecen en Airbnb (nombre, URL y foto de perfil, descripción y ubicación declarada), redistribuidos por Inside Airbnb.
- El conjunto procesado (`data/processed/`) **no incluye** nombres, URLs, descripciones ni la ubicación declarada de los anfitriones. Conserva `host_id` (identificador numérico público de Airbnb) porque es imprescindible para el análisis: agrupar los errores de la regresión, validar el modelo sin repetir anfitriones entre entrenamiento y prueba, el bootstrap por anfitrión y la composición de la oferta.
- El dashboard de Excel solo incluye columnas de los anuncios con precio necesarias para el análisis: sin nombres, URLs, fotos, coordenadas ni `host_id`.
- Los datos del INE son agregados estadísticos por territorio (distrito censal y municipio), sin datos de personas.
- No se cruzan los datos con otras fuentes de personas ni se intenta reidentificar a nadie.

## Fuente de datos, atribución y licencias

- **Inside Airbnb** ([insideairbnb.com](https://insideairbnb.com/), «Get the Data»), <!--f:descriptivo.n_ciudades:n0-->9<!--/f--> ciudades y regiones de España, snapshots del <!--f:meta.snapshot_min:fecha-->2026-09-17<!--/f--> al <!--f:meta.snapshot_max:fecha-->2026-09-28<!--/f-->. Licencia: **Creative Commons Attribution 4.0 International (CC BY 4.0)** ([https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)), verificada en la página de descargas el <!--f:dataset.fuente_airbnb.licencia.fecha_verificacion:fecha-->2026-10-27<!--/f-->. La página de Inside Airbnb es ambigua: indica esa licencia y, en su pie, también una renuncia de derechos del autor; por prudencia se cita y se respeta la licencia con atribución. Los datos se han limpiado y transformado para este proyecto.
- **INE**, Atlas de Distribución de Renta de los Hogares ([INEbase](https://www.ine.es/dynt3/inebase/es/index.htm?padre=7132)). Elaboración propia con datos extraídos del sitio web del INE: www.ine.es. Fecha de la última actualización del ADRH: <!--f:dataset.fuente_ine.actualizacion.fecha:fecha-->2026-10-19<!--/f--> ([página de resultados](https://ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736177088&idp=1254735976608&menu=resultados)); tablas descargadas el <!--f:dataset.fuente_ine.fechas_descarga.1:fecha-->2026-10-27<!--/f-->. Reutilización autorizada por el [aviso legal del INE](https://www.ine.es/ss/Satellite?L=es_ES&c=Page&cid=1254735849170&p=1254735849170&pagename=Ayuda%2FINELayout) citando la fuente, sin desnaturalizar el sentido de los datos y sin sugerir que el INE patrocina este trabajo.
- **Código:** licencia MIT, ver [LICENSE](LICENSE).
