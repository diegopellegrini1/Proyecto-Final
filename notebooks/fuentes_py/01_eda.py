# %% [markdown]
# # 01 · Análisis exploratorio (EDA): Airbnb en Madrid
#
# ## Introducción
#
# Este notebook recorre el análisis exploratorio del proyecto final del Máster en Data Analytics. La pregunta de
# fondo es doble: **con qué se asocia el precio por noche** de un anuncio de Airbnb en Madrid y **cómo está
# organizada la oferta** (qué se alquila, dónde y quién lo alquila). Antes de cualquier contraste formal conviene
# mirar los datos con calma: cómo se distribuye el precio, qué columnas faltan, qué grupos son pequeños y qué
# relaciones aparecen a simple vista.
#
# El notebook **no repite la limpieza**. Lee el dataset procesado (`data/processed/listings_limpio.parquet`) y los
# resultados (`reports/findings.json`) que genera `python run_pipeline.py`, y llama a las mismas funciones de
# `src/` que el pipeline, de modo que las cifras del notebook y las del informe salen del mismo código. Además de
# leer resultados, aquí se calculan **operaciones pandas explícitas** (`describe`, `value_counts`,
# `groupby().agg`, `crosstab`, `pivot_table`, `merge`) directamente sobre el DataFrame, para que se vea cómo se
# obtienen las tablas.
#
# Una advertencia de lectura que vale para todo el notebook: los datos son **observacionales**. Cuando decimos que
# el precio «se asocia» con algo, no afirmamos que ese algo lo cause. Las interpretaciones que siguen a cada figura
# se generan con cifras leídas de `findings.json` o calculadas en la propia celda; ninguna está escrita a mano.
#
# Se genera desde `notebooks/fuentes_py/01_eda.py` con `python scripts/ejecutar_notebooks.py`.

# %%
import sys
from pathlib import Path

sys.dont_write_bytecode = True
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from src import analisis, carga, config, formato as fmt, visualizacion as viz

%matplotlib inline
viz.configurar_estilo()
plt.rcParams["figure.dpi"] = 72  # imágenes incrustadas ligeras (las figuras del informe se guardan a 150 dpi)
pd.set_option("display.max_columns", 30)

df = pd.read_parquet(config.RUTA_PARQUET)
with open(config.RUTA_FINDINGS, encoding="utf-8") as f:
    findings = json.load(f)
with open(config.RUTA_CALIDAD, encoding="utf-8") as f:
    calidad = json.load(f)
D, E = findings["descriptivo"], findings["estadistica"]


def lectura(texto: str) -> None:
    """Muestra una interpretación en Markdown (las cifras van dentro del f-string, nunca a mano)."""
    display(Markdown(texto))


print(f"Dataset procesado: {fmt.n0(df.shape[0])} filas × {fmt.n0(df.shape[1])} columnas")

# %% [markdown]
# ## 1. Datos y requisitos mínimos
#
# El primer paso es comprobar que el conjunto de datos cumple los requisitos del proyecto: un volumen suficiente
# de filas y columnas y variedad de tipos de variable (numéricas, categóricas, fechas, texto y coordenadas). La
# clasificación por tipo se hace por código en `src/calidad.py` sobre el fichero original, no a ojo.

# %%
meta, dataset = findings["meta"], findings["dataset"]
print("Snapshot (fecha de la URL):", meta["fecha_snapshot_url"])
print("Fechas de scrape (last_scraped):", meta["last_scraped_min"][:10], "→", meta["last_scraped_max"][:10])
print("Licencia:", dataset["licencia"]["nombre"], "| verificada en la página:", dataset["licencia"]["verificada_en_pagina"])
tipos = pd.Series(dataset["tipos_raw"], name="columnas").sort_values(ascending=False)
tipos.to_frame().assign(pct=lambda t: (t["columnas"] / t["columnas"].sum() * 100).round(1))

# %%
lectura(f"**Lectura.** El fichero original tiene {fmt.n0(dataset['filas_raw'])} anuncios y "
        f"{fmt.n0(dataset['columnas_raw'])} columnas; la tabla anterior suma exactamente "
        f"{fmt.n0(tipos.sum())} columnas repartidas en {len(tipos)} tipos lógicos, entre ellos "
        f"{fmt.n0(dataset['tipos_raw']['identificador'])} identificadores. Predominan las numéricas "
        f"({fmt.n0(dataset['tipos_raw']['numerico'])}), lo esperable en un scrape con muchas métricas de "
        "disponibilidad y reseñas.")

# %% [markdown]
# ## 2. Calidad de datos antes y después de limpiar
#
# La limpieza sigue un orden documentado: perfilar, convertir tipos, derivar variables, eliminar columnas y aplicar
# reglas de calidad. Cada regla registra cuántas filas afecta. Comparar el perfil «antes» y «después» permite ver
# que la limpieza no ha perdido filas sin contabilizarlas y que los nulos que quedan tienen una decisión explícita
# (conservar, imputar con indicador o eliminar la columna).

# %%
resumen = pd.DataFrame({
    "antes": {k: calidad["antes"][k] for k in ("filas", "columnas", "nulos_total", "pct_nulos_total", "columnas_con_nulos")},
    "después": {k: calidad["despues"][k] for k in ("filas", "columnas", "nulos_total", "pct_nulos_total", "columnas_con_nulos")},
})
resumen

# %%
pd.Series(calidad["reglas"], name="filas afectadas").to_frame()

# %%
pd.DataFrame(calidad["politica_nulos"])[["columna", "nulos", "decision"]].head(20)

# %%
perfil_antes = {"nulos_por_columna_pct": (carga.leer_listings().isna().mean() * 100).to_dict()}
viz.fig_calidad_nulos(perfil_antes);

# %%
nulos = pd.Series(perfil_antes["nulos_por_columna_pct"])
lectura(f"**Lectura.** {fmt.n0((nulos == 100).sum())} columnas vienen completamente vacías en este snapshot "
        f"(en rojo) y se eliminan con su motivo. El precio falta en el {fmt.pct1(findings['limpieza']['precio']['pct_nulo'])} "
        f"de los anuncios, y el {fmt.pct1(findings['limpieza']['precio']['pct_nulos_sin_disponibilidad'])} de esos "
        "nulos corresponde a anuncios sin ningún día disponible: los nulos no son aleatorios, por eso el precio no "
        "se imputa y el análisis describe solo la oferta reservable.")

# %% [markdown]
# ## 3. Qué es exactamente el precio
#
# Antes de describir el precio hay que saber qué mide. En Inside Airbnb, `price` es el precio por noche de una
# **cotización concreta**: Airbnb calcula cuánto costaría una estancia en unas fechas determinadas (normalmente la
# primera ventana reservable, con la duración mínima del anuncio). Ese precio ya lleva aplicados los descuentos
# (semanal, mensual, de última hora) y no incluye tasas. Las cifras siguientes se calculan en la limpieza a partir
# del JSON `price_quote_raw`.

# %%
pn = findings["precio_naturaleza"]
pd.Series({k: v for k, v in pn.items() if k != "partidas_descuento"}, name="valor").to_frame()

# %%
lectura(f"**Lectura.** El {fmt.pct1(pn['pct_con_descuento'])} de las cotizaciones con precio incluye algún "
        f"descuento (mediana {fmt.pct1(pn['mediana_pct_descuento'])} del subtotal). La estancia cotizada coincide "
        f"con la estancia mínima en el {fmt.pct1(pn['pct_noches_igual_minimo'])} de los casos, y entre los anuncios "
        f"con estancia mínima de {config.MIN_NIGHTS_LARGA_ESTANCIA} noches o más el "
        f"{fmt.pct1(pn['pct_mensual_entre_min30'])} usa tarifa mensual. Es decir: el «precio por noche» mezcla "
        "tarifas de estancia corta con tarifas mensuales repartidas, algo que habrá que controlar.")

# %% [markdown]
# ## 4. Distribución del precio
#
# Se describen solo los anuncios con precio publicado (`precio_valido`). La tabla compara el precio, el precio sin
# outliers (vallas de Tukey sobre el logaritmo, que se marcan pero no se borran) y el logaritmo del precio, que es
# la escala de la regresión. Usamos `describe` de pandas y, para la asimetría y la curtosis, la función del
# proyecto; ambas se comprueban contra `findings.json`.

# %%
con_precio = df[df["precio_valido"]]
con_precio["price"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2)

# %%
desc = pd.DataFrame({
    "precio": analisis.descriptivos(con_precio["price"]),
    "precio sin outliers": analisis.descriptivos(con_precio.loc[~con_precio["es_outlier_precio"], "price"]),
    "ln(precio)": analisis.descriptivos(con_precio["log_precio"]),
})
# Comprobación: el notebook reproduce las mismas cifras que findings.json.
assert abs(desc.loc["mediana", "precio"] - D["precio"]["mediana"]) < 1e-9
assert abs(desc.loc["asimetria", "precio"] - D["precio"]["asimetria"]) < 1e-9
desc.round(2)

# %%
viz.fig_distribucion_precio(df);

# %%
lectura(f"**Lectura.** La mediana es {fmt.eur2(desc.loc['mediana', 'precio'])} y la media "
        f"{fmt.eur2(desc.loc['media', 'precio'])}: la media queda por encima porque la cola derecha es muy larga "
        f"(asimetría {fmt.n2(desc.loc['asimetria', 'precio'])}, máximo {fmt.eur0(desc.loc['max', 'precio'])}). "
        f"En logaritmo la asimetría baja a {fmt.n2(desc.loc['asimetria', 'ln(precio)'])}, por eso la regresión "
        "trabaja con `ln(precio)` y los contrastes de grupos usan rangos. La mediana es el resumen más honesto del "
        "«precio típico».")

# %% [markdown]
# ### Sensibilidad: sin estancias mensuales
#
# Como el precio de las estancias de un mes o más suele ser una tarifa mensual repartida, repetimos los
# descriptivos sin esos anuncios ni las cotizaciones mensuales. Si las conclusiones descriptivas cambiaran mucho,
# habría que separar los dos mercados en todo el análisis.

# %%
corta = con_precio[analisis.mascara_sin_mensuales(con_precio)]
comparacion = pd.DataFrame({"todos": con_precio["price"].describe(), "sin estancias mensuales": corta["price"].describe()})
comparacion.round(2)

# %%
lectura(f"**Lectura.** Al quitar {fmt.n0(len(con_precio) - len(corta))} anuncios de estancia mensual, la mediana "
        f"pasa de {fmt.eur2(con_precio['price'].median())} a {fmt.eur2(corta['price'].median())}. Los precios de "
        "estancia corta son más altos por noche, como cabía esperar; el dashboard excluye por defecto ese mercado.")

# %% [markdown]
# ## 5. Otras variables clave
#
# Capacidad, estancia mínima, disponibilidad, reseñas y valoración. Las variables de conteo suelen ser asimétricas y
# con valores extremos (estancias mínimas de más de un año, anuncios con cientos de reseñas), lo que vuelve a
# aconsejar medianas y rangos intercuartílicos.

# %%
pd.DataFrame({c: analisis.descriptivos(df[c]) for c in
              ["accommodates", "minimum_nights", "availability_365", "number_of_reviews", "review_scores_rating"]}).round(2)

# %% [markdown]
# ## 6. Precio por tipo de habitación, distrito, barrio y capacidad
#
# Aquí aparecen las primeras operaciones de agregación explícitas. `groupby().agg` calcula n, mediana, media y
# cuartiles por grupo; `value_counts` muestra cuántos anuncios hay de cada tipo (incluidos los que no tienen precio).

# %%
df["tipo_habitacion"].value_counts().to_frame("anuncios").assign(
    pct=lambda t: (t["anuncios"] / t["anuncios"].sum() * 100).round(1))

# %%
por_tipo = (con_precio.groupby("tipo_habitacion", observed=True)["price"]
            .agg(n="size", mediana="median", media="mean", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
            .sort_values("mediana", ascending=False))
assert por_tipo["mediana"].round(6).tolist() == [round(g["mediana"], 6) for g in D["por_tipo"]]
por_tipo.round(2)

# %%
viz.fig_precio_por_tipo(df);

# %%
mayor, menor = por_tipo.index[0], por_tipo.index[-1]
lectura(f"**Lectura.** La mediana más alta es la de {mayor.lower()} ({fmt.eur2(por_tipo.loc[mayor, 'mediana'])}, "
        f"n = {fmt.n0(por_tipo.loc[mayor, 'n'])}) y la más baja la de {menor.lower()} "
        f"({fmt.eur2(por_tipo.loc[menor, 'mediana'])}, n = {fmt.n0(por_tipo.loc[menor, 'n'])}). Las cajas casi no se "
        "solapan entre alojamiento entero y habitación privada; los tipos con muy pocos anuncios se describen pero "
        "se agrupan para la inferencia.")

# %% [markdown]
# ### Distritos
#
# Para los distritos usamos `pivot_table`, que cruza distrito y tipo de habitación con la mediana del precio. Así
# se ve si un distrito es caro porque todo es caro o porque tiene más alojamientos enteros. Solo se ordenan los
# distritos con al menos `MIN_N_GRUPO` anuncios con precio.

# %%
pivote = pd.pivot_table(con_precio, index="neighbourhood_group_cleansed", columns="tipo_habitacion_agrupado",
                        values="price", aggfunc="median", observed=True)
pivote["n"] = con_precio.groupby("neighbourhood_group_cleansed", observed=True).size()
pivote = pivote[pivote["n"] >= config.MIN_N_GRUPO].sort_values("Alojamiento entero", ascending=False)
pivote.round(1)

# %%
viz.fig_precio_por_distrito(findings);

# %%
d0, d1, db = D["por_distrito"][0], D["por_distrito"][1], D["distrito_mas_barato"]
empate = E["top_distritos_empate"]
lectura(f"**Lectura.** {d0['grupo']} ({fmt.eur2(d0['mediana'])}) y {d1['grupo']} ({fmt.eur2(d1['mediana'])}) tienen "
        f"las medianas más altas y {db['grupo']} ({fmt.eur2(db['mediana'])}, n = {fmt.n0(db['n'])}) la más baja. "
        + (f"Entre los dos primeros no hay diferencia estadística (Dunn-Bonferroni p = {fmt.n3(empate['p_ajustado'])}). "
           if empate["empate"] else "")
        + "La tabla cruzada muestra que el orden se mantiene en buena medida dentro del alojamiento entero, así que "
          "no se explica solo por la mezcla de tipos.")

# %% [markdown]
# ### Barrios y capacidad
#
# Con más de un centenar de barrios, muchos tienen pocos anuncios y su mediana es inestable. La figura solo muestra barrios con al
# menos `MIN_N_GRUPO` anuncios con precio y el informe da intervalos bootstrap para los extremos.

# %%
viz.fig_precio_por_barrio(df, config.MIN_N_GRUPO);

# %%
b = E["bootstrap"]["barrios_extremos"]
lectura(f"**Lectura.** El barrio de mediana más alta es {b[0]['grupo']} ({fmt.eur2(b[0]['mediana'])}, n = "
        f"{fmt.n0(b[0]['n'])}; IC 95 % por anfitrión {fmt.eur0(b[0]['ic_inf_cluster'])}–{fmt.eur0(b[0]['ic_sup_cluster'])}) "
        f"y el de mediana más baja {b[2]['grupo']} ({fmt.eur2(b[2]['mediana'])}, n = {fmt.n0(b[2]['n'])}). "
        + ("Sus intervalos se solapan con los del segundo barrio de cada lista, así que el orden exacto no es "
           "distinguible." if not (E["bootstrap"]["barrio_top_distinguible"] and E["bootstrap"]["barrio_bottom_distinguible"])
           else "Ambos se distinguen del segundo de su lista."))

# %%
viz.fig_precio_por_capacidad(findings);

# %%
cap = pd.DataFrame(D["por_capacidad"]).set_index("grupo")
lectura(f"**Lectura.** La mediana sube con la capacidad, desde {fmt.eur2(cap['mediana'].iloc[0])} en "
        f"{cap.index[0]} hasta {fmt.eur2(cap['mediana'].iloc[-1])} en {cap.index[-1]}. La capacidad es, además, la "
        "variable más correlacionada con el precio (sección 10).")

# %% [markdown]
# ### Fecha de la cotización
#
# Como cada anuncio se cotiza para fechas distintas, el precio mezcla temporadas. La tabla agrupa por mes de
# entrada de la cotización.

# %%
por_mes = con_precio.groupby("mes_cotizacion")["price"].agg(n="size", mediana="median")
por_mes.round(2)

# %%
lectura(f"**Lectura.** La mayoría de cotizaciones se concentra en los primeros meses "
        f"({fmt.pct1(D['mes_cotizacion']['pct_primeros_dos_meses'])} en los dos primeros). Las medianas mensuales "
        f"van de {fmt.eur2(D['mes_cotizacion']['mes_mediana_min']['mediana'])} a "
        f"{fmt.eur2(D['mes_cotizacion']['mes_mediana_max']['mediana'])} entre los meses con suficientes anuncios; "
        "por eso los modelos del notebook 02 incluyen el mes como control.")

# %% [markdown]
# ## 7. Composición de anfitriones
#
# Un mismo anfitrión puede tener muchos anuncios. Con `groupby("host_id").size()` contamos anuncios por anfitrión y
# con `merge` añadimos ese recuento a cada anuncio para comparar el precio según el tamaño de la cartera. Tener
# varios anuncios mide concentración de la oferta, no prueba gestión profesional.

# %%
por_host = df.groupby("host_id").size().rename("anuncios_del_host").reset_index()
por_host["anuncios_del_host"].describe().round(2)

# %%
con_host = con_precio.merge(por_host, on="host_id", how="left", validate="many_to_one")
con_host.groupby("segmento_host", observed=True).agg(
    anuncios=("price", "size"), mediana=("price", "median"), pct_superhost=("host_is_superhost", "mean")).round(3)

# %%
viz.fig_composicion_hosts(df, findings);

# %%
h = D["hosts"]
lectura(f"**Lectura.** Hay {fmt.n0(h['n_hosts'])} anfitriones; el {fmt.pct1(h['pct_hosts_un_anuncio'])} tiene un solo "
        f"anuncio, pero el {fmt.pct1(h['pct_anuncios_multianuncio'])} de los anuncios pertenece a cuentas con varios y "
        f"el mayor tiene {fmt.n0(h['anuncios_por_host_max'])}. Es una oferta concentrada: por eso los contrastes del "
        "notebook 02 tienen en cuenta la dependencia entre anuncios del mismo anfitrión.")

# %% [markdown]
# ### Tipo de habitación y superhost: tabla cruzada
#
# `crosstab` con `normalize="index"` da el porcentaje de superhosts dentro de cada tipo de habitación. Es la tabla
# que después se contrasta con chi-cuadrado.

# %%
pd.crosstab(df["tipo_habitacion_agrupado"], df["host_is_superhost"], normalize="index", margins=True).mul(100).round(1)

# %% [markdown]
# ## 8. Mapas de densidad y de precio
#
# La densidad muestra dónde hay más anuncios; la coropleta, la mediana del precio por barrio (en gris los barrios
# con muy pocos anuncios). Las coordenadas están desplazadas por Airbnb unos cientos de metros, así que los mapas
# sirven para patrones de zona, no para ubicaciones exactas.

# %%
geojson = carga.leer_geojson()
ciudad = config.CIUDADES[config.CIUDAD_POR_DEFECTO]
viz.fig_mapa_densidad(df, geojson, ciudad.centro, ciudad.centro_nombre);

# %%
viz.fig_mapa_precio_barrio(df, geojson, config.MIN_N_BARRIO_MAPA);

# %%
centro = (df["neighbourhood_group_cleansed"] == ciudad.distrito_referencia).mean() * 100
lectura(f"**Lectura.** El distrito {ciudad.distrito_referencia} concentra el {fmt.pct1(centro)} de los anuncios y la "
        "densidad cae rápidamente al alejarse de la Puerta del Sol. Los barrios de mediana más alta están en el "
        "centro y el norte; la relación con la distancia se cuantifica con la regresión del notebook 02.")

# %% [markdown]
# ## 9. Días no disponibles e ingreso hipotético
#
# `365 − availability_365` cuenta los días **no disponibles en el próximo año**, que incluyen reservas y bloqueos
# del anfitrión: no es ocupación. Inside Airbnb estima la ocupación de los **últimos doce meses** a partir de
# reseñas. Son ventanas distintas, así que la comparación es orientativa. Ambas medias se calculan sobre las mismas
# filas.

# %%
oi = D["ocupacion_ingresos"]
pd.Series({k: v for k, v in oi.items() if not isinstance(v, str)}, name="valor").to_frame().round(3)

# %%
viz.fig_ocupacion_ingresos(df, findings);

# %%
lectura(f"**Lectura.** Los días no disponibles promedian un {fmt.pct1(oi['no_disponible_media'])} frente a una "
        f"ocupación estimada por Inside Airbnb del {fmt.pct1(oi['ocupacion_inside_airbnb_media'])} "
        f"(n = {fmt.n0(oi['n_comparacion_ocupacion'])} en ambos casos). La diferencia mezcla bloqueos y ventana "
        f"temporal; la correlación de Spearman del ingreso hipotético con el de Inside Airbnb es "
        f"{fmt.n2(oi['spearman_propio_vs_inside'])}, moderada. Ninguna de las dos es un ingreso observado.")

# %% [markdown]
# ## 10. Amenities y correlaciones
#
# Los indicadores de amenities se calculan elemento a elemento sobre la lista JSON (así «Dishwasher» no cuenta como
# lavadora ni «Pool table» como piscina). La prima de precio de cada amenity es una diferencia bruta de medianas,
# que puede reflejar tipo de alojamiento o zona.

# %%
pd.DataFrame(D["amenities"])[["nombre", "prevalencia_pct", "mediana_con", "mediana_sin", "prima_pct"]].round(1)

# %%
viz.fig_amenities(findings);

# %%
am = pd.DataFrame(D["amenities"])
lectura(f"**Lectura.** Las amenities más frecuentes son {', '.join(am['nombre'].str.lower().head(3))} "
        f"(entre {fmt.pct1(am['prevalencia_pct'].iloc[2])} y {fmt.pct1(am['prevalencia_pct'].iloc[0])}). Las primas "
        "brutas más grandes no deben leerse como el valor de añadir esa amenity: la regresión las compara a igualdad "
        "de tipo, capacidad y zona.")

# %%
pd.DataFrame(D["correlaciones"]["con_precio"]).round(3)

# %%
viz.fig_correlaciones(findings);

# %%
c0 = D["correlaciones"]["con_precio"][0]
lectura(f"**Lectura.** La variable más correlacionada con el precio es `{c0['variable']}` (ρ de Spearman = "
        f"{fmt.n2(c0['spearman'])}; Pearson sobre el precio en bruto = {fmt.n2(c0['pearson'])}). Pearson en escala "
        "original queda muy por debajo de Spearman porque los precios extremos dominan la covarianza; con este "
        "precio tan asimétrico, rangos o logaritmos son la elección adecuada.")

# %% [markdown]
# ## Conclusión del EDA
#
# El precio por noche es muy asimétrico, con una cola de precios extremos que aleja la media de la mediana, y es el
# precio de una cotización concreta: ya incluye descuentos, depende de la duración cotizada (a menudo la estancia
# mínima) y de la fecha. Las diferencias más visibles aparecen entre tipos de alojamiento, por capacidad y entre
# zonas, con los distritos céntricos en la parte alta; los barrios pequeños tienen medianas inestables.
# La oferta está concentrada en cuentas con varios anuncios, lo que obliga a tratar con cuidado la independencia
# de las observaciones. Los días no disponibles y el ingreso hipotético son proxies con ventanas temporales
# distintas a las de Inside Airbnb, útiles solo como orientación.
#
# Estas observaciones son **hipótesis**: el notebook `02_analisis_estadistico.ipynb` las contrasta formalmente, con
# tamaños de efecto, sensibilidad a la dependencia por anfitrión y modelos multivariantes. La interpretación
# completa y las limitaciones están en `reports/informe.md`.
