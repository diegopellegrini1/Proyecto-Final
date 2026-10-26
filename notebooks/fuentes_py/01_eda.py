# %% [markdown]
# # 01 · Análisis exploratorio (EDA): Airbnb en Madrid
#
# Este notebook recorre el análisis exploratorio del proyecto. **No repite la limpieza**: lee el dataset
# procesado (`data/processed/listings_limpio.parquet`) y los resultados (`reports/findings.json`) que genera
# `python run_pipeline.py`, y llama a las mismas funciones de `src/` que el pipeline.
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

from src import analisis, carga, config, formato as fmt, visualizacion as viz

%matplotlib inline
viz.configurar_estilo()
pd.set_option("display.max_columns", 30)

df = pd.read_parquet(config.RUTA_PARQUET)
with open(config.RUTA_FINDINGS, encoding="utf-8") as f:
    findings = json.load(f)
with open(config.RUTA_CALIDAD, encoding="utf-8") as f:
    calidad = json.load(f)
print(f"Dataset procesado: {fmt.n0(df.shape[0])} filas × {fmt.n0(df.shape[1])} columnas")

# %% [markdown]
# ## 1. Datos y requisitos mínimos
#
# Snapshot, licencia verificada y comprobación de los umbrales del dataset (calculados por `src/calidad.py`
# sobre el fichero original).

# %%
meta, dataset = findings["meta"], findings["dataset"]
print("Snapshot (fecha de la URL):", meta["fecha_snapshot_url"])
print("Fechas de scrape (last_scraped):", meta["last_scraped_min"][:10], "→", meta["last_scraped_max"][:10])
print("Licencia:", dataset["licencia"]["nombre"], "| verificada en la página:", dataset["licencia"]["verificada_en_pagina"])
print("Filas y columnas originales:", fmt.n0(dataset["filas_raw"]), "×", dataset["columnas_raw"])
pd.Series(dataset["tipos_raw"], name="columnas por tipo lógico").to_frame()

# %%
pd.DataFrame(dataset["requisitos"]["tipos_presentes"], index=["presente"]).T

# %% [markdown]
# ## 2. Calidad de datos antes y después de limpiar
#
# El detalle de cada regla (conversiones, columnas eliminadas con su motivo, política de nulos y filas afectadas)
# está en `reports/calidad_datos.json`.

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
pd.Series(calidad["columnas_eliminadas"], name="motivo").to_frame()

# %%
perfil_antes = {"nulos_por_columna_pct": (carga.leer_listings().isna().mean() * 100).to_dict()}
viz.fig_calidad_nulos(perfil_antes);

# %% [markdown]
# ## 3. Distribución del precio
#
# Se describen solo los anuncios con precio publicado (`precio_valido`). Los precios nulos se concentran en
# anuncios sin disponibilidad y no se imputan.

# %%
con_precio = df[df["precio_valido"]]
desc = pd.DataFrame({
    "precio": analisis.descriptivos(con_precio["price"]),
    "precio sin outliers": analisis.descriptivos(con_precio.loc[~con_precio["es_outlier_precio"], "price"]),
    "ln(precio)": analisis.descriptivos(con_precio["log_precio"]),
})
# Comprobación: el notebook reproduce las mismas cifras que findings.json.
assert abs(desc.loc["mediana", "precio"] - findings["descriptivo"]["precio"]["mediana"]) < 1e-9
assert abs(desc.loc["asimetria", "precio"] - findings["descriptivo"]["precio"]["asimetria"]) < 1e-9
desc.round(2)

# %%
viz.fig_distribucion_precio(df);

# %%
print("Vallas IQR sobre ln(precio):", fmt.eur2(findings["limpieza"]["outliers"]["valla_inf"]), "y",
      fmt.eur2(findings["limpieza"]["outliers"]["valla_sup"]))
print("Outliers marcados (no borrados):", fmt.n0(findings["limpieza"]["outliers"]["n_total"]))

# %% [markdown]
# ## 4. Otras variables clave

# %%
pd.DataFrame({c: analisis.descriptivos(df[c]) for c in
              ["accommodates", "minimum_nights", "availability_365", "number_of_reviews", "review_scores_rating"]}).round(2)

# %% [markdown]
# ## 5. Precio por tipo de habitación, distrito, barrio y capacidad

# %%
pd.DataFrame(findings["descriptivo"]["por_tipo"]).round(2)

# %%
viz.fig_precio_por_tipo(df);

# %%
viz.fig_precio_por_distrito(findings);

# %%
viz.fig_precio_por_barrio(df, config.MIN_N_GRUPO);

# %%
viz.fig_precio_por_capacidad(findings);

# %% [markdown]
# ## 6. Composición de anfitriones

# %%
pd.Series(findings["descriptivo"]["hosts"]).drop(["segmentos", "segmentos_pct"]).to_frame("valor")

# %%
viz.fig_composicion_hosts(df, findings);

# %% [markdown]
# ## 7. Mapas de densidad y de precio

# %%
geojson = carga.leer_geojson()
ciudad = config.CIUDADES[config.CIUDAD_POR_DEFECTO]
viz.fig_mapa_densidad(df, geojson, ciudad.centro, ciudad.centro_nombre);

# %%
viz.fig_mapa_precio_barrio(df, geojson, config.MIN_N_BARRIO_MAPA);

# %% [markdown]
# ## 8. Ocupación e ingresos estimados
#
# `365 − availability_365` **no es ocupación real**; se compara con la estimación de Inside Airbnb.

# %%
pd.Series(findings["descriptivo"]["ocupacion_ingresos"]).to_frame("valor").round(3)

# %%
viz.fig_ocupacion_ingresos(df, findings);

# %% [markdown]
# ## 9. Amenities

# %%
pd.DataFrame(findings["descriptivo"]["amenities"]).round(1)

# %%
viz.fig_amenities(findings);

# %% [markdown]
# ## 10. Correlaciones (Pearson y Spearman)
#
# Spearman es más adecuado con un precio tan asimétrico; Pearson sobre el precio original queda dominado por los
# valores extremos.

# %%
pd.DataFrame(findings["descriptivo"]["correlaciones"]["con_precio"]).round(3)

# %%
viz.fig_correlaciones(findings);

# %% [markdown]
# ## Conclusión del EDA
#
# El precio es muy asimétrico, depende sobre todo del tipo de alojamiento, la capacidad y la ubicación, y la oferta
# está muy concentrada en anfitriones con varios anuncios. Estas hipótesis se contrastan formalmente en
# `02_analisis_estadistico.ipynb`. Las cifras y su interpretación completa están en `reports/informe.md`.
