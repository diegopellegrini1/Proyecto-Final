# %% [markdown]
# # 02 · Análisis estadístico: Airbnb en Madrid
#
# Contrastes de hipótesis, tamaños de efecto, intervalos bootstrap y regresión. Para cada test se muestran
# **H0 y H1 antes de ejecutarlo** (definidas en `src/estadistica.py`), α = 0,05, los supuestos comprobados y la
# conclusión. Se usan exactamente las mismas funciones que el pipeline, con la misma semilla, y se comprueba que
# los resultados coinciden con `reports/findings.json`.

# %%
import sys
from pathlib import Path

sys.dont_write_bytecode = True
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import json

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from src import analisis, config, estadistica as est, formato as fmt, visualizacion as viz

%matplotlib inline
viz.configurar_estilo()
df = pd.read_parquet(config.RUTA_PARQUET)
with open(config.RUTA_FINDINGS, encoding="utf-8") as f:
    findings = json.load(f)
cp = df[df["precio_valido"]]
rng = np.random.default_rng(config.SEMILLA)


def hipotesis(clave: str) -> None:
    """Muestra H0 y H1 de un contraste antes de ejecutarlo."""
    h0, h1 = est.HIPOTESIS[clave]
    display(Markdown(f"**H0:** {h0}  \n**H1:** {h1}  \n**α = {fmt.n2(config.ALFA)}**"))


def mostrar(resultado: est.ResultadoTest) -> None:
    """Resume un ResultadoTest."""
    efecto = resultado.tamano_efecto
    display(Markdown(
        f"- Estadístico = {fmt.n3(resultado.estadistico)}, {fmt.p_texto(resultado.p_valor)}\n"
        f"- {efecto['nombre']} = {fmt.n3(efecto['valor'])} (efecto {efecto['interpretacion']})\n"
        f"- Supuestos: {resultado.supuestos}\n"
        f"- **Conclusión:** {resultado.conclusion}"))

# %% [markdown]
# ## 1. Normalidad del precio y de ln(precio)
#
# Shapiro-Wilk sobre una muestra aleatoria acotada (semilla fija) y D'Agostino-K² sobre todos los datos.

# %%
hipotesis("normalidad")
r_precio = est.test_normalidad(cp["price"], "el precio", rng)
r_log = est.test_normalidad(cp["log_precio"], "el logaritmo del precio", rng)
mostrar(r_precio)
mostrar(r_log)
assert np.isclose(r_precio.estadistico, findings["estadistica"]["normalidad_precio"]["estadistico"])

# %%
viz.fig_distribucion_precio(df);

# %% [markdown]
# ## 2. Mann-Whitney U: superhosts frente a no superhosts

# %%
hipotesis("mann_whitney_superhost")
sh = cp["host_is_superhost"]
r_mw = est.mann_whitney(cp.loc[sh, "price"], cp.loc[~sh, "price"], "superhosts", "no superhosts",
                        "mann_whitney_superhost")
mostrar(r_mw)
assert np.isclose(r_mw.tamano_efecto["valor"], findings["estadistica"]["mann_whitney_superhost"]["tamano_efecto"]["valor"])

# %% [markdown]
# **Sensibilidad (independencia):** el pipeline repite el contraste con un anuncio por anfitrión.

# %%
s = findings["estadistica"]["sensibilidad_un_anuncio_por_host"]
print("n =", fmt.n0(s["n"]), "| r =", fmt.n3(s["mann_whitney_superhost"]["tamano_efecto"]["valor"]),
      "|", fmt.p_texto(s["mann_whitney_superhost"]["p_valor"]), "| misma decisión:", s["misma_decision_superhost"])

# %% [markdown]
# ## 3. Kruskal-Wallis: precio por tipo de habitación y por distrito
#
# Las categorías con pocos anuncios con precio (compartida y hotel, por debajo de `config.MIN_N_CATEGORIA_RARA`)
# se agrupan; los grupos por debajo de `config.MIN_N_GRUPO` se excluyen de la inferencia.

# %%
print("Umbral de agrupación:", config.MIN_N_CATEGORIA_RARA, "| mínimo por grupo:", config.MIN_N_GRUPO)

# %%
hipotesis("kruskal_tipo")
r_kt = est.kruskal(cp, "price", "tipo_habitacion_agrupado", "kruskal_tipo", "tipos de habitación")
mostrar(r_kt)

# %%
hipotesis("kruskal_distrito")
r_kd = est.kruskal(cp, "price", "neighbourhood_group_cleansed", "kruskal_distrito", "distritos")
mostrar(r_kd)
assert np.isclose(r_kd.estadistico, findings["estadistica"]["kruskal_distrito"]["estadistico"])

# %% [markdown]
# ### Post-hoc de Dunn con corrección de Bonferroni

# %%
hipotesis("posthoc")
ph = findings["estadistica"]["posthoc_distrito"]
print(ph["metodo"], "|", ph["n_pares_significativos"], "de", ph["n_pares"], "pares significativos")
pd.DataFrame(ph["pares"]).sort_values("p_ajustado").head(10).round(4)

# %%
viz.fig_posthoc_distritos(findings);

# %% [markdown]
# ## 4. Chi-cuadrado con V de Cramér

# %%
for clave, fila, col, etiqueta in [
    ("chi2_tipo_superhost", "tipo_habitacion_agrupado", "host_is_superhost", "tipo de habitación y ser superhost"),
    ("chi2_multi_tipo", "es_multianuncio", "tipo_habitacion_agrupado", "ser multi-anuncio y tipo de habitación"),
]:
    hipotesis(clave)
    mostrar(est.chi_cuadrado(df, fila, col, clave, etiqueta))

# %%
hipotesis("chi2_distrito_tipo")
r = findings["estadistica"]["chi2_distrito_tipo"]
print("V de Cramér =", fmt.n3(r["tamano_efecto"]["valor"]), "|", fmt.p_texto(r["p_valor"]),
      "| % celdas con esperada < 5:", fmt.pct1(r["supuestos"]["pct_celdas_esperada_menor_5"]))
print("Conclusión:", r["conclusion"])

# %% [markdown]
# ## 5. Intervalos de confianza bootstrap de la mediana

# %%
b = findings["estadistica"]["bootstrap"]
pd.concat([pd.DataFrame(b["tipo"]), pd.DataFrame(b["distritos"])]).round(2)

# %%
viz.fig_ic_bootstrap(findings);

# %% [markdown]
# ## 6. Regresión OLS sobre ln(precio)
#
# Errores agrupados por anfitrión (y HC3 como comparación), Breusch-Pagan, Jarque-Bera, VIF y coeficientes
# interpretados como cambio porcentual `100·(e^β − 1)`.

# %%
hipotesis("ols")
ols = est.ajustar_ols(df, config.CIUDADES[config.CIUDAD_POR_DEFECTO].distrito_referencia)
artefactos = {"residuos": ols.pop("_residuos"), "ajustados": ols.pop("_ajustados")}
print("n efectivo:", fmt.n0(ols["n_efectivo"]), "| R² ajustado:", fmt.n3(ols["r2_ajustado"]))
print("Diagnóstico:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ols["diagnostico"].items()})
assert np.isclose(ols["r2_ajustado"], findings["estadistica"]["ols"]["r2_ajustado"])
coef = pd.DataFrame(ols["coeficientes"])
coef.loc[~coef["es_distrito"], ["nombre", "pct", "ic_inf_pct", "ic_sup_pct", "p_cluster", "p_hc3"]].round(4)

# %%
pd.DataFrame(ols["vif"])[["nombre", "vif"]].round(2)

# %%
viz.fig_ols_residuos(artefactos);

# %%
viz.fig_ols_coeficientes(findings);

# %% [markdown]
# ## 7. Modelo predictivo (complementario)
#
# HistGradientBoosting con lista blanca de variables (sin fuga del precio), partición train/test con semilla y
# línea base. La importancia por permutación es predictiva, no causal.

# %%
ml = findings["estadistica"]["ml"]
if ml["ejecutado"]:
    print("R² test:", fmt.n3(ml["r2_test"]), "| R² línea base:", fmt.n3(ml["r2_base"]),
          "| MAE ≈", fmt.eur0(ml["mae_eur_test"]), "(base", fmt.eur0(ml["mae_eur_base"]) + ")")
    viz.fig_importancia_variables(findings)
else:
    print("Modelo no ejecutado:", ml["motivo"])

# %% [markdown]
# ## Conclusión
#
# Las diferencias por tipo de habitación y distrito son significativas y relevantes; la de superhost es significativa
# pero despreciable y desaparece al controlar en la regresión. La interpretación completa y las limitaciones están en
# `reports/informe.md`.
