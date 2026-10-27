# %% [markdown]
# # 02 · Análisis estadístico: Airbnb en Madrid
#
# ## Introducción
#
# El EDA dejó varias hipótesis: el precio cambia mucho con el tipo de alojamiento y con la zona, los superhosts
# parecen cobrar algo más y la oferta está concentrada en cuentas con muchos anuncios. Este notebook las contrasta
# formalmente. Para cada contraste se muestran **H0 y H1 antes de ejecutarlo** (definidas en `src/estadistica.py`),
# el nivel de significación α, los supuestos comprobados, el **tamaño del efecto** y la conclusión.
#
# Tres ideas guían la lectura. Primera: con miles de anuncios con precio, cualquier diferencia mínima
# sale «significativa», así que lo que importa es el tamaño del efecto. Segunda: los anuncios de un mismo anfitrión
# no son independientes; por eso los p-valores por anuncio se acompañan de una sensibilidad con un anuncio por
# anfitrión, intervalos bootstrap por anfitrión y errores agrupados en la regresión. Tercera: todo es
# observacional; los coeficientes describen asociaciones manteniendo constantes otras variables, no efectos causales.
#
# Se usan exactamente las mismas funciones que el pipeline, con la misma semilla, y se comprueba con `assert` que
# los resultados coinciden con `reports/findings.json`. Las interpretaciones tras cada resultado se generan con
# cifras del propio cálculo o de `findings.json`.

# %%
import sys
from pathlib import Path

sys.dont_write_bytecode = True
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from src import config, estadistica as est, formato as fmt, visualizacion as viz

%matplotlib inline
viz.configurar_estilo()
plt.rcParams["figure.dpi"] = 72  # imágenes incrustadas ligeras
df = pd.read_parquet(config.RUTA_PARQUET)
with open(config.RUTA_FINDINGS, encoding="utf-8") as f:
    findings = json.load(f)
E = findings["estadistica"]
cp = df[df["precio_valido"]]
rng = np.random.default_rng(config.SEMILLA)


def hipotesis(clave: str) -> None:
    """Muestra H0 y H1 de un contraste antes de ejecutarlo."""
    h0, h1 = est.HIPOTESIS[clave]
    display(Markdown(f"**H0:** {h0}  \n**H1:** {h1}  \n**α = {fmt.n2(config.ALFA)}**"))


def mostrar(resultado: est.ResultadoTest, decimales_efecto: int = 3) -> None:
    """Resume un ResultadoTest."""
    efecto = resultado.tamano_efecto
    display(Markdown(
        f"- Estadístico = {fmt.n3(resultado.estadistico)}, {fmt.p_texto(resultado.p_valor)}\n"
        f"- {efecto['nombre']} = {fmt.numero(efecto['valor'], decimales_efecto)} (efecto {efecto['interpretacion']})\n"
        f"- Supuestos: {resultado.supuestos}\n"
        f"- **Conclusión:** {resultado.conclusion}"))


def lectura(texto: str) -> None:
    """Interpretación en Markdown con cifras calculadas (nunca escritas a mano)."""
    display(Markdown(texto))

# %% [markdown]
# ## 1. Normalidad del precio y de ln(precio)
#
# Antes de elegir contrastes conviene saber si el precio es aproximadamente normal. Se usa Shapiro-Wilk sobre una
# muestra aleatoria acotada (con muestras enormes el test es demasiado sensible) y D'Agostino-K² sobre todos los
# datos. La decisión no se apoya solo en el p-valor, sino también en la asimetría y en los gráficos.

# %%
hipotesis("normalidad")
r_precio = est.test_normalidad(cp["price"], "el precio", rng)
r_log = est.test_normalidad(cp["log_precio"], "el logaritmo del precio", rng)
mostrar(r_precio)
mostrar(r_log)
assert np.isclose(r_precio.estadistico, E["normalidad_precio"]["estadistico"])

# %%
viz.fig_distribucion_precio(df);

# %%
lectura(f"**Lectura.** Se rechaza la normalidad del precio (asimetría {fmt.n2(r_precio.tamano_efecto['valor'])}) y "
        f"también la de `ln(precio)`, aunque su asimetría es solo {fmt.n2(r_log.tamano_efecto['valor'])}. Decisión: "
        "comparar grupos con contrastes de rangos (Mann-Whitney, Kruskal-Wallis) y modelar `ln(precio)` en la "
        "regresión con errores robustos.")

# %% [markdown]
# ## 2. Mann-Whitney U: superhosts frente a no superhosts
#
# Mann-Whitney compara si un grupo tiende a tener valores más altos que el otro. El tamaño de efecto es la
# correlación rank-biserial: 0 indica que no hay tendencia y ±1 que un grupo supera siempre al otro.

# %%
hipotesis("mann_whitney_superhost")
sh = cp["host_is_superhost"]
r_mw = est.mann_whitney(cp.loc[sh, "price"], cp.loc[~sh, "price"], "superhosts", "no superhosts",
                        "mann_whitney_superhost")
mostrar(r_mw, decimales_efecto=4)
assert np.isclose(r_mw.tamano_efecto["valor"], E["mann_whitney_superhost"]["tamano_efecto"]["valor"])

# %%
umbral = E["umbrales_efecto"]["rank_biserial"][0]
lectura(f"**Lectura.** La diferencia es significativa pero el efecto es despreciable: r = "
        f"{fmt.n4(r_mw.tamano_efecto['valor'])}, justo por debajo del umbral de {fmt.n1(umbral)} que separa "
        f"despreciable de pequeño (mediana {fmt.eur2(r_mw.extra['mediana_a'])} frente a "
        f"{fmt.eur2(r_mw.extra['mediana_b'])}). Es una asociación bruta: no controla tipo ni zona.")

# %% [markdown]
# ### Sensibilidad: un anuncio por anfitrión
#
# Para comprobar que la dependencia entre anuncios del mismo anfitrión no fabrica los resultados, el pipeline
# repite **todos** los contrastes principales con un anuncio elegido al azar por anfitrión.

# %%
s = E["sensibilidad_un_anuncio_por_host"]
pd.DataFrame({
    "decisión se mantiene": s["decisiones"],
    "p (un anuncio por host)": {k: s[k if k not in ("superhost", "tipo", "distrito") else
                                    {"superhost": "mann_whitney_superhost", "tipo": "kruskal_tipo",
                                     "distrito": "kruskal_distrito"}[k]]["p_valor"] for k in s["decisiones"]},
})

# %%
lectura(f"**Lectura.** Con {fmt.n0(s['n'])} anuncios (uno por anfitrión con precio), "
        + ("todas las decisiones se mantienen" if s["todas_mismas_decisiones"] else "alguna decisión cambia")
        + f"; el post-hoc de distritos conserva {s['posthoc_distrito']['n_pares_significativos']} de "
          f"{s['posthoc_distrito']['n_pares']} pares significativos, menos que con todos los anuncios, lo que "
          "confirma que los p-valores por anuncio son algo optimistas.")

# %% [markdown]
# ## 3. Kruskal-Wallis: precio por tipo de habitación y por distrito
#
# Kruskal-Wallis generaliza Mann-Whitney a más de dos grupos. Su tamaño de efecto, ε², se interpreta con umbrales
# convencionales (pequeño, mediano, grande). Las categorías con pocos anuncios con precio (compartida y hotel) se
# agrupan, y los grupos por debajo del mínimo se excluyen de la inferencia.

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
assert np.isclose(r_kd.estadistico, E["kruskal_distrito"]["estadistico"])

# %%
lectura(f"**Lectura.** El tipo de habitación tiene un efecto {r_kt.tamano_efecto['interpretacion']} "
        f"(ε² = {fmt.n3(r_kt.tamano_efecto['valor'])}) y el distrito un efecto "
        f"{r_kd.tamano_efecto['interpretacion']} (ε² = {fmt.n3(r_kd.tamano_efecto['valor'])}). Al ser la misma medida "
        "sobre la misma muestra, la comparación es homogénea: entre tipos hay más diferencia que entre distritos.")

# %% [markdown]
# ### Post-hoc de Dunn con corrección de Bonferroni
#
# Un Kruskal-Wallis significativo dice que algún grupo difiere, no cuál. Dunn compara cada par y Bonferroni ajusta
# los p-valores por el número de pares, para que la tasa de falsos positivos de la familia no se dispare.

# %%
hipotesis("posthoc")
ph = E["posthoc_distrito"]
print(ph["metodo"], "|", ph["n_pares_significativos"], "de", ph["n_pares"], "pares significativos")
pd.DataFrame(ph["pares"]).sort_values("p_ajustado", ascending=False).head(10).round(4)

# %%
viz.fig_posthoc_distritos(findings);

# %%
emp = E["top_distritos_empate"]
lectura(f"**Lectura.** {ph['n_pares_significativos']} de {ph['n_pares']} pares de distritos difieren. Los dos distritos "
        f"de mediana más alta, {emp['a']} y {emp['b']}, tienen p ajustado = {fmt.n3(emp['p_ajustado'])}: "
        + ("están **empatados estadísticamente**." if emp["empate"] else "sí se distinguen."))

# %% [markdown]
# ## 4. Chi-cuadrado con V de Cramér
#
# El chi-cuadrado de independencia contrasta si dos variables categóricas están asociadas; la V de Cramér mide la
# fuerza de la asociación. Se comprueba que las frecuencias esperadas no sean demasiado bajas.

# %%
for clave, fila, col, etiqueta in [
    ("chi2_tipo_superhost", "tipo_habitacion_agrupado", "host_is_superhost", "tipo de habitación y ser superhost"),
    ("chi2_multi_tipo", "es_multianuncio", "tipo_habitacion_agrupado", "ser multi-anuncio y tipo de habitación"),
]:
    hipotesis(clave)
    mostrar(est.chi_cuadrado(df, fila, col, clave, etiqueta))

# %%
hipotesis("chi2_distrito_tipo")
r = E["chi2_distrito_tipo"]
print("V de Cramér =", fmt.n3(r["tamano_efecto"]["valor"]), "|", fmt.p_texto(r["p_valor"]),
      "| % celdas con esperada < 5:", fmt.pct1(r["supuestos"]["pct_celdas_esperada_menor_5"]))
print("Conclusión:", r["conclusion"])

# %%
lectura(f"**Lectura.** Las tres asociaciones son significativas, pero solo la de distrito y tipo de habitación "
        f"tiene un efecto no despreciable (V = {fmt.n3(r['tamano_efecto']['valor'])}): la mezcla de alojamientos "
        f"cambia de una zona a otra. Tipo × superhost (V = {fmt.n3(E['chi2_tipo_superhost']['tamano_efecto']['valor'])}) "
        "es estadísticamente significativa sin relevancia práctica.")

# %% [markdown]
# ## 5. Intervalos de confianza bootstrap de la mediana
#
# El bootstrap remuestrea los datos muchas veces para ver cuánto varía la mediana. Se hace de dos formas:
# remuestreando anuncios (supone independencia) y remuestreando **anfitriones** con todos sus anuncios, que respeta
# la dependencia y da intervalos más anchos y realistas.

# %%
b = E["bootstrap"]
tabla_ic = pd.concat([pd.DataFrame(b["tipo"]), pd.DataFrame(b["distritos"])])
tabla_ic[["grupo", "n", "n_anfitriones", "mediana", "ic_inf", "ic_sup", "ic_inf_cluster", "ic_sup_cluster"]].round(2)

# %%
viz.fig_ic_bootstrap(findings);

# %%
t0, t1 = b["tipo"][0], b["tipo"][1]
lectura(f"**Lectura.** El IC por anfitrión de {t0['grupo'].lower()} ({fmt.eur0(t0['ic_inf_cluster'])}–"
        f"{fmt.eur0(t0['ic_sup_cluster'])}) es más ancho que el IC por anuncio ({fmt.eur0(t0['ic_inf'])}–"
        f"{fmt.eur0(t0['ic_sup'])}), como debe ser con observaciones dependientes, y aun así no se solapa con el de "
        f"{t1['grupo'].lower()} ({fmt.eur0(t1['ic_inf_cluster'])}–{fmt.eur0(t1['ic_sup_cluster'])}).")

# %% [markdown]
# ## 6. Regresión OLS sobre ln(precio)
#
# La regresión permite comparar a igualdad de otras variables. Se modela `ln(precio)`, así que cada coeficiente β se
# interpreta como un cambio porcentual `100·(e^β − 1)`. Controles: tipo, capacidad, dormitorios, baños, amenities,
# distancia, superhost, multi-anuncio, estancia larga, licencia, distrito y **mes de la cotización**. Los errores
# se agrupan por anfitrión (con HC3 como comparación) y el contraste global usa esa misma covarianza.

# %%
hipotesis("ols")
ols = est.ajustar_ols(df, config.CIUDADES[config.CIUDAD_POR_DEFECTO].distrito_referencia)
artefactos = {"residuos": ols.pop("_residuos"), "ajustados": ols.pop("_ajustados")}
ols.pop("_indices")
print("n efectivo:", fmt.n0(ols["n_efectivo"]), "| R² ajustado:", fmt.n3(ols["r2_ajustado"]),
      "| F (cluster):", fmt.n1(ols["f_cluster"]), "| F no robusto:", fmt.n1(ols["f_no_robusto"]))
print("Diagnóstico:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ols["diagnostico"].items()})
assert np.isclose(ols["r2_ajustado"], E["ols"]["r2_ajustado"])
coef = pd.DataFrame(ols["coeficientes"])
coef.loc[~coef["es_distrito"] & ~coef["es_mes"], ["nombre", "pct", "ic_inf_pct", "ic_sup_pct", "p_cluster", "p_holm"]].round(4)

# %%
lectura(f"**Lectura.** El modelo explica un {fmt.pct1(ols['r2_ajustado'] * 100)} de la varianza de `ln(precio)` "
        f"dentro de muestra. De los {ols['n_coeficientes']} coeficientes, {ols['n_significativos']} son significativos "
        f"con α y {ols['n_significativos_holm']} tras la corrección de Holm por multiplicidad. El superhost se asocia "
        f"con {fmt.signo_pct(E['ols']['coef_superhost']['pct'])} (IC {fmt.signo_pct(E['ols']['coef_superhost']['ic_inf_pct'])}"
        f" a {fmt.signo_pct(E['ols']['coef_superhost']['ic_sup_pct'])}, p Holm = {fmt.n3(E['ols']['coef_superhost']['p_holm'])}): "
        "evidencia débil de una prima pequeña, no robusta.")

# %% [markdown]
# ### Multicolinealidad: VIF y GVIF
#
# Sin dummies de distrito el VIF es bajo, pero la distancia al centro y el distrito están muy relacionados. Con la
# matriz completa, el GVIF por término mide esa colinealidad; por eso el coeficiente de distancia se lee como un
# efecto **dentro** de cada distrito.

# %%
pd.DataFrame(ols["gvif"])[["nombre", "gl", "gvif", "gvif_ajustado"]].head(6).round(2)

# %%
td, ref = E["ols"]["top_distritos"][0], E["ols"]["referencia_distrito"]
lectura(f"**Lectura.** El VIF de la distancia con dummies es {fmt.n2(ols['vif_completo_distancia'])} (sin ellas, el máximo "
        f"es {fmt.n2(ols['vif_max']['vif'])}). Esto explica un resultado que parece paradójico: "
        f"{td['nombre'].replace('Distrito: ', '')} tiene un coeficiente de {fmt.signo_pct(td['pct'])} frente a "
        f"{E['ols']['referencias']['distrito']}, pero su mediana bruta es {fmt.eur2(td['mediana_bruta'])} frente a "
        f"{fmt.eur2(ref['mediana_bruta'])}. Está a {fmt.n1(td['distancia_media_km'])} km de media y el término de "
        f"distancia ya implica un cambio de {fmt.signo_pct(td['efecto_distancia_pct'])} a esa distancia: la dummy compensa esa penalización.")

# %%
viz.fig_ols_residuos(artefactos);

# %%
lectura(f"**Lectura.** Los residuos no son normales (colas pesadas) y hay heterocedasticidad (Breusch-Pagan "
        f"{fmt.p_texto(ols['diagnostico']['breusch_pagan_p'])}), de ahí los errores robustos. El borde diagonal del "
        "gráfico de residuos frente a ajustados refleja el truncamiento por excluir outliers de la variable dependiente.")

# %%
viz.fig_ols_coeficientes(findings);

# %% [markdown]
# ### Sensibilidades del OLS
#
# Un coeficiente es más creíble si no depende de decisiones concretas de análisis. Se comparan cuatro
# especificaciones alternativas: sin control de mes, con outliers, sin observaciones influyentes (distancia de Cook
# alta) y sin estancias mensuales.

# %%
o = E["ols"]
filas = []
for t, c in o["sensibilidad_sin_mes"]["coeficientes"].items():
    filas.append({"variable": c["nombre"], "principal": c["pct_principal"], "sin mes": c["pct_alternativo"],
                  "sin influyentes": o["sensibilidad_sin_influyentes"]["coeficientes"].get(t, {}).get("pct"),
                  "sin mensuales": o["sensibilidad_sin_mensuales"]["coeficientes"].get(t, {}).get("pct_alternativo")})
pd.DataFrame(filas).set_index("variable").round(2)

# %%
lectura(f"**Lectura.** Los signos se mantienen en todas las especificaciones (con outliers, "
        f"{o['sensibilidad_con_outliers']['n_signos_coinciden']} de {o['sensibilidad_con_outliers']['n_significativos']} "
        "coeficientes significativos conservan el signo). El coeficiente que más cambia es el de la estancia mínima "
        "larga al quitar el control de mes, porque las cotizaciones lejanas son sobre todo estancias mensuales.")

# %% [markdown]
# ## 7. Modelo predictivo (complementario)
#
# Un HistGradientBoosting con una lista blanca de variables (sin ninguna derivada del precio) mide cuánto se puede
# predecir el precio. La validación agrupa por anfitrión: ningún anfitrión está a la vez en entrenamiento y prueba,
# y la imputación y la agrupación de categorías raras se aprenden dentro del Pipeline solo con el entrenamiento. Por
# coste, este notebook lee el resultado de `findings.json` en lugar de reentrenar.

# %%
ml = E["ml"]
if ml["ejecutado"]:
    cv, cm = ml["cv"], ml["comparacion_misma_muestra"]
    display(pd.DataFrame({
        "R² medio": {"HGB, GroupKFold (todas las filas con precio)": cv["r2_media"],
                     "Línea base (mediana)": ml["cv_base"]["r2_media"],
                     "HGB, partición aleatoria (optimista)": ml["aleatorio"]["r2"],
                     "Lineal, variables del OLS (mismas filas y particiones)": cm["lineal_vars_ols"]["r2_media"],
                     "HGB, variables del OLS (mismas filas y particiones)": cm["hgb_vars_ols"]["r2_media"],
                     "HGB, todas las variables (mismas filas y particiones)": cm["hgb_todas"]["r2_media"]},
    }).round(3))
    viz.fig_importancia_variables(findings)
else:
    print("Modelo no ejecutado:", ml["motivo"])

# %%
if ml["ejecutado"]:
    lectura(f"**Lectura.** Para anfitriones no vistos, R² = {fmt.n3(cv['r2_media'])} ± {fmt.n3(cv['r2_std'])} y mediana "
            f"del error absoluto ≈ {fmt.eur0(cv['medae_eur_media'])} por noche. Con partición aleatoria saldría "
            f"{fmt.n3(ml['aleatorio']['r2'])}, porque el {fmt.pct1(ml['aleatorio']['pct_test_con_anfitrion_en_train'])} "
            "de los anuncios de prueba comparte anfitrión con el entrenamiento. En la comparación homogénea, el no "
            "lineal supera al lineal con las mismas variables; la importancia por permutación es predictiva, no causal.")

# %% [markdown]
# ## Conclusión
#
# Las diferencias de precio por tipo de habitación y por distrito son significativas y relevantes, y resisten la
# sensibilidad con un anuncio por anfitrión; el tipo de habitación muestra la mayor diferencia entre grupos con la
# misma medida de efecto. La diferencia bruta por superhost es significativa pero despreciable, y al controlar en
# la regresión la evidencia de una prima es débil y no robusta: si existe, es pequeña. Los coeficientes de distrito
# hay que leerlos condicionados a la distancia, que dentro de un distrito se asocia con precios más bajos. Los
# resultados son estables frente a outliers, observaciones influyentes, el control de mes y la exclusión de
# estancias mensuales. El modelo predictivo, validado sin repetir anfitriones, mejora claramente a la línea base,
# pero su capacidad de generalizar es menor que la que sugeriría una partición aleatoria.
#
# Todo ello son asociaciones en datos observacionales de un único snapshot. La interpretación completa, las
# limitaciones y las recomendaciones (formuladas como hipótesis) están en `reports/informe.md`.
