"""Análisis descriptivo y orquestación de la estadística → diccionario de hallazgos.

``ejecutar`` devuelve ``(hallazgos, artefactos)``: ``hallazgos`` es serializable y va a
``findings.json``; ``artefactos`` contiene arrays que solo necesitan las figuras.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import config, estadistica as est
from src.features import ETIQUETAS_CAPACIDAD, agrupar_raras

logger = logging.getLogger(__name__)

VARIABLES_CORRELACION: tuple[str, ...] = (
    "price", "log_precio", "accommodates", "bedrooms", "beds", "bathrooms", "n_amenities",
    "distancia_centro_km", "number_of_reviews", "reviews_per_month", "review_scores_rating",
    "availability_365", "minimum_nights", "antiguedad_host_anios", "calculated_host_listings_count",
)
ETIQUETAS_VARIABLES: dict[str, str] = {
    "price": "Precio", "log_precio": "ln(precio)", "accommodates": "Capacidad", "bedrooms": "Dormitorios",
    "beds": "Camas", "bathrooms": "Baños", "n_amenities": "Nº amenities", "distancia_centro_km": "Distancia centro",
    "number_of_reviews": "Nº reseñas", "reviews_per_month": "Reseñas/mes", "review_scores_rating": "Valoración",
    "availability_365": "Disponibilidad 365", "minimum_nights": "Noches mínimas",
    "antiguedad_host_anios": "Antigüedad host", "calculated_host_listings_count": "Anuncios del host",
}
ETIQUETA_OTROS_DISTRITOS = "Otros distritos"


def descriptivos(serie: pd.Series) -> dict[str, float]:
    """n, media, mediana, desviación, cuartiles, IQR, mín., máx., asimetría, curtosis de exceso y CV."""
    x = serie.dropna().astype("float64")
    if x.empty:
        return {"n": 0}
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    media = float(x.mean())
    return {
        "n": int(len(x)), "media": media, "mediana": float(x.median()), "std": float(x.std()),
        "q1": q1, "q3": q3, "iqr": q3 - q1, "min": float(x.min()), "max": float(x.max()),
        "p5": float(x.quantile(0.05)), "p95": float(x.quantile(0.95)),
        "asimetria": float(stats.skew(x)), "curtosis_exceso": float(stats.kurtosis(x)),
        "cv": float(x.std() / media) if media else float("nan"),
    }


def obtener_subconjuntos(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Subconjuntos de análisis: todos, con precio, con precio sin outliers y activos con precio."""
    con_precio = df[df["precio_valido"]]
    return {
        "todos": df,
        "con_precio": con_precio,
        "con_precio_sin_outliers": con_precio[~con_precio["es_outlier_precio"]],
        "activos_con_precio": con_precio[con_precio["es_activo_probable"]],
    }


def precio_por_grupo(df: pd.DataFrame, grupo: str, min_n: int = 1) -> list[dict[str, Any]]:
    """n, mediana, media y cuartiles de precio por ``grupo`` (orden: mediana desc., nombre)."""
    g = df.groupby(grupo, observed=True)["price"]
    tabla = pd.DataFrame({"n": g.size(), "mediana": g.median(), "media": g.mean(),
                          "q1": g.quantile(0.25), "q3": g.quantile(0.75)})
    tabla = tabla[tabla["n"] >= min_n].reset_index().rename(columns={grupo: "grupo"})
    tabla["grupo"] = tabla["grupo"].astype(str)
    tabla = tabla.sort_values(["mediana", "grupo"], ascending=[False, True], kind="mergesort")
    return [{k: (int(v) if k == "n" else v) for k, v in fila.items()} for fila in tabla.to_dict("records")]


def composicion_hosts(df: pd.DataFrame) -> dict[str, Any]:
    """Composición de anfitriones a nivel anuncio y a nivel anfitrión."""
    por_host = df.groupby("host_id").size()
    superhost_host = df.groupby("host_id")["host_is_superhost"].first()
    top10 = por_host.sort_values(ascending=False, kind="mergesort").head(10).sum()
    segmentos = df["segmento_host"].value_counts(sort=False)
    return {
        "n_anuncios": int(len(df)), "n_hosts": int(por_host.size),
        "anuncios_por_host_media": float(por_host.mean()), "anuncios_por_host_mediana": float(por_host.median()),
        "anuncios_por_host_max": int(por_host.max()),
        "pct_hosts_un_anuncio": float((por_host == 1).mean() * 100),
        "pct_anuncios_multianuncio": float(df["es_multianuncio"].mean() * 100),
        "pct_anuncios_superhost": float(df["host_is_superhost"].mean() * 100),
        "pct_hosts_superhost": float(superhost_host.mean() * 100),
        "pct_anuncios_hosts_6_o_mas": float((df["calculated_host_listings_count"] >= 6).mean() * 100),
        "pct_anuncios_top10_hosts": float(top10 / len(df) * 100),
        "segmentos": {str(k): int(v) for k, v in segmentos.items()},
        "segmentos_pct": {str(k): float(v / len(df) * 100) for k, v in segmentos.items()},
    }


def ocupacion_ingresos(activos: pd.DataFrame) -> dict[str, Any]:
    """Ocupación e ingresos estimados de anuncios activos con precio, frente a la estimación de Inside Airbnb."""
    propio = activos["ingresos_estimados_anuales"]
    externo = activos["estimated_revenue_l365d"]
    ambos = propio.notna() & externo.notna()
    rho, p = stats.spearmanr(propio[ambos], externo[ambos])
    med_ext = float(externo.median())
    return {
        "n_activos_con_precio": int(len(activos)),
        "ocupacion_estimada_media": float(activos["ocupacion_estimada"].mean() * 100),
        "ocupacion_estimada_mediana": float(activos["ocupacion_estimada"].median() * 100),
        "ocupacion_inside_airbnb_media": float(activos["ocupacion_inside_airbnb"].mean() * 100),
        "ocupacion_inside_airbnb_mediana": float(activos["ocupacion_inside_airbnb"].median() * 100),
        "ingresos_propios_mediana": float(propio.median()),
        "ingresos_inside_airbnb_mediana": med_ext,
        "cociente_medianas": float(propio.median() / med_ext) if med_ext else float("nan"),
        "spearman_propio_vs_inside": float(rho), "spearman_p": float(p),
    }


def amenities(con_precio: pd.DataFrame, todos: pd.DataFrame) -> list[dict[str, Any]]:
    """Prevalencia (todos los anuncios) y prima de precio mediana con/sin cada amenity."""
    filas = []
    for col in config.PATRONES_AMENITIES:
        con = con_precio.loc[con_precio[col], "price"]
        sin = con_precio.loc[~con_precio[col], "price"]
        med_con, med_sin = float(con.median()), float(sin.median())
        filas.append({"amenity": col, "nombre": config.NOMBRES_AMENITIES.get(col, col), "prevalencia_pct": float(todos[col].mean() * 100),
                      "mediana_con": med_con, "mediana_sin": med_sin,
                      "prima_pct": float((med_con / med_sin - 1) * 100) if med_sin else float("nan")})
    return sorted(filas, key=lambda r: -r["prevalencia_pct"])


def correlaciones(con_precio: pd.DataFrame) -> dict[str, Any]:
    """Matrices de Pearson y Spearman (pairwise) y pares con mayor |ρ| de Spearman."""
    variables = [v for v in VARIABLES_CORRELACION if v in con_precio]
    datos = con_precio[variables].astype("float64")
    pearson = datos.corr(method="pearson")
    spearman = datos.corr(method="spearman")
    pares = []
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            if {a, b} == {"price", "log_precio"}:
                continue
            pares.append({"a": a, "b": b, "spearman": float(spearman.loc[a, b]), "pearson": float(pearson.loc[a, b])})
    pares.sort(key=lambda r: (-abs(r["spearman"]), r["a"], r["b"]))
    con_precio_rho = sorted(
        ({"variable": v, "spearman": float(spearman.loc["price", v]), "pearson_log": float(pearson.loc["log_precio", v]),
          "pearson": float(pearson.loc["price", v])}
         for v in variables if v not in ("price", "log_precio")),
        key=lambda r: -abs(r["spearman"]))
    return {"variables": variables, "pearson": pearson.round(6).to_dict(), "spearman": spearman.round(6).to_dict(),
            "top_pares": pares[:10], "con_precio": con_precio_rho, "n": int(len(datos))}


def descriptivo_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Bloque descriptivo completo de ``findings``."""
    sub = obtener_subconjuntos(df)
    cp, cps = sub["con_precio"], sub["con_precio_sin_outliers"]
    barrios = precio_por_grupo(cp, "neighbourhood_cleansed", min_n=config.MIN_N_GRUPO)
    por_distrito = precio_por_grupo(cp, "neighbourhood_group_cleansed")
    orden_cap = list(ETIQUETAS_CAPACIDAD)
    por_capacidad = sorted(precio_por_grupo(cp, "categoria_capacidad"), key=lambda g: orden_cap.index(g["grupo"]))
    corr = correlaciones(cp)
    corr["con_precio_por_variable"] = {r["variable"]: r for r in corr["con_precio"]}
    return {
        "subconjuntos": {k: int(len(v)) for k, v in sub.items()},
        "precio": descriptivos(cp["price"]),
        "precio_sin_outliers": descriptivos(cps["price"]),
        "log_precio": descriptivos(cp["log_precio"]),
        "precio_por_persona": descriptivos(cp["precio_por_persona"]),
        "accommodates": descriptivos(df["accommodates"]),
        "minimum_nights": descriptivos(df["minimum_nights"]),
        "availability_365": descriptivos(df["availability_365"]),
        "number_of_reviews": descriptivos(df["number_of_reviews"]),
        "review_scores_rating": descriptivos(df["review_scores_rating"]),
        "distancia_centro_km": descriptivos(df["distancia_centro_km"]),
        "n_amenities": descriptivos(df["n_amenities"]),
        "por_tipo": precio_por_grupo(cp, "tipo_habitacion"),
        "por_tipo_agrupado": precio_por_grupo(cp, "tipo_habitacion_agrupado"),
        "anuncios_por_tipo": {str(k): int(v) for k, v in df["tipo_habitacion"].value_counts().items()},
        "por_distrito": por_distrito,
        "distrito_mas_barato": por_distrito[-1],
        "por_distrito_n": sorted(por_distrito, key=lambda g: (-g["n"], g["grupo"])),
        "barrios_n_min": config.MIN_N_GRUPO,
        "n_barrios_con_n_min": len(barrios),
        "barrios_top": barrios[:10],
        "barrios_bottom": barrios[-10:][::-1],
        "por_capacidad": por_capacidad,
        "por_licencia": precio_por_grupo(cp, "licencia_tipo"),
        "pct_sin_licencia": float((df["licencia_tipo"] == "Sin licencia").mean() * 100),
        "hosts": composicion_hosts(df),
        "ocupacion_ingresos": ocupacion_ingresos(sub["activos_con_precio"]),
        "pct_activos_probables": float(df["es_activo_probable"].mean() * 100),
        "n_no_activos": int((~df["es_activo_probable"]).sum()),
        "amenities": amenities(cp, df),
        "correlaciones": corr,
    }


def _chi2_distrito_tipo(df: pd.DataFrame) -> est.ResultadoTest:
    """Chi² distrito × tipo; si > 20 % de celdas esperadas < 5, agrupa distritos pequeños."""
    res = est.chi_cuadrado(df, "neighbourhood_group_cleansed", "tipo_habitacion_agrupado",
                           "chi2_distrito_tipo", "distrito y tipo de habitación")
    if res.supuestos["cumple_esperadas"]:
        return res
    agrupado, info = agrupar_raras(df["neighbourhood_group_cleansed"], pd.Series(True, index=df.index),
                                   config.MIN_N_CATEGORIA_RARA, ETIQUETA_OTROS_DISTRITOS)
    return est.chi_cuadrado(df.assign(distrito_agrupado=agrupado), "distrito_agrupado", "tipo_habitacion_agrupado",
                            "chi2_distrito_tipo", "distrito y tipo de habitación", extra={"agrupacion": info})


def ejecutar(df: pd.DataFrame, semilla: int, n_boot: int, con_ml: bool, distrito_ref: str
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ejecuta descriptivos y toda la estadística. Devuelve (hallazgos serializables, artefactos para figuras)."""
    rng = np.random.default_rng(semilla)
    descriptivo = descriptivo_dataset(df)
    sub = obtener_subconjuntos(df)
    cp = sub["con_precio"]
    r: dict[str, Any] = {}

    logger.info("Normalidad…")
    r["normalidad_precio"] = est.test_normalidad(cp["price"], "el precio", rng).a_dict()
    r["normalidad_log_precio"] = est.test_normalidad(cp["log_precio"], "el logaritmo del precio", rng).a_dict()

    logger.info("Mann-Whitney y Kruskal-Wallis…")
    sh = cp["host_is_superhost"]
    r["mann_whitney_superhost"] = est.mann_whitney(cp.loc[sh, "price"], cp.loc[~sh, "price"], "superhosts",
                                                   "no superhosts", "mann_whitney_superhost").a_dict()
    agrup = {"agrupacion": {"columna": "tipo_habitacion_agrupado",
                            "descripcion": f"categorías con menos de {config.MIN_N_CATEGORIA_RARA} anuncios con precio "
                                           f"agrupadas en «{config.ETIQUETA_OTRAS_TIPO}»"}}
    r["kruskal_tipo"] = est.kruskal(cp, "price", "tipo_habitacion_agrupado", "kruskal_tipo",
                                    "tipos de habitación", extra=agrup).a_dict()
    r["kruskal_distrito"] = est.kruskal(cp, "price", "neighbourhood_group_cleansed", "kruskal_distrito",
                                        "distritos").a_dict()
    r["tipos_solo_descriptivos"] = [g for g in descriptivo["por_tipo"]
                                    if g["grupo"] in ("Habitación compartida", "Habitación de hotel")]

    logger.info("Post-hoc…")
    tipos = list(r["kruskal_tipo"]["n"].keys())
    r["posthoc_tipo"] = est.posthoc(cp, "price", "tipo_habitacion_agrupado", tipos)
    conteo_d = cp["neighbourhood_group_cleansed"].value_counts()
    top_d = sorted(conteo_d.sort_values(ascending=False, kind="mergesort").head(config.TOP_N_DISTRITOS).index)
    r["posthoc_distrito"] = est.posthoc(cp, "price", "neighbourhood_group_cleansed", list(top_d))

    logger.info("Chi-cuadrado…")
    r["chi2_tipo_superhost"] = est.chi_cuadrado(df, "tipo_habitacion_agrupado", "host_is_superhost",
                                                "chi2_tipo_superhost", "tipo de habitación y ser superhost").a_dict()
    r["chi2_distrito_tipo"] = _chi2_distrito_tipo(df).a_dict()
    r["chi2_multi_tipo"] = est.chi_cuadrado(df, "es_multianuncio", "tipo_habitacion_agrupado", "chi2_multi_tipo",
                                            "ser multi-anuncio y tipo de habitación").a_dict()

    logger.info("Bootstrap (%d remuestreos)…", n_boot)
    boot_tipo = [{"grupo": g, **est.bootstrap_mediana(cp.loc[cp["tipo_habitacion_agrupado"] == g, "price"].to_numpy(),
                                                      n_boot, rng)} for g in tipos]
    boot_dist = [{"grupo": d, **est.bootstrap_mediana(cp.loc[cp["neighbourhood_group_cleansed"] == d, "price"].to_numpy(),
                                                      n_boot, rng)} for d in top_d]
    boot_dist.sort(key=lambda x: (-x["mediana"], x["grupo"]))
    r["bootstrap"] = {"n_boot": n_boot, "metodo": "percentil, IC 95 %", "tipo": boot_tipo, "distritos": boot_dist}

    logger.info("Sensibilidad: un anuncio por anfitrión…")
    uno = est.un_anuncio_por_host(cp, rng)
    sh1 = uno["host_is_superhost"]
    mw1 = est.mann_whitney(uno.loc[sh1, "price"], uno.loc[~sh1, "price"], "superhosts", "no superhosts",
                           "mann_whitney_superhost").a_dict()
    kt1 = est.kruskal(uno, "price", "tipo_habitacion_agrupado", "kruskal_tipo", "tipos de habitación").a_dict()
    r["sensibilidad_un_anuncio_por_host"] = {
        "n": int(len(uno)), "mann_whitney_superhost": mw1, "kruskal_tipo": kt1,
        "mismo_signo_superhost": bool(np.sign(mw1["tamano_efecto"]["valor"])
                                      == np.sign(r["mann_whitney_superhost"]["tamano_efecto"]["valor"])),
        "misma_decision_superhost": mw1["significativo"] == r["mann_whitney_superhost"]["significativo"],
        "misma_decision_tipo": kt1["significativo"] == r["kruskal_tipo"]["significativo"],
    }

    logger.info("Regresión OLS…")
    ols = est.ajustar_ols(df, distrito_ref)
    ols_sens = est.ajustar_ols(df, distrito_ref, incluir_outliers=True)
    artefactos = {"residuos": ols.pop("_residuos"), "ajustados": ols.pop("_ajustados")}
    ols["sensibilidad_con_outliers"] = est.comparar_signos(ols, ols_sens)
    no_distrito = [c for c in ols["coeficientes"] if not c["es_distrito"]]
    dist = sorted((c for c in ols["coeficientes"] if c["es_distrito"] and c["significativo"]),
                  key=lambda c: -abs(c["beta"]))
    ols["coef_superhost"] = next(c for c in no_distrito if c["termino"] == "host_is_superhost")
    ols["coef_distancia"] = next(c for c in no_distrito if c["termino"] == "distancia_centro_km")
    ols["coef_privada"] = next(c for c in no_distrito if c["nombre"] == "Tipo: Habitación privada")
    ols["coef_accommodates"] = next(c for c in no_distrito if c["termino"] == "accommodates")
    ols["coef_por_termino"] = {c["termino"]: c for c in no_distrito if c["termino"] in est.VARIABLES_OLS_NUMERICAS}
    ols["top_distritos"] = dist[:5]
    ols["r2_ajustado_pct"] = ols["r2_ajustado"] * 100
    ols["n_coef_significativos"] = int(sum(c["significativo"] for c in no_distrito))
    ols["n_coef_no_distrito"] = len(no_distrito)
    r["ols"] = ols

    if con_ml:
        logger.info("Modelo ML…")
        r["ml"] = est.ajustar_modelo_ml(df, semilla)
    else:
        r["ml"] = {"ejecutado": False, "motivo": "omitido con --sin-ml"}
    r["hipotesis"] = {k: {"h0": v[0], "h1": v[1]} for k, v in est.HIPOTESIS.items()}
    return {"descriptivo": descriptivo, "estadistica": r}, artefactos
