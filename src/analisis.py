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
    """Días no disponibles e ingreso hipotético propios frente a las estimaciones de Inside Airbnb.

    Ventanas distintas: lo propio mira a los PRÓXIMOS 365 días (calendario futuro) y lo de Inside Airbnb
    a los ÚLTIMOS 12 meses (reseñas). Cada comparación usa la MISMA máscara de filas (ambos valores
    presentes) para medianas, cociente y correlación, y se informan los n.
    """
    nd, ocu = activos["no_disponible_365"], activos["ocupacion_inside_airbnb"]
    m_ocu = nd.notna() & ocu.notna()
    propio, externo = activos["ingreso_hipotetico_anual"], activos["estimated_revenue_l365d"]
    m_ing = propio.notna() & externo.notna()
    rho, p = stats.spearmanr(propio[m_ing], externo[m_ing])
    rho_o, p_o = stats.spearmanr(nd[m_ocu], ocu[m_ocu])
    med_prop, med_ext = float(propio[m_ing].median()), float(externo[m_ing].median())
    return {
        "n_activos_con_precio": int(len(activos)),
        "n_comparacion_ocupacion": int(m_ocu.sum()),
        "no_disponible_media": float(nd[m_ocu].mean() * 100),
        "no_disponible_mediana": float(nd[m_ocu].median() * 100),
        "ocupacion_inside_airbnb_media": float(ocu[m_ocu].mean() * 100),
        "ocupacion_inside_airbnb_mediana": float(ocu[m_ocu].median() * 100),
        "spearman_ocupacion": float(rho_o), "spearman_ocupacion_p": float(p_o),
        "n_comparacion_ingresos": int(m_ing.sum()),
        "ingreso_hipotetico_mediana": med_prop,
        "ingresos_inside_airbnb_mediana": med_ext,
        "cociente_medianas": float(med_prop / med_ext) if med_ext else float("nan"),
        "spearman_propio_vs_inside": float(rho), "spearman_p": float(p),
        "ventana_propia": "próximos 365 días (availability_365, calendario futuro)",
        "ventana_inside_airbnb": "últimos 12 meses (estimated_*_l365d, modelo basado en reseñas)",
    }


def por_mes_cotizacion(cp: pd.DataFrame) -> dict[str, Any]:
    """Mediana del precio por mes de entrada de la cotización (temporada) y extremos con n ≥ ``MIN_N_GRUPO``."""
    filas = sorted(precio_por_grupo(cp, "mes_cotizacion"), key=lambda g: g["grupo"])
    validos = [g for g in filas if g["n"] >= config.MIN_N_GRUPO]
    total = sum(g["n"] for g in filas)
    return {
        "meses": filas, "n_meses": len(filas), "n_con_mes": total,
        "mes_mediana_min": min(validos, key=lambda g: g["mediana"]) if validos else None,
        "mes_mediana_max": max(validos, key=lambda g: g["mediana"]) if validos else None,
        "pct_primeros_dos_meses": float(sum(g["n"] for g in filas[:2]) / total * 100) if total else float("nan"),
    }


def mascara_sin_mensuales(df: pd.DataFrame) -> pd.Series:
    """True para anuncios con estancia mínima < 30 noches y cotización no mensual (tarifa de corta estancia)."""
    return ~(df["minimum_nights"].ge(config.MIN_NIGHTS_LARGA_ESTANCIA).fillna(False).astype(bool)
             | df["es_tarifa_mensual"].astype(bool))


def sensibilidad_sin_mensuales(cp: pd.DataFrame) -> dict[str, Any]:
    """Descriptivos del precio excluyendo estancias mínimas ≥ 30 noches y tarifas mensuales."""
    sub = cp[mascara_sin_mensuales(cp)]
    distritos = precio_por_grupo(sub, "neighbourhood_group_cleansed", min_n=config.MIN_N_GRUPO)
    original = precio_por_grupo(cp, "neighbourhood_group_cleansed", min_n=config.MIN_N_GRUPO)
    return {
        "n": int(len(sub)), "n_excluidos": int(len(cp) - len(sub)),
        "pct_excluidos": float((len(cp) - len(sub)) / len(cp) * 100),
        "precio": descriptivos(sub["price"]),
        "top_distritos": distritos[:3], "distrito_mas_barato": distritos[-1] if distritos else None,
        "mismo_top2": [g["grupo"] for g in distritos[:2]] == [g["grupo"] for g in original[:2]],
        "mismo_mas_barato": bool(distritos and original and distritos[-1]["grupo"] == original[-1]["grupo"]),
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
    # Mismo suelo de muestra que la inferencia (R21): los distritos con n < MIN_N_GRUPO no entran en el ranking.
    por_distrito = precio_por_grupo(cp, "neighbourhood_group_cleansed", min_n=config.MIN_N_GRUPO)
    conteo_d = cp["neighbourhood_group_cleansed"].value_counts()
    distritos_excluidos = {str(k): int(v) for k, v in conteo_d[conteo_d < config.MIN_N_GRUPO].items()}
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
        "distritos_n_min": config.MIN_N_GRUPO,
        "distritos_excluidos_ranking": distritos_excluidos,
        "n_distritos_excluidos_ranking": len(distritos_excluidos),
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
        "mes_cotizacion": por_mes_cotizacion(cp),
        "sin_mensuales": sensibilidad_sin_mensuales(cp),
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

    logger.info("Bootstrap (%d remuestreos, por anuncio y por anfitrión)…", n_boot)

    def _boot(mascara: pd.Series, grupo: str) -> dict[str, Any]:
        x = cp.loc[mascara, "price"].to_numpy()
        simple = est.bootstrap_mediana(x, n_boot, rng)
        clus = est.bootstrap_mediana_cluster(x, cp.loc[mascara, "host_id"].to_numpy(), n_boot, rng)
        return {"grupo": grupo, **simple, "n_anfitriones": clus["n_anfitriones"],
                "ic_inf_cluster": clus["ic_inf"], "ic_sup_cluster": clus["ic_sup"]}

    boot_tipo = [_boot(cp["tipo_habitacion_agrupado"] == g, g) for g in tipos]
    boot_dist = [_boot(cp["neighbourhood_group_cleansed"] == d, d) for d in top_d]
    boot_dist.sort(key=lambda x: (-x["mediana"], x["grupo"]))
    barrio = cp["neighbourhood_cleansed"].astype(str)
    extremos = [descriptivo["barrios_top"][0], descriptivo["barrios_top"][1],
                descriptivo["barrios_bottom"][0], descriptivo["barrios_bottom"][1]]
    boot_barrios = [_boot(barrio == g["grupo"], g["grupo"]) for g in extremos]
    r["bootstrap"] = {
        "n_boot": n_boot, "metodo": "percentil, IC 95 %; por anuncio y por anfitrión (cluster)",
        "tipo": boot_tipo, "distritos": boot_dist, "barrios_extremos": boot_barrios,
        # El barrio más caro (o más barato) solo se distingue del segundo si sus IC por anfitrión no se solapan.
        "barrio_top_distinguible": boot_barrios[0]["ic_inf_cluster"] > boot_barrios[1]["ic_sup_cluster"],
        "barrio_bottom_distinguible": boot_barrios[2]["ic_sup_cluster"] < boot_barrios[3]["ic_inf_cluster"],
    }

    # Empate estadístico entre los dos distritos de mayor mediana (B1).
    d1, d2 = descriptivo["por_distrito"][0]["grupo"], descriptivo["por_distrito"][1]["grupo"]
    par = next((p for p in r["posthoc_distrito"]["pares"] if {p["grupo_a"], p["grupo_b"]} == {d1, d2}), None)
    ic = {b["grupo"]: b for b in boot_dist}
    solapan = (d1 in ic and d2 in ic and ic[d1]["ic_inf_cluster"] <= ic[d2]["ic_sup_cluster"]
               and ic[d2]["ic_inf_cluster"] <= ic[d1]["ic_sup_cluster"])
    r["top_distritos_empate"] = {
        "a": d1, "b": d2, "en_posthoc": par is not None,
        "p_ajustado": par["p_ajustado"] if par else None, "ic_solapan": bool(solapan),
        "empate": bool(par is not None and not par["significativo"] and solapan),
    }

    logger.info("Sensibilidad: un anuncio por anfitrión…")
    uno = est.un_anuncio_por_host(cp, rng)
    uno_todos = est.un_anuncio_por_host(df, rng)
    sh1 = uno["host_is_superhost"]
    mw1 = est.mann_whitney(uno.loc[sh1, "price"], uno.loc[~sh1, "price"], "superhosts", "no superhosts",
                           "mann_whitney_superhost").a_dict()
    kt1 = est.kruskal(uno, "price", "tipo_habitacion_agrupado", "kruskal_tipo", "tipos de habitación").a_dict()
    kd1 = est.kruskal(uno, "price", "neighbourhood_group_cleansed", "kruskal_distrito", "distritos").a_dict()
    top_d1 = [d for d in top_d if (uno["neighbourhood_group_cleansed"] == d).sum() >= config.MIN_N_GRUPO]
    ph1 = est.posthoc(uno, "price", "neighbourhood_group_cleansed", top_d1)
    chi1 = {
        "chi2_tipo_superhost": est.chi_cuadrado(uno_todos, "tipo_habitacion_agrupado", "host_is_superhost",
                                                "chi2_tipo_superhost", "tipo de habitación y ser superhost").a_dict(),
        "chi2_distrito_tipo": _chi2_distrito_tipo(uno_todos).a_dict(),
        "chi2_multi_tipo": est.chi_cuadrado(uno_todos, "es_multianuncio", "tipo_habitacion_agrupado",
                                            "chi2_multi_tipo", "ser multi-anuncio y tipo de habitación").a_dict(),
    }
    decisiones = {
        "superhost": mw1["significativo"] == r["mann_whitney_superhost"]["significativo"],
        "tipo": kt1["significativo"] == r["kruskal_tipo"]["significativo"],
        "distrito": kd1["significativo"] == r["kruskal_distrito"]["significativo"],
        **{k: v["significativo"] == r[k]["significativo"] for k, v in chi1.items()},
    }
    r["sensibilidad_un_anuncio_por_host"] = {
        "n": int(len(uno)), "n_todos": int(len(uno_todos)), "mann_whitney_superhost": mw1, "kruskal_tipo": kt1,
        "kruskal_distrito": kd1, "posthoc_distrito": {k: ph1[k] for k in ("metodo", "n_pares", "n_pares_significativos")},
        **chi1,
        "mismo_signo_superhost": bool(np.sign(mw1["tamano_efecto"]["valor"])
                                      == np.sign(r["mann_whitney_superhost"]["tamano_efecto"]["valor"])),
        "misma_decision_superhost": decisiones["superhost"], "misma_decision_tipo": decisiones["tipo"],
        "decisiones": decisiones, "todas_mismas_decisiones": all(decisiones.values()),
    }

    logger.info("Regresión OLS y especificaciones alternativas…")
    ols = est.ajustar_ols(df, distrito_ref)
    artefactos = {"residuos": ols.pop("_residuos"), "ajustados": ols.pop("_ajustados")}
    filas_ols = ols.pop("_indices")
    ols["sensibilidad_con_outliers"] = est.comparar_signos(
        ols, est.ajustar_ols(df, distrito_ref, incluir_outliers=True, diagnostico=False))
    sin_mes = est.ajustar_ols(df, distrito_ref, con_mes=False, diagnostico=False)
    ols["sensibilidad_sin_mes"] = {"r2_ajustado": sin_mes["r2_ajustado"], "n_efectivo": sin_mes["n_efectivo"],
                                   "coeficientes": est.coeficientes_clave(ols, sin_mes)}
    sin_dist = est.ajustar_ols(df, distrito_ref, con_distrito=False, diagnostico=False)
    ols["sin_distritos"] = {"r2_ajustado": sin_dist["r2_ajustado"], "n_efectivo": sin_dist["n_efectivo"],
                            "coef_distancia": next(c for c in sin_dist["coeficientes"]
                                                   if c["termino"] == "distancia_centro_km")}
    sin_mens = est.ajustar_ols(df, distrito_ref, mascara_extra=mascara_sin_mensuales(df),
                               excluir=("alquiler_larga_estancia",), diagnostico=False)
    ols["sensibilidad_sin_mensuales"] = {"r2_ajustado": sin_mens["r2_ajustado"], "n_efectivo": sin_mens["n_efectivo"],
                                         "coeficientes": est.coeficientes_clave(ols, sin_mens)}
    variables = [c for c in ols["coeficientes"] if not c["es_distrito"] and not c["es_mes"]]
    por_termino = {c["termino"]: c for c in variables}
    ols["coef_superhost"] = por_termino["host_is_superhost"]
    ols["coef_distancia"] = por_termino["distancia_centro_km"]
    ols["coef_privada"] = next(c for c in variables if c["nombre"] == "Tipo: Habitación privada")
    ols["coef_accommodates"] = por_termino["accommodates"]
    ols["coef_por_termino"] = {t: c for t, c in por_termino.items() if t in est.VARIABLES_OLS_NUMERICAS}
    # Los coeficientes de distrito son CONDICIONALES a la distancia (M1): se acompañan de la mediana
    # bruta y la distancia media del distrito para que no se lean como «más caro que el centro».
    cps_ols = df.loc[filas_ols]
    resumen_d = cps_ols.groupby("neighbourhood_group_cleansed", observed=True).agg(
        mediana_bruta=("price", "median"), distancia_media_km=("distancia_centro_km", "mean"))
    top = sorted((c for c in ols["coeficientes"] if c["es_distrito"] and c["significativo"]),
                 key=lambda c: -abs(c["beta"]))[:5]
    for c in top:
        nombre = c["nombre"].replace("Distrito: ", "")
        c["mediana_bruta"] = float(resumen_d.loc[nombre, "mediana_bruta"])
        c["distancia_media_km"] = float(resumen_d.loc[nombre, "distancia_media_km"])
        c["efecto_distancia_pct"] = 100 * (np.exp(ols["coef_distancia"]["beta"] * c["distancia_media_km"]) - 1)
    ols["top_distritos"] = top
    ols["referencia_distrito"] = {"mediana_bruta": float(resumen_d.loc[distrito_ref, "mediana_bruta"]),
                                  "distancia_media_km": float(resumen_d.loc[distrito_ref, "distancia_media_km"])}
    ols["r2_ajustado_pct"] = ols["r2_ajustado"] * 100
    ols["n_coef_significativos"] = int(sum(c["significativo"] for c in variables))
    ols["n_coef_variables"] = len(variables)
    ols["n_coef_distrito"] = int(sum(c["es_distrito"] for c in ols["coeficientes"]))
    ols["n_coef_mes"] = int(sum(c["es_mes"] for c in ols["coeficientes"]))
    r["ols"] = ols

    # Multiplicidad (R24): nº de contrastes y efecto de Holm sobre la familia de coeficientes del OLS.
    omnibus = ["mann_whitney_superhost", "kruskal_tipo", "kruskal_distrito", "chi2_tipo_superhost",
               "chi2_distrito_tipo", "chi2_multi_tipo"]
    r["multiplicidad"] = {
        "n_contrastes_principales": len(omnibus),
        "n_principales_p_menor_0001": int(sum(r[k]["p_valor"] < 0.001 for k in omnibus)),
        "n_pares_posthoc": r["posthoc_tipo"]["n_pares"] + r["posthoc_distrito"]["n_pares"],
        "n_coeficientes_ols": ols["n_coeficientes"], "n_significativos_ols": ols["n_significativos"],
        "n_significativos_ols_holm": ols["n_significativos_holm"],
        "n_hipotesis_total": len(omnibus) + r["posthoc_tipo"]["n_pares"] + r["posthoc_distrito"]["n_pares"]
        + ols["n_coeficientes"],
    }

    if con_ml:
        logger.info("Modelo ML (validación agrupada por anfitrión)…")
        r["ml"] = est.ajustar_modelo_ml(df, semilla, filas_ols=filas_ols)
    else:
        r["ml"] = {"ejecutado": False, "motivo": "omitido con --sin-ml"}
    r["hipotesis"] = {k: {"h0": v[0], "h1": v[1]} for k, v in est.HIPOTESIS.items()}
    r["umbrales_efecto"] = {"rank_biserial": list(est.UMBRALES_RB), "epsilon2": list(est.UMBRALES_EPS2),
                            "etiquetas": list(est.ETIQUETAS_EFECTO)}
    return {"descriptivo": descriptivo, "estadistica": r}, artefactos
