"""Tests de funciones estadísticas sobre muestras construidas (no son datos de análisis)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import config, estadistica as est
from src.errores import ErrorDatos
from src.features import agrupar_raras


def test_mann_whitney_efecto_maximo():
    a = pd.Series(np.arange(100, 130, dtype=float))
    b = pd.Series(np.arange(0, 30, dtype=float))
    r = est.mann_whitney(a, b, "A", "B", "mann_whitney_superhost")
    assert r.tamano_efecto["valor"] == pytest.approx(1.0)
    assert r.tamano_efecto["cles"] == pytest.approx(1.0)
    assert r.significativo and r.extra["grupo_mas_caro"] == "A"


def test_epsilon_cuadrado_valor_calculado_a_mano():
    # Grupos [1,2,3] y [4,5,6]: rangos 6 y 15 → H = 12/(6·7)·(6²/3 + 15²/3) − 3·7 = 27/7.
    h = stats.kruskal([1, 2, 3], [4, 5, 6]).statistic
    assert h == pytest.approx(27 / 7)
    assert est.epsilon_cuadrado(h, 6) == pytest.approx(27 / 35)  # ε² = H/(n − 1)


def test_kruskal_excluye_grupos_pequenos():
    df = pd.DataFrame({"v": np.arange(70.0), "g": ["a"] * 35 + ["b"] * 30 + ["c"] * 5})
    r = est.kruskal(df, "v", "g", "kruskal_tipo", "grupos")
    assert r.extra["grupos_excluidos"] == {"c": 5}


def test_kruskal_un_grupo_falla():
    with pytest.raises(ErrorDatos):
        est.kruskal(pd.DataFrame({"v": np.arange(40.0), "g": ["a"] * 40}), "v", "g", "kruskal_tipo", "x")


def test_cramer_v_tabla_conocida_e_independencia():
    v, _ = est.cramer_v(np.array([[10, 20], [20, 10]]))
    assert v == pytest.approx(1 / 3, rel=1e-6)  # chi² = 6,667 con n = 60 → V = √(chi²/n)
    v0, _ = est.cramer_v(np.array([[50, 50], [50, 50]]))
    assert v0 == pytest.approx(0.0, abs=1e-12)


def test_chi2_con_etiquetas_booleanas():
    df = pd.DataFrame({"a": ["x", "y"] * 50, "b": [True, False, True, True] * 25})
    r = est.chi_cuadrado(df, "a", "b", "chi2_tipo_superhost", "a y b")
    assert r.extra["dimensiones"] == [2, 2]


def test_bootstrap_reproducible_y_cubre():
    x = np.random.default_rng(1).normal(10, 2, 2000)
    r1 = est.bootstrap_mediana(x, 500, np.random.default_rng(5))
    r2 = est.bootstrap_mediana(x, 500, np.random.default_rng(5))
    assert r1 == r2 and r1["ic_inf"] < 10 < r1["ic_sup"]


def test_bootstrap_cluster_mas_ancho_con_dependencia():
    # 60 anfitriones con 25 anuncios idénticos cada uno: la información efectiva es de 60, no de 1.500.
    rng = np.random.default_rng(2)
    efecto = rng.normal(100, 20, 60)
    x, g = np.repeat(efecto, 25), np.repeat(np.arange(60), 25)
    simple = est.bootstrap_mediana(x, 400, np.random.default_rng(3))
    clus = est.bootstrap_mediana_cluster(x, g, 400, np.random.default_rng(3))
    assert clus["n_anfitriones"] == 60 and clus["n"] == 1500
    assert (clus["ic_sup"] - clus["ic_inf"]) > 2 * (simple["ic_sup"] - simple["ic_inf"])
    assert clus["ic_inf"] <= np.median(x) <= clus["ic_sup"]
    assert est.bootstrap_mediana_cluster(x, g, 400, np.random.default_rng(3)) == clus  # reproducible


def test_agrupar_raras_sobre_validas():
    s = pd.Series(["A"] * 10 + ["B"] * 4 + ["C"] * 3)
    validas = pd.Series([True] * 10 + [True, True, False, False] + [True] * 3)
    r, info = agrupar_raras(s, validas, 3, "Otras")
    assert info["agrupadas"] == ["B"] and (r == "Otras").sum() == 4


@pytest.mark.parametrize("p,efecto,esperado", [
    (0.5, "grande", "No se rechaza H0 (p ≥ α). N."),
    (0.001, "despreciable", "Se rechaza H0 (p < α): resultado estadísticamente significativo pero con efecto "
                            "despreciable, sin relevancia práctica. S."),
    (0.001, "mediano", "Se rechaza H0 (p < α) con un efecto mediano. S."),
])
def test_redactar_conclusion(p, efecto, esperado):
    assert est.redactar_conclusion(p, 0.05, efecto, "S.", "N.") == esperado


@pytest.mark.parametrize("variable", ["price", "precio_por_persona", "price_quote_total_price",
                                      "ingreso_hipotetico_anual", "es_outlier_alto", "estimated_revenue_l365d"])
def test_guarda_antifuga_detecta_cada_variable_prohibida(variable):
    with pytest.raises(ErrorDatos, match=variable):
        est.comprobar_fuga(["accommodates", variable])


def test_modelo_ml_aborta_si_la_lista_blanca_incluye_el_objetivo(monkeypatch):
    monkeypatch.setattr(config, "FEATURES_ML", (*config.FEATURES_ML, "precio_por_persona"))
    df = pd.DataFrame({"precio_por_persona": [1.0], "accommodates": [2], "precio_valido": [True]})
    with pytest.raises(ErrorDatos, match="precio_por_persona"):
        est.ajustar_modelo_ml(df, 0)


def _datos_ols(n: int = 900, semilla: int = 3, n_hosts: int | None = None, beta_renta: float = 0.3) -> pd.DataFrame:
    """Datos sintéticos con dos ciudades, 30 distritos y una renta por distrito (ln(precio) conocido)."""
    rng = np.random.default_rng(semilla)
    distrito = rng.integers(0, 30, n)
    renta_dis = rng.uniform(10_000, 30_000, 30)
    df = pd.DataFrame({
        "accommodates": rng.integers(1, 7, n).astype(float), "bedrooms": rng.integers(1, 4, n).astype(float),
        "bathrooms": rng.integers(1, 3, n).astype(float), "bano_compartido": rng.random(n) < 0.2,
        "n_amenities": rng.integers(10, 60, n), "distancia_centro_km": rng.uniform(0, 10, n),
        "host_is_superhost": rng.random(n) < 0.3, "es_multianuncio": rng.random(n) < 0.5,
        "alquiler_larga_estancia": rng.random(n) < 0.1,
        **{c: rng.random(n) < 0.5 for c in ("tiene_aire_acondicionado", "tiene_piscina", "tiene_terraza_balcon",
                                            "tiene_parking_cualquiera")},
        "number_of_reviews": rng.integers(0, 100, n),
        "tipo_habitacion_agrupado": rng.choice(["Alojamiento entero", "Habitación privada"], n),
        "licencia_tipo": rng.choice(["Sin licencia", "Exenta"], n),
        "ciudad": np.where(distrito < 15, "madrid", "barcelona"),
        "codigo_distrito_ine": [f"0801{d:03d}" for d in distrito],
        "ine_dis_renta_neta_persona": renta_dis[distrito],
        "ine_dis_gini": rng.uniform(25, 40, 30)[distrito], "ine_dis_edad_media": rng.uniform(35, 50, 30)[distrito],
        "ine_dis_pct_hogares_unipersonales": rng.uniform(20, 40, 30)[distrito],
        "mes_cotizacion_agrupado": rng.choice(["2026-09", "2026-10"], n),
        "host_id": rng.integers(0, n_hosts, n) if n_hosts else np.arange(n),
        "precio_valido": True, "coords_validas": True, "es_outlier_precio": False,
    })
    efecto_host = rng.normal(0, 0.3, df["host_id"].max() + 1)[df["host_id"]] if n_hosts else 0
    df["log_precio"] = (4 + 0.1 * df["accommodates"] + beta_renta * np.log(df["ine_dis_renta_neta_persona"])
                        + 0.2 * (df["ciudad"] == "barcelona") + efecto_host + rng.normal(0, 0.05, n))
    df["price"] = np.exp(df["log_precio"])
    return df


def test_ols_recupera_coeficientes_renta_y_holm():
    r = est.ajustar_ols(_datos_ols())
    c = r["coef_por_termino"]
    assert c["accommodates"]["beta"] == pytest.approx(0.1, abs=0.01)
    assert c["accommodates"]["pct"] == pytest.approx(100 * (np.exp(c["accommodates"]["beta"]) - 1))
    # La renta entra en logaritmo: β es una elasticidad y se traduce a una renta un 10 % mayor.
    renta = c["ln_renta_dis"]
    assert renta["beta"] == pytest.approx(0.3, abs=0.03)
    assert renta["pct_por_10pct_renta"] == pytest.approx(100 * (1.1 ** renta["beta"] - 1))
    assert "p_cluster_distrito" in renta and renta["significativo_distrito"]
    assert r["n_significativos_holm"] <= r["n_significativos"]
    assert any(x["es_mes"] for x in r["coeficientes"]) and any(x["es_ciudad"] for x in r["coeficientes"])
    assert r["referencias"]["ciudad"] == config.CIUDAD_REFERENCIA
    assert "sensibilidad_sin_influyentes" in r and r["gvif"] and "vif_completo_renta" in r


def test_ols_con_y_sin_ine_usa_las_mismas_filas():
    df = _datos_ols()
    df.loc[df.index[:20], "ine_dis_gini"] = np.nan
    con = est.ajustar_ols(df, diagnostico=False)
    sin = est.ajustar_ols(df, con_ine=False, diagnostico=False)
    assert con["n_efectivo"] == sin["n_efectivo"] == len(df) - 20
    assert con["r2_ajustado"] > sin["r2_ajustado"]
    assert "ln_renta_dis" not in sin["coef_por_termino"]


def test_ols_renta_sin_asociacion_no_es_significativa():
    r = est.ajustar_ols(_datos_ols(beta_renta=0.0, semilla=11), diagnostico=False)
    assert abs(r["coef_por_termino"]["ln_renta_dis"]["beta"]) < 0.05


def test_f_global_usa_covarianza_agrupada():
    import statsmodels.formula.api as smf

    df = _datos_ols(n=900, semilla=7, n_hosts=90)
    r = est.ajustar_ols(df, diagnostico=False)
    # Wald calculado a mano: F = b' V⁻¹ b / q con la covarianza agrupada por anfitrión (sin intercepto).
    datos = est._matriz_ols(df, ())
    ajuste = smf.ols(r["formula"], data=datos).fit(cov_type="cluster", cov_kwds={"groups": datos["host_id"]})
    nombres = [t for t in ajuste.params.index if t != "Intercept"]
    b, v = ajuste.params[nombres].to_numpy(), ajuste.cov_params().loc[nombres, nombres].to_numpy()
    f_manual = float(b @ np.linalg.solve(v, b) / len(nombres))
    assert r["f_cluster"] == pytest.approx(f_manual, rel=1e-6)
    assert r["f_cluster"] != pytest.approx(r["f_no_robusto"], rel=0.05)


# --------------------------------------------------------------------------
# Relación renta-precio (Spearman con bootstrap por clusters)
# --------------------------------------------------------------------------
def test_rangos_ponderados_equivalen_a_replicar_filas():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 15, 300).astype(float)
    y = x + rng.normal(0, 4, 300)
    w = rng.integers(0, 4, 300).astype(float)
    gx, kx = est._grupos_empate(x)
    gy, ky = est._grupos_empate(y)
    rho = est._pearson_ponderado(est._rangos_ponderados(gx, kx, w), est._rangos_ponderados(gy, ky, w), w)
    assert rho == pytest.approx(stats.spearmanr(np.repeat(x, w.astype(int)), np.repeat(y, w.astype(int))).statistic)


def test_spearman_cluster_valor_puntual_e_ic_mas_ancho_por_zona():
    rng = np.random.default_rng(1)
    zona = np.repeat(np.arange(40), 50)
    renta = rng.normal(size=40)[zona]
    precio = 0.3 * renta + rng.normal(0, 1, 40)[zona] + rng.normal(0, 0.5, len(zona))
    x, y = pd.Series(renta), pd.Series(precio)
    r = est.spearman_cluster(x, y, {"cluster_zona": pd.Series(zona), "cluster_anfitrion": pd.Series(np.arange(len(zona)))},
                             300, np.random.default_rng(2))
    assert r["rho"] == pytest.approx(stats.spearmanr(renta, precio).statistic)
    ancho_zona = r["cluster_zona"]["ic_sup"] - r["cluster_zona"]["ic_inf"]
    ancho_fila = r["cluster_anfitrion"]["ic_sup"] - r["cluster_anfitrion"]["ic_inf"]
    assert ancho_zona > 2 * ancho_fila  # la renta solo varía entre zonas: remuestrear filas subestima la incertidumbre
    assert r["cluster_zona"]["n_clusters"] == 40 and 0 < r["cluster_zona"]["p_boot"] <= 1


def test_spearman_cluster_no_evaluable_con_renta_constante():
    r = est.spearman_cluster(pd.Series([1.0] * 50), pd.Series(range(50), dtype=float),
                             {"cluster_zona": pd.Series([0] * 50)}, 10, np.random.default_rng(0))
    assert r["evaluable"] is False


def test_holm_valores_conocidos():
    r = est.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # Holm: orden 0,01·3 = 0,03; 0,03·2 = 0,06; max(0,06, 0,04·1) = 0,06.
    assert r["a"]["p_holm"] == pytest.approx(0.03) and r["c"]["p_holm"] == pytest.approx(0.06)
    assert r["b"]["p_holm"] == pytest.approx(0.06)
    assert r["a"]["significativo_holm"] and not r["b"]["significativo_holm"]


def test_correlacion_ecologica_un_punto_por_zona():
    from src import analisis

    filas = []
    for z in range(8):
        filas += [{"ciudad": "madrid", "codigo_distrito_ine": f"d{z}", "renta": 1000 * z, "price": 50 + 10 * z + k}
                  for k in range(config.MIN_N_GRUPO)]
    filas += [{"ciudad": "madrid", "codigo_distrito_ine": "pequeno", "renta": 99_999, "price": 1.0}]
    r = analisis.correlacion_ecologica(pd.DataFrame(filas), "renta", "codigo_distrito_ine")
    assert r["evaluable"] and r["n_zonas"] == 8 and r["rho"] == pytest.approx(1.0)  # la zona pequeña no entra


def test_renta_precio_global_por_ciudad_y_sensibilidades():
    from src import analisis

    df = _datos_ols(n=1200, semilla=5, n_hosts=300)
    df = df.assign(codigo_municipio_ine=np.where(df["ciudad"] == "madrid", "28079", "08019"),
                   ine_mun_renta_neta_persona=np.where(df["ciudad"] == "madrid", 20_000.0, 18_000.0),
                   minimum_nights=2.0, es_tarifa_mensual=False)
    r = analisis.renta_precio(df, 50, np.random.default_rng(0))
    assert r["global_distrito"]["rho"] == pytest.approx(
        stats.spearmanr(df["ine_dis_renta_neta_persona"], df["price"]).statistic)
    assert {c["clave"] for c in r["por_ciudad"]} == {"madrid", "barcelona"}
    assert all("holm" in c["distrito"] for c in r["por_ciudad"])
    assert r["n_ciudades_evaluables"] == 2 and r["rho_ciudad_max"]["distrito"]["rho"] >= r["rho_ciudad_min"]["distrito"]["rho"]
    assert set(r["sensibilidades"]) == {"sin_estancias_30", "sin_outliers", "un_anuncio_por_anfitrion", "solo_urbanas"}
    assert r["sensibilidades"]["un_anuncio_por_anfitrion"]["n"] == df["host_id"].nunique()
    assert r["sensibilidades_mismo_signo"]
    # El ρ global usa más réplicas (con un generador propio) y lleva su lectura prudente.
    gz = r["global_distrito"]["cluster_zona"]
    assert gz["n_boot"] == r["n_boot_global"] == 50 * config.FACTOR_N_BOOT_RHO_GLOBAL
    assert gz["lectura"] in {"significativa", "en el límite", "no significativa"}
    assert r["global_distrito"]["cluster_zona_replicas_generales"]["n_replicas_validas"] == 50
    assert r["n_ciudades_positivas"] <= r["n_ciudades_evaluables"]


@pytest.mark.parametrize("p,lo,hi,esperada", [
    (0.04995, 0.001, 0.138, "en el límite"),   # el caso del proyecto: p justo por debajo de α e IC que roza 0
    (0.054, -0.001, 0.14, "en el límite"),     # la misma ρ con otra semilla: p por encima de α
    (0.02, 0.005, 0.2, "en el límite"),        # p pequeño pero el IC roza el 0
    (0.002, 0.10, 0.30, "significativa"),
    (0.40, -0.20, 0.30, "no significativa"),
])
def test_lectura_significacion_no_depende_del_lado_de_alfa(p, lo, hi, esperada):
    from src import estadistica as est

    r = est.lectura_significacion({"p_boot": p, "ic_inf": lo, "ic_sup": hi}, 10_000)
    assert r["lectura"] == esperada
    assert r["en_el_limite"] is (esperada == "en el límite")
    assert r["p_error_montecarlo"] == pytest.approx(np.sqrt(p * (1 - p) / 10_000))


def test_gvif_coincide_con_vif_para_un_grado_de_libertad():
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    rng = np.random.default_rng(4)
    x1 = rng.normal(size=500)
    x = pd.DataFrame({"Intercept": 1.0, "a": x1, "b": x1 + rng.normal(scale=0.5, size=500), "c": rng.normal(size=500)})
    g = {f["termino"]: f["gvif"] for f in est.gvif(x)}
    for i, col in enumerate(x.columns):
        if col != "Intercept":
            assert g[col] == pytest.approx(variance_inflation_factor(x.to_numpy(), i), rel=1e-6)


def test_particiones_por_anfitrion_sin_solape():
    grupos = np.repeat(np.arange(50), np.random.default_rng(0).integers(1, 20, 50))
    particiones = est.particiones_por_anfitrion(grupos, 5, 42)
    assert len(particiones) == 5
    for tr, te in particiones:
        assert not set(grupos[tr]) & set(grupos[te])
    assert sorted(np.concatenate([te for _, te in particiones]).tolist()) == list(range(len(grupos)))


def test_comprobar_sin_solape_detecta_fuga():
    grupos = np.array([1, 1, 2, 2])
    with pytest.raises(ErrorDatos, match="anfitriones en train y test"):
        est.comprobar_sin_solape(grupos, [(np.array([0, 2]), np.array([1, 3]))])


def test_pipeline_aprende_imputacion_y_raras_solo_con_train():
    features = ["accommodates", "bedrooms", "tipo_habitacion", "ciudad"]
    tr = pd.DataFrame({"accommodates": [2.0] * 40, "bedrooms": [1.0] * 39 + [np.nan],
                       "tipo_habitacion": ["Alojamiento entero"] * 40,
                       "ciudad": ["madrid"] * 35 + ["menorca"] * 5})
    te = pd.DataFrame({"accommodates": [2.0] * 3, "bedrooms": [np.nan, 50.0, 50.0],
                       "tipo_habitacion": ["Alojamiento entero"] * 3, "ciudad": ["menorca"] * 3})
    modelo = est.crear_pipeline(features, "lineal", 0).fit(tr, np.ones(40))
    imputador, agrupador = modelo.named_steps["imputar"], modelo.named_steps["raras"]
    # Las medianas y las categorías frecuentes salen solo del train: los 50 dormitorios del test no cuentan.
    assert imputador.medianas_global_ == {"bedrooms": 1.0}
    assert agrupador.frecuentes_["ciudad"] == {"madrid"}
    transformado = agrupador.transform(imputador.transform(te))
    assert transformado["bedrooms"].tolist() == [1.0, 50.0, 50.0]
    assert transformado["ciudad"].eq("Otras").all()


def test_preparar_x_devuelve_nan_donde_se_imputo():
    df = pd.DataFrame({"bedrooms": [1.0, 2.0], "bedrooms_imputado": [False, True], "accommodates": [2, 3],
                       "tipo_habitacion": ["Alojamiento entero"] * 2})
    X = est.preparar_X(df, ["bedrooms", "accommodates", "tipo_habitacion"])
    assert X["bedrooms"].iloc[0] == 1.0 and np.isnan(X["bedrooms"].iloc[1])


def test_modelo_ml_validacion_agrupada_en_datos_sinteticos():
    rng = np.random.default_rng(0)
    n = 400
    host = rng.integers(0, 80, n)
    df = pd.DataFrame({c: rng.random(n) for c in ("bathrooms", "n_amenities", "distancia_centro_km",
                                                  "number_of_reviews", "review_scores_rating", "minimum_nights",
                                                  "availability_365", "antiguedad_host_anios", "beds",
                                                  *config.VARIABLES_INE_ML)})
    df = df.assign(accommodates=rng.integers(1, 6, n).astype(float), bedrooms=rng.integers(1, 3, n).astype(float),
                   bano_compartido=False, host_is_superhost=rng.random(n) < 0.3, es_multianuncio=True,
                   tipo_habitacion="Alojamiento entero", ciudad=rng.choice(["madrid", "barcelona"], n),
                   licencia_tipo="Sin licencia", mes_cotizacion="2026-09", host_id=host, precio_valido=True,
                   **{c: rng.random(n) < 0.5 for c in config.PATRONES_AMENITIES})
    df[config.RENTA_PRINCIPAL] = rng.uniform(10_000, 30_000, n)
    df["log_precio"] = 4 + 0.2 * df["accommodates"] + rng.normal(0, 0.1, n)
    r = est.ajustar_modelo_ml(df, 0)
    assert r["ejecutado"] and r["n_folds"] == config.N_FOLDS_ML and len(r["cv"]["r2_folds"]) == config.N_FOLDS_ML
    assert r["cv"]["r2_media"] > r["cv_base"]["r2_media"]
    assert r["importancia_permutacion"][0]["variable"] == "accommodates"
    assert r["importancia_permutacion"][0]["nombre"] == config.ETIQUETAS_COLUMNAS["accommodates"]
    assert 0 < r["aleatorio"]["pct_test_con_anfitrion_en_train"] <= 100
    # Sin las variables del INE (que aquí son ruido) el R² apenas cambia.
    assert abs(r["aporte_r2_ine"]) < 0.05 and r["features_ine"] == list(config.VARIABLES_INE_ML)
    assert 1 <= r["rango_ine_en_importancia"] <= r["n_features"]
