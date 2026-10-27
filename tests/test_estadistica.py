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


def _datos_ols(n: int = 600, semilla: int = 3, n_hosts: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({
        "accommodates": rng.integers(1, 7, n).astype(float), "bedrooms": rng.integers(1, 4, n).astype(float),
        "bathrooms": rng.integers(1, 3, n).astype(float), "bano_compartido": rng.random(n) < 0.2,
        "n_amenities": rng.integers(10, 60, n), "distancia_centro_km": rng.uniform(0, 10, n),
        "host_is_superhost": rng.random(n) < 0.3, "es_multianuncio": rng.random(n) < 0.5,
        "alquiler_larga_estancia": rng.random(n) < 0.1,
        **{c: rng.random(n) < 0.5 for c in ("tiene_aire_acondicionado", "tiene_ascensor", "tiene_terraza_balcon",
                                            "tiene_parking_cualquiera", "tiene_wifi")},
        "number_of_reviews": rng.integers(0, 100, n),
        "tipo_habitacion_agrupado": rng.choice(["Alojamiento entero", "Habitación privada"], n),
        "licencia_tipo": rng.choice(["Sin licencia", "Exenta"], n),
        "neighbourhood_group_cleansed": rng.choice(["Centro", "Retiro"], n),
        "mes_cotizacion_agrupado": rng.choice(["2026-09", "2026-10"], n),
        "host_id": rng.integers(0, n_hosts, n) if n_hosts else np.arange(n),
        "precio_valido": True, "coords_validas": True, "es_outlier_precio": False,
    })
    efecto_host = rng.normal(0, 0.3, df["host_id"].max() + 1)[df["host_id"]] if n_hosts else 0
    df["log_precio"] = (4 + 0.1 * df["accommodates"] - 0.05 * df["distancia_centro_km"] + efecto_host
                        + rng.normal(0, 0.05, n))
    return df


def test_ols_recupera_coeficiente_y_holm():
    r = est.ajustar_ols(_datos_ols(), "Centro")
    c = next(x for x in r["coeficientes"] if x["termino"] == "accommodates")
    assert c["beta"] == pytest.approx(0.1, abs=0.01)
    assert c["pct"] == pytest.approx(100 * (np.exp(c["beta"]) - 1))
    assert r["n_significativos_holm"] <= r["n_significativos"]
    assert any(x["es_mes"] for x in r["coeficientes"]) and r["referencias"]["mes"] in ("2026-09", "2026-10")
    assert "sensibilidad_sin_influyentes" in r and r["gvif"]


def test_f_global_usa_covarianza_agrupada():
    import statsmodels.formula.api as smf

    df = _datos_ols(n=800, semilla=7, n_hosts=80)
    r = est.ajustar_ols(df, "Centro")
    # Wald calculado a mano: F = b' V⁻¹ b / q con la covarianza agrupada por anfitrión (sin intercepto).
    datos = est._matriz_ols(df)
    ajuste = smf.ols(r["formula"], data=datos).fit(cov_type="cluster", cov_kwds={"groups": datos["host_id"]})
    nombres = [t for t in ajuste.params.index if t != "Intercept"]
    b, v = ajuste.params[nombres].to_numpy(), ajuste.cov_params().loc[nombres, nombres].to_numpy()
    f_manual = float(b @ np.linalg.solve(v, b) / len(nombres))
    assert r["f_cluster"] == pytest.approx(f_manual, rel=1e-6)
    assert r["f_cluster"] != pytest.approx(r["f_no_robusto"], rel=0.05)


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
    features = ["accommodates", "bedrooms", "tipo_habitacion", "neighbourhood_group_cleansed"]
    tr = pd.DataFrame({"accommodates": [2.0] * 40, "bedrooms": [1.0] * 39 + [np.nan],
                       "tipo_habitacion": ["Alojamiento entero"] * 40,
                       "neighbourhood_group_cleansed": ["Centro"] * 35 + ["Raro"] * 5})
    te = pd.DataFrame({"accommodates": [2.0] * 3, "bedrooms": [np.nan, 50.0, 50.0],
                       "tipo_habitacion": ["Alojamiento entero"] * 3, "neighbourhood_group_cleansed": ["Raro"] * 3})
    modelo = est.crear_pipeline(features, "lineal", 0).fit(tr, np.ones(40))
    imputador, agrupador = modelo.named_steps["imputar"], modelo.named_steps["raras"]
    # Las medianas y las categorías frecuentes salen solo del train: los 50 dormitorios del test no cuentan.
    assert imputador.medianas_global_ == {"bedrooms": 1.0}
    assert agrupador.frecuentes_["neighbourhood_group_cleansed"] == {"Centro"}
    transformado = agrupador.transform(imputador.transform(te))
    assert transformado["bedrooms"].tolist() == [1.0, 50.0, 50.0]
    assert transformado["neighbourhood_group_cleansed"].eq("Otras").all()


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
                                                  "number_of_reviews", "review_scores_rating", "review_scores_location",
                                                  "review_scores_cleanliness", "minimum_nights", "availability_365",
                                                  "antiguedad_host_anios", "beds")})
    df = df.assign(accommodates=rng.integers(1, 6, n).astype(float), bedrooms=rng.integers(1, 3, n).astype(float),
                   bano_compartido=False, host_is_superhost=rng.random(n) < 0.3, es_multianuncio=True,
                   tipo_habitacion="Alojamiento entero", neighbourhood_group_cleansed=rng.choice(["Centro", "Retiro"], n),
                   licencia_tipo="Sin licencia", mes_cotizacion="2026-09", host_id=host, precio_valido=True,
                   **{c: rng.random(n) < 0.5 for c in config.PATRONES_AMENITIES})
    df["log_precio"] = 4 + 0.2 * df["accommodates"] + rng.normal(0, 0.1, n)
    r = est.ajustar_modelo_ml(df, 0)
    assert r["ejecutado"] and r["n_folds"] == config.N_FOLDS_ML and len(r["cv"]["r2_folds"]) == config.N_FOLDS_ML
    assert r["cv"]["r2_media"] > r["cv_base"]["r2_media"]
    assert r["importancia_permutacion"][0]["variable"] == "accommodates"
    assert r["importancia_permutacion"][0]["nombre"] == config.ETIQUETAS_COLUMNAS["accommodates"]
    assert 0 < r["aleatorio"]["pct_test_con_anfitrion_en_train"] <= 100
