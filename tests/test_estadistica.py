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


def test_epsilon_cuadrado_formula():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": np.r_[rng.normal(0, 1, 40), rng.normal(1, 1, 40)], "g": ["a"] * 40 + ["b"] * 40})
    r = est.kruskal(df, "v", "g", "kruskal_tipo", "grupos")
    h, _ = stats.kruskal(df.v[:40], df.v[40:])
    assert r.tamano_efecto["valor"] == pytest.approx(h * 81 / (80 ** 2 - 1))


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


def test_agrupar_raras_sobre_validas():
    s = pd.Series(["A"] * 10 + ["B"] * 4 + ["C"] * 3)
    validas = pd.Series([True] * 10 + [True, True, False, False] + [True] * 3)
    r, info = agrupar_raras(s, validas, 3, "Otras")
    assert info["agrupadas"] == ["B"] and (r == "Otras").sum() == 4


@pytest.mark.parametrize("p,efecto,inicio", [
    (0.5, "grande", "No se rechaza"), (0.001, "despreciable", "Se rechaza H0 (p < α): resultado estadísticamente"),
    (0.001, "mediano", "Se rechaza H0 (p < α) con un efecto mediano"),
])
def test_redactar_conclusion(p, efecto, inicio):
    texto = est.redactar_conclusion(p, 0.05, efecto, "S.", "N.")
    assert texto.startswith(inicio) and "causa" not in texto


def test_guarda_antifuga():
    assert not set(config.FEATURES_ML) & config.COLUMNAS_PROHIBIDAS_FUGA
    assert not [f for f in config.FEATURES_ML if f.startswith(config.PREFIJOS_PROHIBIDOS_FUGA)]
    with pytest.raises(ErrorDatos):
        est.comprobar_fuga(["accommodates", "precio_por_persona"])


def test_ols_recupera_coeficiente():
    rng = np.random.default_rng(3)
    n = 600
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
        "host_id": np.arange(n), "precio_valido": True, "coords_validas": True, "es_outlier_precio": False,
    })
    df["log_precio"] = 4 + 0.1 * df["accommodates"] - 0.05 * df["distancia_centro_km"] + rng.normal(0, 0.05, n)
    r = est.ajustar_ols(df, "Centro")
    c = next(x for x in r["coeficientes"] if x["termino"] == "accommodates")
    assert c["beta"] == pytest.approx(0.1, abs=0.01)
    assert c["pct"] == pytest.approx(100 * (np.exp(c["beta"]) - 1))
