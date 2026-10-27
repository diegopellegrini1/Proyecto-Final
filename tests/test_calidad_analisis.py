"""Tests de la clasificación de columnas (calidad) y de los descriptivos de analisis.py con valores conocidos."""

import numpy as np
import pandas as pd
import pytest

from src import analisis, calidad, config


def test_clasificar_columnas_tipos_conocidos():
    df = pd.DataFrame({
        "id": [1, 2, 3], "host_id": [7, 7, 8], "listing_url": ["u", "v", "w"],
        "latitude": [40.4, 40.5, 40.6], "longitude": [-3.7, -3.6, -3.5],
        "price": ["$10.00", "$1,200.00", "$5"], "accommodates": [1, 2, 3],
        "host_is_superhost": ["t", "f", "t"], "last_scraped": ["2026-09-19", "2026-09-20", "2026-09-21"],
        "name": ["Piso", "Ático", "Estudio"], "room_type": ["Entire home/apt", "Private room", "Private room"],
        "bathrooms": [np.nan, np.nan, np.nan], "vacia_texto": pd.Series([None, None, None], dtype=object),
    })
    assert calidad.clasificar_columnas(df) == {
        "id": "identificador", "host_id": "identificador", "listing_url": "identificador",
        "latitude": "geoespacial", "longitude": "geoespacial", "price": "numerico", "accommodates": "numerico",
        "host_is_superhost": "booleano", "last_scraped": "fecha", "name": "texto", "room_type": "categorico",
        "bathrooms": "numerico", "vacia_texto": "vacia",
    }


def test_comprobar_requisitos_y_conteo():
    perfil = {"filas": 1000, "columnas": 20, "tipos_logicos": calidad.contar_tipos(
        {"a": "numerico", "b": "booleano", "c": "fecha", "d": "texto", "e": "geoespacial", "f": "numerico"})}
    r = calidad.comprobar_requisitos(perfil)
    assert perfil["tipos_logicos"] == {"booleano": 1, "fecha": 1, "geoespacial": 1, "numerico": 2, "texto": 1}
    assert r["cumple_filas"] and r["cumple_columnas"] and r["cumple_tipos"]
    assert not calidad.comprobar_requisitos({**perfil, "filas": 999})["cumple_filas"]


def test_descriptivos_valores_calculados_a_mano():
    d = analisis.descriptivos(pd.Series([1.0, 2, 3, 4, 100, np.nan]))
    assert d["n"] == 5 and d["media"] == 22.0 and d["mediana"] == 3.0
    assert (d["q1"], d["q3"], d["iqr"], d["min"], d["max"]) == (2.0, 4.0, 2.0, 1.0, 100.0)
    assert d["std"] == pytest.approx(np.sqrt(7610 / 4))  # desviaciones −21, −20, −19, −18 y 78
    assert d["cv"] == pytest.approx(np.sqrt(7610 / 4) / 22)
    assert analisis.descriptivos(pd.Series([], dtype=float)) == {"n": 0}


def test_precio_por_grupo_orden_y_suelo():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 2 + ["c"], "price": [10.0, 20, 30, 100, 200, 5]})
    filas = analisis.precio_por_grupo(df, "g")
    assert [(f["grupo"], f["n"], f["mediana"]) for f in filas] == [("b", 2, 150.0), ("a", 3, 20.0), ("c", 1, 5.0)]
    assert [f["grupo"] for f in analisis.precio_por_grupo(df, "g", min_n=2)] == ["b", "a"]


def test_composicion_hosts_valores_conocidos():
    df = pd.DataFrame({
        "host_id": [1, 1, 1, 2, 3, 3], "host_is_superhost": [True, True, True, False, False, False],
        "es_multianuncio": [True, True, True, False, True, True],
        "calculated_host_listings_count": [3, 3, 3, 1, 2, 2],
        "segmento_host": pd.Categorical(["2-5 anuncios"] * 3 + ["1 anuncio"] + ["2-5 anuncios"] * 2),
    })
    h = analisis.composicion_hosts(df)
    assert (h["n_anuncios"], h["n_hosts"], h["anuncios_por_host_max"]) == (6, 3, 3)
    assert h["pct_hosts_un_anuncio"] == pytest.approx(100 / 3)
    assert h["pct_anuncios_multianuncio"] == pytest.approx(500 / 6)
    assert h["pct_anuncios_superhost"] == pytest.approx(50.0) and h["pct_hosts_superhost"] == pytest.approx(100 / 3)
    assert h["pct_anuncios_hosts_6_o_mas"] == 0.0 and h["pct_anuncios_top10_hosts"] == 100.0


def test_ocupacion_ingresos_misma_mascara():
    activos = pd.DataFrame({
        "no_disponible_365": [0.5, 0.2, 0.8, 0.1], "ocupacion_inside_airbnb": [0.3, 0.1, 0.4, np.nan],
        "ingreso_hipotetico_anual": [1000.0, 200.0, 3000.0, 50.0],
        "estimated_revenue_l365d": [500.0, 100.0, np.nan, 10_000.0],
    })
    r = analisis.ocupacion_ingresos(activos)
    assert r["n_comparacion_ocupacion"] == 3 and r["no_disponible_media"] == pytest.approx(50.0)
    assert r["ocupacion_inside_airbnb_media"] == pytest.approx(80 / 3)
    # Ingresos: solo filas 0, 1 y 3 tienen ambos valores → medianas 200 y 500 sobre las mismas filas.
    assert r["n_comparacion_ingresos"] == 3
    assert (r["ingreso_hipotetico_mediana"], r["ingresos_inside_airbnb_mediana"]) == (200.0, 500.0)
    assert r["cociente_medianas"] == pytest.approx(0.4)


def test_sensibilidad_sin_mensuales_excluye_estancias_largas_y_tarifas_mensuales():
    n = config.MIN_N_GRUPO
    cp = pd.DataFrame({
        "price": [100.0] * n + [50.0] * n + [10.0] * 5,
        "neighbourhood_group_cleansed": ["A"] * n + ["B"] * n + ["A"] * 5,
        "minimum_nights": [2] * (2 * n) + [30, 31, 2, 2, 90],
        "es_tarifa_mensual": [False] * (2 * n) + [True, True, True, True, False],
    })
    assert analisis.mascara_sin_mensuales(cp).sum() == 2 * n
    r = analisis.sensibilidad_sin_mensuales(cp)
    assert (r["n"], r["n_excluidos"]) == (2 * n, 5)
    assert [g["grupo"] for g in r["top_distritos"]] == ["A", "B"] and r["precio"]["mediana"] == 75.0
