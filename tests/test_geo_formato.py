"""Tests de geo (haversine, bbox) y del formato es-ES."""

import numpy as np
import pandas as pd
import pytest

from src import config, formato as fmt
from src.geo import dentro_bbox, haversine_km

SOL = config.CIUDADES_ESPANA["madrid"].centro


def test_centros_dentro_de_su_bbox_y_regiones_sin_centro():
    for c in config.CIUDADES_ESPANA.values():
        if c.tipo == config.TIPO_URBANA:
            assert c.centro is not None
            assert dentro_bbox(pd.Series([c.centro[0]]), pd.Series([c.centro[1]]), c.bbox).all()
        else:
            assert c.centro is None


def test_haversine_cero_en_puerta_del_sol():
    assert haversine_km(*SOL, *SOL) == pytest.approx(0.0, abs=1e-9)


def test_haversine_un_grado_en_ecuador():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=0.01)


def test_haversine_madrid_barcelona():
    assert haversine_km(40.4168, -3.7038, 41.3874, 2.1686) == pytest.approx(505, abs=3)


def test_haversine_vectorizado():
    d = haversine_km(np.array([SOL[0], 0.0]), np.array([SOL[1], 0.0]), SOL[0], SOL[1])
    assert d.shape == (2,) and d[0] == pytest.approx(0, abs=1e-9) and d[1] > 4000


def test_dentro_bbox_con_nulos():
    lat = pd.Series([40.4, 41.0, np.nan])
    lon = pd.Series([-3.7, -3.7, -3.7])
    assert dentro_bbox(lat, lon, (40.31, 40.65, -3.9, -3.5)).tolist() == [True, False, False]


@pytest.mark.parametrize("func,valor,esperado", [
    (fmt.n0, 1234567, "1.234.567"), (fmt.n2, 1234.5, "1.234,50"), (fmt.pvalor, 1e-5, "< 0,001"),
    (fmt.pvalor, 0.1234, "0,123"), (fmt.pct1, 12.34, "12,3 %"), (fmt.eur2, 151, "151,00 €"),
    (fmt.n0, 0, "0"), (fmt.n1, -0.04, "0,0"), (fmt.n2, -3.5, "-3,50"), (fmt.n0, float("nan"), "–"),
    (fmt.eur2, None, "–"), (fmt.signo_pct, 12.3, "+12,3 %"), (fmt.signo_pct, -4.7, "-4,7 %"),
    (fmt.p_texto, 1e-9, "p < 0,001"), (fmt.p_texto, 0.107, "p = 0,107"), (fmt.fecha, "2026-09-17T00:00:00", "2026-09-17"),
])
def test_formato(func, valor, esperado):
    assert func(valor) == esperado
