"""Tests de conversiones, reglas de limpieza, features y diccionario."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, diccionario, features, limpieza
from src.errores import ErrorDatos

CIUDAD = config.CIUDADES["madrid"]


def test_parse_precio():
    s = pd.Series(["$1,234.00", "$364.75", "", np.nan, "basura", "$1.2.3"])
    r = limpieza.parse_precio(s)
    assert r.dtype == "float64"
    assert r.iloc[0] == 1234.0 and r.iloc[1] == 364.75
    assert r.iloc[2:].isna().all()


def test_parse_porcentaje():
    r = limpieza.parse_porcentaje(pd.Series(["95%", "N/A", "150%", np.nan, "0%"]))
    assert r.iloc[0] == 95.0 and r.iloc[4] == 0.0
    assert r.iloc[[1, 2, 3]].isna().all()


def test_a_booleano_sin_y_con_nulos():
    assert limpieza.a_booleano(pd.Series(["t", "f"])).dtype == bool
    r = limpieza.a_booleano(pd.Series(["t", None, "x"]))
    assert str(r.dtype) == "boolean" and r.iloc[0] and r.iloc[1:].isna().all()


def test_a_fecha():
    r = limpieza.a_fecha(pd.Series(["2026-09-17", "no", None]))
    assert pd.api.types.is_datetime64_any_dtype(r) and r.iloc[1:].isna().all()


def test_parse_banios_variantes():
    s = pd.Series(["1.5 baths", "Half-bath", "Shared half-bath", "1 shared bath", "0 baths", np.nan,
                   "Private half-bath", "2 shared baths"])
    r = limpieza.parse_banios(s)
    assert r["bathrooms"].tolist()[:5] == [1.5, 0.5, 0.5, 1.0, 0.0]
    assert np.isnan(r["bathrooms"].iloc[5]) and r["bathrooms"].iloc[6] == 0.5 and r["bathrooms"].iloc[7] == 2.0
    assert r["bano_compartido"].tolist() == [False, False, True, True, False, False, False, True]


@pytest.mark.parametrize("valor,esperado", [
    ('["Wifi", "Kitchen"]', ["Wifi", "Kitchen"]), ('["Wifi", "Kit', ["Wifi"]), ("", []), (None, []), (3, []),
])
def test_parse_amenities_robusto(valor, esperado):
    assert limpieza.parse_amenities(valor) == esperado


def test_flags_amenities():
    r = limpieza.flags_amenities(pd.Series(['["Wifi", "Free parking on premises", "Pool table"]', np.nan]))
    assert r.loc[0, "tiene_wifi"] and r.loc[0, "tiene_parking_cualquiera"] and not r.loc[0, "tiene_cocina"]
    assert not r.loc[1].any()


def test_vallas_iqr_log():
    precio = pd.Series([np.e ** k for k in (1, 2, 2, 2, 3, 10)])
    inf, sup = limpieza.vallas_iqr_log(precio)
    assert inf < np.e and sup < np.e ** 10


def test_politica_nulos_siempre_resuelve():
    for col in ("price", "review_scores_rating", "beds", "columna_nueva_futura", "price_quote_total_price"):
        assert limpieza.regla_nulos(col)[1]
    assert limpieza.regla_nulos("columna_nueva_futura")[0] == ".*"


def _listings_sinteticos() -> pd.DataFrame:
    """Entrada de prueba mínima con el formato de Inside Airbnb (no son datos de análisis)."""
    n = 6
    return pd.DataFrame({
        "id": [1, 2, 3, 3, 5, 6], "name": ["A", "B<br/>", "C", "C", "D", "E"], "host_id": [10, 10, 11, 11, 12, 13],
        "price": ["$100.00", "$0.00", np.nan, "$50.00", "$1,000.00", "$80.00"],
        "room_type": ["Entire home/apt"] * 3 + ["Private room"] * 3, "accommodates": [2, 4, 0, 1, 2, 3],
        "latitude": [40.42, 40.40, 41.5, 40.41, 40.43, 40.44], "longitude": [-3.70] * n,
        "neighbourhood_cleansed": ["Sol"] * n, "neighbourhood_group_cleansed": ["Centro"] * n,
        "host_is_superhost": ["t", "f", "f", "f", "t", "f"], "availability_365": [0, 365, 100, 0, 200, 50],
        "number_of_reviews_ltm": [0, 1, 2, 3, 0, 1], "number_of_reviews": [0, 5, 2, 3, 1, 1],
        "amenities": ['["Wifi"]', '["Kitchen", "Heating"]', '["Washer"]', "[malo", '["Elevator", "Air conditioning"]',
                      '["Balcony", "Pool", "TV", "Dedicated workspace", "Pets allowed", "Self check-in", "Free parking"]'],
        "minimum_nights": [1, 400, 30, np.nan, 2, 3], "bathrooms_text": ["1 bath"] * n, "bathrooms": [np.nan] * n,
        "bedrooms": [1, np.nan, 2, 1, 1, 1], "beds": [1, 2, np.nan, 1, 1, 1],
        "last_scraped": ["2026-09-20"] * n, "last_review": ["2026-01-01"] * n,
        "calculated_host_listings_count": [2, 2, 2, 2, 1, 7], "license": [np.nan, "Exempt", "En proceso",
                                                                          "Spain - National registration number<br />X", np.nan, "VT"],
        "hosts_time_as_host_years": [1, 2, 3, 3, 4, 5], "hosts_time_as_host_months": [6, 0, 0, 0, 0, 0],
        "estimated_occupancy_l365d": [0, 10, 20, 30, 40, 50],
        "review_scores_rating": [np.nan, 4.5, 4.0, 4.0, 5.0, 4.8],
    })


def test_limpiar_contabilidad_no_muta_y_reglas():
    raw = _listings_sinteticos()
    copia = raw.copy()
    barrios = pd.DataFrame({"neighbourhood_group": ["Centro"], "neighbourhood": ["Sol"]})
    df, reg = limpieza.limpiar(raw, barrios, CIUDAD)
    pd.testing.assert_frame_equal(raw, copia)
    assert reg["duplicados"]["eliminados"] == 1 and len(df) == 5
    assert reg["contabilidad_filas"] == {"filas_raw": 6, "duplicados_eliminados": 1, "filas_procesadas": 5}
    assert reg["precio"]["cero"] == 1 and reg["precio"]["nulo"] == 1
    assert df["precio_valido"].sum() == 3
    assert reg["minimum_nights"]["extremo_mayor_365"] == 1
    assert reg["coordenadas"]["fuera_bbox_o_nulas"] == 1
    assert reg["imputados"] == {"bedrooms": 1, "beds": 1}
    assert reg["conversiones"]["amenities"]["malformadas_json"] == 1  # se cuenta antes de quitar duplicados
    assert df["price"].dtype == "float64" and df["host_is_superhost"].dtype == bool
    assert "amenities" not in df and "bathrooms" in df and "bathrooms" not in reg["columnas_eliminadas"]


def test_features_y_validacion():
    raw = _listings_sinteticos()
    barrios = pd.DataFrame({"neighbourhood_group": ["Centro"], "neighbourhood": ["Sol"]})
    df, _ = limpieza.limpiar(raw, barrios, CIUDAD)
    df, meta = features.agregar_features(df, CIUDAD, pd.Timestamp("2026-09-20"))
    assert df["ocupacion_estimada"].between(0, 1).all()
    assert df.loc[df["accommodates"] == 0, "precio_por_persona"].isna().all()
    assert df.loc[~df["coords_validas"], "distancia_centro_km"].isna().all()
    assert meta["antiguedad_fuente"].startswith("hosts_time_as_host_years")
    assert df.loc[df["id"] == 1, "antiguedad_host_anios"].iloc[0] == pytest.approx(1.5)
    sin_doc = limpieza.validar_procesado(df, diccionario.documentadas())
    assert isinstance(sin_doc, list)


def test_validar_procesado_detecta_violaciones():
    raw = _listings_sinteticos()
    barrios = pd.DataFrame({"neighbourhood_group": ["Centro"], "neighbourhood": ["Sol"]})
    df, _ = limpieza.limpiar(raw, barrios, CIUDAD)
    df, _ = features.agregar_features(df, CIUDAD, pd.Timestamp("2026-09-20"))
    roto = df.assign(ocupacion_estimada=df["ocupacion_estimada"] + 5)
    with pytest.raises(ErrorDatos):
        limpieza.validar_procesado(roto)


def test_ocupacion_limites():
    r = features.ocupacion_estimada(pd.Series([0, 365, 400]))
    assert r.tolist() == [1.0, 0.0, 0.0]


def test_antiguedad_con_host_since():
    df = pd.DataFrame({"host_since": pd.to_datetime(["2020-09-20", None]),
                       "hosts_time_as_host_years": [9, 2], "hosts_time_as_host_months": [0, 6]})
    r, fuente = features.antiguedad_host(df, pd.Timestamp("2026-09-20"))
    assert fuente == "host_since" and r.iloc[0] == pytest.approx(6, abs=0.01) and r.iloc[1] == 2.5


def test_licencia_tipo():
    s = pd.Series([np.nan, "Exempt", "Madrid - Regional registration number<br />Exempt - seasonal",
                   "Spain - National registration number<br />ESF", "En proceso", "VT"])
    assert features.licencia_tipo(s).tolist() == ["Sin licencia", "Exenta", "Exenta", "Con nº de registro",
                                                  "Otro / en proceso", "Otro / en proceso"]


def test_categoria_precio_cuartiles():
    p = pd.Series([10.0, 20, 30, 40, 50, 60, 70, 80, np.nan])
    cat, cortes = features.categoria_precio(p, p.notna())
    assert len(cortes) == 5 and cat.iloc[0] == "Económico" and cat.iloc[7] == "Alto" and pd.isna(cat.iloc[8])


def test_columnas_config_documentadas():
    assert diccionario.columnas_requeridas_documentadas() == []
