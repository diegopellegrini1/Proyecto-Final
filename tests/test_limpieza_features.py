"""Tests de conversiones, reglas de limpieza, features y diccionario."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, diccionario, features, limpieza
from src.errores import ErrorDatos

BARRIOS = {"madrid": {"Sol"}, "mallorca": {"Palma de Mallorca"}, "euskadi": {"Donostia-San Sebastián"}}


@pytest.mark.parametrize("texto,esperado", [
    ("$1,234.00", 1234.0), ("$364.75", 364.75), ("1,234.50", 1234.5), ("$-10.00", -10.0), ("-$5", -5.0),
    ("€80.00", 80.0), ("1234", 1234.0),
    # No convertibles (política explícita: no se adivina el formato).
    ("EUR -25,00", np.nan), ("abc123xyz", np.nan), ("$1.2.3", np.nan), ("12,34", np.nan), ("", np.nan),
    ("basura", np.nan), (np.nan, np.nan),
])
def test_parse_precio(texto, esperado):
    r = limpieza.parse_precio(pd.Series([texto]))
    assert r.dtype == "float64"
    if np.isnan(esperado):
        assert np.isnan(r.iloc[0])
    else:
        assert r.iloc[0] == esperado


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


# Cadenas reales del snapshot (incluidos los casos ambiguos) y el flag esperado para cada patrón.
CASOS_AMENITIES = [
    ("tiene_tv", "TV", True), ("tiene_tv", "32 inch HDTV with Netflix", True), ("tiene_tv", "HDTV", True),
    ("tiene_tv", "TV with standard cable", True), ("tiene_tv", "Netflix", False),
    ("tiene_lavadora", "Washer", True), ("tiene_lavadora", "Free washer – in unit", True),
    ("tiene_lavadora", "Washer –\xa0in building", True), ("tiene_lavadora", "Dishwasher", False),
    ("tiene_piscina", "Pool", True), ("tiene_piscina", "Shared outdoor pool - available seasonally, open specific hours", True),
    ("tiene_piscina", "Pool table", False), ("tiene_piscina", "Pool toys", False), ("tiene_piscina", "Pool view", False),
    ("tiene_piscina", "Whirlpool refrigerator", False),
    ("tiene_wifi", "Wifi", True), ("tiene_wifi", "Fast wifi – 500 Mbps", True), ("tiene_wifi", "Pocket wifi", False),
    ("tiene_cocina", "Kitchen", True), ("tiene_cocina", "Kitchenette", True), ("tiene_cocina", "Outdoor kitchen", False),
    ("tiene_aire_acondicionado", "Central air conditioning", True), ("tiene_aire_acondicionado", "Ceiling fan", False),
    ("tiene_aire_acondicionado", "Air conditioning", True),
    ("tiene_aire_acondicionado", "AC - split type ductless system", True),
    ("tiene_aire_acondicionado", "Window AC unit", True),
    ("tiene_aire_acondicionado", "Portable air conditioning", True),
    ("tiene_aire_acondicionado", "Aire acondicionado dentro del salón Mitsubishi. conditioner", True),
    ("tiene_aire_acondicionado", "Heating - split type ductless system", False),
    ("tiene_aire_acondicionado", "Conditioner", False), ("tiene_aire_acondicionado", "Hair dryer", False),
    ("tiene_aire_acondicionado", "High chair", False),
    ("tiene_calefaccion", "Heating - split type ductless system", True), ("tiene_calefaccion", "Hot water", False),
    ("tiene_calefaccion", "Heating", True), ("tiene_calefaccion", "Central heating", True),
    ("tiene_calefaccion", "Radiant heating", True), ("tiene_calefaccion", "Portable heater", True),
    ("tiene_calefaccion", "Heat pump", True), ("tiene_calefaccion", "AC - split type ductless system", False),
    ("tiene_calefaccion", "Movie theater", False),
    ("tiene_calefaccion", "Oven with a fast-heating, small oven with grill when something smaller is required. oven",
     False),
    ("tiene_ascensor", "Elevator", True), ("tiene_ascensor", "Stairs", False),
    ("tiene_terraza_balcon", "Private patio or balcony", True), ("tiene_terraza_balcon", "Backyard", False),
    ("tiene_parking_cualquiera", "Free street parking", True),
    ("tiene_parking_cualquiera", "Free residential garage on premises – 1 space", True),
    ("tiene_parking_cualquiera", "EV charger", False),
    ("tiene_espacio_trabajo", "Dedicated workspace", True), ("tiene_espacio_trabajo", "Desk", False),
    ("admite_mascotas", "Pets allowed", True), ("admite_mascotas", "Pets live on this property", False),
    ("llegada_autonoma", "Self check-in", True), ("llegada_autonoma", "Smart lock", False),
]


@pytest.mark.parametrize("columna,cadena,esperado", CASOS_AMENITIES)
def test_flag_amenity_por_patron(columna, cadena, esperado):
    r = limpieza.flags_amenities(pd.Series([json.dumps([cadena])]))
    assert bool(r.loc[0, columna]) is esperado


def test_todos_los_patrones_tienen_caso_positivo_y_negativo():
    for col in config.PATRONES_AMENITIES:
        valores = {e for c, _, e in CASOS_AMENITIES if c == col}
        assert valores == {True, False}, col


def test_flags_amenities_lista_no_mezcla_elementos():
    # «Dish» y «washer» en elementos distintos no forman una lavadora; NaN → todo False.
    r = limpieza.flags_amenities(pd.Series(['["Dishwasher", "Pool table", "HDTV"]', np.nan]))
    assert not r.loc[0, "tiene_lavadora"] and not r.loc[0, "tiene_piscina"] and r.loc[0, "tiene_tv"]
    assert not r.loc[1].any()


def test_comprobar_prevalencias():
    ok = {c: (lo + hi) / 2 for c, (lo, hi) in config.COTAS_PREVALENCIA_AMENITIES.items()}
    assert limpieza.comprobar_prevalencias(ok, 10_000) == []
    fuera = {**ok, "tiene_piscina": 60.0}
    assert len(limpieza.comprobar_prevalencias(fuera, 10_000)) == 1
    assert limpieza.comprobar_prevalencias(fuera, 10) == []        # pocas filas: sin cotas
    assert len(limpieza.comprobar_prevalencias({**ok, "tiene_tv": 0.0}, 10)) == 1  # el cero siempre falla


def test_vallas_iqr_log_valores_exactos():
    # ln = [1, 2, 2, 2, 3, 10] → Q1 = 2, Q3 = 2,75 (cuantiles lineales), IQR = 0,75.
    precio = pd.Series([np.e ** k for k in (1, 2, 2, 2, 3, 10)])
    inf, sup = limpieza.vallas_iqr_log(precio)
    assert inf == pytest.approx(np.exp(2 - 1.5 * 0.75)) and sup == pytest.approx(np.exp(2.75 + 1.5 * 0.75))


def test_politica_nulos_siempre_resuelve():
    for col in ("price", "review_scores_rating", "beds", "columna_nueva_futura", "price_quote_total_price"):
        assert limpieza.regla_nulos(col)[1]
    assert limpieza.regla_nulos("columna_nueva_futura")[0] == ".*"


def _cotizacion(subtotal: float, descuento: float, mensual: bool) -> str:
    return json.dumps({"quote": {"nightly_subtotal": str(subtotal), "discount_amount": str(descuento),
                                 "is_monthly_priced": mensual, "currency": "EUR",
                                 "raw_price_line_items": [{"item_type": "discount_amount",
                                                           "description": "Weekly stay discount"}] if descuento else []}})


def _listings_sinteticos() -> pd.DataFrame:
    """Entrada de prueba mínima con el formato del conjunto unido (no son datos de análisis).

    Filas 0-4 en Madrid (la fila 2 con latitud fuera de la envolvente de Madrid), fila 5 en Mallorca (región:
    sin distancia al centro) y fila 6 en Euskadi con el nombre doblemente codificado.
    """
    n = 7
    base = pd.DataFrame({
        "ciudad": ["madrid"] * 5 + ["mallorca", "euskadi"], "fecha_snapshot": ["2026-09-17"] * n,
        "id": [1, 2, 3, 3, 5, 6, 7], "name": ["A", "B<br/>", "C", "C", "D", "E", "F"],
        "host_id": [10, 10, 11, 11, 12, 13, 14],
        "price": ["$100.00", "$0.00", np.nan, "$50.00", "$1,000.00", "$-80.00", "$120.00"],
        "room_type": ["Entire home/apt"] * 3 + ["Private room"] * 4, "accommodates": [2, 4, 0, 1, 2, 3, 2],
        "latitude": [40.42, 40.40, 41.5, 40.41, 40.43, 39.57, 43.32],
        "longitude": [-3.70] * 5 + [2.65, -1.98],
        "neighbourhood_cleansed": ["Sol"] * 5 + ["Palma de Mallorca", "Donostia-San SebastiÃ¡n"],
        "neighbourhood_group_cleansed": ["Centro"] * 5 + [np.nan, "GuipÃºzcoa"],
        "host_is_superhost": ["t", "f", "f", "f", "t", "f", "f"], "availability_365": [0, 365, 100, 0, 200, 50, 10],
        "number_of_reviews_ltm": [0, 1, 2, 3, 0, 1, 1], "number_of_reviews": [0, 5, 2, 3, 1, 1, 1],
        "amenities": ['["Wifi"]', '["Kitchen", "Heating"]', '["Washer"]', "[malo", '["Elevator", "Air conditioning"]',
                      '["Balcony", "Pool", "TV", "Dedicated workspace", "Pets allowed", "Self check-in", "Free parking"]',
                      '["Wifi"]'],
        "minimum_nights": [1, 400, 30, np.nan, 2, 3, 1], "bathrooms_text": ["1 bath"] * n, "bathrooms": [np.nan] * n,
        "bedrooms": [1, np.nan, 2, 1, 1, 1, 1], "beds": [1, 2, np.nan, 1, 1, 1, 1],
        "last_scraped": ["2026-09-20"] * n, "last_review": ["2026-01-01"] * n,
        "calculated_host_listings_count": [2, 2, 2, 2, 1, 7, 1],
        "license": [np.nan, "Exempt", "En proceso", "Spain - National registration number<br />X", np.nan, "VT", np.nan],
        "hosts_time_as_host_years": [1, 2, 3, 3, 4, 5, 1], "hosts_time_as_host_months": [6, 0, 0, 0, 0, 0, 0],
        "estimated_occupancy_l365d": [0, 10, 20, 30, 40, 50, 0],
        "estimated_revenue_l365d": [0, 100, 200, 300, 400, 500, 0],
        "review_scores_rating": [np.nan, 4.5, 4.0, 4.0, 5.0, 4.8, 4.9],
        "price_quote_checkin_date": ["2026-09-20", "2026-10-01", None, "2026-09-25", "2026-11-02", "2026-09-21",
                                     "2026-09-28"],
        "price_quote_checkout_date": ["2026-09-21", "2027-11-05", None, "2026-09-27", "2026-11-04", "2026-09-24",
                                      "2026-09-29"],
        "price_quote_raw": [_cotizacion(100, 0, False), _cotizacion(0, 0, False), np.nan, _cotizacion(330, 30, True),
                            _cotizacion(2000, 0, False), "no es json", _cotizacion(120, 0, False)],
        "codigo_distrito_ine": ["2807901"] * 5 + ["0704001", "2006901"],
        "codigo_municipio_ine": ["28079"] * 5 + ["07040", "20069"],
        "municipio_ine": ["Madrid"] * 5 + ["Palma", "Donostia/San Sebastián"],
    })
    for col in config.COLUMNAS_INE_FINALES:
        base[col] = np.linspace(10, 20, n)
    return base


def test_limpiar_contabilidad_no_muta_y_reglas():
    raw = _listings_sinteticos()
    copia = raw.copy()
    df, reg = limpieza.limpiar(raw, BARRIOS)
    pd.testing.assert_frame_equal(raw, copia)
    assert reg["duplicados"]["eliminados"] == 1 and len(df) == 6
    assert reg["contabilidad_filas"] == {"filas_raw": 7, "duplicados_eliminados": 1, "filas_procesadas": 6}
    assert reg["precio"]["cero"] == 1 and reg["precio"]["nulo"] == 1 and reg["precio"]["negativo"] == 1
    assert df["precio_valido"].sum() == 3  # 100 €, 1.000 € y 120 €; el negativo y el cero no son válidos
    assert reg["minimum_nights"]["extremo_mayor_365"] == 1
    assert reg["coordenadas"]["fuera_bbox_o_nulas"] == 1
    assert reg["imputados"] == {"bedrooms": 1, "beds": 1}
    assert reg["conversiones"]["amenities"]["malformadas_json"] == 1  # se cuenta antes de quitar duplicados
    assert df["price"].dtype == "float64" and df["host_is_superhost"].dtype == bool
    assert "amenities" not in df and "bathrooms" in df and "bathrooms" not in reg["columnas_eliminadas"]
    assert "host_location" not in df


def test_minimum_nights_vacio_se_rellena_con_la_media_de_los_proximos_30_dias():
    raw = _listings_sinteticos()
    raw.loc[4, "minimum_nights"] = np.nan  # anuncio id 5, con precio
    raw["minimum_nights_avg_ntm"] = [1.0, 400.0, 30.0, 45.0, 45.0, 3.0, 1.0]
    df, reg = limpieza.limpiar(raw, BARRIOS)
    fila = df.loc[df["id"] == 5].iloc[0]
    assert fila["minimum_nights"] == 45.0 and fila["minimum_nights_imputado"] and fila["alquiler_larga_estancia"]
    mn = reg["minimum_nights"]
    assert mn["rellenados_avg_ntm"] == int(df["minimum_nights_imputado"].sum()) >= 1
    assert mn["rellenados_con_precio"] == mn["rellenados_con_precio_larga_estancia"] == 1 and mn["nulos"] == 0
    assert mn["pct_concordancia_larga_estancia_avg_ntm"] == 100.0
    # Sin la columna alternativa no se inventa nada: el nulo se conserva.
    df2, reg2 = limpieza.limpiar(_listings_sinteticos(), BARRIOS)
    assert reg2["minimum_nights"]["nulos"] == reg2["minimum_nights"]["nulos_originales"]
    assert not df2["minimum_nights_imputado"].any()


def test_limpieza_por_ciudad_bbox_propio_y_recuentos():
    df, reg = limpieza.limpiar(_listings_sinteticos(), BARRIOS)
    # La latitud 41,5 está fuera de Madrid; el anuncio de Mallorca (39,57) y el de Euskadi son válidos en SU bbox.
    assert df.groupby("ciudad")["coords_validas"].sum().to_dict() == {"euskadi": 1, "madrid": 3, "mallorca": 1}
    pc = reg["por_ciudad"]
    assert set(pc) == {"madrid", "mallorca", "euskadi"}
    assert pc["madrid"]["filas"] == 4 and pc["madrid"]["fuera_bbox_o_nulas"] == 1
    assert pc["madrid"]["precio_cero"] == 1 and pc["madrid"]["precio_nulo"] == 1
    assert pc["mallorca"]["precio_valido"] == 0 and pc["euskadi"]["precio_valido"] == 1
    assert sum(v["filas"] for v in pc.values()) == len(df)
    assert reg["barrios"]["por_ciudad"]["euskadi"]["no_oficiales"] == []  # el nombre reparado sí es oficial


def test_reparar_mojibake_en_nombres():
    df, reg = limpieza.limpiar(_listings_sinteticos(), BARRIOS)
    eus = df[df["ciudad"] == "euskadi"].iloc[0]
    assert eus["neighbourhood_cleansed"] == "Donostia-San Sebastián" and eus["neighbourhood_group_cleansed"] == "Guipúzcoa"
    assert reg["mojibake_reparado"] == {"neighbourhood_cleansed": 1, "neighbourhood_group_cleansed": 1}
    assert limpieza.reparar_nombres(pd.Series(["Sol", np.nan])).tolist()[0] == "Sol"


def test_naturaleza_del_precio_desde_cotizacion():
    df, reg = limpieza.limpiar(_listings_sinteticos(), BARRIOS)
    pn = reg["precio_naturaleza"]
    # Con precio > 0 y cotización parseable: 100 €, 50 € (30 € de descuento sobre 330 €, mensual), 1.000 € y 120 €.
    assert pn["n_cotizaciones_con_precio"] == 4
    assert pn["n_con_descuento"] == 1 and pn["pct_con_descuento"] == pytest.approx(25.0)
    assert pn["mediana_pct_descuento"] == pytest.approx(30 / 330 * 100)
    assert pn["n_tarifa_mensual"] == 1 and pn["partidas_descuento"] == {"Weekly stay discount": 1}
    assert df.loc[df["id"] == 1, "noches_cotizadas"].iloc[0] == 1.0
    assert df["es_tarifa_mensual"].dtype == bool


def test_limpiar_columna_obligatoria_ausente_da_error_accionable():
    with pytest.raises(ErrorDatos, match="name"):
        limpieza.limpiar(_listings_sinteticos().drop(columns="name"), BARRIOS)


def _procesado() -> tuple[pd.DataFrame, dict]:
    df, _ = limpieza.limpiar(_listings_sinteticos(), BARRIOS)
    return features.agregar_features(df, pd.Timestamp("2026-09-20"))


def test_features_y_validacion():
    df, meta = _procesado()
    assert df["no_disponible_365"].between(0, 1).all()
    assert df.loc[~df["coords_validas"], "distancia_centro_km"].isna().all()
    assert meta["antiguedad_fuente"].startswith("hosts_time_as_host_years")
    assert df.loc[df["id"] == 1, "antiguedad_host_anios"].iloc[0] == pytest.approx(1.5)
    assert df.loc[df["id"] == 1, "mes_cotizacion"].iloc[0] == "2026-09"
    assert limpieza.validar_procesado(df, diccionario.documentadas()) == sorted(
        set(df.columns) - diccionario.documentadas())
    assert set(config.COLUMNAS_FINALES) <= diccionario.documentadas()


def test_distancia_centro_solo_en_ciudades_urbanas():
    df, meta = _procesado()
    madrid = df[(df["ciudad"] == "madrid") & df["coords_validas"]]
    assert madrid["distancia_centro_km"].notna().all() and (madrid["distancia_centro_km"] < 5).all()
    assert df.loc[df["ciudad"].isin(["mallorca", "euskadi"]), "distancia_centro_km"].isna().all()
    assert set(meta["centros"]) == set(config.CIUDADES_URBANAS)


def test_validar_procesado_rechaza_distancia_en_region():
    df, _ = _procesado()
    roto = df.assign(distancia_centro_km=df["distancia_centro_km"].fillna(3.0))
    with pytest.raises(ErrorDatos, match="región"):
        limpieza.validar_procesado(roto)


def test_licencia_tipo_sin_columna_license_es_serie_string():
    df, _ = limpieza.limpiar(_listings_sinteticos().drop(columns="license"), BARRIOS)
    df, meta = features.agregar_features(df, pd.Timestamp("2026-09-20"))
    assert str(df["licencia_tipo"].dtype) == "string"
    assert df["licencia_tipo"].eq("Sin licencia").all() and meta["licencia_tipo_conteo"] == {"Sin licencia": 6}
    assert not df["licencia_registro_nacional"].any()


def test_validar_procesado_detecta_violaciones():
    df, _ = _procesado()
    roto = df.assign(no_disponible_365=df["no_disponible_365"] + 5)
    with pytest.raises(ErrorDatos):
        limpieza.validar_procesado(roto)
    with pytest.raises(ErrorDatos, match="falta la columna final"):
        limpieza.validar_procesado(df.drop(columns=config.RENTA_PRINCIPAL))


def test_no_disponible_limites():
    r = features.no_disponible_365(pd.Series([0, 365, 400, 73]))
    assert r.tolist() == [1.0, 0.0, 0.0, pytest.approx(0.8)]


def test_antiguedad_con_host_since():
    df = pd.DataFrame({"host_since": pd.to_datetime(["2020-09-20", None]),
                       "hosts_time_as_host_years": [9, 2], "hosts_time_as_host_months": [0, 6]})
    r, fuente = features.antiguedad_host(df, pd.Timestamp("2026-09-20"))
    assert fuente == "host_since" and r.iloc[0] == pytest.approx(6, abs=0.01) and r.iloc[1] == 2.5


def test_licencia_tipo():
    s = pd.Series([np.nan, "Exempt", "Madrid - Regional registration number<br />Exempt - seasonal",
                   "Spain - National registration number<br />ESF", "En proceso", "VT",
                   "Spain - National registration number<br />ESF<br />Madrid - Regional registration number<br />Exempt"])
    assert features.licencia_tipo(s).tolist() == ["Sin licencia", "Exenta", "Exenta", "Con nº de registro",
                                                  "Otro / en proceso", "Otro / en proceso", "Con nº de registro"]
    assert features.registro_nacional(s).tolist() == [False, False, False, True, False, False, True]


def test_categoria_precio_cuartiles():
    p = pd.Series([10.0, 20, 30, 40, 50, 60, 70, 80, np.nan])
    cat, cortes = features.categoria_precio(p, p.notna())
    assert cortes == [10.0, 27.5, 45.0, 62.5, 80.0]
    assert cat.tolist()[:8] == ["Económico", "Económico", "Medio-bajo", "Medio-bajo", "Medio-alto", "Medio-alto",
                                "Alto", "Alto"] and pd.isna(cat.iloc[8])
