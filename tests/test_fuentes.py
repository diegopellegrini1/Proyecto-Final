"""Tests de las dos fuentes (Inside Airbnb España + INE ADRH) y de su unión. Sin red: sesiones falsas."""

import gzip
import hashlib
import io

import numpy as np
import pandas as pd
import pytest

from src import config, fuentes
from src.errores import ErrorDatos, ErrorDescarga


# --------------------------------------------------------------------------
# URLs con tilde y descubrimiento
# --------------------------------------------------------------------------
MOJIBAKE = "andaluc\u00c3\u00ada"  # 'í' en UTF-8 leído como Latin-1 (como lo sirve la página)


def test_reparar_mojibake_y_codificar_url():
    assert fuentes.reparar_mojibake(MOJIBAKE) == "andalucía"
    assert fuentes.reparar_mojibake("andalucía") == "andalucía"          # idempotente
    assert fuentes.reparar_mojibake("madrid") == "madrid"
    triple = "andalucía".encode("utf-8").decode("latin-1").encode("utf-8").decode("latin-1")
    assert fuentes.reparar_mojibake(triple) == "andalucía"
    url = f"https://data.insideairbnb.com/spain/{MOJIBAKE}/malaga/2026-09-28/data/listings.csv.gz"
    esperado = "https://data.insideairbnb.com/spain/andaluc%C3%ADa/malaga/2026-09-28/data/listings.csv.gz"
    assert fuentes.codificar_url(url) == esperado
    assert fuentes.codificar_url(esperado) == esperado                  # no recodifica %XX


def _html(*urls):
    return "".join(f'<a href="{u}">x</a>' for u in urls)


def test_descubrir_urls_espana_elige_snapshot_completo_mas_reciente(caplog):
    base = f"https://data.insideairbnb.com/spain/{MOJIBAKE}/malaga"
    otra = "https://data.insideairbnb.com/spain/canarias/tenerife/2026-01-01/data/listings.csv.gz"
    html = _html(
        f"{base}/2026-06-01/data/listings.csv.gz", f"{base}/2026-06-01/visualisations/neighbourhoods.csv",
        f"{base}/2026-06-01/visualisations/neighbourhoods.geojson",
        f"{base}/2026-09-28/data/listings.csv.gz", f"{base}/2026-09-28/visualisations/neighbourhoods.csv",
        f"{base}/2026-09-28/visualisations/neighbourhoods.geojson",
        f"{base}/2026-12-01/data/listings.csv.gz",                    # incompleto: se ignora
        f"{base}/2026-09-28/data/reviews.csv.gz", otra)
    res = fuentes.descubrir_urls_espana(html)
    assert set(res) == {"malaga"}
    assert res["malaga"]["fecha_snapshot"] == "2026-09-28"
    assert set(res["malaga"]["urls"]) == set(config.FICHEROS_OBLIGATORIOS)
    assert all("/andaluc%C3%ADa/" in u for u in res["malaga"]["urls"].values())
    assert "canarias/tenerife" in caplog.text


# --------------------------------------------------------------------------
# Descarga verificada
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status=200, contenido=b"", headers=None):
        self.status_code, self._c, self.headers = status, contenido, headers or {}
        self.text = contenido.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._c), chunk_size):
            yield self._c[i:i + chunk_size]

    def close(self):
        pass


class _Sesion:
    def __init__(self, respuestas):
        self.respuestas, self.urls = respuestas, []

    def get(self, url, **kw):
        self.urls.append(url)
        return self.respuestas[url].pop(0)


CSV_INE = ("\ufeffMunicipios;Distritos;Secciones;Indicadores de renta media y mediana;Periodo;Total\n"
           "28079 Madrid;;;Renta neta media por persona;2024;20.123\n"
           "28079 Madrid;2807901 Madrid distrito 01;;Renta neta media por persona;2024;23.342\n"
           "28079 Madrid;2807901 Madrid distrito 01;2807901001 Madrid sección 01001;Renta neta media por persona;2024;.\n"
           ).encode("utf-8")
URL_INE = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31097.csv"


def test_descarga_ine_comprime_sin_perdida_y_es_determinista(tmp_path):
    hashes = []
    for i in range(2):
        sesion = _Sesion({URL_INE: [_Resp(contenido=CSV_INE, headers={"Content-Length": str(len(CSV_INE))})]})
        destino = tmp_path / f"t{i}.csv.gz"
        e = fuentes.descargar_verificado(URL_INE, destino, sesion, config.DOMINIOS_PERMITIDOS_INE,
                                         validar=fuentes.validar_csv_ine, comprimir=True, espera_base=0)
        assert gzip.decompress(destino.read_bytes()) == CSV_INE
        assert e["sha256_original"] == hashlib.sha256(CSV_INE).hexdigest()
        assert e["bytes_original"] == len(CSV_INE) and e["bytes"] == destino.stat().st_size
        hashes.append(e["sha256"])
        assert not list(tmp_path.glob("*.part"))
    assert hashes[0] == hashes[1]


def test_descarga_truncada_o_corrupta_reintenta_y_falla(tmp_path):
    truncada = _Resp(contenido=CSV_INE[:-10], headers={"Content-Length": str(len(CSV_INE))})
    corrupta = _Resp(contenido=b"<html>error</html>")
    buena = _Resp(contenido=CSV_INE)
    sesion = _Sesion({URL_INE: [truncada, corrupta, buena]})
    fuentes.descargar_verificado(URL_INE, tmp_path / "x.csv", sesion, config.DOMINIOS_PERMITIDOS_INE,
                                 validar=fuentes.validar_csv_ine, espera_base=0)
    assert len(sesion.urls) == 3
    sesion = _Sesion({URL_INE: [_Resp(contenido=b"basura")] * 3})
    with pytest.raises(ErrorDescarga, match="tras 3 intentos"):
        fuentes.descargar_verificado(URL_INE, tmp_path / "y.csv", sesion, config.DOMINIOS_PERMITIDOS_INE,
                                     validar=fuentes.validar_csv_ine, espera_base=0)
    assert not (tmp_path / "y.csv").exists()


def test_descarga_rechaza_dominio_y_404(tmp_path):
    with pytest.raises(ErrorDescarga, match="no permitida"):
        fuentes.descargar_verificado("https://evil.example/x.csv", tmp_path / "x", _Sesion({}),
                                     config.DOMINIOS_PERMITIDOS_INE, validar=fuentes.validar_csv_ine)
    with pytest.raises(ErrorDescarga, match="no permitida"):
        fuentes.descargar_verificado("http://www.ine.es/x.csv", tmp_path / "x", _Sesion({}),
                                     config.DOMINIOS_PERMITIDOS_INE, validar=fuentes.validar_csv_ine)
    sesion = _Sesion({URL_INE: [_Resp(status=404)]})
    with pytest.raises(ErrorDescarga, match="404"):
        fuentes.descargar_verificado(URL_INE, tmp_path / "x", sesion, config.DOMINIOS_PERMITIDOS_INE,
                                     validar=fuentes.validar_csv_ine)


def test_redireccion_a_otro_dominio_se_rechaza(tmp_path):
    sesion = _Sesion({URL_INE: [_Resp(status=302, headers={"Location": "https://evil.example/x"})]})
    with pytest.raises(ErrorDescarga, match="no permitida"):
        fuentes.descargar_verificado(URL_INE, tmp_path / "x", sesion, config.DOMINIOS_PERMITIDOS_INE,
                                     validar=fuentes.validar_csv_ine)


def test_verificar_manifiesto_detecta_cambios(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"hola")
    fuentes.escribir_manifiesto({"ficheros": [{"archivo": "a.txt", "sha256": hashlib.sha256(b"hola").hexdigest()}]},
                                tmp_path)
    fuentes.verificar_manifiesto(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"adios")
    with pytest.raises(ErrorDatos, match="SHA-256"):
        fuentes.verificar_manifiesto(tmp_path)


def test_descubrir_tablas_adrh():
    html = ('<a href="/jaxiT3/Tabla.htm?t=31097&amp;L=0">Indicadores de renta media y mediana</a>'
            '<a href="/jaxiT3/Tabla.htm?t=37727&amp;L=0">\u00cdndice de Gini y Distribuci&oacute;n de la renta P80/P20</a>')
    tablas = fuentes.descubrir_tablas_adrh(html)
    assert tablas == {"Indicadores de renta media y mediana": "31097",
                      "Índice de Gini y Distribución de la renta P80/P20": "37727"}


# --------------------------------------------------------------------------
# Parseo de tablas del INE
# --------------------------------------------------------------------------
def test_parsear_numero_ine():
    s = pd.Series(["17.216", "39,9", "1.297", ".", "", "2,5", None])
    np.testing.assert_allclose(fuentes.parsear_numero_ine(s).to_numpy(),
                               [17216, 39.9, 1297, np.nan, np.nan, 2.5, np.nan])
    with pytest.raises(ErrorDatos, match="no numéricos"):
        fuentes.parsear_numero_ine(pd.Series(["12", "abc"]))


def _escribir_gz(ruta, texto):
    ruta.write_bytes(gzip.compress(texto.encode("utf-8"), mtime=0))
    return ruta


def test_leer_tabla_adrh_niveles_sexo_y_fila_descolocada(tmp_path):
    texto = ("\ufeffMunicipios;Distritos;Secciones;Sexo;Distribución de la renta por unidad de consumo;Periodo;Total\n"
             "07040 Palma;;;Total;Población con ingresos por unidad de consumo por debajo 60% de la mediana;2024;20,9\n"
             "07040 Palma;;;Hombres;Población con ingresos por unidad de consumo por debajo 60% de la mediana;2024;19,0\n"
             "07040 Palma;0704001 Palma distrito 01;;Total;Población con ingresos por unidad de consumo por debajo 60% de la mediana;2024;.\n"
             "0311902006 Sant Joan d'Alacant sección 02006;;;Total;Población con ingresos por unidad de consumo por debajo 60% de la mediana;2024;12,4\n")
    df = fuentes.leer_tabla_adrh(_escribir_gz(tmp_path / "u.csv.gz", texto))
    assert df["nivel"].tolist() == ["municipio", "distrito", "seccion"]
    assert df["codigo"].tolist() == ["07040", "0704001", "0311902006"]
    assert df["nombre"].iloc[0] == "Palma"
    assert df["valor"].iloc[0] == pytest.approx(20.9) and np.isnan(df["valor"].iloc[1])
    assert df["periodo"].dtype == "int64"


def test_leer_tabla_adrh_estructura_invalida(tmp_path):
    with pytest.raises(ErrorDatos, match="Estructura inesperada"):
        fuentes.leer_tabla_adrh(_escribir_gz(tmp_path / "m.csv.gz", "a;b;c\n1;2;3\n"))
    malo = "Municipios;Distritos;Secciones;Ind;Periodo;Total\n123 X;;;Población;2024;5\n"
    with pytest.raises(ErrorDatos, match="longitud inesperada"):
        fuentes.leer_tabla_adrh(_escribir_gz(tmp_path / "n.csv.gz", malo))


def _largo_sintetico():
    filas = []
    for anio, cobertura in ((2023, True), (2024, True), (2025, False)):
        for ind in config.INDICADORES_ADRH:
            for nivel, codigo in (("municipio", "28079"), ("distrito", "2807901"), ("distrito", "2807902")):
                valor = float(len(ind) + anio) if cobertura or ind != "Índice de Gini" else np.nan
                filas.append({"nivel": nivel, "codigo": codigo, "nombre": "Madrid", "indicador": ind,
                              "periodo": anio, "valor": valor})
    return pd.DataFrame(filas)


def test_elegir_anio_e_indicadores_adrh():
    largo = _largo_sintetico()
    assert fuentes.elegir_anio_adrh(largo) == 2024          # 2025 no tiene Gini
    ancho = fuentes.indicadores_adrh(largo, 2024)
    assert len(ancho) == 3
    assert set(config.INDICADORES_ADRH.values()) <= set(ancho.columns)
    with pytest.raises(ErrorDatos, match="duplicadas"):
        fuentes.indicadores_adrh(pd.concat([largo, largo.iloc[[0]]]), 2023)


# --------------------------------------------------------------------------
# Mapeo anuncio -> sección y unión
# --------------------------------------------------------------------------
def _cuadrados_utm(lat, lon, lado=200.0):
    import shapely
    from pyproj import Transformer

    t = Transformer.from_crs(config.CRS_LISTINGS, config.CRS_SECCIONADO, always_xy=True)
    x, y = t.transform(lon, lat)
    return [shapely.box(xi - lado / 2, yi - lado / 2, xi + lado / 2, yi + lado / 2) for xi, yi in zip(x, y)]


def test_asignar_seccion_punto_en_poligono():
    lat = pd.Series([40.4169, 41.3870, 0.0, np.nan])
    lon = pd.Series([-3.7035, 2.1700, 0.0, np.nan])
    geoms = _cuadrados_utm(lat[:2].to_numpy(), lon[:2].to_numpy())
    res = fuentes.asignar_seccion(lat, lon, geoms, np.array(["2807901001", "0801901001"]))
    assert res.iloc[0] == "2807901001" and res.iloc[1] == "0801901001"
    assert res.iloc[2:].isna().all()


def _listings(n):
    return pd.DataFrame({"ciudad": "madrid", "id": np.arange(n), "latitude": 40.4, "longitude": -3.7,
                         "price": "$50.00", **{f"c{i}": i for i in range(17)}})


def test_unir_fuentes_conserva_filas_y_deja_nan(caplog):
    listings = _listings(4)
    asignacion = listings[["ciudad", "id"]].assign(
        codigo_seccion_ine=pd.array(["2807901001", "2807902001", "2807999001", pd.NA], dtype="string"))
    indicadores = fuentes.indicadores_adrh(_largo_sintetico(), 2024)
    unido = fuentes.unir_fuentes(listings, asignacion, indicadores, 2024)
    assert len(unido) == 4
    assert unido["codigo_distrito_ine"].tolist()[:3] == ["2807901", "2807902", "2807999"]
    assert unido["ine_dis_renta_neta_persona"].notna().tolist() == [True, True, False, False]
    assert unido["ine_mun_renta_neta_persona"].notna().tolist() == [True, True, True, False]
    assert (unido["municipio_ine"].iloc[:3] == "Madrid").all()
    tasas = fuentes.tasas_emparejamiento(unido)
    assert tasas.loc["TOTAL", "pct_distrito"] == 50.0
    assert "sin dato del INE" in caplog.text


def test_comprobar_requisitos_union():
    ok = pd.DataFrame({f"c{i}": 0 for i in range(19)} | {"ine_dis_x": 1.0}, index=range(config.MIN_FILAS_UNION))
    fuentes.comprobar_requisitos_union(ok)
    with pytest.raises(ErrorDatos, match="filas"):
        fuentes.comprobar_requisitos_union(ok.iloc[:-1])
    with pytest.raises(ErrorDatos, match="columnas"):
        fuentes.comprobar_requisitos_union(ok.drop(columns="c0"))
    with pytest.raises(ErrorDatos, match="Ningún anuncio"):
        fuentes.comprobar_requisitos_union(ok.assign(ine_dis_x=np.nan))
    with pytest.raises(ErrorDatos, match="Ningún anuncio"):
        fuentes.comprobar_requisitos_union(ok.rename(columns={"ine_dis_x": "c19"}))


def test_normalizacion_de_nombres_sin_similitud_difusa():
    assert fuentes.coincide_nombre("L'Escala", "Escala, L'")
    assert fuentes.coincide_nombre("Donostia-San Sebastián", "Donostia/San Sebastián")
    assert fuentes.coincide_nombre("Santa Maria del Camí", "Santa María del Camí")
    assert fuentes.coincide_nombre("Alicante/Alacant", "Alacant")      # variante bilingüe con barra
    assert not fuentes.coincide_nombre("Maó-Mahón", "Mahón")           # el guion sin espacios no se parte
    assert not fuentes.coincide_nombre("Palma de Mallorca", "Palma")   # solo vía alias revisado
    assert not fuentes.coincide_nombre("Sopelana", "Sopela")
    assert not fuentes.coincide_nombre(None, "Madrid")


def test_concordancia_distritos_y_municipios():
    unido = pd.DataFrame({
        "ciudad": ["madrid", "madrid", "mallorca", "mallorca", "mallorca"], "id": [1, 2, 3, 4, 5],
        "latitude": 0.0, "longitude": 0.0,
        "neighbourhood_group_cleansed": ["Centro", "Retiro", None, None, None],
        "neighbourhood_cleansed": ["Sol", "Ibiza", "Palma de Mallorca", "Campos", "Campos"],
        "codigo_distrito_ine": pd.array(["2807901", "2807904", "0704001", "0701401", pd.NA], dtype="string"),
        "municipio_ine": pd.array([None, None, "Palma", "Santanyí", pd.NA], dtype="string"),
        "codigo_seccion_ine": pd.array(["x"] * 4 + [pd.NA], dtype="string"),
    })
    tabla = fuentes.concordancia(unido)
    assert tabla["coincide"].tolist()[:4] == [True, False, True, False]
    assert pd.isna(tabla["coincide"].iloc[4])
    res = fuentes.resumen_concordancia(tabla)
    assert res.loc["TOTAL", "validados"] == 4 and res.loc["TOTAL", "pct_coinciden"] == 50.0


def test_alias_y_distritos_oficiales_bien_formados():
    assert all(len(c) == 7 for d in config.DISTRITOS_OFICIALES.values() for c in d.values())
    assert all(len(set(d.values())) == len(d) for d in config.DISTRITOS_OFICIALES.values())
    assert set(config.DISTRITOS_OFICIALES) <= set(config.CIUDADES_ESPANA)
    assert all(not fuentes.coincide_nombre(a, b) for a, b in config.ALIAS_MUNICIPIOS_INE.items())


# --------------------------------------------------------------------------
# Datos reales (si están en data/raw)
# --------------------------------------------------------------------------
requiere_datos = pytest.mark.skipif(
    not (config.DIR_RAW_AIRBNB / "MANIFEST.json").exists() or not (config.DIR_RAW_INE / "MANIFEST.json").exists(),
    reason="faltan los datos crudos de España")


@requiere_datos
def test_datos_reales_dos_fuentes_y_tamano_minimo():
    m_airbnb = fuentes.verificar_manifiesto(config.DIR_RAW_AIRBNB)
    m_ine = fuentes.verificar_manifiesto(config.DIR_RAW_INE)
    assert len({e["ciudad"] for e in m_airbnb["ficheros"]}) == len(config.CIUDADES_ESPANA)
    assert all((config.DIR_RAW_AIRBNB / e["archivo"]).stat().st_size < config.LIMITE_SUBIDA_BYTES
               for e in m_airbnb["ficheros"])
    assert all((config.DIR_RAW_INE / e["archivo"]).stat().st_size < config.LIMITE_SUBIDA_BYTES for e in m_ine["ficheros"])
    assert {e["tabla"] for e in m_ine["ficheros"]} == set(config.TABLAS_ADRH)
    listings = fuentes.leer_listings_espana()
    assert len(listings) >= config.MIN_FILAS_UNION
    asignacion = pd.read_csv(config.RUTA_ASIGNACION_SECCIONES, dtype={"codigo_seccion_ine": "string"})
    assert len(asignacion) == len(listings)
    assert asignacion["codigo_seccion_ine"].notna().mean() > 0.99


@requiere_datos
def test_datos_reales_parseo_ine_madrid():
    ruta = next(config.DIR_RAW_INE.glob("adrh_28_renta_*.csv.gz"))
    df = fuentes.leer_tabla_adrh(ruta)
    centro = df[(df["codigo"] == "2807901") & (df["indicador"] == "Renta neta media por persona")]
    assert centro["periodo"].is_unique and centro["valor"].notna().any()
    assert set(df["nivel"]) == {"municipio", "distrito", "seccion"}
