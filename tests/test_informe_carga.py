"""Tests del render de documentos, del verificador y de la descarga (con sesión HTTP falsa, sin red)."""

import dataclasses
import gzip
import importlib.util
import io
import json
import re

import pandas as pd
import pytest

from src import carga, config, informe
from src.errores import ErrorDatos, ErrorDescarga, ErrorInforme

_spec = importlib.util.spec_from_file_location("verificar_consistencia", config.RAIZ / "scripts" / "verificar_consistencia.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def _entorno(tmp_path, plantilla: str, findings: dict):
    (tmp_path / "t.md.j2").write_text(plantilla, encoding="utf-8")
    return informe.crear_entorno(findings, tmp_path).get_template("t.md.j2")


def test_render_cifra_con_marcador(tmp_path):
    texto = _entorno(tmp_path, '{{ cifra("a.b", "n0") }}', {"a": {"b": 1234}}).render()
    assert texto == "<!--f:a.b:n0-->1.234<!--/f-->"


def test_strict_undefined_y_ruta_inexistente(tmp_path):
    with pytest.raises(Exception):
        _entorno(tmp_path, "{{ no_existe }}", {}).render()
    with pytest.raises(ErrorInforme):
        _entorno(tmp_path, '{{ cifra("x.y", "n0") }}', {"x": {}}).render()


def test_afirmacion_no_evaluable_lanza():
    with pytest.raises(ErrorInforme):
        informe.evaluar_afirmaciones({})


requiere_findings = pytest.mark.skipif(not config.RUTA_FINDINGS.exists(), reason="falta reports/findings.json")


def _findings() -> dict:
    return json.loads(config.RUTA_FINDINGS.read_text(encoding="utf-8"))


@requiere_findings
def test_invariante_falsa_detiene_el_pipeline():
    f = _findings()
    f["moneda"]["moneda"] = "USD"
    with pytest.raises(ErrorInforme, match="moneda_eur"):
        informe.evaluar_afirmaciones(f)


@requiere_findings
def test_interpretativa_falsa_avisa_y_no_aborta(tmp_path, caplog):
    f = _findings()
    f["estadistica"]["kruskal_tipo"]["extra"]["grupo_mas_caro"] = "Habitación privada"
    resultado = informe.evaluar_afirmaciones(f)
    assert resultado["entero_mas_caro"] is False and "entero_mas_caro" in caplog.text
    f["afirmaciones"] = resultado
    texto = _entorno(tmp_path, '{{ afirmacion("entero_mas_caro") }}{% if cumple("entero_mas_caro") %}A{% else %}B'
                               '{% endif %}', f).render()
    assert "Aviso automático" in texto and texto.endswith("B")


def test_invariantes_e_interpretativas_no_se_solapan():
    ids_int = {i for i, _, _ in informe.INTERPRETATIVAS}
    assert not ids_int & informe.IDS_INVARIANTES
    assert "ml_mejor_que_ols" not in ids_int | informe.IDS_INVARIANTES  # comparación no homogénea retirada


def test_plantillas_usan_solo_afirmaciones_existentes():
    usadas = set()
    for nombre in informe.PLANTILLAS:
        texto = (config.DIR_PLANTILLAS / nombre).read_text(encoding="utf-8")
        usadas |= set(re.findall(r'(?:afirmacion|cumple)\("([^"]+)"\)', texto))
    assert usadas and usadas <= set(informe.DESCRIPCIONES)


def test_verificador_detecta_cifra_alterada():
    findings = {"a": {"b": 1234.5}}
    ok = "<!--f:a.b:n2-->1.234,50<!--/f-->"
    assert vc.verificar_marcadores(ok, findings, informe.obtener, informe.formatear)[0] == []
    mal = "<!--f:a.b:n2-->1.234,51<!--/f-->"
    assert vc.verificar_marcadores(mal, findings, informe.obtener, informe.formatear)[0]


def test_verificador_enlaces_y_placeholders(tmp_path):
    (tmp_path / "existe.md").write_text("x", encoding="utf-8")
    texto = "[a](existe.md) [b](no.md) [c](https://x.org) ![d](falta.png) [e](#ancla)"
    assert vc.enlaces_rotos(texto, tmp_path) == ["no.md", "falta.png"]
    assert vc.placeholders("Texto TODO y [completar]") == ["TODO", "[completar]"]
    assert vc.placeholders("Todo correcto") == []


def test_lint_plantilla_detecta_literal():
    assert vc.lint_plantilla("Hay 23.109 anuncios") != []
    assert vc.lint_plantilla('Hay {{ cifra("a", "n0") }} anuncios, IC 95 %, H0 y `x_365`') == []


class _Resp:
    def __init__(self, status=200, contenido=b"", headers=None, texto=""):
        self.status_code, self._c, self.headers, self.text = status, contenido, headers or {}, texto

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._c), chunk_size):
            yield self._c[i:i + chunk_size]

    def close(self):
        pass


class _Sesion:
    """Sesión falsa: devuelve respuestas en orden para cada URL."""

    def __init__(self, respuestas):
        self.respuestas, self.urls = respuestas, []

    def get(self, url, **kw):
        self.urls.append(url)
        return self.respuestas[url].pop(0)


def _gz_csv() -> bytes:
    buf = io.BytesIO()
    with gzip.open(buf, "wb") as f:
        f.write(b"id,price\n1,$10.00\n")
    return buf.getvalue()


URL = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2026-09-17/data/listings.csv.gz"


def test_descarga_reintenta_tras_gzip_truncado(tmp_path):
    bueno = _gz_csv()
    sesion = _Sesion({URL: [_Resp(contenido=bueno[:-6]), _Resp(contenido=bueno, headers={"Content-Length": str(len(bueno))})]})
    destino = carga.descargar_fichero(URL, tmp_path / "listings.csv.gz", sesion, espera_base=0)
    assert destino.read_bytes() == bueno and len(sesion.urls) == 2
    assert not list(tmp_path.glob("*.part"))


def test_descarga_tamano_incompleto_agota_reintentos(tmp_path):
    bueno = _gz_csv()
    sesion = _Sesion({URL: [_Resp(contenido=bueno, headers={"Content-Length": "999999"}) for _ in range(3)]})
    with pytest.raises(ErrorDescarga):
        carga.descargar_fichero(URL, tmp_path / "listings.csv.gz", sesion, espera_base=0)


def test_redireccion_a_dominio_no_permitido(tmp_path):
    sesion = _Sesion({URL: [_Resp(status=302, headers={"Location": "https://malicioso.example/x.gz"})]})
    with pytest.raises(ErrorDescarga):
        carga.descargar_fichero(URL, tmp_path / "x.csv.gz", sesion, espera_base=0)


def test_validar_url():
    with pytest.raises(ErrorDescarga):
        carga.validar_url("http://data.insideairbnb.com/x")
    carga.validar_url(URL)


def test_extraer_urls_y_licencia():
    base = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2026-09-17"
    html = (f'<a href="{base}/data/listings.csv.gz"></a><a href="{base}/visualisations/neighbourhoods.csv"></a>'
            f'<a href="{base}/visualisations/neighbourhoods.geojson"></a><a href="{base}/data/neighbourhoods.csv"></a>'
            '<a href="http://creativecommons.org/licenses/by/4.0/">CC</a>')
    urls = carga.extraer_urls(html, config.CIUDADES["madrid"])
    assert set(urls["2026-09-17"]) == set(config.FICHEROS_OBLIGATORIOS)
    assert urls["2026-09-17"]["neighbourhoods.csv"].endswith("/visualisations/neighbourhoods.csv")
    assert carga.detectar_licencia(html)["verificada_en_pagina"] is True


BASE_A = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2026-09-17"
BASE_B = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2026-06-10"


def _ok(contenido: bytes) -> "_Resp":
    return _Resp(contenido=contenido, headers={"Content-Length": str(len(contenido))})


def _ficheros_validos() -> dict[str, bytes]:
    return {"listings.csv.gz": _gz_csv(), "neighbourhoods.csv": b"neighbourhood_group,neighbourhood\nCentro,Sol\n",
            "neighbourhoods.geojson": json.dumps({"features": [{"properties": {"neighbourhood": "Sol"}}]}).encode()}


def test_snapshot_con_fechas_distintas_no_toca_data_raw(tmp_path):
    """Si el fallback trae un fichero de otro snapshot, se aborta y data/raw queda intacto."""
    dir_raw = tmp_path / "raw"
    dir_raw.mkdir()
    previo = {n: f"previo {n}".encode() for n in config.FICHEROS_OBLIGATORIOS}
    for n, b in previo.items():
        (dir_raw / n).write_bytes(b)
    f = _ficheros_validos()
    urls = {n: f"{BASE_A}/{config.FICHEROS_OBLIGATORIOS[n]}/{n}" for n in config.FICHEROS_OBLIGATORIOS}
    directa_geo = f"{BASE_B}/visualisations/neighbourhoods.geojson"
    ciudad = dataclasses.replace(config.CIUDADES["madrid"], url_directa_conocida={"neighbourhoods.geojson": directa_geo})
    sesion = _Sesion({urls["listings.csv.gz"]: [_ok(f["listings.csv.gz"])],
                      urls["neighbourhoods.csv"]: [_ok(f["neighbourhoods.csv"])],
                      urls["neighbourhoods.geojson"]: [_Resp(status=404)],
                      directa_geo: [_ok(f["neighbourhoods.geojson"])]})
    snapshot = carga.UrlsSnapshot(fecha="2026-09-17", urls=urls, licencia={}, origen="pagina")
    with pytest.raises(ErrorDescarga, match="snapshots distintos"):
        carga.descargar_snapshot(ciudad, snapshot, dir_raw, sesion)
    assert {n: (dir_raw / n).read_bytes() for n in previo} == previo
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".descarga_")]  # temporal borrado


def test_snapshot_coherente_registra_url_y_fecha_por_fichero(tmp_path):
    f = _ficheros_validos()
    urls = {n: f"{BASE_A}/{config.FICHEROS_OBLIGATORIOS[n]}/{n}" for n in config.FICHEROS_OBLIGATORIOS}
    sesion = _Sesion({urls[n]: [_ok(f[n])] for n in urls})
    snapshot = carga.UrlsSnapshot(fecha="2026-09-17", urls=urls, licencia={}, origen="pagina")
    ficheros = carga.descargar_snapshot(config.CIUDADES["madrid"], snapshot, tmp_path / "raw", sesion)
    assert {x["archivo"]: (x["url"], x["fecha_snapshot"]) for x in ficheros} == {n: (urls[n], "2026-09-17") for n in urls}
    assert (tmp_path / "raw" / "listings.csv.gz").read_bytes() == f["listings.csv.gz"]


class _SinRed:
    def get(self, url, **kw):
        raise AssertionError(f"petición de red en modo offline: {url}")


def test_extras_en_modo_offline_no_usan_la_red(tmp_path):
    manifiesto = {"fecha_snapshot": "2026-09-17", "extras": []}
    with pytest.raises(ErrorDescarga, match="sin descarga"):
        carga.asegurar_extras(config.CIUDADES["madrid"], manifiesto, ["reviews"], sin_descarga=True,
                              dir_extra=tmp_path, dir_raw=tmp_path, sesion=_SinRed())
    # Con el extra ya en caché y registrado en el manifiesto, tampoco hay red (ni en modo normal).
    (tmp_path / "reviews.csv.gz").write_bytes(b"x")
    manifiesto["extras"] = [{"archivo": "reviews.csv.gz", "sha256": carga.sha256_fichero(tmp_path / "reviews.csv.gz")}]
    for offline in (True, False):
        assert carga.asegurar_extras(config.CIUDADES["madrid"], manifiesto, ["reviews"], sin_descarga=offline,
                                     dir_extra=tmp_path, dir_raw=tmp_path, sesion=_SinRed()) is manifiesto


def test_gzip_con_demasiados_bytes_descomprimidos(tmp_path):
    ruta = tmp_path / "grande.csv.gz"
    with gzip.open(ruta, "wb") as f:
        f.write(b"id,price\n" + b"1,$10.00\n" * 400_000)
    carga.validar_contenido(ruta, "grande.csv.gz")  # con el tope por defecto, pasa
    with pytest.raises(ErrorDescarga, match="descomprimidos"):
        carga.validar_contenido(ruta, "grande.csv.gz", max_descomprimido=1_000_000)


def test_leer_listings_sin_columna_obligatoria(tmp_path):
    ruta = tmp_path / "listings.csv.gz"
    pd.DataFrame({c: [1] for c in config.COLUMNAS_OBLIGATORIAS if c != "estimated_revenue_l365d"}).to_csv(
        ruta, index=False, compression="gzip")
    with pytest.raises(ErrorDatos, match="estimated_revenue_l365d"):
        carga.leer_listings(ruta)


def test_manifiesto_real_coincide_con_hash():
    if not config.RUTA_MANIFIESTO.exists():
        pytest.skip("data/raw no descargado")
    manifiesto = carga.verificar_manifiesto()
    assert {f["archivo"] for f in manifiesto["ficheros"]} == set(config.FICHEROS_OBLIGATORIOS)
    assert json.loads(config.RUTA_MANIFIESTO.read_text(encoding="utf-8"))["licencia"]["nombre"]
