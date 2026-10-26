"""Tests del render de documentos, del verificador y de la descarga (con sesión HTTP falsa, sin red)."""

import gzip
import importlib.util
import io
import json

import pytest

from src import carga, config, informe
from src.errores import ErrorDescarga, ErrorInforme

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


def test_afirmacion_falsa_lanza():
    with pytest.raises(ErrorInforme):
        informe.evaluar_afirmaciones({})


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


def test_manifiesto_real_coincide_con_hash():
    if not config.RUTA_MANIFIESTO.exists():
        pytest.skip("data/raw no descargado")
    manifiesto = carga.verificar_manifiesto()
    assert {f["archivo"] for f in manifiesto["ficheros"]} == set(config.FICHEROS_OBLIGATORIOS)
    assert json.loads(config.RUTA_MANIFIESTO.read_text(encoding="utf-8"))["licencia"]["nombre"]
