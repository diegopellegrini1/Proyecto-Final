"""Tests del render de documentos, del verificador y de la descarga (con sesión HTTP falsa, sin red)."""

import gzip
import importlib.util
import io
import json
import re

import pandas as pd
import pytest

from src import carga, config, fuentes, informe
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
    f["descriptivo"]["por_capacidad"][0]["mediana"] = 1e9
    resultado = informe.evaluar_afirmaciones(f)
    assert resultado["capacidad_creciente"] is False and "capacidad_creciente" in caplog.text
    f["afirmaciones"] = resultado
    texto = _entorno(tmp_path, '{{ afirmacion("capacidad_creciente") }}{% if cumple("capacidad_creciente") %}A'
                               '{% else %}B{% endif %}', f).render()
    assert "Aviso automático" in texto and texto.endswith("B")


@requiere_findings
def test_requisitos_del_enunciado_son_invariantes():
    f = _findings()
    f["dataset"]["requisitos"]["cumple_filas"] = False
    with pytest.raises(ErrorInforme, match="dataset_cumple_requisitos"):
        informe.evaluar_afirmaciones(f)


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


def _descargar(sesion, destino):
    return fuentes.descargar_verificado(URL, destino, sesion, config.DOMINIOS_PERMITIDOS,
                                        validar=fuentes._validador_airbnb("listings.csv.gz"), espera_base=0)


def test_descarga_reintenta_tras_gzip_truncado(tmp_path):
    bueno = _gz_csv()
    sesion = _Sesion({URL: [_Resp(contenido=bueno[:-6]), _Resp(contenido=bueno, headers={"Content-Length": str(len(bueno))})]})
    entrada = _descargar(sesion, tmp_path / "listings.csv.gz")
    assert (tmp_path / "listings.csv.gz").read_bytes() == bueno and len(sesion.urls) == 2
    assert entrada["sha256"] == carga.sha256_fichero(tmp_path / "listings.csv.gz")
    assert not list(tmp_path.glob("*.part"))


def test_descarga_tamano_incompleto_agota_reintentos(tmp_path):
    bueno = _gz_csv()
    sesion = _Sesion({URL: [_Resp(contenido=bueno, headers={"Content-Length": "999999"}) for _ in range(3)]})
    with pytest.raises(ErrorDescarga):
        _descargar(sesion, tmp_path / "listings.csv.gz")


def test_redireccion_a_dominio_no_permitido(tmp_path):
    sesion = _Sesion({URL: [_Resp(status=302, headers={"Location": "https://malicioso.example/x.gz"})]})
    with pytest.raises(ErrorDescarga):
        _descargar(sesion, tmp_path / "x.csv.gz")


def test_validar_url():
    with pytest.raises(ErrorDescarga):
        fuentes.validar_url("http://data.insideairbnb.com/x", config.DOMINIOS_PERMITIDOS)
    fuentes.validar_url(URL, config.DOMINIOS_PERMITIDOS)


def test_detectar_licencia():
    assert carga.detectar_licencia('<a href="http://creativecommons.org/licenses/by/4.0/">CC</a>')["verificada_en_pagina"]
    assert carga.detectar_licencia("<p>sin licencia</p>")["verificada_en_pagina"] is False


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


def test_manifiestos_reales_coinciden_con_hash():
    for directorio in (config.DIR_RAW_AIRBNB, config.DIR_RAW_INE):
        if not (directorio / "MANIFEST.json").exists():
            pytest.skip("data/raw no descargado")
        manifiesto = fuentes.verificar_manifiesto(directorio)
        assert manifiesto["licencia"]["nombre"] and manifiesto["ficheros"]
