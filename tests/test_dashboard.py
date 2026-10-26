"""Tests del dashboard: funciones puras y ejecución de la app con AppTest sobre los datos reales."""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import filtros as fl
from src import config, diccionario

RAIZ = config.RAIZ


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "neighbourhood_group_cleansed": ["Centro", "Centro", "Retiro", "Retiro"],
        "neighbourhood_cleansed": ["Sol", "Cortes", "Ibiza", "Jerónimos"],
        "tipo_habitacion": ["Alojamiento entero", "Habitación privada", "Alojamiento entero", "Habitación privada"],
        "price": [100.0, 50.0, 200.0, 9.13], "name": ["Piso (céntrico)", "Habitación [a-", "*Ático*", None],
        "es_activo_probable": [True, True, False, True], "ocupacion_estimada": [0.5, 0.2, 1.0, 0.0],
        "host_is_superhost": [True, False, False, True], "es_multianuncio": [True, True, False, False],
        "review_scores_rating": [4.5, np.nan, 5.0, 4.0],
    })


def test_aplicar_filtros_vacios_y_combinados():
    df = _df()
    assert len(fl.aplicar_filtros(df, [], [], [], None)) == 4
    assert len(fl.aplicar_filtros(df, ["Centro"], [], ["Habitación privada"], (0, 1000))) == 1
    assert len(fl.aplicar_filtros(df, [], ["Ibiza"], [], (0, 150))) == 0
    assert len(fl.aplicar_filtros(df, [], [], [], (50, 100))) == 2


@pytest.mark.parametrize("texto,n", [("(", 1), ("[a-", 1), ("*", 1), ("x" * 10_000, 0), ("  ", 4), ("PISO", 1)])
def test_busqueda_texto_literal(texto, n):
    assert len(fl.aplicar_filtros(_df(), [], [], [], None, texto)) == n


def test_opciones_barrio_en_cascada():
    assert fl.opciones_barrio(_df(), ["Retiro"]) == ["Ibiza", "Jerónimos"]
    assert len(fl.opciones_barrio(_df(), [])) == 4


def test_kpis_vacio_y_normal():
    vacio = fl.calcular_kpis(_df().iloc[0:0])
    assert vacio["anuncios"] == 0 and vacio["precio_mediano"] is None
    assert fl.resumen_kpis_texto(vacio)["precio_mediano"] == "–"
    k = fl.calcular_kpis(_df())
    assert k["anuncios"] == 4 and k["pct_superhost"] == 50.0
    assert k["ocupacion_media"] == pytest.approx(100 * (0.5 + 0.2 + 0.0) / 3)


def test_limites_slider_incluyen_extremos():
    lo, hi, (dlo, dhi) = fl.limites_precio(_df())
    assert lo <= 9.13 and hi >= 200 and lo % 5 == 0 and hi % 5 == 0 and dlo == lo and dhi <= hi


def test_csv_neutraliza_formulas():
    df = pd.DataFrame({"name": ["=HYPERLINK(1)", "+1", "-x", "@a", "normal"], "price": [-1.0, 1, 2, 3, 4]})
    csv = fl.a_csv(df).decode("utf-8-sig")
    assert "'=HYPERLINK" in csv and "'+1" in csv and "'-x" in csv and "'@a" in csv and "\nnormal," in csv
    assert "-1.0" in csv  # los números no se tocan


def test_columnas_dashboard_documentadas():
    assert set(config.COLUMNAS_DASHBOARD) <= set(diccionario.DICCIONARIO)
    assert set(config.COLUMNAS_TABLA) <= set(config.COLUMNAS_DASHBOARD)


def test_sin_use_container_width():
    for p in (RAIZ / "dashboard").glob("*.py"):
        assert "use_container_width" not in p.read_text(encoding="utf-8"), p.name


# ----------------------------------------------------------------- AppTest (datos reales)
requiere_datos = pytest.mark.skipif(not config.RUTA_PARQUET.exists(),
                                    reason="falta data/processed: ejecuta python run_pipeline.py")
APP = str(RAIZ / "dashboard" / "app.py")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=120)


def _metricas(at) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


@requiere_datos
def test_app_arranca_con_filtros_y_kpis():
    at = _app().run()
    assert not at.exception
    for clave in (fl.CLAVE_DISTRITO, fl.CLAVE_BARRIO, fl.CLAVE_TIPO):
        assert at.multiselect(key=clave) is not None
    assert at.slider(key=fl.CLAVE_PRECIO) is not None
    assert set(fl.ETIQUETAS_KPI.values()) <= set(_metricas(at))
    assert len(at.get("plotly_chart")) >= 2


@requiere_datos
def test_filtro_cambia_kpis_y_coincide_con_pandas():
    at = _app().run()
    antes = _metricas(at)["Anuncios"]
    at.multiselect(key=fl.CLAVE_TIPO).set_value(["Habitación privada"]).run()
    assert not at.exception
    despues = _metricas(at)["Anuncios"]
    df = pd.read_parquet(config.RUTA_PARQUET, columns=list(config.COLUMNAS_DASHBOARD))
    base = df[df["precio_valido"]]
    _, _, defecto = fl.limites_precio(base)
    esperado = len(fl.aplicar_filtros(base, [], [], ["Habitación privada"], defecto))
    assert despues != antes and int(despues.replace(".", "")) == esperado


@requiere_datos
def test_filtro_sin_resultados_no_lanza():
    at = _app().run()
    at.text_input(key=fl.CLAVE_TEXTO).set_value("[(*zzzz-no-existe").run()
    assert not at.exception
    assert any("No hay anuncios" in w.value for w in at.warning)
    assert _metricas(at)["Precio mediano"] == "–"


@requiere_datos
def test_vista_barrios_sin_excepcion():
    at = _app().run()
    at.radio(key=fl.CLAVE_VISTA_MAPA).set_value("Barrios").run()
    assert not at.exception


def test_sin_parquet_muestra_mensaje(tmp_path):
    from streamlit.testing.v1 import AppTest
    codigo = Path(APP).read_text(encoding="utf-8").replace(
        "RUTA_PARQUET: Path = config.RUTA_PARQUET", f"RUTA_PARQUET: Path = Path({str(tmp_path / 'no.parquet')!r})")
    codigo = codigo.replace("RAIZ = Path(__file__).resolve().parents[1]", f"RAIZ = Path({str(RAIZ)!r})")
    at = AppTest.from_string(codigo, default_timeout=60).run()
    assert not at.exception
    assert any("run_pipeline.py" in e.value for e in at.error)
