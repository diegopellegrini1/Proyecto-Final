"""Tests del dashboard: funciones puras y ejecución de la app con AppTest sobre los datos reales."""

import io
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
        "accommodates": [2, 1, 4, 2], "minimum_nights": [1, 2, 30, 3],
        "es_tarifa_mensual": [False, False, True, True],
        "es_activo_probable": [True, True, False, True], "no_disponible_365": [0.5, 0.2, 1.0, 0.0],
        "host_is_superhost": [True, False, False, True], "es_multianuncio": [True, True, False, False],
        "review_scores_rating": [4.5, np.nan, 5.0, 4.0],
    })


def test_aplicar_filtros_vacios_y_combinados():
    df = _df()
    assert len(fl.aplicar_filtros(df, [], [], [], None)) == 4
    assert len(fl.aplicar_filtros(df, ["Centro"], [], ["Habitación privada"], (0, 1000))) == 1
    assert len(fl.aplicar_filtros(df, [], ["Ibiza"], [], (0, 150))) == 0
    assert len(fl.aplicar_filtros(df, [], [], [], (50, 100))) == 2
    assert len(fl.aplicar_filtros(df, [], [], [], (100, 50))) == 2  # rango invertido se ordena


def test_filtros_capacidad_superhost_y_estancias_largas():
    df = _df()
    assert fl.aplicar_filtros(df, [], [], [], None, capacidad=(2, 2))["price"].tolist() == [100.0, 9.13]
    assert fl.aplicar_filtros(df, [], [], [], None, superhost=True)["price"].tolist() == [100.0, 9.13]
    assert fl.aplicar_filtros(df, [], [], [], None, superhost=False)["price"].tolist() == [50.0, 200.0]
    # Fila 2: mínimo 30 noches; fila 3: tarifa mensual con mínimo corto. Ambas se excluyen.
    assert fl.aplicar_filtros(df, [], [], [], None, excluir_largas=True)["price"].tolist() == [100.0, 50.0]


@pytest.mark.parametrize("texto,n", [("(", 1), ("[a-", 1), ("*", 1), ("x" * 10_000, 0), ("  ", 4), ("PISO", 1)])
def test_busqueda_texto_literal(texto, n):
    assert len(fl.aplicar_filtros(_df(), [], [], [], None, texto)) == n


def test_opciones_barrio_en_cascada_y_seleccion_invalidada():
    assert fl.opciones_barrio(_df(), ["Retiro"]) == ["Ibiza", "Jerónimos"]
    assert len(fl.opciones_barrio(_df(), [])) == 4
    assert fl.seleccion_invalidada(["Sol", "Ibiza"], ["Ibiza", "Jerónimos"]) == (["Ibiza"], ["Sol"])
    assert fl.seleccion_invalidada(None, ["Ibiza"]) == ([], [])


def test_kpis_vacio_y_normal():
    vacio = fl.calcular_kpis(_df().iloc[0:0])
    assert vacio["anuncios"] == 0 and vacio["precio_mediano"] is None
    assert fl.resumen_kpis_texto(vacio)["precio_mediano"] == "–"
    k = fl.calcular_kpis(_df())
    assert k["anuncios"] == 4 and k["pct_superhost"] == 50.0
    # Media de no_disponible_365 solo en activos: (0,5 + 0,2 + 0,0) / 3, en %.
    assert k["no_disponible_media"] == pytest.approx(70.0 / 3)
    assert k["precio_mediano"] == pytest.approx(75.0) and k["valoracion_media"] == pytest.approx(4.5)


def test_comparables_valores_y_aviso():
    df = _df()
    r = fl.comparables(df, "Alojamiento entero", 2, "neighbourhood_group_cleansed", "Centro", min_n=30)
    assert r == {"n": 1, "mediana": 100.0, "p25": 100.0, "p75": 100.0, "aviso": True}
    r = fl.comparables(df, None, None, None, None, min_n=2)
    assert r["n"] == 4 and r["mediana"] == pytest.approx(75.0) and r["aviso"] is False
    assert r["p25"] == pytest.approx(39.7825) and r["p75"] == pytest.approx(125.0)  # cuantiles lineales a mano
    assert fl.comparables(df, "Habitación de hotel", 2, None, None, min_n=1)["n"] == 0


def test_limites_precio_y_capacidad():
    lo, hi, (dlo, dhi) = fl.limites_precio(_df())
    assert (lo, hi) == (5.0, 200.0) and dlo == lo and dhi == 200.0
    assert fl.limites_capacidad(_df()) == (1, 4)


def test_csv_punto_y_coma_bom_coma_decimal_y_formulas():
    df = pd.DataFrame({"Nombre": ["=HYPERLINK(1)", "+1", "-x", "@a", "normal"], "Precio": [-1.5, 1, 2, 3, 4]})
    crudo = fl.a_csv(df)
    assert crudo.startswith("\ufeff".encode("utf-8"))
    texto = crudo.decode("utf-8-sig")
    assert texto.splitlines()[0] == "Nombre;Precio"
    assert "'=HYPERLINK(1);-1,5" in texto and "'+1" in texto and "'-x" in texto and "'@a" in texto
    leido = pd.read_csv(io.StringIO(texto), sep=";", decimal=",")
    assert leido["Precio"].tolist() == [-1.5, 1, 2, 3, 4]  # los números no se tocan


def test_etiquetas_en_espanol():
    tabla = fl.con_etiquetas(pd.DataFrame(columns=list(config.COLUMNAS_TABLA)), config.ETIQUETAS_COLUMNAS)
    assert not set(tabla.columns) & set(config.COLUMNAS_TABLA)  # ninguna columna conserva el nombre crudo
    assert "Precio por noche (€)" in tabla.columns and "Distrito" in tabla.columns


def test_columnas_dashboard_documentadas_y_usadas():
    assert set(config.COLUMNAS_DASHBOARD) <= set(diccionario.DICCIONARIO)
    assert set(config.COLUMNAS_TABLA) <= set(config.COLUMNAS_DASHBOARD)
    # Cada columna cargada se usa: está en la tabla, es un flag de amenities o aparece en el código del dashboard.
    codigo = "".join(p.read_text(encoding="utf-8") for p in (RAIZ / "dashboard").glob("*.py"))
    no_usadas = [c for c in config.COLUMNAS_DASHBOARD
                 if c not in config.COLUMNAS_TABLA and c not in config.PATRONES_AMENITIES and f'"{c}"' not in codigo]
    assert no_usadas == []
    assert "host_id" not in config.COLUMNAS_DASHBOARD  # minimización: el dashboard no carga el identificador


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


def _app_con(reemplazos: dict[str, str]):
    """AppTest sobre una copia del código de la app con algunas líneas sustituidas."""
    from streamlit.testing.v1 import AppTest
    codigo = Path(APP).read_text(encoding="utf-8")
    codigo = codigo.replace("RAIZ = Path(__file__).resolve().parents[1]", f"RAIZ = Path({str(RAIZ)!r})")
    for viejo, nuevo in reemplazos.items():
        assert viejo in codigo, viejo
        codigo = codigo.replace(viejo, nuevo)
    return AppTest.from_string(codigo, default_timeout=120)


def _metricas(at) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _base_real() -> pd.DataFrame:
    df = pd.read_parquet(config.RUTA_PARQUET, columns=list(config.COLUMNAS_DASHBOARD))
    return df[df["precio_valido"]]


@requiere_datos
def test_app_arranca_con_filtros_kpis_y_nota():
    at = _app().run()
    assert not at.exception
    for clave in (fl.CLAVE_DISTRITO, fl.CLAVE_BARRIO, fl.CLAVE_TIPO):
        assert at.multiselect(key=clave) is not None
    assert at.number_input(key=fl.CLAVE_PRECIO_MIN) is not None and at.slider(key=fl.CLAVE_CAPACIDAD) is not None
    assert at.checkbox(key=fl.CLAVE_EXCLUIR_LARGAS).value is True
    assert set(fl.ETIQUETAS_KPI.values()) <= set(_metricas(at))
    assert any("P99" in c.value and "con precio" in c.value for c in at.caption)
    assert len(at.get("plotly_chart")) >= 2


@requiere_datos
def test_kpis_por_defecto_coinciden_con_pandas():
    at = _app().run()
    base = _base_real()
    _, _, defecto = fl.limites_precio(base)
    esperado = fl.aplicar_filtros(base, [], [], [], defecto, excluir_largas=True)
    assert int(_metricas(at)["Anuncios"].replace(".", "")) == len(esperado)
    at.checkbox(key=fl.CLAVE_EXCLUIR_LARGAS).uncheck().run()
    sin_excluir = fl.aplicar_filtros(base, [], [], [], defecto)
    assert int(_metricas(at)["Anuncios"].replace(".", "")) == len(sin_excluir) > len(esperado)


@requiere_datos
def test_filtro_cambia_kpis_y_coincide_con_pandas():
    at = _app().run()
    antes = _metricas(at)["Anuncios"]
    at.multiselect(key=fl.CLAVE_TIPO).set_value(["Habitación privada"]).run()
    assert not at.exception
    despues = _metricas(at)["Anuncios"]
    base = _base_real()
    _, _, defecto = fl.limites_precio(base)
    esperado = len(fl.aplicar_filtros(base, [], [], ["Habitación privada"], defecto, excluir_largas=True))
    assert despues != antes and int(despues.replace(".", "")) == esperado


@requiere_datos
def test_filtro_sin_resultados_no_lanza():
    at = _app().run()
    at.text_input(key=fl.CLAVE_TEXTO).set_value("[(*zzzz-no-existe").run()
    assert not at.exception
    assert any("No hay anuncios" in w.value for w in at.warning)
    assert _metricas(at)["Precio mediano"] == "–"


@requiere_datos
def test_cascada_avisa_al_invalidar_barrio():
    at = _app().run()
    at.multiselect(key=fl.CLAVE_BARRIO).set_value(["Sol"]).run()
    at.multiselect(key=fl.CLAVE_DISTRITO).set_value(["Barajas"]).run()
    assert not at.exception
    assert any("Sol" in i.value for i in at.info)
    assert at.multiselect(key=fl.CLAVE_BARRIO).value == []


@requiere_datos
def test_comparador_con_pocos_comparables_avisa():
    at = _app().run()
    base = _base_real()
    base = base[~fl.es_estancia_larga(base)]
    # Combinación real con n pequeño: el barrio con menos alojamientos enteros de 2 personas (al menos 1).
    sub = base[(base["tipo_habitacion"] == "Alojamiento entero") & (base["accommodates"] == 2)]
    conteo = sub["neighbourhood_cleansed"].value_counts()
    barrio = sorted(conteo[conteo < config.MIN_N_COMPARADOR].index)[0]
    at.selectbox(key=fl.CLAVE_COMP_TIPO).set_value("Alojamiento entero").run()
    at.number_input(key=fl.CLAVE_COMP_CAPACIDAD).set_value(2).run()
    at.radio(key=fl.CLAVE_COMP_NIVEL).set_value("Barrio").run()
    at.selectbox(key=fl.CLAVE_COMP_ZONA).set_value(barrio).run()
    assert not at.exception
    assert _metricas(at)["Anuncios comparables"] == str(int(conteo[barrio]))
    assert any("poco fiables" in w.value for w in at.warning)


@requiere_datos
def test_vista_barrios_sin_excepcion():
    at = _app().run()
    at.radio(key=fl.CLAVE_VISTA_MAPA).set_value("Barrios").run()
    assert not at.exception


@requiere_datos
def test_geojson_ausente_desactiva_solo_vista_barrios(tmp_path):
    at = _app_con({"RUTA_GEOJSON: Path = config.RUTA_GEOJSON":
                   f"RUTA_GEOJSON: Path = Path({str(tmp_path / 'no.geojson')!r})"}).run()
    assert not at.exception
    assert any("vista de barrios queda desactivada" in i.value for i in at.info)
    assert at.radio(key=fl.CLAVE_VISTA_MAPA).options == ["Puntos"]
    assert len(at.get("plotly_chart")) >= 2  # el resto del dashboard sigue funcionando


def test_sin_parquet_muestra_mensaje(tmp_path):
    at = _app_con({"RUTA_PARQUET: Path = config.RUTA_PARQUET":
                   f"RUTA_PARQUET: Path = Path({str(tmp_path / 'no.parquet')!r})"}).run()
    assert not at.exception
    assert any("run_pipeline.py" in e.value for e in at.error)
