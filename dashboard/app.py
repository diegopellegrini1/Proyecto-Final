"""Dashboard operativo de Airbnb Madrid (Streamlit + Plotly).

Ejecutar desde la raíz del repositorio:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

from dashboard import filtros as fl, graficos as gr  # noqa: E402
from dashboard.datos import cargar_datos, cargar_geojson, columnas_disponibles  # noqa: E402
from src import config  # noqa: E402

st.set_page_config(page_title="Airbnb Madrid · Dashboard", page_icon="🏠", layout="wide")

CIUDAD = config.CIUDADES[config.CIUDAD_POR_DEFECTO]
RUTA_PARQUET: Path = config.RUTA_PARQUET
RUTA_GEOJSON: Path = config.RUTA_GEOJSON
MENSAJE_SIN_DATOS = ("No se encuentra data/processed/listings_limpio.parquet. Ejecuta primero: "
                     "python run_pipeline.py")

st.title(f"Airbnb en {CIUDAD.nombre}: dashboard del mercado")
st.caption("Fuente: Inside Airbnb (CC BY 4.0). Precio = precio por noche publicado (cotización), no precio pagado.")

# ---------------------------------------------------------------- datos
if not RUTA_PARQUET.exists():
    st.error(MENSAJE_SIN_DATOS)
    st.stop()
faltan = set(config.COLUMNAS_DASHBOARD) - columnas_disponibles(RUTA_PARQUET)
if faltan:
    st.error(f"El Parquet no tiene las columnas necesarias ({', '.join(sorted(faltan))}). "
             "Vuelve a ejecutar: python run_pipeline.py")
    st.stop()

datos = cargar_datos(str(RUTA_PARQUET), RUTA_PARQUET.stat().st_mtime)
base = datos[datos["precio_valido"]]
n_sin_precio = int(len(datos) - len(base))

# ---------------------------------------------------------------- filtros
with st.sidebar:
    st.header("Filtros")
    distritos = st.multiselect("Distrito", sorted(base["neighbourhood_group_cleansed"].unique()),
                               key=fl.CLAVE_DISTRITO, placeholder="Todos")
    barrios = st.multiselect("Barrio", fl.opciones_barrio(base, distritos), key=fl.CLAVE_BARRIO, placeholder="Todos")
    tipos = st.multiselect("Tipo de habitación", sorted(base["tipo_habitacion"].unique()), key=fl.CLAVE_TIPO,
                           placeholder="Todos")
    lo, hi, defecto = fl.limites_precio(base)
    rango = st.slider("Precio por noche (€)", min_value=lo, max_value=hi, value=defecto, step=fl.PASO_PRECIO,
                      key=fl.CLAVE_PRECIO,
                      help="Por defecto hasta el percentil 99; mueve el control para incluir los precios extremos.")
    texto = st.text_input("Buscar en el nombre del anuncio", key=fl.CLAVE_TEXTO, max_chars=fl.MAX_LONGITUD_TEXTO)

filtrado = fl.aplicar_filtros(base, distritos, barrios, tipos, rango, texto)
st.sidebar.caption(f"{len(filtrado):,} de {len(base):,} anuncios con precio publicado. "
                   f"{n_sin_precio:,} anuncios del fichero no publican precio y no aparecen.".replace(",", "."))

kpis = fl.calcular_kpis(filtrado)
textos = fl.resumen_kpis_texto(kpis)
cols = st.columns(len(fl.ETIQUETAS_KPI))
ayudas = {"ocupacion_media": "(365 − días disponibles) / 365 en anuncios activos. No es ocupación real.",
          "valoracion_media": "Media de review_scores_rating (1-5) de los anuncios con reseñas."}
for col, (clave, etiqueta) in zip(cols, fl.ETIQUETAS_KPI.items()):
    col.metric(etiqueta, textos[clave], help=ayudas.get(clave))

if filtrado.empty:
    st.warning("No hay anuncios con los filtros seleccionados. Amplía el rango de precio o quita filtros.")
    st.stop()

# ---------------------------------------------------------------- pestañas
t_resumen, t_mapa, t_precios, t_hosts, t_datos = st.tabs(
    ["Resumen", "Mapa", "Precios", "Anfitriones y calidad", "Datos"])

with t_resumen:
    c1, c2 = st.columns(2)
    c1.plotly_chart(gr.fig_histograma_precio(filtrado), width="stretch")
    c2.plotly_chart(gr.fig_barras_distrito(filtrado), width="stretch")

with t_mapa:
    vista = st.radio("Vista", ["Puntos", "Barrios"], horizontal=True, key=fl.CLAVE_VISTA_MAPA)
    if vista == "Puntos":
        puntos, muestreado = fl.muestrear(filtrado, config.MAX_PUNTOS_MAPA, config.SEMILLA)
        if muestreado:
            st.caption(f"Se muestran {config.MAX_PUNTOS_MAPA:,} anuncios elegidos al azar (semilla fija) "
                       f"de {len(filtrado):,} para mantener la fluidez.".replace(",", "."))
        st.plotly_chart(gr.fig_mapa_puntos(puntos, CIUDAD.centro), width="stretch")
    else:
        geojson = cargar_geojson(str(RUTA_GEOJSON), RUTA_GEOJSON.stat().st_mtime)
        st.plotly_chart(gr.fig_mapa_barrios(filtrado, geojson, CIUDAD.centro), width="stretch")
    st.caption("Color recortado a los percentiles 5-95 (solo visual). El mapa base se descarga en el navegador "
               "(necesita conexión a internet). Las coordenadas están anonimizadas por Airbnb.")

with t_precios:
    st.plotly_chart(gr.fig_box_tipo(filtrado), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(gr.fig_dispersion_distancia(filtrado, config.MAX_PUNTOS_MAPA, config.SEMILLA), width="stretch")
    c2.plotly_chart(gr.fig_barras_barrio(filtrado, config.MIN_N_GRUPO), width="stretch")

with t_hosts:
    c1, c2 = st.columns(2)
    c1.plotly_chart(gr.fig_segmentos_host(filtrado), width="stretch")
    c2.plotly_chart(gr.fig_valoraciones(filtrado), width="stretch")
    st.plotly_chart(gr.fig_amenities(filtrado), width="stretch")

with t_datos:
    tabla = filtrado[list(config.COLUMNAS_TABLA)]
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.download_button("Descargar CSV filtrado", data=fl.a_csv(tabla), file_name="airbnb_madrid_filtrado.csv",
                       mime="text/csv", key="descarga_csv")
