"""Dashboard operativo de Airbnb Madrid (Streamlit + Plotly).

Ejecutar desde la raíz del repositorio (solo accesible desde este equipo):

    streamlit run dashboard/app.py --server.address 127.0.0.1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Airbnb Madrid · Dashboard", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------- dependencias
try:
    from dashboard import filtros as fl, graficos as gr  # noqa: E402
    from dashboard.datos import cargar_datos, columnas_disponibles, validar_geojson  # noqa: E402
    from src import config, formato as fmt  # noqa: E402
except ImportError as exc:  # p. ej. falta plotly o pyarrow
    st.error(f"Falta una dependencia del dashboard ({exc.name or exc}). Instálalas con: "
             "pip install -r requirements.txt")
    st.stop()

CIUDAD = config.CIUDADES[config.CIUDAD_POR_DEFECTO]
RUTA_PARQUET: Path = config.RUTA_PARQUET
RUTA_GEOJSON: Path = config.RUTA_GEOJSON
MENSAJE_SIN_DATOS = ("No se encuentra data/processed/listings_limpio.parquet. Ejecuta primero: "
                     "python run_pipeline.py")
VISTA_PUNTOS, VISTA_BARRIOS = "Puntos", "Barrios"

st.title(f"Airbnb en {CIUDAD.nombre}: dashboard del mercado")
st.caption("Fuente: Inside Airbnb (CC BY 4.0). Precio = precio por noche de una cotización publicada, ya con "
           "descuentos y sin tasas; no es el precio pagado.")

# ---------------------------------------------------------------- datos
if not RUTA_PARQUET.exists():
    st.error(MENSAJE_SIN_DATOS)
    st.stop()
faltan = set(config.COLUMNAS_DASHBOARD) - columnas_disponibles(RUTA_PARQUET)
if faltan:
    st.error(f"El Parquet no tiene las columnas necesarias ({', '.join(sorted(faltan))}). "
             "Vuelve a ejecutar: python run_pipeline.py")
    st.stop()
geojson, aviso_geojson = validar_geojson(RUTA_GEOJSON)

datos = cargar_datos(str(RUTA_PARQUET), RUTA_PARQUET.stat().st_mtime)
base = datos[datos["precio_valido"]]
n_sin_precio = int(len(datos) - len(base))

# ---------------------------------------------------------------- filtros
with st.sidebar:
    st.header("Filtros")
    distritos = st.multiselect("Distrito", sorted(base["neighbourhood_group_cleansed"].unique()),
                               key=fl.CLAVE_DISTRITO, placeholder="Todos")
    opciones_b = fl.opciones_barrio(base, distritos)
    validos, invalidados = fl.seleccion_invalidada(st.session_state.get(fl.CLAVE_BARRIO), opciones_b)
    if invalidados:
        st.session_state[fl.CLAVE_BARRIO] = validos
        st.info(f"Se ha quitado del filtro de barrio {', '.join(invalidados)} porque no pertenece a los "
                "distritos elegidos.")
    barrios = st.multiselect("Barrio", opciones_b, key=fl.CLAVE_BARRIO, placeholder="Todos")
    tipos = st.multiselect("Tipo de habitación", sorted(base["tipo_habitacion"].unique()), key=fl.CLAVE_TIPO,
                           placeholder="Todos")
    cap_lo, cap_hi = fl.limites_capacidad(base)
    capacidad = st.slider("Capacidad (personas)", min_value=cap_lo, max_value=cap_hi, value=(cap_lo, cap_hi),
                          step=1, key=fl.CLAVE_CAPACIDAD)
    superhost = fl.SUPERHOST_OPCIONES[st.selectbox("Superhost", list(fl.SUPERHOST_OPCIONES), key=fl.CLAVE_SUPERHOST)]
    lo, hi, defecto = fl.limites_precio(base)
    c_min, c_max = st.columns(2)
    precio_min = c_min.number_input("Precio mín. (€)", min_value=lo, max_value=hi, value=defecto[0],
                                    step=fl.PASO_PRECIO, key=fl.CLAVE_PRECIO_MIN)
    precio_max = c_max.number_input("Precio máx. (€)", min_value=lo, max_value=hi, value=defecto[1],
                                    step=fl.PASO_PRECIO, key=fl.CLAVE_PRECIO_MAX,
                                    help="Por defecto hasta el percentil 99; escribe un valor mayor para incluir "
                                         "los precios extremos.")
    if precio_min > precio_max:
        st.warning("El precio mínimo es mayor que el máximo: se intercambian.")
    excluir_largas = st.checkbox(f"Excluir estancias de {fl.NOCHES_LARGA_ESTANCIA} noches o más",
                                 value=True, key=fl.CLAVE_EXCLUIR_LARGAS,
                                 help="Quita los anuncios con estancia mínima de 30 noches o más y las cotizaciones "
                                      "con tarifa mensual: su precio por noche es una tarifa mensual repartida y "
                                      "no se compara con el de estancias cortas.")
    texto = st.text_input("Buscar en el nombre del anuncio", key=fl.CLAVE_TEXTO, max_chars=fl.MAX_LONGITUD_TEXTO)

filtrado = fl.aplicar_filtros(base, distritos, barrios, tipos, (precio_min, precio_max), texto,
                              capacidad=capacidad, superhost=superhost, excluir_largas=excluir_largas)
st.sidebar.caption(f"{fmt.n0(len(filtrado))} de {fmt.n0(len(base))} anuncios con precio publicado. "
                   f"{fmt.n0(n_sin_precio)} anuncios del fichero no publican precio y no aparecen.")

kpis = fl.calcular_kpis(filtrado)
textos = fl.resumen_kpis_texto(kpis)
cols = st.columns(len(fl.ETIQUETAS_KPI))
ayudas = {"no_disponible_media": "(365 − días disponibles) / 365 en anuncios activos: mezcla reservas y bloqueos "
                                 "del anfitrión y mira al próximo año. No es ocupación real.",
          "valoracion_media": "Media de review_scores_rating (escala de 1 a 5) de los anuncios con reseñas."}
for col, (clave, etiqueta) in zip(cols, fl.ETIQUETAS_KPI.items()):
    col.metric(etiqueta, textos[clave], help=ayudas.get(clave))
st.caption("Por qué estos KPIs no coinciden con el informe: solo cuentan anuncios **con precio**; por defecto se "
           "excluyen los precios por encima del **P99** (el 1 % más caro, que dispara la media) y las **estancias de "
           f"{fl.NOCHES_LARGA_ESTANCIA} noches o más**. Con todos los precios, sin excluir estancias largas y sin "
           "otros filtros, el número de anuncios y las medianas son los del informe.")

if filtrado.empty:
    st.warning("No hay anuncios con los filtros seleccionados. Amplía el rango de precio o quita filtros.")
    st.stop()

# ---------------------------------------------------------------- pestañas
t_resumen, t_comp, t_mapa, t_precios, t_hosts, t_datos = st.tabs(
    ["Resumen", "Comparador", "Mapa", "Precios", "Anfitriones y calidad", "Datos"])

with t_resumen:
    c1, c2 = st.columns(2)
    c1.plotly_chart(gr.fig_histograma_precio(filtrado), width="stretch")
    c2.plotly_chart(gr.fig_barras_distrito(filtrado), width="stretch")

with t_comp:
    st.subheader("¿Qué precio tienen anuncios comparables?")
    st.caption("Compara con anuncios del mismo tipo, la misma capacidad y la misma zona (respeta el filtro de "
               "estancias largas de la barra lateral, no los demás). Es una referencia descriptiva, no una tarifa "
               "recomendada.")
    universo = base[~fl.es_estancia_larga(base)] if excluir_largas else base
    k1, k2, k3, k4 = st.columns(4)
    tipo_c = k1.selectbox("Tipo", sorted(universo["tipo_habitacion"].unique()), key=fl.CLAVE_COMP_TIPO)
    cap_c = int(k2.number_input("Capacidad (personas)", min_value=cap_lo, max_value=cap_hi, value=min(max(2, cap_lo), cap_hi),
                                step=1, key=fl.CLAVE_COMP_CAPACIDAD))
    nivel = k3.radio("Zona", ["Distrito", "Barrio"], horizontal=True, key=fl.CLAVE_COMP_NIVEL)
    col_zona = "neighbourhood_group_cleansed" if nivel == "Distrito" else "neighbourhood_cleansed"
    zona = k4.selectbox(nivel, sorted(universo[col_zona].dropna().astype(str).unique()), key=fl.CLAVE_COMP_ZONA)
    comp = fl.comparables(universo, tipo_c, cap_c, col_zona, zona, config.MIN_N_COMPARADOR)
    m1, m2, m3 = st.columns(3)
    m1.metric("Anuncios comparables", fmt.n0(comp["n"]))
    m2.metric("Mediana", fmt.eur2(comp["mediana"]))
    m3.metric("Rango P25–P75", f"{fmt.eur0(comp['p25'])} – {fmt.eur0(comp['p75'])}" if comp["n"] else "–")
    if comp["n"] == 0:
        st.info("No hay anuncios comparables con esa combinación. Prueba con el distrito en lugar del barrio.")
    elif comp["aviso"]:
        st.warning(f"Solo hay {fmt.n0(comp['n'])} anuncios comparables (menos de {config.MIN_N_COMPARADOR}): la "
                   "mediana y el rango son poco fiables.")

with t_mapa:
    opciones_vista = [VISTA_PUNTOS] if geojson is None else [VISTA_PUNTOS, VISTA_BARRIOS]
    if aviso_geojson:
        st.info(aviso_geojson)
    vista = st.radio("Vista", opciones_vista, horizontal=True, key=fl.CLAVE_VISTA_MAPA)
    if vista == VISTA_PUNTOS:
        puntos, muestreado = fl.muestrear(filtrado, config.MAX_PUNTOS_MAPA, config.SEMILLA)
        if muestreado:
            st.caption(f"Se muestran {fmt.n0(config.MAX_PUNTOS_MAPA)} anuncios elegidos al azar (semilla fija) "
                       f"de {fmt.n0(len(filtrado))} para mantener la fluidez.")
        st.plotly_chart(gr.fig_mapa_puntos(puntos, CIUDAD.centro), width="stretch")
    else:
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
    tabla = fl.con_etiquetas(filtrado[list(config.COLUMNAS_TABLA)], config.ETIQUETAS_COLUMNAS)
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.download_button("Descargar CSV filtrado (separador «;», para Excel en español)", data=fl.a_csv(tabla),
                       file_name="airbnb_madrid_filtrado.csv", mime="text/csv", key="descarga_csv")
