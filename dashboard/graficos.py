"""Constructores de figuras Plotly del dashboard (funciones puras: DataFrame → Figure)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.filtros import mediana_por, rango_color
from src import config

PALETA = px.colors.qualitative.Safe
ESCALA = "Viridis"
ESTILO_MAPA = "carto-positron"
ETIQ = {"price": "Precio por noche (€)", "tipo_habitacion": "Tipo de habitación",
        "neighbourhood_group_cleansed": "Distrito", "neighbourhood_cleansed": "Barrio",
        "distancia_centro_km": "Distancia a la Puerta del Sol (km)", "name": "Anuncio", "n": "Anuncios",
        "mediana": "Mediana del precio (€)", "review_scores_rating": "Valoración (1-5)",
        "segmento_host": "Segmento del anfitrión", "accommodates": "Capacidad"}


def figura_vacia(titulo: str) -> go.Figure:
    """Figura con la anotación «Sin datos»."""
    fig = go.Figure()
    fig.add_annotation(text="Sin datos con los filtros actuales", showarrow=False, font={"size": 16})
    fig.update_layout(title=titulo, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def fig_histograma_precio(df: pd.DataFrame) -> go.Figure:
    """Histograma del precio por noche (recortado al P99 del filtro)."""
    titulo = "Distribución del precio por noche"
    if df.empty:
        return figura_vacia(titulo)
    p99 = df["price"].quantile(0.99)
    fig = px.histogram(df[df["price"] <= p99], x="price", nbins=60, labels=ETIQ, color_discrete_sequence=PALETA,
                       title=f"{titulo} (hasta el percentil 99)")
    fig.update_layout(yaxis_title="Anuncios", bargap=0.05)
    return fig


def fig_barras_distrito(df: pd.DataFrame) -> go.Figure:
    """Mediana de precio por distrito."""
    titulo = "Mediana del precio por distrito"
    m = mediana_por(df, "neighbourhood_group_cleansed")
    if m.empty:
        return figura_vacia(titulo)
    fig = px.bar(m, x="mediana", y="neighbourhood_group_cleansed", orientation="h", hover_data=["n"],
                 labels=ETIQ, title=titulo, color_discrete_sequence=PALETA)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(350, 24 * len(m)))
    return fig


def fig_box_tipo(df: pd.DataFrame) -> go.Figure:
    """Boxplot del precio (escala log) por tipo de habitación."""
    titulo = "Precio por tipo de habitación (escala logarítmica)"
    if df.empty:
        return figura_vacia(titulo)
    fig = px.box(df, x="tipo_habitacion", y="price", color="tipo_habitacion", log_y=True, labels=ETIQ,
                 title=titulo, color_discrete_sequence=PALETA, points=False)
    fig.update_layout(showlegend=False)
    return fig


def fig_dispersion_distancia(df: pd.DataFrame, maximo: int, semilla: int) -> go.Figure:
    """Precio frente a distancia al centro (muestra reproducible)."""
    titulo = "Precio frente a distancia al centro"
    d = df.dropna(subset=["distancia_centro_km"])
    if d.empty:
        return figura_vacia(titulo)
    if len(d) > maximo:
        d = d.sample(n=maximo, random_state=semilla)
    fig = px.scatter(d, x="distancia_centro_km", y="price", color="tipo_habitacion", log_y=True, opacity=0.4,
                     labels=ETIQ, title=titulo, color_discrete_sequence=PALETA, hover_data=["name"])
    fig.update_traces(marker={"size": 4})
    return fig


def fig_barras_barrio(df: pd.DataFrame, min_n: int, top: int = 15) -> go.Figure:
    """Los barrios más caros por mediana (n ≥ min_n dentro del filtro)."""
    titulo = f"Barrios más caros (al menos {min_n} anuncios en el filtro)"
    m = mediana_por(df, "neighbourhood_cleansed", min_n=min_n).head(top)
    if m.empty:
        return figura_vacia(titulo)
    fig = px.bar(m, x="mediana", y="neighbourhood_cleansed", orientation="h", hover_data=["n"], labels=ETIQ,
                 title=titulo, color_discrete_sequence=PALETA)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def fig_segmentos_host(df: pd.DataFrame) -> go.Figure:
    """Anuncios por segmento de anfitrión."""
    titulo = "Anuncios según el tamaño del anfitrión"
    if df.empty:
        return figura_vacia(titulo)
    c = df["segmento_host"].value_counts(sort=False).rename_axis("segmento_host").reset_index(name="n")
    fig = px.bar(c, x="segmento_host", y="n", labels=ETIQ, title=titulo, color_discrete_sequence=PALETA, text="n")
    return fig


def fig_valoraciones(df: pd.DataFrame) -> go.Figure:
    """Distribución de la valoración global."""
    titulo = "Valoración global de los anuncios con reseñas"
    d = df.dropna(subset=["review_scores_rating"])
    if d.empty:
        return figura_vacia(titulo)
    fig = px.histogram(d, x="review_scores_rating", nbins=40, labels=ETIQ, title=titulo, color_discrete_sequence=PALETA)
    fig.update_layout(yaxis_title="Anuncios")
    return fig


def fig_amenities(df: pd.DataFrame) -> go.Figure:
    """Prevalencia de amenities en el filtro."""
    titulo = "Amenities (% de anuncios del filtro)"
    if df.empty:
        return figura_vacia(titulo)
    filas = [{"amenity": config.NOMBRES_AMENITIES[c], "pct": float(df[c].mean() * 100)} for c in config.PATRONES_AMENITIES]
    d = pd.DataFrame(filas).sort_values("pct")
    fig = px.bar(d, x="pct", y="amenity", orientation="h", title=titulo, labels={"pct": "% de anuncios", "amenity": ""},
                 color_discrete_sequence=PALETA)
    return fig


def fig_mapa_puntos(df: pd.DataFrame, centro: tuple[float, float]) -> go.Figure:
    """Mapa de puntos coloreado por precio (escala de color P5-P95)."""
    titulo = "Anuncios coloreados por precio"
    d = df[df["coords_validas"]]
    if d.empty:
        return figura_vacia(titulo)
    fig = px.scatter_map(d, lat="latitude", lon="longitude", color="price", range_color=rango_color(d["price"]),
                         color_continuous_scale=ESCALA, hover_name="name",
                         hover_data={"neighbourhood_cleansed": True, "tipo_habitacion": True, "price": ":.2f",
                                     "latitude": False, "longitude": False},
                         labels=ETIQ, zoom=11, center={"lat": centro[0], "lon": centro[1]}, map_style=ESTILO_MAPA,
                         height=650, title=titulo)
    fig.update_traces(marker={"size": 5, "opacity": 0.7})
    return fig


def fig_mapa_barrios(df: pd.DataFrame, geojson: dict[str, Any], centro: tuple[float, float]) -> go.Figure:
    """Coropleta de la mediana del precio por barrio."""
    titulo = "Mediana del precio por barrio"
    m = mediana_por(df, "neighbourhood_cleansed")
    if m.empty:
        return figura_vacia(titulo)
    fig = px.choropleth_map(m, geojson=geojson, locations="neighbourhood_cleansed",
                            featureidkey="properties.neighbourhood", color="mediana",
                            color_continuous_scale=ESCALA, range_color=rango_color(m["mediana"]),
                            hover_data={"n": True}, labels=ETIQ, zoom=10, center={"lat": centro[0], "lon": centro[1]},
                            map_style=ESTILO_MAPA, opacity=0.7, height=650, title=titulo)
    return fig
