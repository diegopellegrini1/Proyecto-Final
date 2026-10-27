"""Figuras estáticas (matplotlib/seaborn) del EDA y de la estadística.

Cada ``fig_*`` devuelve una ``Figure``; ``guardar_figura`` la escribe en PNG y la cierra.
Los notebooks llaman a las mismas funciones y las muestran en línea.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from scipy import stats

from src import config, formato as fmt
from src.analisis import ETIQUETAS_VARIABLES

logger = logging.getLogger(__name__)

DPI = 150
PALETA = sns.color_palette("colorblind")
NOMBRES_AMENITIES = config.NOMBRES_AMENITIES


def configurar_estilo() -> None:
    """Estilo común: seaborn whitegrid, paleta apta para daltonismo y tamaños legibles."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 13, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                         "axes.formatter.use_locale": False})


def guardar_figura(fig: Figure, nombre: str, directorio: Path) -> Path:
    """Guarda ``fig`` como ``directorio/nombre`` (PNG, 150 dpi) y la cierra."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _cp(df: pd.DataFrame) -> pd.DataFrame:
    """Anuncios con precio válido."""
    return df[df["precio_valido"]]


def _eje_euros(ax, eje: str = "x") -> None:
    """Formatea un eje con separador de miles es-ES y símbolo €."""
    f = plt.FuncFormatter(lambda v, _: f"{fmt.n0(v)} €")
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(f)


# --------------------------------------------------------------------------
# Figuras descriptivas
# --------------------------------------------------------------------------
def fig_distribucion_precio(df: pd.DataFrame) -> Figure:
    """Histograma del precio (recortado a P99) y de ln(precio) con curva normal."""
    cp = _cp(df)
    p = cp["price"]
    p99 = p.quantile(0.99)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.hist(p[p <= p99], bins=60, color=PALETA[0], alpha=0.85)
    a1.axvline(p.median(), color=PALETA[1], lw=2, label=f"Mediana: {fmt.eur0(p.median())}")
    a1.axvline(p.mean(), color=PALETA[2], lw=2, ls="--", label=f"Media: {fmt.eur0(p.mean())}")
    a1.set(title=f"Precio por noche (hasta el P99 = {fmt.eur0(p99)})", xlabel="Precio por noche", ylabel="Anuncios")
    _eje_euros(a1)
    a1.legend()
    lp = cp["log_precio"]
    a2.hist(lp, bins=60, density=True, color=PALETA[0], alpha=0.85, label="ln(precio)")
    xs = np.linspace(lp.min(), lp.max(), 300)
    a2.plot(xs, stats.norm.pdf(xs, lp.mean(), lp.std()), color=PALETA[3], lw=2, label="Normal teórica")
    a2.set(title="Logaritmo natural del precio", xlabel="ln(precio)", ylabel="Densidad")
    a2.legend()
    fig.suptitle(f"Distribución del precio (n = {fmt.n0(len(cp))} anuncios con precio)", y=1.02)
    return fig


def fig_precio_por_tipo(df: pd.DataFrame) -> Figure:
    """Boxplot del precio (escala log) por tipo de habitación, con n."""
    cp = _cp(df)
    orden = cp.groupby("tipo_habitacion")["price"].median().sort_values(ascending=False).index.tolist()
    datos = [cp.loc[cp["tipo_habitacion"] == t, "price"].to_numpy() for t in orden]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(datos, orientation="horizontal", patch_artist=True, showfliers=True,
                    flierprops={"markersize": 2, "alpha": 0.3})
    for caja, color in zip(bp["boxes"], PALETA):
        caja.set_facecolor(color)
        caja.set_alpha(0.7)
    ax.set_yticks(range(1, len(orden) + 1), [f"{t}\n(n = {fmt.n0(len(d))})" for t, d in zip(orden, datos)])
    ax.set_xscale("log")
    ax.invert_yaxis()
    _eje_euros(ax)
    ax.set(title="Precio por noche según tipo de habitación (escala logarítmica)", xlabel="Precio por noche")
    return fig


def fig_precio_por_distrito(hallazgos: dict[str, Any]) -> Figure:
    """Barras horizontales de la mediana de precio por distrito, con n."""
    filas = hallazgos["descriptivo"]["por_distrito"]
    fig, ax = plt.subplots(figsize=(10, 7))
    nombres = [f"{f['grupo']} (n = {fmt.n0(f['n'])})" for f in filas]
    ax.barh(nombres, [f["mediana"] for f in filas], color=PALETA[0])
    ax.invert_yaxis()
    _eje_euros(ax)
    ax.set(title="Mediana del precio por noche por distrito", xlabel="Mediana del precio")
    return fig


def fig_precio_por_barrio(df: pd.DataFrame, min_n: int, top: int = 15) -> Figure:
    """Los ``top`` barrios más caros y más baratos por mediana (n ≥ min_n)."""
    cp = _cp(df)
    g = cp.groupby("neighbourhood_cleansed")["price"].agg(["median", "size"])
    g = g[g["size"] >= min_n].sort_values("median", ascending=False)
    caros, baratos = g.head(top), g.tail(top)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, datos, color, titulo in ((a1, caros, PALETA[3], f"{top} barrios más caros"),
                                     (a2, baratos, PALETA[2], f"{top} barrios más baratos")):
        ax.barh([f"{b} ({fmt.n0(n)})" for b, n in zip(datos.index, datos["size"])], datos["median"], color=color)
        ax.invert_yaxis()
        _eje_euros(ax)
        ax.set(title=titulo, xlabel="Mediana del precio")
    fig.suptitle(f"Mediana del precio por barrio (barrios con al menos {min_n} anuncios con precio; n entre paréntesis)",
                 y=1.01)
    fig.tight_layout()
    return fig


def fig_precio_por_capacidad(hallazgos: dict[str, Any]) -> Figure:
    """Mediana y rango intercuartílico del precio por tramo de capacidad."""
    filas = hallazgos["descriptivo"]["por_capacidad"]
    filas = sorted(filas, key=lambda f: f["grupo"])
    x = np.arange(len(filas))
    med = np.array([f["mediana"] for f in filas])
    err = np.array([[f["mediana"] - f["q1"] for f in filas], [f["q3"] - f["mediana"] for f in filas]])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, med, color=PALETA[0], alpha=0.85)
    ax.errorbar(x, med, yerr=err, fmt="none", ecolor="black", capsize=6, label="Rango intercuartílico (Q1–Q3)")
    ax.set_xticks(x, [f"{f['grupo']}\n(n = {fmt.n0(f['n'])})" for f in filas])
    _eje_euros(ax, "y")
    ax.set(title="Precio por noche según capacidad del alojamiento", ylabel="Mediana del precio")
    ax.legend()
    return fig


def fig_composicion_hosts(df: pd.DataFrame, hallazgos: dict[str, Any]) -> Figure:
    """Anuncios por anfitrión (escala log) y reparto de anuncios por segmento de anfitrión."""
    h = hallazgos["descriptivo"]["hosts"]
    por_host = df.groupby("host_id").size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.hist(por_host, bins=np.logspace(0, np.log10(por_host.max() + 1), 30), color=PALETA[0])
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set(title=f"Anuncios por anfitrión ({fmt.n0(h['n_hosts'])} anfitriones)",
           xlabel="Nº de anuncios del anfitrión (escala log)", ylabel="Nº de anfitriones (escala log)")
    seg = h["segmentos_pct"]
    barras = a2.bar(list(seg.keys()), list(seg.values()), color=PALETA[:len(seg)])
    for b, v in zip(barras, seg.values()):
        a2.text(b.get_x() + b.get_width() / 2, v + 1, fmt.pct1(v), ha="center")
    a2.set(title="Reparto de anuncios según el tamaño del anfitrión", ylabel="% de anuncios", ylim=(0, 100))
    a2.text(0.98, 0.95, f"Anuncios de superhosts: {fmt.pct1(h['pct_anuncios_superhost'])}\n"
                        f"Anuncios de multi-anuncio: {fmt.pct1(h['pct_anuncios_multianuncio'])}",
            transform=a2.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.8})
    return fig


def _poligonos(geojson: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Lista de (barrio, anillo exterior) de todos los polígonos del GeoJSON."""
    salida = []
    for ft in geojson["features"]:
        nombre = ft["properties"]["neighbourhood"]
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            salida.append((nombre, np.asarray(poly[0])))
    return salida


def fig_mapa_densidad(df: pd.DataFrame, geojson: dict[str, Any], centro: tuple[float, float], centro_nombre: str) -> Figure:
    """Mapa hexbin de densidad de anuncios con los límites de los barrios."""
    d = df[df["coords_validas"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    hb = ax.hexbin(d["longitude"], d["latitude"], gridsize=70, bins="log", cmap="viridis", mincnt=1)
    ax.add_collection(PatchCollection([Polygon(a, closed=True) for _, a in _poligonos(geojson)],
                                      facecolor="none", edgecolor="grey", linewidth=0.4))
    ax.plot(centro[1], centro[0], marker="*", color="red", markersize=14, label=centro_nombre)
    fig.colorbar(hb, ax=ax, shrink=0.7, label="Anuncios por celda (escala log)")
    ax.set(title=f"Densidad de anuncios (n = {fmt.n0(len(d))})", xlabel="Longitud", ylabel="Latitud")
    ax.set_aspect(1 / np.cos(np.radians(centro[0])))
    ax.legend(loc="lower left")
    return fig


def fig_mapa_precio_barrio(df: pd.DataFrame, geojson: dict[str, Any], min_n: int) -> Figure:
    """Coropleta de la mediana de precio por barrio (gris si n < min_n)."""
    cp = _cp(df)
    g = cp.groupby("neighbourhood_cleansed")["price"].agg(["median", "size"])
    med = g.loc[g["size"] >= min_n, "median"]
    polis = _poligonos(geojson)
    norm = LogNorm(vmin=med.min(), vmax=med.max())
    cmap = plt.get_cmap("viridis")
    colores = [cmap(norm(med[n])) if n in med.index else (0.85, 0.85, 0.85, 1) for n, _ in polis]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(PatchCollection([Polygon(a, closed=True) for _, a in polis], facecolor=colores,
                                      edgecolor="white", linewidth=0.4))
    ax.autoscale_view()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, label="Mediana del precio por noche (€, escala log)")
    formateador = plt.FuncFormatter(lambda v, _: fmt.eur0(v))
    cb.ax.yaxis.set_major_formatter(formateador)
    cb.ax.yaxis.set_minor_formatter(formateador)
    ax.set(title=f"Mediana del precio por barrio (gris: menos de {min_n} anuncios con precio)",
           xlabel="Longitud", ylabel="Latitud")
    ax.set_aspect(1 / np.cos(np.radians(40.4)))
    return fig


def fig_correlaciones(hallazgos: dict[str, Any]) -> Figure:
    """Heatmaps de Pearson y Spearman."""
    c = hallazgos["descriptivo"]["correlaciones"]
    variables = c["variables"]
    etiquetas = [ETIQUETAS_VARIABLES.get(v, v) for v in variables]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    for ax, clave, titulo in ((axes[0], "pearson", "Pearson"), (axes[1], "spearman", "Spearman")):
        m = pd.DataFrame(c[clave]).loc[variables, variables].to_numpy()
        sns.heatmap(m, ax=ax, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", annot_kws={"size": 7},
                    xticklabels=etiquetas, yticklabels=etiquetas, cbar_kws={"shrink": 0.7})
        ax.set_title(f"Correlación de {titulo} (anuncios con precio, n = {fmt.n0(c['n'])})")
    fig.tight_layout()
    return fig


def fig_ic_bootstrap(hallazgos: dict[str, Any]) -> Figure:
    """Forest plot del IC 95 % bootstrap de la mediana por tipo y por distrito."""
    b = hallazgos["estadistica"]["bootstrap"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, filas, titulo in ((a1, b["tipo"], "Tipo de habitación"), (a2, b["distritos"], "Distritos con más anuncios")):
        y = np.arange(len(filas))
        med = np.array([f["mediana"] for f in filas])
        err = np.array([[f["mediana"] - f["ic_inf"] for f in filas], [f["ic_sup"] - f["mediana"] for f in filas]])
        err_c = np.array([[f["mediana"] - f["ic_inf_cluster"] for f in filas],
                          [f["ic_sup_cluster"] - f["mediana"] for f in filas]])
        ax.errorbar(med, y + 0.15, xerr=err_c, fmt="s", color=PALETA[2], ecolor=PALETA[2], capsize=4,
                    label="Por anfitrión (cluster)")
        ax.errorbar(med, y - 0.15, xerr=err, fmt="o", color=PALETA[0], ecolor=PALETA[1], capsize=4,
                    label="Por anuncio")
        ax.set_yticks(y, [f"{f['grupo']} (n = {fmt.n0(f['n'])})" for f in filas])
        ax.invert_yaxis()
        _eje_euros(ax)
        ax.set(title=titulo, xlabel="Mediana del precio e IC 95 %")
        ax.legend(loc="lower right")
    fig.suptitle(f"IC bootstrap (percentil) de la mediana del precio ({fmt.n0(b['n_boot'])} remuestreos; "
                 "remuestreando anuncios o anfitriones)", y=1.02)
    fig.tight_layout()
    return fig


def fig_ols_residuos(artefactos: dict[str, Any]) -> Figure:
    """Q-Q de residuos, histograma y residuos frente a ajustados."""
    res, aj = artefactos["residuos"], artefactos["ajustados"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    stats.probplot(res, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set(markersize=2, alpha=0.4, color=PALETA[0])
    axes[0].get_lines()[1].set(color=PALETA[3])
    axes[0].set(title="Gráfico Q-Q de los residuos", xlabel="Cuantiles teóricos (normal)", ylabel="Cuantiles observados")
    axes[1].hist(res, bins=60, density=True, color=PALETA[0], alpha=0.85)
    xs = np.linspace(res.min(), res.max(), 300)
    axes[1].plot(xs, stats.norm.pdf(xs, res.mean(), res.std()), color=PALETA[3], label="Normal teórica")
    axes[1].set(title="Histograma de residuos", xlabel="Residuo (ln €)", ylabel="Densidad")
    axes[1].legend()
    axes[2].scatter(aj, res, s=2, alpha=0.25, color=PALETA[0])
    axes[2].axhline(0, color=PALETA[3])
    axes[2].set(title="Residuos frente a valores ajustados", xlabel="ln(precio) ajustado", ylabel="Residuo")
    fig.suptitle(f"Diagnóstico de residuos de la regresión OLS sobre ln(precio) (n = {fmt.n0(len(res))})", y=1.02)
    fig.tight_layout()
    return fig


def fig_ols_coeficientes(hallazgos: dict[str, Any]) -> Figure:
    """Cambio porcentual asociado a cada variable (sin distritos) con IC 95 % (cluster por anfitrión)."""
    ols = hallazgos["estadistica"]["ols"]
    filas = sorted((c for c in ols["coeficientes"] if not c["es_distrito"] and not c.get("es_mes")),
                   key=lambda c: c["pct"])
    y = np.arange(len(filas))
    fig, ax = plt.subplots(figsize=(10, 8))
    colores = [PALETA[0] if c["significativo"] else (0.6, 0.6, 0.6) for c in filas]
    for yi, c, col in zip(y, filas, colores):
        ax.errorbar(c["pct"], yi, xerr=[[c["pct"] - c["ic_inf_pct"]], [c["ic_sup_pct"] - c["pct"]]],
                    fmt="o", color=col, ecolor=col, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [c["nombre"] for c in filas])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{fmt.n0(v)} %"))
    ax.set(title=f"Regresión OLS: cambio % del precio asociado a cada variable\n"
                 f"(IC 95 % con errores agrupados por anfitrión; gris = no significativo; R² ajustado = "
                 f"{fmt.n3(ols['r2_ajustado'])}; distritos y mes de cotización como control)",
           xlabel="Cambio porcentual en el precio, 100·(e^β − 1)")
    return fig


def fig_importancia_variables(hallazgos: dict[str, Any]) -> Figure:
    """Importancia por permutación (descenso de R² en las particiones de test agrupadas por anfitrión)."""
    ml = hallazgos["estadistica"]["ml"]
    cv = ml["cv"]
    filas = ml["importancia_permutacion"][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh([f["nombre"] for f in filas], [f["importancia"] for f in filas],
            xerr=[f["std"] for f in filas], color=PALETA[2])
    ax.set(title=f"Importancia por permutación (top 10; media de {ml['n_folds']} particiones por anfitrión)\n"
                 f"HistGradientBoosting: R² = {fmt.n3(cv['r2_media'])} ± {fmt.n3(cv['r2_std'])} (GroupKFold); "
                 "importancia predictiva, no causal",
           xlabel="Descenso medio del R² al permutar la variable (barra: desviación entre particiones)")
    return fig


def fig_ocupacion_ingresos(df: pd.DataFrame, hallazgos: dict[str, Any]) -> Figure:
    """Días no disponibles (próximos 365) vs ocupación de Inside Airbnb (últimos 12 meses), e ingresos."""
    a = df[df["precio_valido"] & df["es_activo_probable"]]
    oi = hallazgos["descriptivo"]["ocupacion_ingresos"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    bins = np.linspace(0, 100, 41)
    m_o = a["no_disponible_365"].notna() & a["ocupacion_inside_airbnb"].notna()
    a1.hist(a.loc[m_o, "no_disponible_365"] * 100, bins=bins, alpha=0.6, color=PALETA[0],
            label="Propia: % días no disponibles, próximos 365")
    a1.hist(a.loc[m_o, "ocupacion_inside_airbnb"] * 100, bins=bins, alpha=0.6, color=PALETA[1],
            label="Inside Airbnb: ocupación, últimos 12 meses")
    a1.set(title=f"Ventanas temporales distintas (mismos {fmt.n0(oi['n_comparacion_ocupacion'])} anuncios)",
           xlabel="% del año", ylabel="Anuncios")
    a1.legend()
    m = a["ingreso_hipotetico_anual"].notna() & a["estimated_revenue_l365d"].notna() & (a["estimated_revenue_l365d"] > 0) \
        & (a["ingreso_hipotetico_anual"] > 0)
    a2.scatter(a.loc[m, "estimated_revenue_l365d"], a.loc[m, "ingreso_hipotetico_anual"], s=3, alpha=0.2, color=PALETA[0])
    a2.set_xscale("log")
    a2.set_yscale("log")
    lim = [max(1, a.loc[m, ["estimated_revenue_l365d", "ingreso_hipotetico_anual"]].min().min()),
           a.loc[m, ["estimated_revenue_l365d", "ingreso_hipotetico_anual"]].max().max()]
    a2.plot(lim, lim, color=PALETA[3], label="y = x")
    a2.set(title=f"Ingreso hipotético propio vs Inside Airbnb (ρ de Spearman = "
                 f"{fmt.n2(oi['spearman_propio_vs_inside'])})",
           xlabel="Inside Airbnb, últimos 12 meses (€/año, log)",
           ylabel="Proxy: precio × días no disponibles futuros (€/año, log)")
    a2.legend()
    fig.tight_layout()
    return fig


def fig_posthoc_distritos(hallazgos: dict[str, Any]) -> Figure:
    """Heatmap de p-valores ajustados (Dunn-Bonferroni) entre los distritos con más anuncios."""
    ph = hallazgos["estadistica"]["posthoc_distrito"]
    g = ph["grupos"]
    m = pd.DataFrame(ph["matriz_p"]).loc[g, g].to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(m, ax=ax, cmap="viridis_r", vmin=0, vmax=0.1, annot=True, fmt=".3f", annot_kws={"size": 7},
                xticklabels=g, yticklabels=g, cbar_kws={"label": "p ajustado (escala recortada a 0,1)"})
    ax.set_title(f"Post-hoc: {ph['metodo']}\n{ph['n_pares_significativos']} de {ph['n_pares']} pares "
                 f"significativos (α = 0,05)")
    return fig


def fig_calidad_nulos(perfil_antes: dict[str, Any], top: int = 20) -> Figure:
    """% de nulos por columna (top) en el dataset original."""
    nulos = pd.Series(perfil_antes["nulos_por_columna_pct"]).sort_values(ascending=False, kind="mergesort").head(top)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(nulos.index, nulos.to_numpy(), color=[PALETA[3] if v == 100 else PALETA[0] for v in nulos])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{fmt.n0(v)} %"))
    ax.set(title=f"Columnas con más nulos en el dataset original (top {top}; rojo = 100 % nulas)",
           xlabel="% de valores nulos", xlim=(0, 105))
    return fig


def fig_amenities(hallazgos: dict[str, Any]) -> Figure:
    """Prevalencia de amenities y prima de precio mediana (con vs sin)."""
    filas = hallazgos["descriptivo"]["amenities"]
    nombres = [NOMBRES_AMENITIES.get(f["amenity"], f["amenity"]) for f in filas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    a1.barh(nombres, [f["prevalencia_pct"] for f in filas], color=PALETA[0])
    a1.invert_yaxis()
    a1.set(title="Prevalencia de amenities (% de anuncios)", xlabel="% de anuncios", xlim=(0, 100))
    a2.barh(nombres, [f["prima_pct"] for f in filas], color=[PALETA[2] if f["prima_pct"] >= 0 else PALETA[3] for f in filas])
    a2.axvline(0, color="black", lw=0.8)
    a2.set(title="Diferencia de la mediana de precio (con vs sin)", xlabel="% (asociación bruta, no causal)")
    fig.tight_layout()
    return fig


def generar_todas(df: pd.DataFrame, hallazgos: dict[str, Any], artefactos: dict[str, Any], geojson: dict[str, Any],
                  perfil_antes: dict[str, Any], centro: tuple[float, float], centro_nombre: str,
                  directorio: Path, min_n_barrio: int, min_n_mapa: int) -> list[str]:
    """Genera y guarda todas las figuras. Devuelve la lista de nombres de archivo."""
    configurar_estilo()
    especificaciones: list[tuple[str, Callable[[], Figure]]] = [
        ("01_distribucion_precio.png", lambda: fig_distribucion_precio(df)),
        ("02_precio_por_tipo_habitacion.png", lambda: fig_precio_por_tipo(df)),
        ("03_precio_por_distrito.png", lambda: fig_precio_por_distrito(hallazgos)),
        ("04_precio_por_barrio.png", lambda: fig_precio_por_barrio(df, min_n_barrio)),
        ("05_precio_por_capacidad.png", lambda: fig_precio_por_capacidad(hallazgos)),
        ("06_composicion_hosts.png", lambda: fig_composicion_hosts(df, hallazgos)),
        ("07_mapa_densidad.png", lambda: fig_mapa_densidad(df, geojson, centro, centro_nombre)),
        ("08_mapa_precio_barrio.png", lambda: fig_mapa_precio_barrio(df, geojson, min_n_mapa)),
        ("09_correlaciones.png", lambda: fig_correlaciones(hallazgos)),
        ("10_ic_bootstrap_mediana.png", lambda: fig_ic_bootstrap(hallazgos)),
        ("11_ols_diagnostico_residuos.png", lambda: fig_ols_residuos(artefactos)),
        ("12_ols_coeficientes.png", lambda: fig_ols_coeficientes(hallazgos)),
        ("14_ocupacion_ingresos.png", lambda: fig_ocupacion_ingresos(df, hallazgos)),
        ("15_posthoc_dunn_distritos.png", lambda: fig_posthoc_distritos(hallazgos)),
        ("16_calidad_nulos.png", lambda: fig_calidad_nulos(perfil_antes)),
        ("17_amenities.png", lambda: fig_amenities(hallazgos)),
    ]
    if hallazgos["estadistica"]["ml"].get("ejecutado"):
        especificaciones.insert(12, ("13_importancia_variables.png", lambda: fig_importancia_variables(hallazgos)))
    nombres = []
    for nombre, construir in especificaciones:
        guardar_figura(construir(), nombre, directorio)
        nombres.append(nombre)
        logger.info("Figura %s", nombre)
    return nombres


def validar_figuras(directorio: Path, nombres: list[str], min_bytes: int = 5000) -> list[str]:
    """Devuelve los problemas encontrados (inexistente, < 5 KB o imagen en blanco)."""
    import matplotlib.image as mpimg
    problemas = []
    for n in nombres:
        ruta = directorio / n
        if not ruta.exists():
            problemas.append(f"{n}: no existe")
            continue
        if ruta.stat().st_size < min_bytes:
            problemas.append(f"{n}: pesa menos de {min_bytes} bytes")
            continue
        if float(np.std(mpimg.imread(ruta))) == 0:
            problemas.append(f"{n}: imagen en blanco")
    return problemas
