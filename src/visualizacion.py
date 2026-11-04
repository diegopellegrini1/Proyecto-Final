"""Figuras estáticas (matplotlib/seaborn) del análisis descriptivo y estadístico de las 9 ciudades.

Cada ``fig_*`` devuelve una ``Figure``; ``guardar_figura`` la escribe en PNG y la cierra.
Los notebooks llaman a las mismas funciones y las muestran en línea. Todos los ejes de precio están en
euros y las escalas logarítmicas se indican en el título o en el eje.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from scipy import stats

from src import config, formato as fmt
from src.analisis import ETIQUETAS_VARIABLES, NOMBRE_CIUDAD

logger = logging.getLogger(__name__)

DPI = 150
PALETA = sns.color_palette("colorblind")
GRIS = (0.6, 0.6, 0.6)
COLORES_CIUDAD: dict[str, Any] = {n: c for n, c in zip(sorted(NOMBRE_CIUDAD.values()),
                                                       sns.color_palette("tab10", len(NOMBRE_CIUDAD)))}


def configurar_estilo() -> None:
    """Estilo común: seaborn whitegrid, paleta apta para daltonismo y tamaños legibles."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 12, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                         "axes.formatter.use_locale": False})


def guardar_figura(fig: Figure, nombre: str, directorio: Path) -> Path:
    """Guarda ``fig`` como ``directorio/nombre`` (PNG, 150 dpi) y la cierra."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _cp(df: pd.DataFrame) -> pd.DataFrame:
    """Anuncios con precio válido, con el nombre legible de la ciudad."""
    cp = df[df["precio_valido"]]
    return cp.assign(nombre_ciudad=cp["ciudad"].map(NOMBRE_CIUDAD))


def _eje_euros(ax, eje: str = "x") -> None:
    """Formatea un eje con separador de miles es-ES y símbolo €."""
    f = plt.FuncFormatter(lambda v, _: f"{fmt.n0(v)} €")
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(f)
    (ax.xaxis if eje == "x" else ax.yaxis).set_minor_formatter(plt.NullFormatter())


def _eje_pct(ax, eje: str = "x") -> None:
    """Formatea un eje en porcentaje es-ES."""
    f = plt.FuncFormatter(lambda v, _: f"{fmt.n0(v)} %")
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(f)


def _ticks_log_euros(ax, valores: list[float], eje: str = "x") -> None:
    """Marcas legibles en un eje logarítmico de euros."""
    lim = ax.get_xlim() if eje == "x" else ax.get_ylim()
    marcas = [v for v in valores if lim[0] <= v <= lim[1]]
    (ax.set_xticks if eje == "x" else ax.set_yticks)(marcas)
    _eje_euros(ax, eje)


MARCAS_EUROS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


# --------------------------------------------------------------------------
# Calidad y descriptivo
# --------------------------------------------------------------------------
def fig_calidad_nulos(informe_calidad: dict[str, Any]) -> Figure:
    """Heatmap del % de nulos por ciudad en columnas clave, en el conjunto unido (antes de limpiar)."""
    por = informe_calidad["por_ciudad"]
    tabla = pd.DataFrame({NOMBRE_CIUDAD[c]: v["antes"]["nulos_pct"] for c, v in por.items()}).T.sort_index()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(tabla, ax=ax, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100, annot_kws={"size": 8},
                cbar_kws={"label": "% de anuncios con el valor vacío"})
    ax.set(title="Valores vacíos por ciudad en columnas clave del conjunto unido (antes de limpiar, %)",
           xlabel="Columna", ylabel="")
    ax.tick_params(axis="x", rotation=35)
    plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def fig_distribucion_precio(df: pd.DataFrame) -> Figure:
    """Histograma del precio (recortado a P99) y de ln(precio) con curva normal."""
    cp = _cp(df)
    p = cp["price"]
    p99 = p.quantile(0.99)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.hist(p[p <= p99], bins=60, color=PALETA[0], alpha=0.85)
    a1.axvline(p.median(), color=PALETA[1], lw=2, label=f"Mediana: {fmt.eur0(p.median())}")
    a1.axvline(p.mean(), color=PALETA[2], lw=2, ls="--", label=f"Media: {fmt.eur0(p.mean())}")
    a1.set(title=f"Precio por noche (hasta el P99 = {fmt.eur0(p99)})", xlabel="Precio por noche",
           ylabel="Anuncios")
    _eje_euros(a1)
    a1.legend()
    lp = cp["log_precio"]
    a2.hist(lp, bins=60, density=True, color=PALETA[0], alpha=0.85, label="ln(precio)")
    xs = np.linspace(lp.min(), lp.max(), 300)
    a2.plot(xs, stats.norm.pdf(xs, lp.mean(), lp.std()), color=PALETA[3], lw=2, label="Normal teórica")
    a2.set(title="Logaritmo natural del precio", xlabel="ln(precio en €)", ylabel="Densidad")
    a2.legend()
    fig.suptitle(f"Distribución del precio en las 9 ciudades (n = {fmt.n0(len(cp))} anuncios con precio)", y=1.02)
    return fig


def _boxplot_horizontal(ax, grupos: list[str], datos: list[np.ndarray], colores: list[Any]) -> None:
    """Boxplot horizontal en escala log con n en la etiqueta."""
    bp = ax.boxplot(datos, orientation="horizontal", patch_artist=True, showfliers=False, widths=0.6)
    for caja, color in zip(bp["boxes"], colores):
        caja.set_facecolor(color)
        caja.set_alpha(0.75)
    ax.set_yticks(range(1, len(grupos) + 1), [f"{g}\n(n = {fmt.n0(len(d))})" for g, d in zip(grupos, datos)])
    ax.set_xscale("log")
    ax.invert_yaxis()


def fig_precio_por_ciudad(df: pd.DataFrame) -> Figure:
    """Boxplot del precio (escala log, sin atípicos dibujados) por ciudad, ordenado por mediana."""
    cp = _cp(df)
    orden = cp.groupby("nombre_ciudad")["price"].median().sort_values(ascending=False).index.tolist()
    datos = [cp.loc[cp["nombre_ciudad"] == c, "price"].to_numpy() for c in orden]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _boxplot_horizontal(ax, orden, datos, [COLORES_CIUDAD[c] for c in orden])
    _ticks_log_euros(ax, MARCAS_EUROS)
    ax.set(title="Precio por noche según ciudad o región (escala logarítmica; caja = Q1–Q3, línea = mediana)\n"
                 "Bigotes: hasta el dato más extremo a 1,5 × RIC de la caja; los atípicos no se dibujan",
           xlabel="Precio por noche (escala log)")
    return fig


def fig_precio_por_tipo(df: pd.DataFrame) -> Figure:
    """Boxplot del precio (escala log) por tipo de habitación, con n."""
    cp = _cp(df)
    orden = cp.groupby("tipo_habitacion")["price"].median().sort_values(ascending=False).index.tolist()
    datos = [cp.loc[cp["tipo_habitacion"] == t, "price"].to_numpy() for t in orden]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _boxplot_horizontal(ax, orden, datos, PALETA[:len(orden)])
    _ticks_log_euros(ax, MARCAS_EUROS)
    ax.set(title="Precio por noche según tipo de habitación (escala logarítmica)", xlabel="Precio por noche (escala log)")
    return fig


def _barras_apiladas(ax, tabla: pd.DataFrame, colores: list[Any]) -> None:
    """Barras horizontales apiladas al 100 % con la etiqueta del % si el tramo es visible."""
    izquierda = np.zeros(len(tabla))
    for col, color in zip(tabla.columns, colores):
        v = tabla[col].to_numpy()
        ax.barh(tabla.index, v, left=izquierda, color=color, label=col)
        for i, (x0, w) in enumerate(zip(izquierda, v)):
            if w >= 6:
                ax.text(x0 + w / 2, i, fmt.n0(w), ha="center", va="center", fontsize=8, color="white")
        izquierda += v
    ax.set_xlim(0, 100)
    _eje_pct(ax)
    ax.invert_yaxis()


def fig_tipo_por_ciudad(hallazgos: dict[str, Any]) -> Figure:
    """Composición de la oferta por tipo de habitación en cada ciudad (% de anuncios)."""
    tabla = pd.DataFrame(hallazgos["descriptivo"]["tipo_por_ciudad_pct"]).T.fillna(0)
    tabla = tabla.sort_values(tabla.columns[0], ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _barras_apiladas(ax, tabla, PALETA)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set(title="Tipo de habitación por ciudad (% de anuncios; etiqueta = %)", xlabel="% de anuncios")
    return fig


def fig_precio_por_capacidad(hallazgos: dict[str, Any]) -> Figure:
    """Mediana y rango intercuartílico del precio por tramo de capacidad."""
    filas = hallazgos["descriptivo"]["por_capacidad"]
    x = np.arange(len(filas))
    med = np.array([f["mediana"] for f in filas])
    err = np.array([[f["mediana"] - f["q1"] for f in filas], [f["q3"] - f["mediana"] for f in filas]])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, med, color=PALETA[0], alpha=0.85)
    ax.errorbar(x, med, yerr=err, fmt="none", ecolor="black", capsize=6, label="Rango intercuartílico (Q1–Q3)")
    ax.set_xticks(x, [f"{f['grupo']}\n(n = {fmt.n0(f['n'])})" for f in filas])
    _eje_euros(ax, "y")
    ax.set_ylim(bottom=0)
    ax.set(title="Precio por noche según la capacidad del alojamiento (9 ciudades)", ylabel="Mediana del precio")
    ax.legend()
    return fig


def fig_anfitriones(df: pd.DataFrame, hallazgos: dict[str, Any]) -> Figure:
    """Anuncios por anfitrión (log-log) y % de anuncios de multi-anuncio y de superhost por ciudad."""
    h = hallazgos["descriptivo"]["hosts"]
    por_host = df.groupby("host_id").size()
    filas = sorted(hallazgos["descriptivo"]["por_ciudad"], key=lambda f: -f["pct_multianuncio"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.hist(por_host, bins=np.logspace(0, np.log10(por_host.max() + 1), 30), color=PALETA[0])
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set(title=f"Anuncios por anfitrión ({fmt.n0(h['n_hosts'])} anfitriones)",
           xlabel="Nº de anuncios del anfitrión (escala log)", ylabel="Nº de anfitriones (escala log)")
    y = np.arange(len(filas))
    a2.barh(y - 0.2, [f["pct_multianuncio"] for f in filas], height=0.4, color=PALETA[1],
            label="Anuncios de anfitriones con varios anuncios")
    a2.barh(y + 0.2, [f["pct_superhost"] for f in filas], height=0.4, color=PALETA[2], label="Anuncios de superhosts")
    a2.set_yticks(y, [f["grupo"] for f in filas])
    a2.invert_yaxis()
    a2.set_xlim(0, 100)
    _eje_pct(a2)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    a2.set(title="Anfitriones por ciudad (% de anuncios)", xlabel="% de anuncios de la ciudad")
    fig.tight_layout()
    return fig


def fig_licencias_por_ciudad(hallazgos: dict[str, Any]) -> Figure:
    """Tipo de licencia declarada por ciudad (% de anuncios)."""
    tabla = pd.DataFrame(hallazgos["descriptivo"]["licencia_por_ciudad_pct"]).T.fillna(0)
    orden = [c for c in ("Con nº de registro", "Exenta", "Otro / en proceso", "Sin licencia") if c in tabla.columns]
    tabla = tabla[orden].sort_values("Sin licencia", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _barras_apiladas(ax, tabla, [PALETA[2], PALETA[0], PALETA[4], PALETA[3]])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    ax.set(title="Licencia declarada en el anuncio por ciudad (% de anuncios; texto del anuncio, no verificado)",
           xlabel="% de anuncios")
    return fig


def fig_ocupacion(df: pd.DataFrame, hallazgos: dict[str, Any]) -> Figure:
    """Proxy de ocupación (% días no disponibles, próximos 365) frente a la estimación de Inside Airbnb."""
    a = df[df["precio_valido"] & df["es_activo_probable"]].assign(nombre_ciudad=lambda d: d["ciudad"].map(NOMBRE_CIUDAD))
    tabla = a.groupby("nombre_ciudad").agg(
        proxy=("no_disponible_365", lambda s: s.mean() * 100),
        inside=("estimated_occupancy_l365d", lambda s: (s / 365).clip(0, 1).mean() * 100)).sort_values("proxy")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(tabla))
    ax.barh(y - 0.2, tabla["proxy"], height=0.4, color=PALETA[0],
            label="Proxy propio: % días no disponibles, próximos 365 (reservas + bloqueos)")
    ax.barh(y + 0.2, tabla["inside"], height=0.4, color=PALETA[1],
            label="Inside Airbnb: ocupación estimada, últimos 12 meses")
    ax.set_yticks(y, tabla.index)
    ax.set_xlim(0, 100)
    _eje_pct(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    oc = hallazgos["descriptivo"]["ocupacion"]
    ax.set(title=f"Ocupación: dos medidas con ventanas distintas (media; {fmt.n0(oc['n_comparacion'])} anuncios "
                 "activos con precio)", xlabel="% del año")
    return fig


def _mapas(df: pd.DataFrame, columna: str, norm: Normalize, etiqueta: str, titulo: str,
           cmap: str = "viridis") -> Figure:
    """Mapas pequeños (3×3), uno por ciudad, con los puntos coloreados por ``columna`` y escala común."""
    d = df[df["coords_validas"] & df[columna].notna()]
    ciudades = sorted(d["ciudad"].unique(), key=lambda c: NOMBRE_CIUDAD[c])
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for ax, c in zip(axes.flat, ciudades):
        s = d[d["ciudad"] == c].sort_values(columna)
        sc = ax.scatter(s["longitude"], s["latitude"], c=s[columna], s=1.5, cmap=cmap, norm=norm, alpha=0.8,
                        linewidths=0, rasterized=True)
        ax.set_aspect(1 / np.cos(np.radians(s["latitude"].mean())))
        ax.set_title(f"{NOMBRE_CIUDAD[c]} (n = {fmt.n0(len(s))})", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Longitud", fontsize=8)
        ax.set_ylabel("Latitud", fontsize=8)
    for ax in list(axes.flat)[len(ciudades):]:
        ax.axis("off")
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, label=etiqueta)
    formateador = plt.FuncFormatter(lambda v, _: fmt.eur0(v))
    cb.ax.yaxis.set_major_formatter(formateador)
    cb.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    fig.suptitle(titulo, y=0.93, fontsize=13)
    return fig


def fig_mapa_precio(df: pd.DataFrame) -> Figure:
    """Mapa de puntos por ciudad coloreado por el precio (escala log común a todas las ciudades)."""
    cp = df[df["precio_valido"]]
    lo, hi = cp["price"].quantile([0.02, 0.98])
    return _mapas(cp.assign(precio_recortado=cp["price"].clip(lo, hi)), "precio_recortado", LogNorm(lo, hi),
                  "Precio por noche (€, escala log; recortado a P2–P98)",
                  "Precio por noche de cada anuncio (escala de color común; cada mapa tiene su propia extensión)")


def fig_mapa_renta(df: pd.DataFrame) -> Figure:
    """Mapa de puntos por ciudad coloreado por la renta neta por persona del distrito censal (INE)."""
    r = df[config.RENTA_PRINCIPAL]
    lo, hi = r.quantile([0.01, 0.99])
    return _mapas(df.assign(renta_recortada=r.clip(lo, hi)), "renta_recortada", Normalize(lo, hi),
                  "Renta neta media por persona del distrito censal (€/año, INE)",
                  "Renta de la zona (INE) en la ubicación de cada anuncio (escala de color común)", cmap="cividis")


def fig_renta_precio_zonas(df: pd.DataFrame, hallazgos: dict[str, Any]) -> Figure:
    """Mediana del precio frente a la renta del distrito censal (un punto por distrito con n ≥ 30)."""
    cp = _cp(df)
    g = cp.groupby(["nombre_ciudad", "codigo_distrito_ine"]).agg(
        n=("price", "size"), mediana=("price", "median"), renta=(config.RENTA_PRINCIPAL, "first")).dropna()
    g = g[g["n"] >= config.MIN_N_GRUPO].reset_index()
    eco = hallazgos["estadistica"]["renta_precio"]["ecologico_global"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for c, s in g.groupby("nombre_ciudad"):
        ax.scatter(s["renta"], s["mediana"], s=np.clip(s["n"] / 8, 8, 200), alpha=0.65, color=COLORES_CIUDAD[c],
                   label=c, edgecolors="white", linewidths=0.4)
    ax.set_yscale("log")
    _ticks_log_euros(ax, MARCAS_EUROS, "y")
    _eje_euros(ax, "x")
    ax.legend(ncol=3, loc="upper left", markerscale=0.8)
    ax.set(title=f"Renta del distrito censal y mediana del precio (un punto por distrito con al menos "
                 f"{config.MIN_N_GRUPO} anuncios; tamaño proporcional a n)\nρ de Spearman entre distritos = "
                 f"{fmt.n2(eco['rho'])} ({fmt.n0(eco['n_zonas'])} distritos); asociación, no causalidad",
           xlabel="Renta neta media por persona del distrito (€/año, INE)",
           ylabel="Mediana del precio por noche (escala log)")
    return fig


def fig_spearman_ciudad(hallazgos: dict[str, Any]) -> Figure:
    """Forest plot de la ρ de Spearman renta-precio por ciudad con IC 95 % por clusters de distrito."""
    rp = hallazgos["estadistica"]["renta_precio"]
    filas = [f for f in rp["por_ciudad"] if f["distrito"].get("evaluable")]
    g = rp["global_distrito"]
    etiquetas = [f"{f['ciudad']} (n = {fmt.n0(f['distrito']['n'])}; {fmt.n0(f['distrito']['cluster_zona']['n_clusters'])} distritos)"
                 for f in filas]
    # El total no lleva Holm: se dibuja aparte (negro, cuadrado) con su lectura prudente del p bootstrap.
    gz = g["cluster_zona"]
    etiquetas.append(f"Total {len(filas)} ciudades (n = {fmt.n0(g['n'])})\n"
                     f"{gz.get('lectura', '')}: p bootstrap = {fmt.n3(gz['p_boot'])}")
    rho = [f["distrito"]["rho"] for f in filas] + [g["rho"]]
    inf = [f["distrito"]["cluster_zona"]["ic_inf"] for f in filas] + [gz["ic_inf"]]
    sup = [f["distrito"]["cluster_zona"]["ic_sup"] for f in filas] + [gz["ic_sup"]]
    colores = [PALETA[0] if f["distrito"].get("holm", {}).get("significativo_holm", False) else GRIS
               for f in filas] + ["black"]
    marcas = ["o"] * len(filas) + ["s"]
    y = np.arange(len(rho))
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for yi, r, a, b, color, m in zip(y, rho, inf, sup, colores, marcas):
        ax.errorbar(r, yi, xerr=[[r - a], [b - r]], fmt=m, color=color, ecolor=color, capsize=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(len(filas) - 0.5, color=GRIS, lw=0.6, ls=":")
    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set(title="Renta neta por persona del distrito (INE) y precio: ρ de Spearman por ciudad\n"
                 "IC 95 % bootstrap remuestreando distritos censales; azul = significativo tras Holm, gris = no;\n"
                 "total en negro (sin Holm; su lectura figura en la etiqueta)",
           xlabel="ρ de Spearman (anuncio a anuncio)")
    return fig


def fig_correlaciones(hallazgos: dict[str, Any]) -> Figure:
    """Heatmap de Spearman entre variables del anuncio y de la zona (INE)."""
    c = hallazgos["descriptivo"]["correlaciones"]
    variables = c["variables"]
    etiquetas = [ETIQUETAS_VARIABLES.get(v, v) for v in variables]
    m = pd.DataFrame(c["spearman"]).loc[variables, variables].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(m, ax=ax, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=etiquetas, yticklabels=etiquetas, cbar_kws={"shrink": 0.7, "label": "ρ de Spearman"},
                mask=np.triu(np.ones_like(m, dtype=bool), k=1))
    ax.set_title(f"Correlación de Spearman entre variables del anuncio y de la zona (n = {fmt.n0(c['n'])} anuncios "
                 "con precio; por pares)")
    return fig


def fig_ic_bootstrap(hallazgos: dict[str, Any]) -> Figure:
    """Forest plot del IC 95 % bootstrap de la mediana del precio por ciudad (por anuncio y por anfitrión)."""
    b = hallazgos["estadistica"]["bootstrap"]
    filas = b["ciudades"]
    y = np.arange(len(filas))
    med = np.array([f["mediana"] for f in filas])
    err = np.array([[f["mediana"] - f["ic_inf"] for f in filas], [f["ic_sup"] - f["mediana"] for f in filas]])
    err_c = np.array([[f["mediana"] - f["ic_inf_cluster"] for f in filas],
                      [f["ic_sup_cluster"] - f["mediana"] for f in filas]])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(med, y + 0.15, xerr=err_c, fmt="s", color=PALETA[2], ecolor=PALETA[2], capsize=4,
                label="Remuestreando anfitriones (cluster)")
    ax.errorbar(med, y - 0.15, xerr=err, fmt="o", color=PALETA[0], ecolor=PALETA[0], capsize=4,
                label="Remuestreando anuncios")
    ax.set_yticks(y, [f"{f['grupo']} (n = {fmt.n0(f['n'])})" for f in filas])
    ax.invert_yaxis()
    _eje_euros(ax)
    ax.legend(loc="lower right")
    ax.set(title=f"IC 95 % bootstrap (percentil, {fmt.n0(b['n_boot'])} remuestreos) de la mediana del precio "
                 "por ciudad", xlabel="Mediana del precio por noche")
    return fig


def fig_posthoc_ciudades(hallazgos: dict[str, Any]) -> Figure:
    """Heatmap de p-valores ajustados (Dunn-Bonferroni) entre ciudades."""
    ph = hallazgos["estadistica"]["posthoc_ciudad"]
    g = ph["grupos"]
    m = pd.DataFrame(ph["matriz_p"]).loc[g, g].to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(m, ax=ax, cmap="viridis_r", vmin=0, vmax=0.1, annot=True, fmt=".3f", annot_kws={"size": 7},
                xticklabels=g, yticklabels=g, cbar_kws={"label": "p ajustado (escala de color recortada a 0,1)"},
                mask=np.eye(len(g), dtype=bool))
    ax.set_title(f"Post-hoc de Dunn con Bonferroni entre ciudades: {ph['n_pares_significativos']} de "
                 f"{ph['n_pares']} pares significativos (α = {fmt.n2(ph['alfa'])})")
    return fig


def fig_ols_coeficientes(hallazgos: dict[str, Any]) -> Figure:
    """Cambio porcentual asociado a cada variable (IC 95 %, cluster por anfitrión) y efectos fijos de ciudad."""
    ols = hallazgos["estadistica"]["ols"]
    variables = []
    for c in ols["coeficientes"]:
        if c["es_ciudad"] or c["es_mes"]:
            continue
        if c["termino"] == "ln_renta_dis":
            # En logaritmo: se dibuja el cambio asociado a una renta un 10 % mayor (no a +1 en ln).
            c = {**c, "nombre": "Renta neta por persona del distrito (+10 %)", "pct": c["pct_por_10pct_renta"],
                 "ic_inf_pct": c["ic_inf_10pct"], "ic_sup_pct": c["ic_sup_10pct"]}
        variables.append(c)
    variables.sort(key=lambda c: c["pct"])
    ciudades = sorted(ols["coef_ciudades"], key=lambda c: c["pct"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, filas in ((a1, variables), (a2, ciudades)):
        y = np.arange(len(filas))
        for yi, c in zip(y, filas):
            col = (PALETA[3] if c["es_ine"] else PALETA[0]) if c["significativo"] else GRIS
            ax.errorbar(c["pct"], yi, xerr=[[c["pct"] - c["ic_inf_pct"]], [c["ic_sup_pct"] - c["pct"]]],
                        fmt="o", color=col, ecolor=col, capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y, [c["nombre"].replace("Ciudad: ", "") for c in filas])
        _eje_pct(ax)
    a1.set(title="Variables del anuncio y de la zona (rojo = INE; gris = no significativo)",
           xlabel="Cambio % del precio, 100·(e^β − 1)")
    a2.set(title=f"Efecto fijo de ciudad (referencia: {ols['referencias']['ciudad']})",
           xlabel="Cambio % del precio frente a la referencia")
    fig.suptitle(f"Regresión OLS sobre ln(precio): IC 95 % con errores agrupados por anfitrión "
                 f"(n = {fmt.n0(ols['n_efectivo'])}; R² ajustado = {fmt.n3(ols['r2_ajustado'])}; "
                 "mes de cotización como control)", y=1.0)
    fig.tight_layout()
    return fig


def fig_ols_residuos(artefactos: dict[str, Any]) -> Figure:
    """Q-Q de residuos, histograma y residuos frente a ajustados."""
    res, aj = artefactos["residuos"], artefactos["ajustados"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    stats.probplot(res, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set(markersize=2, alpha=0.4, color=PALETA[0], rasterized=True)
    axes[0].get_lines()[1].set(color=PALETA[3])
    axes[0].set(title="Gráfico Q-Q de los residuos", xlabel="Cuantiles teóricos (normal)",
                ylabel="Cuantiles observados")
    axes[1].hist(res, bins=60, density=True, color=PALETA[0], alpha=0.85)
    xs = np.linspace(res.min(), res.max(), 300)
    axes[1].plot(xs, stats.norm.pdf(xs, res.mean(), res.std()), color=PALETA[3], label="Normal teórica")
    axes[1].set(title="Histograma de residuos", xlabel="Residuo (ln €)", ylabel="Densidad")
    axes[1].legend()
    axes[2].scatter(aj, res, s=2, alpha=0.2, color=PALETA[0], rasterized=True)
    axes[2].axhline(0, color=PALETA[3])
    axes[2].set(title="Residuos frente a valores ajustados", xlabel="ln(precio) ajustado", ylabel="Residuo (ln €)")
    fig.suptitle(f"Diagnóstico de residuos de la regresión OLS sobre ln(precio) (n = {fmt.n0(len(res))})", y=1.02)
    fig.tight_layout()
    return fig


def fig_importancia_variables(hallazgos: dict[str, Any]) -> Figure:
    """Importancia por permutación (descenso de R² en las particiones de test agrupadas por anfitrión)."""
    ml = hallazgos["estadistica"]["ml"]
    cv = ml["cv"]
    filas = ml["importancia_permutacion"][::-1]
    colores = [PALETA[3] if f["variable"] in config.VARIABLES_INE_ML else PALETA[2] for f in filas]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh([f["nombre"] for f in filas], [f["importancia"] for f in filas], xerr=[f["std"] for f in filas],
            color=colores)
    ax.set(title=f"Importancia por permutación (top 10; rojo = variable del INE; media de {ml['n_folds']} "
                 f"particiones por anfitrión)\nHistGradientBoosting: R² = {fmt.n3(cv['r2_media'])} ± "
                 f"{fmt.n3(cv['r2_std'])} (GroupKFold); importancia predictiva, no causal",
           xlabel="Descenso medio del R² al permutar la variable (barra: desviación entre particiones)")
    return fig


def fig_amenities(hallazgos: dict[str, Any]) -> Figure:
    """Prevalencia de amenities y diferencia de la mediana de precio (con vs sin)."""
    filas = hallazgos["descriptivo"]["amenities"]
    nombres = [f["nombre"] for f in filas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    a1.barh(nombres, [f["prevalencia_pct"] for f in filas], color=PALETA[0])
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    _eje_pct(a1)
    a1.set(title="Prevalencia de amenities (% de anuncios)", xlabel="% de anuncios")
    a2.barh(nombres, [f["prima_pct"] for f in filas],
            color=[PALETA[2] if f["prima_pct"] >= 0 else PALETA[3] for f in filas])
    a2.axvline(0, color="black", lw=0.8)
    _eje_pct(a2)
    a2.set(title="Diferencia de la mediana del precio (con vs sin)", xlabel="% (asociación bruta, sin controles)")
    fig.tight_layout()
    return fig


def especificaciones(df: pd.DataFrame, hallazgos: dict[str, Any], artefactos: dict[str, Any],
                     informe_calidad: dict[str, Any]) -> list[tuple[str, Callable[[], Figure]]]:
    """Lista ordenada (nombre de archivo, función que construye la figura)."""
    lista: list[tuple[str, Callable[[], Figure]]] = [
        ("01_calidad_nulos_por_ciudad.png", lambda: fig_calidad_nulos(informe_calidad)),
        ("02_distribucion_precio.png", lambda: fig_distribucion_precio(df)),
        ("03_precio_por_ciudad.png", lambda: fig_precio_por_ciudad(df)),
        ("04_precio_por_tipo_habitacion.png", lambda: fig_precio_por_tipo(df)),
        ("05_tipo_por_ciudad.png", lambda: fig_tipo_por_ciudad(hallazgos)),
        ("06_precio_por_capacidad.png", lambda: fig_precio_por_capacidad(hallazgos)),
        ("07_anfitriones.png", lambda: fig_anfitriones(df, hallazgos)),
        ("08_licencias_por_ciudad.png", lambda: fig_licencias_por_ciudad(hallazgos)),
        ("09_ocupacion_proxy.png", lambda: fig_ocupacion(df, hallazgos)),
        ("10_mapa_precio.png", lambda: fig_mapa_precio(df)),
        ("11_mapa_renta_ine.png", lambda: fig_mapa_renta(df)),
        ("12_renta_precio_distritos.png", lambda: fig_renta_precio_zonas(df, hallazgos)),
        ("13_spearman_renta_por_ciudad.png", lambda: fig_spearman_ciudad(hallazgos)),
        ("14_correlaciones.png", lambda: fig_correlaciones(hallazgos)),
        ("15_ic_bootstrap_ciudades.png", lambda: fig_ic_bootstrap(hallazgos)),
        ("16_posthoc_dunn_ciudades.png", lambda: fig_posthoc_ciudades(hallazgos)),
        ("17_ols_coeficientes.png", lambda: fig_ols_coeficientes(hallazgos)),
        ("18_ols_diagnostico_residuos.png", lambda: fig_ols_residuos(artefactos)),
        ("20_amenities.png", lambda: fig_amenities(hallazgos)),
    ]
    if hallazgos["estadistica"]["ml"].get("ejecutado"):
        lista.insert(18, ("19_importancia_variables.png", lambda: fig_importancia_variables(hallazgos)))
    return lista


def generar_todas(df: pd.DataFrame, hallazgos: dict[str, Any], artefactos: dict[str, Any],
                  informe_calidad: dict[str, Any], directorio: Path) -> list[str]:
    """Genera y guarda todas las figuras. Devuelve la lista de nombres de archivo."""
    configurar_estilo()
    nombres = []
    for nombre, construir in especificaciones(df, hallazgos, artefactos, informe_calidad):
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
