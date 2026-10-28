"""Análisis descriptivo (nacional y por ciudad) y orquestación de la estadística → diccionario de hallazgos.

``ejecutar`` devuelve ``(hallazgos, artefactos)``: ``hallazgos`` es serializable y va a
``findings.json``; ``artefactos`` contiene arrays que solo necesitan las figuras.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import config, estadistica as est
from src.features import ETIQUETAS_CAPACIDAD

logger = logging.getLogger(__name__)

VARIABLES_CORRELACION: tuple[str, ...] = (
    "price", "log_precio", "accommodates", "bedrooms", "bathrooms", "n_amenities", "number_of_reviews",
    "review_scores_rating", "availability_365", "minimum_nights", "calculated_host_listings_count",
    "ine_dis_renta_neta_persona", "ine_dis_gini", "ine_dis_edad_media", "ine_dis_pct_hogares_unipersonales",
    "ine_mun_renta_neta_persona",
)
ETIQUETAS_VARIABLES: dict[str, str] = {
    "price": "Precio", "log_precio": "ln(precio)", "accommodates": "Capacidad", "bedrooms": "Dormitorios",
    "bathrooms": "Baños", "n_amenities": "Nº amenities", "number_of_reviews": "Nº reseñas",
    "review_scores_rating": "Valoración", "availability_365": "Disponibilidad 365",
    "minimum_nights": "Noches mínimas", "calculated_host_listings_count": "Anuncios del host",
    "ine_dis_renta_neta_persona": "Renta distrito", "ine_dis_gini": "Gini distrito",
    "ine_dis_edad_media": "Edad media distrito", "ine_dis_pct_hogares_unipersonales": "% unipersonales distrito",
    "ine_mun_renta_neta_persona": "Renta municipio",
}
NOMBRE_CIUDAD: dict[str, str] = {c.clave: c.nombre for c in config.CIUDADES_ESPANA.values()}


def nombre_ciudad(serie: pd.Series) -> pd.Series:
    """Nombre legible de la ciudad o región a partir de su clave."""
    return serie.map(NOMBRE_CIUDAD).astype("string")


def es_urbana(serie: pd.Series) -> pd.Series:
    """True si la ciudad es urbana (con centro definido) y no una región."""
    return serie.map(lambda c: config.CIUDADES_ESPANA[c].tipo == config.TIPO_URBANA).astype(bool)


def descriptivos(serie: pd.Series) -> dict[str, float]:
    """n, media, mediana, desviación, cuartiles, IQR, mín., máx., asimetría, curtosis de exceso y CV."""
    x = serie.dropna().astype("float64")
    if x.empty:
        return {"n": 0}
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    media = float(x.mean())
    return {
        "n": int(len(x)), "media": media, "mediana": float(x.median()), "std": float(x.std()),
        "q1": q1, "q3": q3, "iqr": q3 - q1, "min": float(x.min()), "max": float(x.max()),
        "p5": float(x.quantile(0.05)), "p95": float(x.quantile(0.95)),
        "asimetria": float(stats.skew(x)), "curtosis_exceso": float(stats.kurtosis(x)),
        "cv": float(x.std() / media) if media else float("nan"),
    }


def obtener_subconjuntos(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Subconjuntos de análisis: todos, con precio, con precio sin outliers y activos con precio."""
    con_precio = df[df["precio_valido"]]
    return {
        "todos": df,
        "con_precio": con_precio,
        "con_precio_sin_outliers": con_precio[~con_precio["es_outlier_precio"]],
        "activos_con_precio": con_precio[con_precio["es_activo_probable"]],
    }


def precio_por_grupo(df: pd.DataFrame, grupo: str, min_n: int = 1) -> list[dict[str, Any]]:
    """n, mediana, media y cuartiles de precio por ``grupo`` (orden: mediana desc., nombre)."""
    g = df.groupby(grupo, observed=True)["price"]
    tabla = pd.DataFrame({"n": g.size(), "mediana": g.median(), "media": g.mean(),
                          "q1": g.quantile(0.25), "q3": g.quantile(0.75)})
    tabla = tabla[tabla["n"] >= min_n].reset_index().rename(columns={grupo: "grupo"})
    tabla["grupo"] = tabla["grupo"].astype(str)
    tabla = tabla.sort_values(["mediana", "grupo"], ascending=[False, True], kind="mergesort")
    return [{k: (int(v) if k == "n" else v) for k, v in fila.items()} for fila in tabla.to_dict("records")]


def resumen_por_ciudad(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Una fila por ciudad: oferta, precio, anfitriones, licencias, ocupación proxy y renta de la zona.

    Se construye con ``groupby().agg`` sobre todos los anuncios y sobre los que tienen precio.
    """
    todos = df.groupby("ciudad").agg(
        n_anuncios=("id", "size"), n_hosts=("host_id", "nunique"),
        pct_entero=("tipo_habitacion", lambda s: (s == "Alojamiento entero").mean() * 100),
        pct_multianuncio=("es_multianuncio", "mean"), pct_superhost=("host_is_superhost", "mean"),
        pct_sin_licencia=("licencia_tipo", lambda s: (s == "Sin licencia").mean() * 100),
        pct_registro_nacional=("licencia_registro_nacional", "mean"),
        pct_larga_estancia=("alquiler_larga_estancia", "mean"),
        no_disponible_media=("no_disponible_365", "mean"),
        ocupacion_ia_media=("estimated_occupancy_l365d", lambda s: (s / 365).clip(0, 1).mean() * 100),
        pct_emparejado_ine=(config.RENTA_PRINCIPAL, lambda s: s.notna().mean() * 100),
        renta_dis_mediana=(config.RENTA_PRINCIPAL, "median"),
        renta_mun_mediana=(config.RENTA_MUNICIPAL, "median"),
        n_distritos_ine=("codigo_distrito_ine", "nunique"), n_municipios_ine=("codigo_municipio_ine", "nunique"),
        capacidad_mediana=("accommodates", "median"),
    )
    for c in ("pct_multianuncio", "pct_superhost", "pct_registro_nacional", "pct_larga_estancia",
              "no_disponible_media"):
        todos[c] = todos[c].astype("float64") * 100
    cp = df[df["precio_valido"]].groupby("ciudad")["price"]
    precio = pd.DataFrame({"n": cp.size(), "mediana": cp.median(), "media": cp.mean(),
                           "q1": cp.quantile(0.25), "q3": cp.quantile(0.75)})
    tabla = todos.join(precio)
    tabla["pct_con_precio"] = tabla["n"] / tabla["n_anuncios"] * 100
    tabla = tabla.reset_index()
    tabla["grupo"] = nombre_ciudad(tabla["ciudad"])
    tabla["tipo"] = tabla["ciudad"].map(lambda c: config.CIUDADES_ESPANA[c].tipo)
    tabla = tabla.sort_values(["mediana", "grupo"], ascending=[False, True], kind="mergesort")
    filas = tabla.rename(columns={"ciudad": "clave"}).to_dict("records")
    for f in filas:
        for k in ("n", "n_anuncios", "n_hosts", "n_distritos_ine", "n_municipios_ine"):
            f[k] = int(f[k])
    return filas


def tabla_porcentajes(df: pd.DataFrame, fila: str, columna: str) -> dict[str, dict[str, float]]:
    """``pd.crosstab`` normalizado por filas (en %): {nombre de ciudad: {categoría: %}}."""
    t = pd.crosstab(nombre_ciudad(df[fila]) if fila == "ciudad" else df[fila], df[columna], normalize="index") * 100
    return {str(i): {str(c): float(v) for c, v in r.items()} for i, r in t.iterrows()}


def composicion_hosts(df: pd.DataFrame) -> dict[str, Any]:
    """Composición de anfitriones a nivel anuncio y a nivel anfitrión (concentración de la oferta)."""
    por_host = df.groupby("host_id").size()
    superhost_host = df.groupby("host_id")["host_is_superhost"].first()
    orden = por_host.sort_values(ascending=False, kind="mergesort")
    n_top1 = max(1, int(np.ceil(len(orden) * 0.01)))
    segmentos = df["segmento_host"].value_counts(sort=False)
    return {
        "n_anuncios": int(len(df)), "n_hosts": int(por_host.size),
        "anuncios_por_host_media": float(por_host.mean()), "anuncios_por_host_mediana": float(por_host.median()),
        "anuncios_por_host_max": int(por_host.max()),
        "pct_hosts_un_anuncio": float((por_host == 1).mean() * 100),
        "pct_anuncios_multianuncio": float(df["es_multianuncio"].mean() * 100),
        "pct_anuncios_superhost": float(df["host_is_superhost"].mean() * 100),
        "pct_hosts_superhost": float(superhost_host.mean() * 100),
        "pct_anuncios_hosts_6_o_mas": float((df["calculated_host_listings_count"] >= 6).mean() * 100),
        "pct_anuncios_top10_hosts": float(orden.head(10).sum() / len(df) * 100),
        "n_hosts_top1pct": n_top1,
        "pct_anuncios_top1pct_hosts": float(orden.head(n_top1).sum() / len(df) * 100),
        "segmentos": {str(k): int(v) for k, v in segmentos.items()},
        "segmentos_pct": {str(k): float(v / len(df) * 100) for k, v in segmentos.items()},
    }


def ocupacion(activos: pd.DataFrame) -> dict[str, Any]:
    """Días no disponibles en los próximos 365 (proxy propio) frente a la ocupación estimada por Inside Airbnb.

    Ventanas distintas: el proxy mira a los PRÓXIMOS 365 días (calendario, mezcla reservas y bloqueos) y la
    estimación de Inside Airbnb a los ÚLTIMOS 12 meses (modelo basado en reseñas). Misma máscara de filas.
    """
    nd = activos["no_disponible_365"]
    ocu = (activos["estimated_occupancy_l365d"] / 365).clip(0, 1)
    m = nd.notna() & ocu.notna()
    rho, p = stats.spearmanr(nd[m], ocu[m])
    return {
        "n_activos_con_precio": int(len(activos)), "n_comparacion": int(m.sum()),
        "no_disponible_media": float(nd[m].mean() * 100), "no_disponible_mediana": float(nd[m].median() * 100),
        "ocupacion_inside_airbnb_media": float(ocu[m].mean() * 100),
        "ocupacion_inside_airbnb_mediana": float(ocu[m].median() * 100),
        "spearman": float(rho), "spearman_p": float(p),
        "ventana_propia": "próximos 365 días (availability_365, calendario futuro)",
        "ventana_inside_airbnb": "últimos 12 meses (estimated_occupancy_l365d, modelo basado en reseñas)",
    }


def por_mes_cotizacion(cp: pd.DataFrame) -> dict[str, Any]:
    """Mediana del precio por mes de entrada de la cotización (temporada) y extremos con n ≥ ``MIN_N_GRUPO``."""
    filas = sorted(precio_por_grupo(cp, "mes_cotizacion"), key=lambda g: g["grupo"])
    validos = [g for g in filas if g["n"] >= config.MIN_N_GRUPO]
    total = sum(g["n"] for g in filas)
    return {
        "meses": filas, "n_meses": len(filas), "n_con_mes": total,
        "mes_mediana_min": min(validos, key=lambda g: g["mediana"]) if validos else None,
        "mes_mediana_max": max(validos, key=lambda g: g["mediana"]) if validos else None,
        "pct_primeros_dos_meses": float(sum(g["n"] for g in filas[:2]) / total * 100) if total else float("nan"),
    }


def mascara_sin_mensuales(df: pd.DataFrame) -> pd.Series:
    """True para anuncios con estancia mínima < 30 noches y cotización no mensual (tarifa de corta estancia)."""
    return ~(df["minimum_nights"].ge(config.MIN_NIGHTS_LARGA_ESTANCIA).fillna(False).astype(bool)
             | df["es_tarifa_mensual"].astype(bool))


def sensibilidad_sin_mensuales(cp: pd.DataFrame) -> dict[str, Any]:
    """Precio sin estancias mínimas ≥ 30 noches ni tarifas mensuales, global y por ciudad."""
    sub = cp[mascara_sin_mensuales(cp)]
    original = precio_por_grupo(cp.assign(nc=nombre_ciudad(cp["ciudad"])), "nc")
    nuevo = precio_por_grupo(sub.assign(nc=nombre_ciudad(sub["ciudad"])), "nc")
    return {
        "n": int(len(sub)), "n_excluidos": int(len(cp) - len(sub)),
        "pct_excluidos": float((len(cp) - len(sub)) / len(cp) * 100),
        "precio": descriptivos(sub["price"]), "por_ciudad": nuevo,
        "mismo_orden_ciudades": [g["grupo"] for g in nuevo] == [g["grupo"] for g in original],
        "misma_ciudad_mas_cara": bool(nuevo and original and nuevo[0]["grupo"] == original[0]["grupo"]),
    }


def amenities(con_precio: pd.DataFrame, todos: pd.DataFrame) -> list[dict[str, Any]]:
    """Prevalencia (todos los anuncios) y diferencia de la mediana de precio con/sin cada amenity."""
    filas = []
    for col in config.PATRONES_AMENITIES:
        con = con_precio.loc[con_precio[col], "price"]
        sin = con_precio.loc[~con_precio[col], "price"]
        med_con, med_sin = float(con.median()), float(sin.median())
        filas.append({"amenity": col, "nombre": config.NOMBRES_AMENITIES.get(col, col),
                      "prevalencia_pct": float(todos[col].mean() * 100), "mediana_con": med_con,
                      "mediana_sin": med_sin,
                      "prima_pct": float((med_con / med_sin - 1) * 100) if med_sin else float("nan")})
    return sorted(filas, key=lambda r: -r["prevalencia_pct"])


def correlaciones(con_precio: pd.DataFrame) -> dict[str, Any]:
    """Matrices de Pearson y Spearman (pairwise) y correlación de cada variable con el precio."""
    variables = [v for v in VARIABLES_CORRELACION if v in con_precio]
    datos = con_precio[variables].astype("float64")
    pearson = datos.corr(method="pearson")
    spearman = datos.corr(method="spearman")
    con_precio_rho = sorted(
        ({"variable": v, "nombre": ETIQUETAS_VARIABLES.get(v, v), "spearman": float(spearman.loc["price", v]),
          "pearson_log": float(pearson.loc["log_precio", v])}
         for v in variables if v not in ("price", "log_precio")),
        key=lambda r: -abs(r["spearman"]))
    return {"variables": variables, "pearson": pearson.round(6).to_dict(), "spearman": spearman.round(6).to_dict(),
            "con_precio": con_precio_rho, "con_precio_por_variable": {r["variable"]: r for r in con_precio_rho},
            "n": int(len(datos))}


def descriptivo_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Bloque descriptivo completo de ``findings`` (nacional y por ciudad)."""
    sub = obtener_subconjuntos(df)
    cp = sub["con_precio"]
    por_ciudad = resumen_por_ciudad(df)
    orden_cap = list(ETIQUETAS_CAPACIDAD)
    por_capacidad = sorted(precio_por_grupo(cp, "categoria_capacidad"), key=lambda g: orden_cap.index(g["grupo"]))
    zonas = precio_por_grupo(cp.assign(zona=nombre_ciudad(cp["ciudad"]) + " · " + cp["neighbourhood_cleansed"]),
                             "zona", min_n=config.MIN_N_ZONA_RANKING)
    return {
        "subconjuntos": {k: int(len(v)) for k, v in sub.items()},
        "n_ciudades": int(df["ciudad"].nunique()),
        "n_urbanas": int(es_urbana(df["ciudad"]).groupby(df["ciudad"]).first().sum()),
        "precio": descriptivos(cp["price"]),
        "log_precio": descriptivos(cp["log_precio"]),
        "accommodates": descriptivos(df["accommodates"]),
        "minimum_nights": descriptivos(df["minimum_nights"]),
        "review_scores_rating": descriptivos(df["review_scores_rating"]),
        "distancia_centro_km": descriptivos(df["distancia_centro_km"]),
        "renta_dis": descriptivos(df[config.RENTA_PRINCIPAL]),
        "por_ciudad": por_ciudad,
        "por_ciudad_clave": {f["clave"]: f for f in por_ciudad},
        "ciudad_mas_cara": por_ciudad[0], "ciudad_mas_barata": por_ciudad[-1],
        "por_tipo": precio_por_grupo(cp, "tipo_habitacion"),
        "por_tipo_agrupado": precio_por_grupo(cp, "tipo_habitacion_agrupado"),
        "anuncios_por_tipo": {str(k): int(v) for k, v in df["tipo_habitacion"].value_counts().items()},
        "pct_entero": float((df["tipo_habitacion"] == "Alojamiento entero").mean() * 100),
        "tipo_por_ciudad_pct": tabla_porcentajes(df, "ciudad", "tipo_habitacion_agrupado"),
        "licencia_por_ciudad_pct": tabla_porcentajes(df, "ciudad", "licencia_tipo"),
        "por_capacidad": por_capacidad,
        "por_licencia": precio_por_grupo(cp, "licencia_tipo"),
        "pct_sin_licencia": float((df["licencia_tipo"] == "Sin licencia").mean() * 100),
        "pct_registro_nacional": float(df["licencia_registro_nacional"].mean() * 100),
        "zonas_top": zonas[:10], "zonas_bottom": zonas[-10:][::-1], "n_zonas_ranking": len(zonas),
        "hosts": composicion_hosts(df),
        "ocupacion": ocupacion(sub["activos_con_precio"]),
        "pct_activos_probables": float(df["es_activo_probable"].mean() * 100),
        "amenities": amenities(cp, df),
        "correlaciones": correlaciones(cp),
        "mes_cotizacion": por_mes_cotizacion(cp),
        "sin_mensuales": sensibilidad_sin_mensuales(cp),
    }


# --------------------------------------------------------------------------
# Relación renta-precio
# --------------------------------------------------------------------------
def correlacion_ecologica(cp: pd.DataFrame, columna_renta: str, clave_zona: str) -> dict[str, Any]:
    """ρ de Spearman entre la renta de cada zona y la MEDIANA del precio de sus anuncios (zonas con n ≥ 30).

    Es un análisis a nivel de zona (ecológico): cada zona cuenta una vez. No permite inferir nada del
    anuncio individual (falacia ecológica), pero no infla la n con anuncios que comparten la misma renta.
    """
    g = cp.groupby(["ciudad", clave_zona]).agg(n=("price", "size"), mediana=("price", "median"),
                                              renta=(columna_renta, "first")).dropna()
    g = g[g["n"] >= config.MIN_N_GRUPO]
    if len(g) < config.MIN_DISTRITOS_ECOLOGICO or g["renta"].nunique() < 2:
        return {"n_zonas": int(len(g)), "evaluable": False}
    r = stats.spearmanr(g["renta"], g["mediana"])
    return {"n_zonas": int(len(g)), "evaluable": True, "rho": float(r.statistic), "p": float(r.pvalue),
            "significativo": bool(r.pvalue < config.ALFA)}


def renta_precio(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """ρ de Spearman renta de la zona (INE) - precio: global y por ciudad, con IC bootstrap por clusters.

    - Anuncio a anuncio con la renta del distrito censal (``ine_dis_*``) y del municipio (``ine_mun_*``).
      IC remuestreando distritos censales (o municipios) y, como contraste, anfitriones.
    - Por ciudad, con Holm sobre los p bootstrap (familia de 9 ciudades).
    - A nivel de zona (ecológico): distritos censales con al menos 30 anuncios con precio.
    - Sensibilidades del ρ global: sin estancias ≥ 30 noches, sin outliers, un anuncio por anfitrión,
      solo ciudades urbanas.
    """
    b = min(n_boot, config.N_BOOT_CORRELACION)
    cp = df[df["precio_valido"]]
    renta_d, renta_m = config.RENTA_PRINCIPAL, config.RENTA_MUNICIPAL

    def _rho(d: pd.DataFrame, columna: str, cluster: str, con_host: bool = True) -> dict[str, Any]:
        clusters = {"cluster_zona": d[cluster].astype("string")}
        if con_host:
            clusters["cluster_anfitrion"] = d["host_id"].astype("string")
        return est.spearman_cluster(d[columna], d["price"], clusters, b, rng)

    logger.info("Spearman renta-precio global (%d réplicas por clusters)…", b)
    salida: dict[str, Any] = {"n_boot": b, "hipotesis": dict(zip(("h0", "h1"), est.HIPOTESIS["spearman_renta"])),
                              "variable_renta": renta_d, "variable_renta_municipal": renta_m}
    salida["global_distrito"] = _rho(cp, renta_d, "codigo_distrito_ine")
    salida["global_municipio"] = _rho(cp, renta_m, "codigo_municipio_ine")

    por_ciudad = []
    for clave, s in cp.groupby("ciudad", sort=True):
        fila: dict[str, Any] = {"clave": str(clave), "ciudad": NOMBRE_CIUDAD[str(clave)],
                                "tipo": config.CIUDADES_ESPANA[str(clave)].tipo,
                                "distrito": _rho(s, renta_d, "codigo_distrito_ine"),
                                "ecologico": correlacion_ecologica(s, renta_d, "codigo_distrito_ine")}
        if config.CIUDADES_ESPANA[str(clave)].tipo == config.TIPO_REGION:
            # Solo en las regiones hay varios municipios; en las ciudades la renta municipal es constante.
            fila["municipio"] = _rho(s, renta_m, "codigo_municipio_ine", con_host=False)
        por_ciudad.append(fila)
    ps = {f["clave"]: f["distrito"]["cluster_zona"]["p_boot"] for f in por_ciudad if f["distrito"].get("evaluable")}
    ajuste = est.holm(ps)
    for f in por_ciudad:
        if f["clave"] in ajuste:
            f["distrito"]["holm"] = ajuste[f["clave"]]
    evaluables = [f for f in por_ciudad if f["distrito"].get("evaluable")]
    salida["por_ciudad"] = sorted(por_ciudad, key=lambda f: (-f["distrito"]["rho"] if f["distrito"].get("evaluable")
                                                             else np.inf, f["ciudad"]))
    salida["por_ciudad_clave"] = {f["clave"]: f for f in por_ciudad}
    salida["n_ciudades_evaluables"] = len(evaluables)
    salida["n_ciudades_positivas_holm"] = int(sum(
        f["distrito"]["rho"] > 0 and f["distrito"].get("holm", {}).get("significativo_holm", False)
        for f in evaluables))
    salida["n_ciudades_negativas_holm"] = int(sum(
        f["distrito"]["rho"] < 0 and f["distrito"].get("holm", {}).get("significativo_holm", False)
        for f in evaluables))
    salida["rho_ciudad_max"] = max(evaluables, key=lambda f: f["distrito"]["rho"]) if evaluables else None
    salida["rho_ciudad_min"] = min(evaluables, key=lambda f: f["distrito"]["rho"]) if evaluables else None
    salida["ecologico_global"] = correlacion_ecologica(cp, renta_d, "codigo_distrito_ine")
    salida["ecologico_municipio"] = correlacion_ecologica(cp, renta_m, "codigo_municipio_ine")

    logger.info("Sensibilidades del ρ renta-precio…")
    sens = {
        "sin_estancias_30": cp[mascara_sin_mensuales(cp)],
        "sin_outliers": cp[~cp["es_outlier_precio"]],
        "un_anuncio_por_anfitrion": est.un_anuncio_por_host(cp, rng),
        "solo_urbanas": cp[es_urbana(cp["ciudad"])],
    }
    salida["sensibilidades"] = {
        k: {"n": int(len(d)), **{kk: vv for kk, vv in _rho(d, renta_d, "codigo_distrito_ine", con_host=False).items()}}
        for k, d in sens.items()}
    salida["sensibilidades_mismo_signo"] = all(
        np.sign(v.get("rho", 0)) == np.sign(salida["global_distrito"]["rho"]) for v in salida["sensibilidades"].values())
    return salida


def _chi2_ciudad_tipo(df: pd.DataFrame) -> est.ResultadoTest:
    """Chi² ciudad × tipo de habitación (agrupado)."""
    return est.chi_cuadrado(df.assign(nc=nombre_ciudad(df["ciudad"])), "nc", "tipo_habitacion_agrupado",
                            "chi2_ciudad_tipo", "ciudad y tipo de habitación")


def ejecutar(df: pd.DataFrame, semilla: int, n_boot: int, con_ml: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ejecuta descriptivos y toda la estadística. Devuelve (hallazgos serializables, artefactos para figuras)."""
    rng = np.random.default_rng(semilla)
    descriptivo = descriptivo_dataset(df)
    cp = df[df["precio_valido"]].assign(nombre_ciudad=lambda d: nombre_ciudad(d["ciudad"]))
    r: dict[str, Any] = {}

    logger.info("Normalidad…")
    r["normalidad_precio"] = est.test_normalidad(cp["price"], "el precio", rng).a_dict()
    r["normalidad_log_precio"] = est.test_normalidad(cp["log_precio"], "el logaritmo del precio", rng).a_dict()

    logger.info("Mann-Whitney y Kruskal-Wallis…")
    sh = cp["host_is_superhost"]
    r["mann_whitney_superhost"] = est.mann_whitney(cp.loc[sh, "price"], cp.loc[~sh, "price"], "superhosts",
                                                   "no superhosts", "mann_whitney_superhost").a_dict()
    agrup = {"agrupacion": {"columna": "tipo_habitacion_agrupado",
                            "descripcion": f"categorías con menos de {config.MIN_N_CATEGORIA_RARA} anuncios con precio "
                                           f"agrupadas en «{config.ETIQUETA_OTRAS_TIPO}»"}}
    r["kruskal_tipo"] = est.kruskal(cp, "price", "tipo_habitacion_agrupado", "kruskal_tipo",
                                    "tipos de habitación", extra=agrup).a_dict()
    r["kruskal_ciudad"] = est.kruskal(cp, "price", "nombre_ciudad", "kruskal_ciudad", "ciudades").a_dict()
    r["tipos_solo_descriptivos"] = [g for g in descriptivo["por_tipo"]
                                    if g["grupo"] in ("Habitación compartida", "Habitación de hotel")]

    logger.info("Post-hoc de Dunn…")
    tipos = list(r["kruskal_tipo"]["n"].keys())
    ciudades = list(r["kruskal_ciudad"]["n"].keys())
    r["posthoc_tipo"] = est.posthoc(cp, "price", "tipo_habitacion_agrupado", tipos)
    r["posthoc_ciudad"] = est.posthoc(cp, "price", "nombre_ciudad", ciudades)

    logger.info("Chi-cuadrado…")
    r["chi2_tipo_superhost"] = est.chi_cuadrado(df, "tipo_habitacion_agrupado", "host_is_superhost",
                                                "chi2_tipo_superhost", "tipo de habitación y ser superhost").a_dict()
    r["chi2_ciudad_tipo"] = _chi2_ciudad_tipo(df).a_dict()
    r["chi2_multi_tipo"] = est.chi_cuadrado(df, "es_multianuncio", "tipo_habitacion_agrupado", "chi2_multi_tipo",
                                            "ser multi-anuncio y tipo de habitación").a_dict()

    logger.info("Bootstrap de medianas (%d remuestreos, por anuncio y por anfitrión)…", n_boot)

    def _boot(mascara: pd.Series, grupo: str) -> dict[str, Any]:
        x = cp.loc[mascara, "price"].to_numpy()
        simple = est.bootstrap_mediana(x, n_boot, rng)
        clus = est.bootstrap_mediana_cluster(x, cp.loc[mascara, "host_id"].to_numpy(), n_boot, rng)
        return {"grupo": grupo, **simple, "n_anfitriones": clus["n_anfitriones"],
                "ic_inf_cluster": clus["ic_inf"], "ic_sup_cluster": clus["ic_sup"]}

    boot_tipo = [_boot(cp["tipo_habitacion_agrupado"] == g, g) for g in tipos]
    boot_tipo.sort(key=lambda x: (-x["mediana"], x["grupo"]))
    boot_ciudad = [_boot(cp["nombre_ciudad"] == c, c) for c in ciudades]
    boot_ciudad.sort(key=lambda x: (-x["mediana"], x["grupo"]))
    r["bootstrap"] = {"n_boot": n_boot, "metodo": "percentil, IC 95 %; por anuncio y por anfitrión (cluster)",
                      "tipo": boot_tipo, "ciudades": boot_ciudad,
                      "top_ciudad_distinguible": boot_ciudad[0]["ic_inf_cluster"] > boot_ciudad[1]["ic_sup_cluster"]}

    r["renta_precio"] = renta_precio(df, n_boot, rng)

    logger.info("Sensibilidad de los contrastes: un anuncio por anfitrión…")
    uno = est.un_anuncio_por_host(cp, rng)
    uno_todos = est.un_anuncio_por_host(df, rng)
    sh1 = uno["host_is_superhost"]
    mw1 = est.mann_whitney(uno.loc[sh1, "price"], uno.loc[~sh1, "price"], "superhosts", "no superhosts",
                           "mann_whitney_superhost").a_dict()
    kt1 = est.kruskal(uno, "price", "tipo_habitacion_agrupado", "kruskal_tipo", "tipos de habitación").a_dict()
    kc1 = est.kruskal(uno, "price", "nombre_ciudad", "kruskal_ciudad", "ciudades").a_dict()
    chi1 = {
        "chi2_tipo_superhost": est.chi_cuadrado(uno_todos, "tipo_habitacion_agrupado", "host_is_superhost",
                                                "chi2_tipo_superhost", "tipo de habitación y ser superhost").a_dict(),
        "chi2_ciudad_tipo": _chi2_ciudad_tipo(uno_todos).a_dict(),
        "chi2_multi_tipo": est.chi_cuadrado(uno_todos, "es_multianuncio", "tipo_habitacion_agrupado",
                                            "chi2_multi_tipo", "ser multi-anuncio y tipo de habitación").a_dict(),
    }
    decisiones = {
        "superhost": mw1["significativo"] == r["mann_whitney_superhost"]["significativo"],
        "tipo": kt1["significativo"] == r["kruskal_tipo"]["significativo"],
        "ciudad": kc1["significativo"] == r["kruskal_ciudad"]["significativo"],
        **{k: v["significativo"] == r[k]["significativo"] for k, v in chi1.items()},
    }
    r["sensibilidad_un_anuncio_por_host"] = {
        "n": int(len(uno)), "n_todos": int(len(uno_todos)), "mann_whitney_superhost": mw1, "kruskal_tipo": kt1,
        "kruskal_ciudad": kc1, **chi1,
        "mismo_signo_superhost": bool(np.sign(mw1["tamano_efecto"]["valor"])
                                      == np.sign(r["mann_whitney_superhost"]["tamano_efecto"]["valor"])),
        "decisiones": decisiones, "todas_mismas_decisiones": all(decisiones.values()),
    }

    logger.info("Regresión OLS y especificaciones alternativas…")
    ols = est.ajustar_ols(df)
    artefactos = {"residuos": ols.pop("_residuos"), "ajustados": ols.pop("_ajustados")}
    filas_ols = ols.pop("_indices")
    en_ols = pd.Series(df.index.isin(filas_ols), index=df.index)
    sin_ine = est.ajustar_ols(df, con_ine=False, diagnostico=False, mascara_extra=en_ols)
    ols["sin_ine"] = {"r2_ajustado": sin_ine["r2_ajustado"], "r2": sin_ine["r2"], "n_efectivo": sin_ine["n_efectivo"],
                      "aic": sin_ine["aic"]}
    ols["delta_r2_ajustado_ine"] = ols["r2_ajustado"] - sin_ine["r2_ajustado"]
    ols["delta_r2_ajustado_ine_pp"] = 100 * ols["delta_r2_ajustado_ine"]
    solo_ine = est.ajustar_ols(df, diagnostico=False, mascara_extra=en_ols, excluir=est.VARIABLES_OLS_NUMERICAS,
                               con_mes=False)
    ols["solo_ciudad_tipo_licencia_ine"] = {"r2_ajustado": solo_ine["r2_ajustado"], "n_efectivo": solo_ine["n_efectivo"]}
    sens = {
        "sin_outliers": est.ajustar_ols(df, diagnostico=False, mascara_extra=~df["es_outlier_precio"]),
        "sin_estancias_30": est.ajustar_ols(df, diagnostico=False, mascara_extra=mascara_sin_mensuales(df),
                                            excluir=("alquiler_larga_estancia",)),
        "un_anuncio_por_anfitrion": est.ajustar_ols(
            df, diagnostico=False,
            mascara_extra=pd.Series(df.index.isin(est.un_anuncio_por_host(df[df["precio_valido"]], rng).index),
                                    index=df.index)),
        "solo_urbanas_con_distancia": est.ajustar_ols(df, diagnostico=False, con_distancia=True,
                                                      mascara_extra=es_urbana(df["ciudad"]) & df["coords_validas"]),
        "sin_mes": est.ajustar_ols(df, diagnostico=False, con_mes=False),
    }
    ols["sensibilidades"] = {k: est.resumen_sensibilidad(ols, v) for k, v in sens.items()}
    urb = sens["solo_urbanas_con_distancia"]["coef_por_termino"]
    ols["sensibilidades"]["solo_urbanas_con_distancia"]["distancia"] = {
        k: urb["distancia_centro_km"][k] for k in ("pct", "ic_inf_pct", "ic_sup_pct", "p_cluster", "significativo")}
    ols["sensibilidades_renta_mismo_signo"] = all(
        np.sign(v["renta"]["beta"]) == np.sign(ols["coef_por_termino"]["ln_renta_dis"]["beta"])
        for v in ols["sensibilidades"].values() if v["renta"])
    ols["sensibilidad_con_outliers_signos"] = est.comparar_signos(ols, sens["sin_outliers"])
    pt = ols["coef_por_termino"]
    ols["coef_renta"] = pt["ln_renta_dis"]
    ols["coef_superhost"] = pt["host_is_superhost"]
    ols["coef_accommodates"] = pt["accommodates"]
    ols["coef_multianuncio"] = pt["es_multianuncio"]
    ols["coef_larga_estancia"] = pt["alquiler_larga_estancia"]
    ols["coef_privada"] = next(c for c in ols["coeficientes"] if c["nombre"] == "Tipo: Habitación privada")
    ols["coef_ciudades"] = sorted((c for c in ols["coeficientes"] if c["es_ciudad"]), key=lambda c: -c["pct"])
    ols["r2_ajustado_pct"] = ols["r2_ajustado"] * 100
    variables = [c for c in ols["coeficientes"] if not c["es_ciudad"] and not c["es_mes"]]
    ols["n_coef_significativos"] = int(sum(c["significativo"] for c in variables))
    ols["n_coef_variables"] = len(variables)
    ols["n_coef_ciudad"] = int(sum(c["es_ciudad"] for c in ols["coeficientes"]))
    ols["n_coef_mes"] = int(sum(c["es_mes"] for c in ols["coeficientes"]))
    ols["n_ine_significativos_distrito"] = int(sum(c.get("significativo_distrito", False)
                                                   for c in ols["coeficientes"] if c["es_ine"]))
    r["ols"] = ols

    # Multiplicidad: nº de contrastes y efecto de Holm sobre las familias (OLS y ρ por ciudad).
    omnibus = ["mann_whitney_superhost", "kruskal_tipo", "kruskal_ciudad", "chi2_tipo_superhost",
               "chi2_ciudad_tipo", "chi2_multi_tipo"]
    r["multiplicidad"] = {
        "n_contrastes_principales": len(omnibus),
        "n_principales_p_menor_0001": int(sum(r[k]["p_valor"] < 0.001 for k in omnibus)),
        "n_pares_posthoc": r["posthoc_tipo"]["n_pares"] + r["posthoc_ciudad"]["n_pares"],
        "n_coeficientes_ols": ols["n_coeficientes"], "n_significativos_ols": ols["n_significativos"],
        "n_significativos_ols_holm": ols["n_significativos_holm"],
        "n_rho_ciudad": r["renta_precio"]["n_ciudades_evaluables"],
        "n_hipotesis_total": len(omnibus) + r["posthoc_tipo"]["n_pares"] + r["posthoc_ciudad"]["n_pares"]
        + ols["n_coeficientes"] + r["renta_precio"]["n_ciudades_evaluables"],
    }

    if con_ml:
        logger.info("Modelo ML (validación agrupada por anfitrión)…")
        r["ml"] = est.ajustar_modelo_ml(df, semilla, filas_ols=filas_ols)
    else:
        r["ml"] = {"ejecutado": False, "motivo": "omitido con --sin-ml"}
    r["hipotesis"] = {k: {"h0": v[0], "h1": v[1]} for k, v in est.HIPOTESIS.items()}
    r["umbrales_efecto"] = {"rank_biserial": list(est.UMBRALES_RB), "epsilon2": list(est.UMBRALES_EPS2),
                            "etiquetas": list(est.ETIQUETAS_EFECTO)}
    return {"descriptivo": descriptivo, "estadistica": r}, artefactos
