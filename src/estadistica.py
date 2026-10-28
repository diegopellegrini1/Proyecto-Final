"""Contrastes estadísticos, tamaños de efecto, bootstrap, regresión OLS y modelo ML opcional.

Todos los contrastes usan α = 0,05, declaran H0/H1 en ``HIPOTESIS`` antes de ejecutarse,
comprueban supuestos y devuelven un ``ResultadoTest`` serializable. Las conclusiones se
redactan por código y nunca usan lenguaje causal.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src import formato as fmt
from src.errores import ErrorDatos

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Hipótesis (definidas antes de ejecutar cada test)
# --------------------------------------------------------------------------
HIPOTESIS: dict[str, tuple[str, str]] = {
    "normalidad": ("La variable procede de una distribución normal.",
                   "La variable no procede de una distribución normal."),
    "mann_whitney_superhost": (
        "El precio de los anuncios de superhosts y de no superhosts tiene la misma distribución "
        "(P(X_superhost > X_no) = 0,5).",
        "Las distribuciones difieren (contraste bilateral)."),
    "kruskal_tipo": ("La distribución del precio es la misma en todos los tipos de habitación.",
                     "Al menos un tipo de habitación tiene una distribución de precio distinta."),
    "kruskal_ciudad": ("La distribución del precio es la misma en todas las ciudades y regiones.",
                       "Al menos una ciudad o región tiene una distribución de precio distinta."),
    "posthoc": ("Para cada par de grupos, no hay diferencia en la distribución del precio.",
                "El par de grupos difiere (p ajustado por Bonferroni < α)."),
    "chi2_tipo_superhost": ("El tipo de habitación y ser superhost son independientes.",
                            "Existe asociación entre tipo de habitación y ser superhost."),
    "chi2_ciudad_tipo": ("La ciudad y el tipo de habitación son independientes.",
                         "Existe asociación entre ciudad y tipo de habitación."),
    "chi2_multi_tipo": ("Ser anfitrión multi-anuncio y el tipo de habitación son independientes.",
                        "Existe asociación entre ser multi-anuncio y el tipo de habitación."),
    "spearman_renta": ("ρ de Spearman entre la renta neta por persona de la zona (INE) y el precio es 0.",
                       "ρ ≠ 0 (contraste bilateral; IC y p por bootstrap remuestreando distritos censales)."),
    "ols": ("Cada coeficiente β es 0 (la variable no se asocia con ln(precio) controlando por el resto).",
            "β ≠ 0."),
}

# Umbrales de interpretación de tamaños de efecto.
UMBRALES_RB: tuple[float, float, float] = (0.1, 0.3, 0.5)
UMBRALES_EPS2: tuple[float, float, float] = (0.01, 0.06, 0.14)
UMBRALES_V: dict[int, tuple[float, float, float]] = {
    1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29), 4: (0.05, 0.15, 0.25),
}
ETIQUETAS_EFECTO: tuple[str, str, str, str] = ("despreciable", "pequeño", "mediano", "grande")


def interpretar(valor: float, umbrales: tuple[float, float, float]) -> str:
    """Etiqueta de magnitud según tres umbrales crecientes (sobre |valor|)."""
    v = abs(valor)
    for umbral, etiqueta in zip(umbrales, ETIQUETAS_EFECTO):
        if v < umbral:
            return etiqueta
    return ETIQUETAS_EFECTO[-1]


def interpretar_v(v: float, df_star: int) -> str:
    """Interpreta la V de Cramér según los umbrales de Cohen para ``df* = min(filas, columnas) − 1``."""
    return interpretar(v, UMBRALES_V[min(max(df_star, 1), 4)])


def redactar_conclusion(p: float, alfa: float, efecto: str, si_signif: str, si_no_signif: str) -> str:
    """Conclusión en lenguaje claro que distingue significación estadística y relevancia práctica."""
    if p >= alfa:
        return f"No se rechaza H0 (p ≥ α). {si_no_signif}"
    if efecto == "despreciable":
        return (f"Se rechaza H0 (p < α): resultado estadísticamente significativo pero con efecto "
                f"despreciable, sin relevancia práctica. {si_signif}")
    return f"Se rechaza H0 (p < α) con un efecto {efecto}. {si_signif}"


@dataclass(frozen=True)
class ResultadoTest:
    """Resultado homogéneo de un contraste."""

    nombre: str
    h0: str
    h1: str
    alfa: float
    supuestos: dict[str, Any]
    estadistico: float
    p_valor: float
    n: Any
    tamano_efecto: dict[str, Any]
    conclusion: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def significativo(self) -> bool:
        """True si p < α."""
        return bool(self.p_valor < self.alfa)

    def a_dict(self) -> dict[str, Any]:
        """Diccionario serializable (incluye ``significativo``)."""
        d = asdict(self)
        d["significativo"] = self.significativo
        return d


# --------------------------------------------------------------------------
# Normalidad
# --------------------------------------------------------------------------
def test_normalidad(valores: pd.Series, etiqueta: str, rng: np.random.Generator,
                    n_shapiro: int = config.N_SHAPIRO, alfa: float = config.ALFA) -> ResultadoTest:
    """Shapiro-Wilk sobre una muestra (n acotado) y D'Agostino-K² sobre todos los datos."""
    x = valores.dropna().to_numpy(dtype="float64")
    muestra = rng.choice(x, size=min(n_shapiro, len(x)), replace=False)
    w, p_sw = stats.shapiro(muestra)
    k2, p_k2 = stats.normaltest(x)
    asim = float(stats.skew(x))
    curt = float(stats.kurtosis(x))
    efecto = "despreciable" if abs(asim) < 0.5 and abs(curt) < 1 else "grande"
    conclusion = redactar_conclusion(
        p_k2, alfa, efecto,
        f"{etiqueta[:1].upper()}{etiqueta[1:]} no es normal (asimetría {fmt.n2(asim)}, curtosis de exceso "
        f"{fmt.n2(curt)}). Con n tan grande el test "
        "rechaza incluso desviaciones leves; la decisión de usar contrastes no paramétricos se apoya también en la "
        "asimetría y en los gráficos Q-Q.",
        f"No hay evidencia contra la normalidad de {etiqueta}.",
    )
    return ResultadoTest(
        nombre=f"Normalidad de {etiqueta} (Shapiro-Wilk en muestra + D'Agostino-K²)",
        h0=HIPOTESIS["normalidad"][0], h1=HIPOTESIS["normalidad"][1], alfa=alfa,
        supuestos={"n_total": len(x), "n_muestra_shapiro": len(muestra), "semilla_muestra": "rng del pipeline"},
        estadistico=float(k2), p_valor=float(p_k2), n=len(x),
        tamano_efecto={"nombre": "asimetría y curtosis", "valor": asim, "curtosis_exceso": curt,
                       "interpretacion": efecto},
        conclusion=conclusion,
        extra={"shapiro_w": float(w), "shapiro_p": float(p_sw), "dagostino_k2": float(k2), "dagostino_p": float(p_k2)},
    )


# --------------------------------------------------------------------------
# Mann-Whitney
# --------------------------------------------------------------------------
def efecto_mann_whitney(u1: float, n1: int, n2: int) -> tuple[float, float]:
    """Devuelve (rank-biserial r = 2·U₁/(n₁n₂) − 1, CLES = U₁/(n₁n₂))."""
    cles = u1 / (n1 * n2)
    return 2 * cles - 1, cles


def mann_whitney(a: pd.Series, b: pd.Series, etiqueta_a: str, etiqueta_b: str, clave_hipotesis: str,
                 alfa: float = config.ALFA) -> ResultadoTest:
    """Mann-Whitney U bilateral entre dos grupos con rank-biserial y CLES."""
    xa = a.dropna().to_numpy(dtype="float64")
    xb = b.dropna().to_numpy(dtype="float64")
    if len(xa) < 2 or len(xb) < 2:
        raise ErrorDatos(f"Mann-Whitney necesita al menos 2 observaciones por grupo ({etiqueta_a}/{etiqueta_b}).")
    u1, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    r, cles = efecto_mann_whitney(float(u1), len(xa), len(xb))
    efecto = interpretar(r, UMBRALES_RB)
    med_a, med_b = float(np.median(xa)), float(np.median(xb))
    mayor = etiqueta_a if med_a > med_b else etiqueta_b
    conclusion = redactar_conclusion(
        p, alfa, efecto,
        f"Los anuncios de {mayor} tienden a tener precios más altos (mediana de {etiqueta_a}: {fmt.eur2(med_a)}; "
        f"de {etiqueta_b}: {fmt.eur2(med_b)}). "
        "Es una asociación bruta, sin controlar por ubicación ni tipo.",
        "No se observa diferencia en la distribución del precio entre ambos grupos.",
    )
    h0, h1 = HIPOTESIS[clave_hipotesis]
    return ResultadoTest(
        nombre=f"Mann-Whitney U: {etiqueta_a} vs {etiqueta_b}", h0=h0, h1=h1, alfa=alfa,
        supuestos={"datos_ordinales": True,
                   "independencia": "parcial: un anfitrión puede tener varios anuncios (ver análisis de sensibilidad)",
                   "nota": "compara dominancia estocástica, no solo medianas, si las formas difieren"},
        estadistico=float(u1), p_valor=float(p), n={etiqueta_a: len(xa), etiqueta_b: len(xb)},
        tamano_efecto={"nombre": "correlación rank-biserial", "valor": float(r), "cles": float(cles),
                       "interpretacion": efecto},
        conclusion=conclusion,
        extra={"mediana_a": med_a, "mediana_b": med_b, "etiqueta_a": etiqueta_a, "etiqueta_b": etiqueta_b,
               "grupo_mas_caro": mayor},
    )


# --------------------------------------------------------------------------
# Kruskal-Wallis y post-hoc
# --------------------------------------------------------------------------
def epsilon_cuadrado(h: float, n: int) -> float:
    """ε² de Kruskal-Wallis: H·(n+1)/(n²−1)."""
    return float(h * (n + 1) / (n ** 2 - 1))


def grupos_validos(df: pd.DataFrame, valor: str, grupo: str, min_n: int = config.MIN_N_GRUPO
                   ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Separa ``valor`` por ``grupo`` (orden alfabético) y excluye grupos con menos de ``min_n``."""
    incluidos: dict[str, np.ndarray] = {}
    excluidos: dict[str, int] = {}
    for nombre, sub in sorted(df.groupby(grupo, observed=True)[valor], key=lambda t: str(t[0])):
        x = sub.dropna().to_numpy(dtype="float64")
        if len(x) >= min_n:
            incluidos[str(nombre)] = x
        else:
            excluidos[str(nombre)] = len(x)
    return incluidos, excluidos


def kruskal(df: pd.DataFrame, valor: str, grupo: str, clave_hipotesis: str, etiqueta_grupo: str,
            alfa: float = config.ALFA, extra: dict[str, Any] | None = None) -> ResultadoTest:
    """Kruskal-Wallis con ε² sobre grupos con n ≥ ``MIN_N_GRUPO``."""
    incluidos, excluidos = grupos_validos(df, valor, grupo)
    if len(incluidos) < 2:
        raise ErrorDatos(f"Kruskal-Wallis necesita al menos 2 grupos con n ≥ {config.MIN_N_GRUPO}.")
    h, p = stats.kruskal(*incluidos.values())
    n = sum(len(v) for v in incluidos.values())
    eps2 = epsilon_cuadrado(float(h), n)
    efecto = interpretar(eps2, UMBRALES_EPS2)
    medianas = {k: float(np.median(v)) for k, v in incluidos.items()}
    mas_caro = max(medianas, key=medianas.get)
    mas_barato = min(medianas, key=medianas.get)
    conclusion = redactar_conclusion(
        p, alfa, efecto,
        f"El precio difiere entre {etiqueta_grupo}: la mediana más alta es la de {mas_caro} y la más baja la de "
        f"{mas_barato}.",
        f"No hay evidencia de diferencias de precio entre {etiqueta_grupo}.",
    )
    h0, h1 = HIPOTESIS[clave_hipotesis]
    return ResultadoTest(
        nombre=f"Kruskal-Wallis: precio por {etiqueta_grupo}", h0=h0, h1=h1, alfa=alfa,
        supuestos={"independencia": "parcial (anfitriones con varios anuncios)",
                   "min_n_por_grupo": config.MIN_N_GRUPO, "n_grupos": len(incluidos)},
        estadistico=float(h), p_valor=float(p), n={k: len(v) for k, v in incluidos.items()},
        tamano_efecto={"nombre": "épsilon cuadrado", "valor": eps2, "interpretacion": efecto},
        conclusion=conclusion,
        extra={"medianas": medianas, "grupo_mas_caro": mas_caro, "grupo_mas_barato": mas_barato,
               "grupos_excluidos": excluidos, "n_total": n, **(extra or {})},
    )


def posthoc(df: pd.DataFrame, valor: str, grupo: str, grupos: list[str], alfa: float = config.ALFA) -> dict[str, Any]:
    """Post-hoc de Dunn con Bonferroni (plan B: pares Mann-Whitney + Bonferroni propio)."""
    sub = df[df[grupo].astype("string").isin(grupos)][[valor, grupo]].dropna().copy()
    sub[grupo] = sub[grupo].astype(str)
    pares = list(combinations(sorted(grupos), 2))
    try:
        import scikit_posthocs as sp
        matriz = sp.posthoc_dunn(sub, val_col=valor, group_col=grupo, p_adjust="bonferroni")
        metodo = "Dunn con corrección de Bonferroni (scikit-posthocs)"
        p_par = {(a, b): float(matriz.loc[a, b]) for a, b in pares}
    except Exception as exc:  # noqa: BLE001 - plan B documentado
        logger.warning("scikit-posthocs falló (%s); se usa Mann-Whitney por pares + Bonferroni.", exc)
        metodo = "pares Mann-Whitney + Bonferroni"
        p_par = {}
        for a, b in pares:
            _, p = stats.mannwhitneyu(sub.loc[sub[grupo] == a, valor], sub.loc[sub[grupo] == b, valor])
            p_par[(a, b)] = float(min(p * len(pares), 1.0))
    resultados = []
    for a, b in pares:
        xa = sub.loc[sub[grupo] == a, valor].to_numpy()
        xb = sub.loc[sub[grupo] == b, valor].to_numpy()
        u1, _ = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        r, _ = efecto_mann_whitney(float(u1), len(xa), len(xb))
        resultados.append({"grupo_a": a, "grupo_b": b, "p_ajustado": p_par[(a, b)],
                           "significativo": p_par[(a, b)] < alfa,
                           "dif_medianas": float(np.median(xa) - np.median(xb)), "rank_biserial": float(r)})
    matriz_p = {a: {b: (1.0 if a == b else p_par.get((a, b), p_par.get((b, a)))) for b in sorted(grupos)}
                for a in sorted(grupos)}
    return {"metodo": metodo, "h0": HIPOTESIS["posthoc"][0], "h1": HIPOTESIS["posthoc"][1], "alfa": alfa,
            "n_pares": len(pares), "n_pares_significativos": int(sum(r["significativo"] for r in resultados)),
            "pares": resultados, "matriz_p": matriz_p, "grupos": sorted(grupos)}


# --------------------------------------------------------------------------
# Chi-cuadrado
# --------------------------------------------------------------------------
def cramer_v(tabla: np.ndarray) -> tuple[float, float]:
    """Devuelve (V de Cramér, V corregida de sesgo de Bergsma) de una tabla de contingencia."""
    tabla = np.asarray(tabla, dtype="float64")
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    v = np.sqrt(phi2 / min(r - 1, k - 1))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    vc = np.sqrt(phi2c / min(rc - 1, kc - 1)) if min(rc - 1, kc - 1) > 0 else float("nan")
    return float(v), float(vc)


def chi_cuadrado(df: pd.DataFrame, fila: str, columna: str, clave_hipotesis: str, etiqueta: str,
                 alfa: float = config.ALFA, extra: dict[str, Any] | None = None) -> ResultadoTest:
    """Chi² de independencia con V de Cramér, frecuencias esperadas y residuos tipificados extremos."""
    tabla = pd.crosstab(df[fila], df[columna])
    # reindex (no .loc): con etiquetas booleanas .loc las interpretaría como máscara.
    tabla = tabla.reindex(index=sorted(tabla.index, key=str), columns=sorted(tabla.columns, key=str))
    chi2, p, gl, esperadas = stats.chi2_contingency(tabla.to_numpy(), correction=False)
    v, vc = cramer_v(tabla.to_numpy())
    df_star = min(tabla.shape) - 1
    efecto = interpretar_v(v, df_star)
    pct_bajas = float((esperadas < 5).mean() * 100)
    residuos = (tabla.to_numpy() - esperadas) / np.sqrt(esperadas)
    orden = np.argsort(-np.abs(residuos), axis=None)[:5]
    extremos = [{"fila": str(tabla.index[i // tabla.shape[1]]), "columna": str(tabla.columns[i % tabla.shape[1]]),
                 "residuo": float(residuos.flat[i])} for i in orden]
    conclusion = redactar_conclusion(
        p, alfa, efecto,
        f"Hay asociación entre {etiqueta} (V de Cramér = {fmt.n3(v)}).",
        f"No hay evidencia de asociación entre {etiqueta}.",
    )
    h0, h1 = HIPOTESIS[clave_hipotesis]
    return ResultadoTest(
        nombre=f"Chi-cuadrado: {etiqueta}", h0=h0, h1=h1, alfa=alfa,
        supuestos={"min_esperada": float(esperadas.min()), "pct_celdas_esperada_menor_5": pct_bajas,
                   "cumple_esperadas": bool(pct_bajas <= 20 and esperadas.min() >= 1), "correccion_yates": False},
        estadistico=float(chi2), p_valor=float(p), n=int(tabla.to_numpy().sum()),
        tamano_efecto={"nombre": "V de Cramér", "valor": v, "v_corregida": vc, "df_star": int(df_star),
                       "interpretacion": efecto},
        conclusion=conclusion,
        extra={"grados_libertad": int(gl), "dimensiones": list(tabla.shape), "residuos_extremos": extremos,
               **(extra or {})},
    )


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_mediana(valores: np.ndarray, n_boot: int, rng: np.random.Generator, lote: int = 200,
                      nivel: float = 0.95) -> dict[str, float]:
    """IC percentil de la mediana por bootstrap (remuestreo con reemplazo en lotes)."""
    x = np.asarray(valores, dtype="float64")
    x = x[~np.isnan(x)]
    medianas = np.empty(n_boot)
    hecho = 0
    while hecho < n_boot:
        b = min(lote, n_boot - hecho)
        idx = rng.integers(0, len(x), size=(b, len(x)))
        medianas[hecho:hecho + b] = np.median(x[idx], axis=1)
        hecho += b
    a = (1 - nivel) / 2
    return {"n": int(len(x)), "mediana": float(np.median(x)), "ic_inf": float(np.quantile(medianas, a)),
            "ic_sup": float(np.quantile(medianas, 1 - a)), "n_boot": int(n_boot)}



def bootstrap_mediana_cluster(valores: np.ndarray, grupos: np.ndarray, n_boot: int, rng: np.random.Generator,
                              lote: int = 100, nivel: float = 0.95) -> dict[str, float]:
    """IC percentil de la mediana por bootstrap **por anfitrión** (cluster bootstrap).

    Se remuestrean anfitriones con reemplazo (cada uno con todos sus anuncios), lo que respeta la
    dependencia entre anuncios del mismo anfitrión. Cada réplica es la mediana ponderada (inferior)
    con peso = nº de veces que salió el anfitrión. Vectorizado por lotes.
    """
    x = np.asarray(valores, dtype="float64")
    g = np.asarray(grupos)
    ok = ~np.isnan(x)
    x, g = x[ok], g[ok]
    orden = np.argsort(x, kind="mergesort")
    x, g = x[orden], g[orden]
    _, inversa = np.unique(g, return_inverse=True)
    n_grupos = int(inversa.max()) + 1
    medianas = np.empty(n_boot)
    hecho = 0
    while hecho < n_boot:
        b = min(lote, n_boot - hecho)
        cuentas = rng.multinomial(n_grupos, np.full(n_grupos, 1 / n_grupos), size=b)
        acumulado = np.cumsum(cuentas[:, inversa], axis=1)
        idx = (acumulado >= acumulado[:, -1:] / 2).argmax(axis=1)
        medianas[hecho:hecho + b] = x[idx]
        hecho += b
    a = (1 - nivel) / 2
    return {"n": int(len(x)), "n_anfitriones": n_grupos, "mediana": float(np.median(x)),
            "ic_inf": float(np.quantile(medianas, a)), "ic_sup": float(np.quantile(medianas, 1 - a)),
            "n_boot": int(n_boot)}


# --------------------------------------------------------------------------
# Relación renta-precio: Spearman con bootstrap por clusters
# --------------------------------------------------------------------------
def _grupos_empate(v: np.ndarray) -> tuple[np.ndarray, int]:
    """Para cada observación, el índice (0..k-1, en orden creciente de valor) de su grupo de valores iguales."""
    _, inversa = np.unique(v, return_inverse=True)
    return inversa, int(inversa.max()) + 1


def _rangos_ponderados(grupo: np.ndarray, n_grupos: int, w: np.ndarray) -> np.ndarray:
    """Rangos medios de una muestra en la que la observación i aparece ``w[i]`` veces.

    Equivale a ``rankdata`` sobre los datos replicados: cada grupo de valores iguales con peso total W y
    peso acumulado anterior C recibe el rango C + (W + 1) / 2. Coste O(n), sin volver a ordenar.
    """
    peso = np.bincount(grupo, weights=w, minlength=n_grupos)
    rango = np.cumsum(peso) - peso + (peso + 1) / 2
    return rango[grupo]


def _pearson_ponderado(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Correlación de Pearson con pesos de frecuencia ``w``; NaN si alguna variable es constante."""
    total = w.sum()
    mx, my = (w * x).sum() / total, (w * y).sum() / total
    dx, dy = x - mx, y - my
    sxx, syy = (w * dx * dx).sum(), (w * dy * dy).sum()
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return float((w * dx * dy).sum() / np.sqrt(sxx * syy))


def spearman_cluster(x: pd.Series, y: pd.Series, clusters: dict[str, pd.Series], n_boot: int,
                     rng: np.random.Generator, nivel: float = 0.95) -> dict[str, Any]:
    """ρ de Spearman con IC percentil y p bootstrap remuestreando clusters (p. ej. distrito o anfitrión).

    Cada réplica remuestrea clusters con reemplazo (cada uno con todos sus anuncios) y calcula la ρ de
    Spearman de la muestra replicada; se implementa con pesos de frecuencia (exactamente equivalente).
    El p asintótico de ``scipy.stats.spearmanr`` supone observaciones independientes; aquí no lo son
    (la renta es la misma para todos los anuncios de un distrito y un anfitrión tiene varios anuncios), así
    que se informa solo como referencia. La decisión se toma con el bootstrap por clusters: p bilateral
    = 2·min(P(ρ* ≤ 0), P(ρ* ≥ 0)) con la corrección (k + 1)/(B + 1).
    """
    ok = x.notna() & y.notna()
    for g in clusters.values():
        ok &= g.reindex(x.index).notna()
    xv, yv = x[ok].to_numpy(dtype="float64"), y[ok].to_numpy(dtype="float64")
    salida: dict[str, Any] = {"n": int(ok.sum()), "n_boot": int(n_boot)}
    if len(xv) < config.MIN_N_GRUPO or np.unique(xv).size < 2:
        return {**salida, "rho": float("nan"), "evaluable": False}
    rho = float(stats.spearmanr(xv, yv).statistic)
    p_asint = float(stats.spearmanr(xv, yv).pvalue)
    salida.update({"rho": rho, "p_asintotico": p_asint, "evaluable": True,
                   "n_valores_distintos_x": int(np.unique(xv).size)})
    gx, kx = _grupos_empate(xv)
    gy, ky = _grupos_empate(yv)
    a = (1 - nivel) / 2
    for nombre, g in clusters.items():
        _, inversa = np.unique(g[ok].astype(str).to_numpy(), return_inverse=True)
        n_cl = int(inversa.max()) + 1
        reps = np.empty(n_boot)
        for b in range(n_boot):
            veces = np.bincount(rng.integers(0, n_cl, size=n_cl), minlength=n_cl)
            w = veces[inversa].astype("float64")
            reps[b] = _pearson_ponderado(_rangos_ponderados(gx, kx, w), _rangos_ponderados(gy, ky, w), w)
        reps = reps[~np.isnan(reps)]
        k_le, k_ge = int((reps <= 0).sum()), int((reps >= 0).sum())
        p_boot = float(min(1.0, 2 * min((k_le + 1) / (len(reps) + 1), (k_ge + 1) / (len(reps) + 1))))
        salida[nombre] = {"n_clusters": n_cl, "ic_inf": float(np.quantile(reps, a)),
                          "ic_sup": float(np.quantile(reps, 1 - a)), "p_boot": p_boot,
                          "significativo": bool(p_boot < config.ALFA), "n_replicas_validas": int(len(reps))}
    return salida


def holm(pvalores: dict[str, float], alfa: float = config.ALFA) -> dict[str, dict[str, Any]]:
    """Corrección de Holm sobre una familia de p-valores ``{nombre: p}``."""
    from statsmodels.stats.multitest import multipletests

    nombres = [k for k, v in pvalores.items() if v is not None and np.isfinite(v)]
    if not nombres:
        return {}
    rechaza, p_aj, _, _ = multipletests([pvalores[k] for k in nombres], alpha=alfa, method="holm")
    return {k: {"p": float(pvalores[k]), "p_holm": float(p), "significativo_holm": bool(r)}
            for k, p, r in zip(nombres, p_aj, rechaza)}


# --------------------------------------------------------------------------
# OLS
# --------------------------------------------------------------------------
VARIABLES_OLS_NUMERICAS: tuple[str, ...] = (
    "accommodates", "bedrooms", "bathrooms", "bano_compartido", "n_amenities",
    "host_is_superhost", "es_multianuncio", "alquiler_larga_estancia", "tiene_aire_acondicionado",
    "tiene_piscina", "tiene_terraza_balcon", "tiene_parking_cualquiera", "log1p_resenas",
)
# Variables de la zona (INE, distrito censal). La renta entra en logaritmo: β es una elasticidad.
VARIABLES_OLS_INE: tuple[str, ...] = (
    "ln_renta_dis", "ine_dis_gini", "ine_dis_edad_media", "ine_dis_pct_hogares_unipersonales",
)
NOMBRES_TERMINOS: dict[str, str] = {
    "accommodates": "Capacidad (+1 persona)", "bedrooms": "Dormitorios (+1)", "bathrooms": "Baños (+1)",
    "bano_compartido": "Baño compartido", "n_amenities": "Amenities (+1)",
    "distancia_centro_km": "Distancia al centro (+1 km)", "host_is_superhost": "Superhost",
    "es_multianuncio": "Anfitrión multi-anuncio", "alquiler_larga_estancia": "Estancia mínima ≥ 30 noches",
    "tiene_aire_acondicionado": "Aire acondicionado", "tiene_piscina": "Piscina",
    "tiene_terraza_balcon": "Terraza/balcón", "tiene_parking_cualquiera": "Parking (cualquiera)",
    "log1p_resenas": "ln(1 + nº reseñas)",
    "ln_renta_dis": "ln(renta neta por persona del distrito)", "ine_dis_gini": "Gini del distrito (+1 punto)",
    "ine_dis_edad_media": "Edad media del distrito (+1 año)",
    "ine_dis_pct_hogares_unipersonales": "% hogares unipersonales del distrito (+1 punto)",
}
# Términos cuyos coeficientes se comparan entre especificaciones (sensibilidades).
TERMINOS_CLAVE: tuple[str, ...] = (
    "ln_renta_dis", "ine_dis_gini", "accommodates", "host_is_superhost", "es_multianuncio",
    "alquiler_larga_estancia", "bano_compartido",
)
REFERENCIA_LICENCIA = "Sin licencia"


def anadir_derivadas_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas transformadas que usan OLS y ML: ln(1 + reseñas) y ln(renta del distrito)."""
    return df.assign(
        log1p_resenas=np.log1p(df["number_of_reviews"].astype("float64")),
        ln_renta_dis=np.log(df[config.RENTA_PRINCIPAL].astype("float64").where(df[config.RENTA_PRINCIPAL] > 0)),
    )


def _matriz_ols(df: pd.DataFrame, numericas: tuple[str, ...]) -> pd.DataFrame:
    """DataFrame de la regresión (booleanos a 0/1). Casos completos sobre las variables del modelo
    principal (con INE), para que las especificaciones con y sin INE usen exactamente las mismas filas."""
    d = anadir_derivadas_modelo(df)
    todas = dict.fromkeys([*VARIABLES_OLS_NUMERICAS, *VARIABLES_OLS_INE, *numericas])
    salida = pd.DataFrame({
        "log_precio": d["log_precio"],
        **{c: d[c].astype("float64") for c in todas},
        "tipo": d["tipo_habitacion_agrupado"].astype("string"),
        "licencia": d["licencia_tipo"].astype("string"),
        "ciudad": d["ciudad"].map(lambda c: config.CIUDADES_ESPANA[c].nombre).astype("string"),
        "mes": d["mes_cotizacion_agrupado"].astype("string"),
        "host_id": d["host_id"],
        "distrito_ine": d["codigo_distrito_ine"].astype("string"),
    }).dropna()
    for c in ("tipo", "licencia", "ciudad", "mes", "distrito_ine"):
        salida[c] = salida[c].astype(str)
    return salida


def formula_ols(numericas: tuple[str, ...], referencias: dict[str, str | None], con_ciudad: bool) -> str:
    """Fórmula de statsmodels para ln(precio) con efectos fijos de ciudad, tipo, licencia y mes."""
    partes = [f"C(tipo, Treatment('{referencias['tipo']}'))", *numericas,
              f"C(licencia, Treatment('{referencias['licencia']}'))"]
    if con_ciudad:
        partes.append(f"C(ciudad, Treatment('{referencias['ciudad']}'))")
    if referencias.get("mes") is not None:
        partes.append(f"C(mes, Treatment('{referencias['mes']}'))")
    return "log_precio ~ " + " + ".join(partes)


def _nombre_termino(t: str) -> str:
    """Nombre legible en español de un término de statsmodels."""
    if t in NOMBRES_TERMINOS:
        return NOMBRES_TERMINOS[t]
    if "[T." in t:
        nivel = t.split("[T.", 1)[1].rstrip("]")
        for prefijo, etiqueta in (("C(tipo", "Tipo"), ("C(licencia", "Licencia"), ("C(ciudad", "Ciudad"),
                                  ("C(mes", "Mes de cotización")):
            if t.startswith(prefijo):
                return f"{etiqueta}: {nivel}"
    return t


def _termino_de(columna: str) -> str:
    """Término del modelo al que pertenece una columna de la matriz de diseño (para el GVIF)."""
    return columna.split("[T.", 1)[0] if "[T." in columna else columna


def gvif(exog: pd.DataFrame) -> list[dict[str, Any]]:
    """GVIF de Fox y Monette por término (agrupa las dummies de cada factor) sobre la matriz COMPLETA.

    GVIF_j = det(R_jj)·det(R_−j−j) / det(R), con R la matriz de correlaciones de las columnas sin
    intercepto. Para términos de un solo grado de libertad coincide con el VIF. Se informa también
    GVIF^(1/(2·gl)), comparable entre términos con distinto nº de columnas.
    """
    x = exog.drop(columns=[c for c in exog.columns if c == "Intercept"])
    x = x.loc[:, x.std() > 0]
    r = np.corrcoef(x.to_numpy(), rowvar=False)
    _, logdet_r = np.linalg.slogdet(r)
    terminos: dict[str, list[int]] = {}
    for i, c in enumerate(x.columns):
        terminos.setdefault(_termino_de(c), []).append(i)
    filas = []
    for termino, idx in terminos.items():
        resto = [i for i in range(len(x.columns)) if i not in idx]
        _, ld1 = np.linalg.slogdet(r[np.ix_(idx, idx)])
        _, ld2 = np.linalg.slogdet(r[np.ix_(resto, resto)])
        valor = float(np.exp(ld1 + ld2 - logdet_r))
        filas.append({"termino": termino, "nombre": _nombre_termino(termino).split(":")[0],
                      "gl": len(idx), "gvif": valor, "gvif_ajustado": float(valor ** (1 / (2 * len(idx))))})
    return sorted(filas, key=lambda f: -f["gvif_ajustado"])


def _coeficientes(cluster: Any, hc3: Any, alfa: float, cluster_dis: Any | None = None) -> list[dict[str, Any]]:
    """Coeficientes con cambio %, IC 95 % (cluster por anfitrión), p cluster, p HC3 y p de Holm.

    Si se pasa ``cluster_dis`` (errores agrupados por distrito censal), los términos del INE incluyen
    también su p e IC con esa agrupación, la adecuada para variables que solo varían entre distritos.
    """
    from statsmodels.stats.multitest import multipletests

    ic = cluster.conf_int(alpha=alfa)
    ic_d = cluster_dis.conf_int(alpha=alfa) if cluster_dis is not None else None
    terminos = [t for t in cluster.params.index if t != "Intercept"]
    p_holm = multipletests([float(cluster.pvalues[t]) for t in terminos], alpha=alfa, method="holm")[1]
    salida = []
    for t, ph in zip(terminos, p_holm):
        b = float(cluster.params[t])
        fila = {
            "termino": t, "nombre": _nombre_termino(t), "beta": b, "pct": 100 * (np.exp(b) - 1),
            "ic_inf_pct": 100 * (np.exp(ic.loc[t, 0]) - 1), "ic_sup_pct": 100 * (np.exp(ic.loc[t, 1]) - 1),
            "p_cluster": float(cluster.pvalues[t]), "p_hc3": float(hc3.pvalues[t]), "p_holm": float(ph),
            "significativo": bool(cluster.pvalues[t] < alfa), "significativo_holm": bool(ph < alfa),
            "es_ciudad": t.startswith("C(ciudad"), "es_mes": t.startswith("C(mes"),
            "es_ine": t in VARIABLES_OLS_INE,
        }
        if ic_d is not None and t in VARIABLES_OLS_INE:
            fila.update({"p_cluster_distrito": float(cluster_dis.pvalues[t]),
                         "ic_inf_pct_distrito": 100 * (np.exp(ic_d.loc[t, 0]) - 1),
                         "ic_sup_pct_distrito": 100 * (np.exp(ic_d.loc[t, 1]) - 1),
                         "significativo_distrito": bool(cluster_dis.pvalues[t] < alfa)})
        if t == "ln_renta_dis":
            # Elasticidad: cambio % del precio asociado a una renta un 10 % mayor.
            fila["pct_por_10pct_renta"] = 100 * (1.1 ** b - 1)
            fila["ic_inf_10pct"] = 100 * (1.1 ** float(ic.loc[t, 0]) - 1)
            fila["ic_sup_10pct"] = 100 * (1.1 ** float(ic.loc[t, 1]) - 1)
            if ic_d is not None:
                fila["ic_inf_10pct_distrito"] = 100 * (1.1 ** float(ic_d.loc[t, 0]) - 1)
                fila["ic_sup_10pct_distrito"] = 100 * (1.1 ** float(ic_d.loc[t, 1]) - 1)
        salida.append(fila)
    return salida


def ajustar_ols(df: pd.DataFrame, alfa: float = config.ALFA, *, mascara_extra: pd.Series | None = None,
                con_ine: bool = True, con_distancia: bool = False, con_mes: bool = True, con_ciudad: bool = True,
                excluir: tuple[str, ...] = (), diagnostico: bool = True) -> dict[str, Any]:
    """OLS sobre ln(precio) con efectos fijos de ciudad, errores agrupados por anfitrión, HC3 y diagnóstico.

    - ``con_ine``: añade las variables de la zona (``VARIABLES_OLS_INE``). Con y sin INE se usan las
      mismas filas (casos completos sobre las variables del modelo principal), así ΔR² es comparable.
    - ``con_distancia``: añade la distancia al centro (solo tiene sentido con ``mascara_extra`` de
      ciudades urbanas: en las regiones es NaN y esas filas se pierden).
    - Los términos del INE se contrastan además con errores agrupados por distrito censal.
    ``diagnostico=False`` ajusta una especificación alternativa sin diagnóstico (para sensibilidades).
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import jarque_bera

    mascara = df["precio_valido"].astype(bool)
    if mascara_extra is not None:
        mascara &= mascara_extra.reindex(df.index, fill_value=False).astype(bool)
    numericas = tuple(v for v in (*VARIABLES_OLS_NUMERICAS, *(VARIABLES_OLS_INE if con_ine else ()),
                                  *(("distancia_centro_km",) if con_distancia else ())) if v not in excluir)
    base = df[mascara]
    datos = _matriz_ols(base, numericas)
    n_inicial, n_efectivo = int(len(base)), int(len(datos))
    if n_efectivo < config.MIN_N_GRUPO * 10:
        raise ErrorDatos(f"Muy pocas filas completas para la regresión OLS ({n_efectivo}).")
    ciudades = sorted(datos["ciudad"].unique())
    con_ciudad = con_ciudad and len(ciudades) > 1
    referencias: dict[str, str | None] = {
        "tipo": config.TRADUCCION_TIPO_HABITACION["Entire home/apt"], "licencia": REFERENCIA_LICENCIA,
        "ciudad": config.CIUDAD_REFERENCIA if config.CIUDAD_REFERENCIA in ciudades else ciudades[0],
        "mes": str(datos["mes"].value_counts().idxmax()) if con_mes else None,
    }
    formula = formula_ols(numericas, referencias, con_ciudad)

    def _ajustar(d: pd.DataFrame) -> tuple[Any, Any, Any, Any, Any]:
        try:
            modelo = smf.ols(formula, data=d)
            cl_dis = (modelo.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["distrito_ine"])[0]})
                      if con_ine else None)
            return (modelo, modelo.fit(),
                    modelo.fit(cov_type="cluster", cov_kwds={"groups": d["host_id"].to_numpy()}),
                    modelo.fit(cov_type="HC3"), cl_dis)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise ErrorDatos(f"No se pudo ajustar la regresión OLS: {exc}") from exc

    modelo, ajuste, cluster, hc3, cl_dis = _ajustar(datos)
    coeficientes = _coeficientes(cluster, hc3, alfa, cl_dis)
    resultado: dict[str, Any] = {
        "formula": formula, "n_inicial": n_inicial, "n_efectivo": n_efectivo,
        "perdidas_por_nulos": n_inicial - n_efectivo, "con_ine": con_ine, "con_distancia": con_distancia,
        "con_mes": con_mes, "con_ciudad": con_ciudad, "excluidos": list(excluir), "referencias": referencias,
        "ciudades": ciudades, "n_anfitriones": int(datos["host_id"].nunique()),
        "n_distritos_ine": int(datos["distrito_ine"].nunique()),
        "r2": float(ajuste.rsquared), "r2_ajustado": float(ajuste.rsquared_adj), "aic": float(ajuste.aic),
        "f_cluster": float(cluster.fvalue), "f_cluster_p": float(cluster.f_pvalue),
        "f_no_robusto": float(ajuste.fvalue), "f_no_robusto_p": float(ajuste.f_pvalue),
        "n_parametros": int(len(ajuste.params)), "n_coeficientes": len(coeficientes),
        "n_significativos": int(sum(c["significativo"] for c in coeficientes)),
        "n_significativos_holm": int(sum(c["significativo_holm"] for c in coeficientes)),
        "coeficientes": coeficientes,
        "errores": "agrupados por anfitrión (cluster); HC3 como comparación; INE también por distrito censal",
        "h0": HIPOTESIS["ols"][0], "h1": HIPOTESIS["ols"][1], "alfa": alfa,
    }
    resultado["coef_por_termino"] = {c["termino"]: c for c in coeficientes
                                     if not (c["es_ciudad"] or c["es_mes"]) and "[T." not in c["termino"]}
    if not diagnostico:
        return resultado

    residuos = ajuste.resid.to_numpy()
    bp_lm, bp_p, _, _ = het_breuschpagan(ajuste.resid, modelo.exog)
    jb, jb_p, asim, curt = jarque_bera(residuos)
    cook = ajuste.get_influence().cooks_distance[0]
    umbral_cook = 4 / n_efectivo
    resultado["diagnostico"] = {
        "breusch_pagan_lm": float(bp_lm), "breusch_pagan_p": float(bp_p), "heterocedasticidad": bool(bp_p < alfa),
        "jarque_bera": float(jb), "jarque_bera_p": float(jb_p), "asimetria_residuos": float(asim),
        "curtosis_residuos": float(curt), "n_cook_mayor_4_n": int((cook > umbral_cook).sum()),
        "pct_cook_mayor_4_n": float((cook > umbral_cook).mean() * 100),
    }
    # VIF de las variables numéricas y del INE (sin dummies de ciudad ni de mes) …
    exog = pd.DataFrame(modelo.exog, columns=modelo.exog_names)
    cols = [c for c in exog.columns if not c.startswith(("C(ciudad", "C(mes"))]
    x_vif = exog[cols].to_numpy()
    vif = [{"termino": c, "nombre": _nombre_termino(c), "vif": float(variance_inflation_factor(x_vif, i))}
           for i, c in enumerate(cols) if c != "Intercept"]
    vif.sort(key=lambda r: -r["vif"])
    resultado["vif"] = vif
    resultado["vif_max"] = vif[0]
    resultado["n_vif_mayor_5"] = int(sum(r["vif"] > 5 for r in vif))
    resultado["n_vif_mayor_10"] = int(sum(r["vif"] > 10 for r in vif))
    # … y con la matriz COMPLETA (con todas las dummies): VIF de la renta y GVIF por término.
    if "ln_renta_dis" in exog.columns:
        i_renta = list(exog.columns).index("ln_renta_dis")
        resultado["vif_completo_renta"] = float(variance_inflation_factor(exog.to_numpy(), i_renta))
    resultado["gvif"] = gvif(exog)
    resultado["gvif_max"] = resultado["gvif"][0]

    # Sensibilidad: reajuste sin observaciones influyentes (Cook > 4/n).
    sin_infl = datos[cook <= umbral_cook]
    _, aj2, cl2, hc2, cld2 = _ajustar(sin_infl)
    coef2 = {c["termino"]: c for c in _coeficientes(cl2, hc2, alfa, cld2)}
    resultado["sensibilidad_sin_influyentes"] = {
        "n_efectivo": int(len(sin_infl)), "n_excluidas": int(len(datos) - len(sin_infl)),
        "r2_ajustado": float(aj2.rsquared_adj),
        "coeficientes": {t: {"pct": coef2[t]["pct"], "significativo": coef2[t]["significativo"],
                             "pct_principal": next(c["pct"] for c in coeficientes if c["termino"] == t)}
                         for t in TERMINOS_CLAVE if t in coef2},
        **comparar_signos(resultado, {"coeficientes": list(coef2.values()), "r2_ajustado": float(aj2.rsquared_adj),
                                      "n_efectivo": int(len(sin_infl))}),
    }
    resultado["_residuos"] = residuos
    resultado["_ajustados"] = ajuste.fittedvalues.to_numpy()
    resultado["_indices"] = datos.index
    return resultado


def comparar_signos(principal: dict[str, Any], sensibilidad: dict[str, Any]) -> dict[str, Any]:
    """Compara signos de los coeficientes significativos del modelo principal con la sensibilidad."""
    otros = {c["termino"]: c["beta"] for c in sensibilidad["coeficientes"]}
    signif = [c for c in principal["coeficientes"] if c["significativo"] and c["termino"] in otros]
    coinciden = [c["termino"] for c in signif if np.sign(otros[c["termino"]]) == np.sign(c["beta"])]
    return {"n_significativos": len(signif), "n_signos_coinciden": len(coinciden),
            "signos_coinciden": len(coinciden) == len(signif),
            "r2_ajustado": sensibilidad["r2_ajustado"], "n_efectivo": sensibilidad["n_efectivo"]}


def coeficientes_clave(principal: dict[str, Any], alternativo: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Cambio % de los ``TERMINOS_CLAVE`` en el modelo principal y en una especificación alternativa."""
    alt = {c["termino"]: c for c in alternativo["coeficientes"]}
    salida = {}
    for c in principal["coeficientes"]:
        if c["termino"] in TERMINOS_CLAVE and c["termino"] in alt:
            a = alt[c["termino"]]
            salida[c["termino"]] = {"nombre": c["nombre"], "pct_principal": c["pct"], "pct_alternativo": a["pct"],
                                    "ic_inf_alternativo": a["ic_inf_pct"], "ic_sup_alternativo": a["ic_sup_pct"],
                                    "p_alternativo": a["p_cluster"], "significativo_alternativo": a["significativo"],
                                    "mismo_signo": bool(np.sign(a["pct"]) == np.sign(c["pct"]))}
    return salida


def resumen_sensibilidad(principal: dict[str, Any], alternativo: dict[str, Any]) -> dict[str, Any]:
    """Resumen comparable de una especificación alternativa: n, R² ajustado, coeficientes clave y renta."""
    renta = alternativo["coef_por_termino"].get("ln_renta_dis")
    return {"n_efectivo": alternativo["n_efectivo"], "r2_ajustado": alternativo["r2_ajustado"],
            "coeficientes": coeficientes_clave(principal, alternativo),
            "renta": None if renta is None else {k: renta[k] for k in (
                "beta", "pct_por_10pct_renta", "ic_inf_10pct", "ic_sup_10pct", "p_cluster", "significativo")
                if k in renta} | ({"p_cluster_distrito": renta["p_cluster_distrito"],
                                   "significativo_distrito": renta["significativo_distrito"]}
                                  if "p_cluster_distrito" in renta else {}),
            "todos_mismo_signo": all(c["mismo_signo"] for c in coeficientes_clave(principal, alternativo).values())}


# --------------------------------------------------------------------------
# Modelo ML (complementario): validación agrupada por anfitrión
# --------------------------------------------------------------------------
def comprobar_fuga(features: list[str]) -> None:
    """Lanza ``ErrorDatos`` si alguna variable está en la lista negra o tiene un prefijo prohibido."""
    prohibidas = [f for f in features if f in config.COLUMNAS_PROHIBIDAS_FUGA
                  or f.startswith(config.PREFIJOS_PROHIBIDOS_FUGA)]
    if prohibidas:
        raise ErrorDatos(f"Fuga del objetivo en el modelo ML: {', '.join(prohibidas)}")


def particiones_por_anfitrion(grupos: pd.Series | np.ndarray, n_folds: int, semilla: int
                              ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Particiones GroupKFold (barajadas con semilla) en las que ningún anfitrión está en train y test."""
    from sklearn.model_selection import GroupKFold

    g = np.asarray(grupos)
    kf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    particiones = list(kf.split(np.zeros(len(g)), groups=g))
    comprobar_sin_solape(g, particiones)
    return particiones


def comprobar_sin_solape(grupos: np.ndarray, particiones: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Lanza ``ErrorDatos`` si algún anfitrión aparece a la vez en train y test de una partición."""
    g = np.asarray(grupos)
    for i, (tr, te) in enumerate(particiones):
        comunes = set(g[tr]) & set(g[te])
        if comunes:
            raise ErrorDatos(f"Partición {i}: {len(comunes)} anfitriones en train y test (fuga por anfitrión).")


def _clases_pipeline() -> tuple[type, type]:
    """Transformadores con estado aprendido SOLO en ``fit`` (train): imputación y agrupación de raras."""
    from sklearn.base import BaseEstimator, TransformerMixin

    class ImputadorMedianaGrupo(BaseEstimator, TransformerMixin):
        """Imputa con la mediana por (tipo, capacidad) del train; después por tipo y después global."""

        def __init__(self, columnas: tuple[str, ...] = config.FEATURES_ML_IMPUTADAS,
                     grupo: tuple[str, str] = ("tipo_habitacion", "accommodates")):
            self.columnas = columnas
            self.grupo = grupo

        def fit(self, X: pd.DataFrame, y: Any = None) -> "ImputadorMedianaGrupo":
            cols = [c for c in self.columnas if c in X]
            self.medianas_grupo_ = {c: X.groupby(list(self.grupo))[c].median() for c in cols}
            self.medianas_tipo_ = {c: X.groupby(self.grupo[0])[c].median() for c in cols}
            self.medianas_global_ = {c: float(X[c].median()) for c in cols}
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            X = X.copy()
            claves = pd.MultiIndex.from_frame(X[list(self.grupo)])
            for c, med in self.medianas_grupo_.items():
                m1 = pd.Series(med.reindex(claves).to_numpy(dtype="float64"), index=X.index)
                m2 = X[self.grupo[0]].map(self.medianas_tipo_[c]).astype("float64")
                X[c] = X[c].fillna(m1).fillna(m2).fillna(self.medianas_global_[c])
            return X

    class AgrupadorRaras(BaseEstimator, TransformerMixin):
        """Agrupa en ``etiqueta`` las categorías con menos de ``min_n`` filas en el train."""

        def __init__(self, columnas: tuple[str, ...] = (), min_n: int = config.MIN_N_CATEGORIA_RARA_ML,
                     etiqueta: str = "Otras"):
            self.columnas = columnas
            self.min_n = min_n
            self.etiqueta = etiqueta

        def fit(self, X: pd.DataFrame, y: Any = None) -> "AgrupadorRaras":
            self.frecuentes_ = {}
            for c in self.columnas:
                conteo = X[c].value_counts()
                self.frecuentes_[c] = set(conteo.index[conteo >= self.min_n])
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            X = X.copy()
            for c in self.columnas:
                X[c] = X[c].where(X[c].isin(self.frecuentes_[c]), self.etiqueta)
            return X

    return ImputadorMedianaGrupo, AgrupadorRaras


def preparar_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Matriz de entrada SIN parámetros aprendidos: tipos y valores originales (sin imputar).

    Las variables imputadas en la limpieza (``bedrooms``, ``beds``) vuelven a NaN donde se imputaron,
    para que la imputación se aprenda dentro del Pipeline solo con el train de cada partición.
    """
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c == "log1p_resenas":
            X[c] = np.log1p(df["number_of_reviews"].astype("float64"))
        elif c in config.FEATURES_ML_CATEGORICAS or c in ("tipo_habitacion",):
            X[c] = df[c].astype("string").fillna("Desconocido").astype(str)
        else:
            X[c] = df[c].astype("float64")
            flag = f"{c}_imputado"
            if c in config.FEATURES_ML_IMPUTADAS and flag in df:
                X.loc[df[flag].astype(bool), c] = np.nan
    if "tipo_habitacion" not in X and "accommodates" in X:
        X["tipo_habitacion"] = df["tipo_habitacion"].astype("string").fillna("Desconocido").astype(str)
    return X


def crear_pipeline(features: list[str], tipo: str, semilla: int) -> Any:
    """Pipeline sklearn: imputación por grupo → agrupación de raras → codificación → modelo.

    ``tipo``: ``"hgb"`` (HistGradientBoosting) o ``"lineal"`` (regresión lineal con one-hot).
    Todo lo que se aprende de los datos (medianas, categorías frecuentes, codificación) se ajusta
    en ``fit``, es decir, solo con el train de cada partición.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    Imputador, Agrupador = _clases_pipeline()
    cat = [c for c in features if c in config.FEATURES_ML_CATEGORICAS]
    num = [c for c in features if c not in cat]
    pasos: list[tuple[str, Any]] = [("imputar", Imputador()), ("raras", Agrupador(columnas=tuple(cat)))]
    if tipo == "hgb":
        pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
                                 ("num", "passthrough", num)])
        modelo = HistGradientBoostingRegressor(random_state=semilla, max_iter=300, early_stopping=True,
                                               categorical_features=[True] * len(cat) + [False] * len(num))
    elif tipo == "lineal":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                                 ("num", "passthrough", num)])
        modelo = LinearRegression()
    else:
        raise ValueError(f"Tipo de modelo desconocido: {tipo}")
    return Pipeline([*pasos, ("pre", pre), ("modelo", modelo)])


def _metricas(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """R², MAE en ln, MAE en € y mediana del error absoluto en € (robusta a anuncios extremos)."""
    from sklearn.metrics import mean_absolute_error, r2_score

    err_eur = np.abs(np.exp(y) - np.exp(pred))
    return {"r2": float(r2_score(y, pred)), "mae_log": float(mean_absolute_error(y, pred)),
            "mae_eur": float(err_eur.mean()), "medae_eur": float(np.median(err_eur))}


def _resumir(por_fold: list[dict[str, float]]) -> dict[str, Any]:
    """Media, desviación típica (ddof=1) y valores por partición de cada métrica."""
    claves = por_fold[0].keys()
    salida: dict[str, Any] = {}
    for k in claves:
        v = np.array([f[k] for f in por_fold])
        salida[f"{k}_media"] = float(v.mean())
        salida[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        salida[f"{k}_folds"] = [float(x) for x in v]
    return salida


def evaluar_cv(df: pd.DataFrame, features: list[str], tipo: str, particiones: list[tuple[np.ndarray, np.ndarray]],
               semilla: int, importancia: bool = False) -> dict[str, Any]:
    """Evalúa un Pipeline con las particiones dadas (y la línea base mediana en las mismas)."""
    from sklearn.dummy import DummyRegressor
    from sklearn.inspection import permutation_importance

    X = preparar_X(df, features)
    y = df["log_precio"].to_numpy()
    modelo_f, base_f, imps = [], [], []
    for tr, te in particiones:
        m = crear_pipeline(features, tipo, semilla).fit(X.iloc[tr], y[tr])
        modelo_f.append(_metricas(y[te], m.predict(X.iloc[te])))
        base = DummyRegressor(strategy="median").fit(X.iloc[tr], y[tr])
        base_f.append(_metricas(y[te], base.predict(X.iloc[te])))
        if importancia:
            perm = permutation_importance(m, X.iloc[te][features], y[te], scoring="r2", random_state=semilla,
                                          n_repeats=config.N_REPETICIONES_PERMUTACION)
            imps.append(perm.importances_mean)
    salida = {"modelo": _resumir(modelo_f), "base": _resumir(base_f), "n": int(len(y))}
    if importancia:
        matriz = np.vstack(imps)
        salida["importancia"] = sorted(
            ({"variable": f, "nombre": config.ETIQUETAS_COLUMNAS.get(f, f), "importancia": float(matriz[:, i].mean()),
              "std": float(matriz[:, i].std(ddof=1))} for i, f in enumerate(features)),
            key=lambda r: -r["importancia"])
    return salida


def ajustar_modelo_ml(df: pd.DataFrame, semilla: int, filas_ols: pd.Index | None = None) -> dict[str, Any]:
    """HistGradientBoosting sobre ln(precio) validado con GroupKFold por anfitrión.

    - Validación principal: ``N_FOLDS_ML`` particiones por ``host_id`` (ningún anfitrión en train y test).
    - Contraste «optimista»: partición aleatoria por fila (como antes), solo para mostrar la fuga.
    - Comparación homogénea con el lineal: mismas filas (las del OLS), mismas particiones y mismas
      variables (las del OLS) para regresión lineal y HGB, más HGB con todas las variables.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {"ejecutado": False, "motivo": "scikit-learn no está instalado"}

    features = [f for f in config.FEATURES_ML if f in df.columns]
    comprobar_fuga(features)
    datos = anadir_derivadas_modelo(df[df["precio_valido"]])
    particiones = particiones_por_anfitrion(datos["host_id"], config.N_FOLDS_ML, semilla)
    principal = evaluar_cv(datos, features, "hgb", particiones, semilla, importancia=True)
    # Aportación predictiva de la zona: mismo modelo y mismas particiones sin las variables del INE.
    sin_ine = [f for f in features if f not in config.VARIABLES_INE_ML]
    cv_sin_ine = evaluar_cv(datos, sin_ine, "hgb", particiones, semilla)["modelo"]

    # Contraste: partición aleatoria por fila (mide interpolación dentro de anfitriones ya vistos).
    X = preparar_X(datos, features)
    y = datos["log_precio"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=config.TEST_SIZE_ALEATORIO, random_state=semilla)
    m = crear_pipeline(features, "hgb", semilla).fit(X.iloc[idx_tr], y[idx_tr])
    hosts = datos["host_id"].to_numpy()
    aleatorio = {**_metricas(y[idx_te], m.predict(X.iloc[idx_te])), "n_train": int(len(idx_tr)),
                 "n_test": int(len(idx_te)),
                 "pct_test_con_anfitrion_en_train": float(np.isin(hosts[idx_te], hosts[idx_tr]).mean() * 100)}

    comparacion: dict[str, Any] = {}
    if filas_ols is not None and len(filas_ols):
        sub = anadir_derivadas_modelo(df.loc[filas_ols])
        vars_ols = [*VARIABLES_OLS_NUMERICAS, *VARIABLES_OLS_INE, "tipo_habitacion", "licencia_tipo", "ciudad",
                    "mes_cotizacion"]
        part_ols = particiones_por_anfitrion(sub["host_id"], config.N_FOLDS_ML, semilla)
        comparacion = {
            "n": int(len(sub)), "variables_ols": vars_ols,
            "lineal_vars_ols": evaluar_cv(sub, vars_ols, "lineal", part_ols, semilla)["modelo"],
            "hgb_vars_ols": evaluar_cv(sub, vars_ols, "hgb", part_ols, semilla)["modelo"],
            "hgb_todas": evaluar_cv(sub, features, "hgb", part_ols, semilla)["modelo"],
        }
    cv = principal["modelo"]
    return {
        "ejecutado": True, "modelo": "HistGradientBoostingRegressor (max_iter=300, early_stopping)",
        "objetivo": "ln(precio)", "n": principal["n"], "n_anfitriones": int(datos["host_id"].nunique()),
        "n_folds": config.N_FOLDS_ML, "semilla": semilla, "n_features": len(features), "features": features,
        "validacion": "GroupKFold por host_id (barajado con semilla); imputación y agrupación de categorías "
                      "raras aprendidas dentro del Pipeline solo con el train de cada partición",
        "cv": cv, "cv_base": principal["base"], "aleatorio": aleatorio,
        "caida_r2_aleatorio_a_grupos": aleatorio["r2"] - cv["r2_media"],
        "cv_sin_ine": cv_sin_ine, "features_ine": list(config.VARIABLES_INE_ML),
        "aporte_r2_ine": cv["r2_media"] - cv_sin_ine["r2_media"],
        "comparacion_misma_muestra": comparacion,
        "importancia_permutacion": principal["importancia"][:10],
        "importancia_completa": principal["importancia"],
        "rango_ine_en_importancia": min(i for i, r in enumerate(principal["importancia"], 1)
                                         if r["variable"] in config.VARIABLES_INE_ML),
        "nota": "Importancia predictiva (media de las particiones), no causal. El MAE en € tras exp() tiene sesgo "
                "de retransformación y lo dominan pocos anuncios extremos; la mediana del error absoluto es robusta.",
    }


def un_anuncio_por_host(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Selecciona al azar (semilla fija) un anuncio por anfitrión para el análisis de sensibilidad."""
    orden = df.assign(_r=rng.random(len(df))).sort_values(["host_id", "_r"], kind="mergesort")
    return orden.drop_duplicates("host_id", keep="first").drop(columns="_r")
