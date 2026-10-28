"""Recálculo independiente de cifras clave, con código propio que NO importa ``src/``.

Lee directamente los ficheros crudos (``data/raw``), la asignación anuncio-sección (``data/interim``) y el
conjunto final (``data/processed``), recalcula las cifras con pandas, numpy, scipy y scikit-learn, y las compara
con ``reports/findings.json``. Sale con código 0 si todas coinciden (dentro de la tolerancia indicada) y 5 si no.

Uso:  python scripts/recalcular_cifras.py
"""

from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
RAIZ = Path(__file__).resolve().parents[1]
RAW_AIRBNB = RAIZ / "data" / "raw" / "inside_airbnb"
RAW_INE = RAIZ / "data" / "raw" / "ine"
ASIGNACION = RAIZ / "data" / "interim" / "asignacion_secciones.csv.gz"
PARQUET = RAIZ / "data" / "processed" / "airbnb_espana_ine.parquet"
FINDINGS = RAIZ / "reports" / "findings.json"
INDICADOR_RENTA = "Renta neta media por persona"


def leer_listings() -> pd.DataFrame:
    """Precio (texto → número) e id de cada listings.csv.gz, con la ciudad tomada del nombre de la carpeta."""
    partes = []
    for ruta in sorted(RAW_AIRBNB.glob("*/listings.csv.gz")):
        d = pd.read_csv(ruta, usecols=["id", "price"], dtype={"price": "string"})
        d["precio"] = pd.to_numeric(d["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                                    errors="coerce")
        d["ciudad"] = ruta.parent.name
        partes.append(d[["ciudad", "id", "precio"]])
    return pd.concat(partes, ignore_index=True)


def renta_distritos(anio: int) -> pd.Series:
    """Renta neta media por persona de cada distrito censal (7 dígitos) en ``anio``, leída de los CSV del INE."""
    trozos = []
    for ruta in sorted(RAW_INE.glob("adrh_*_renta_*.csv.gz")):
        with gzip.open(ruta, "rb") as f:
            d = pd.read_csv(io.BytesIO(f.read()), sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False)
        indicador = [c for c in d.columns if c not in ("Municipios", "Distritos", "Secciones", "Periodo", "Total")][0]
        d = d[(d[indicador].str.strip() == INDICADOR_RENTA) & (d["Periodo"] == str(anio))
              & (d["Distritos"].str.strip() != "") & (d["Secciones"].str.strip() == "")]
        codigo = d["Distritos"].str.extract(r"^(\d{7})\s")[0]
        valor = pd.to_numeric(d["Total"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                              errors="coerce")
        trozos.append(pd.Series(valor.to_numpy(), index=codigo.to_numpy()))
    serie = pd.concat(trozos)
    return serie[~serie.index.duplicated()].dropna()


def r2_ajustado_ols(df: pd.DataFrame) -> tuple[float, int]:
    """R² ajustado de la regresión principal con mínimos cuadrados de numpy (sin statsmodels ni src)."""
    numericas = ["accommodates", "bedrooms", "bathrooms", "bano_compartido", "n_amenities", "host_is_superhost",
                 "es_multianuncio", "alquiler_larga_estancia", "tiene_aire_acondicionado", "tiene_piscina",
                 "tiene_terraza_balcon", "tiene_parking_cualquiera"]
    ine = ["ine_dis_gini", "ine_dis_edad_media", "ine_dis_pct_hogares_unipersonales"]
    d = df[df["precio_valido"]].copy()
    d["log1p_resenas"] = np.log1p(d["number_of_reviews"].astype(float))
    d["ln_renta"] = np.log(d["ine_dis_renta_neta_persona"])
    cats = ["tipo_habitacion_agrupado", "licencia_tipo", "ciudad", "mes_cotizacion_agrupado"]
    columnas = ["log_precio", *numericas, *ine, "log1p_resenas", "ln_renta", *cats, "host_id", "codigo_distrito_ine"]
    d = d[columnas].dropna()
    x = pd.concat([d[[*numericas, *ine, "log1p_resenas", "ln_renta"]].astype(float),
                   pd.get_dummies(d[cats].astype(str), drop_first=True, dtype=float)], axis=1)
    x.insert(0, "const", 1.0)
    y = d["log_precio"].to_numpy()
    beta, *_ = np.linalg.lstsq(x.to_numpy(), y, rcond=None)
    resid = y - x.to_numpy() @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    n, k = x.shape
    return float(1 - (1 - r2) * (n - 1) / (n - k)), int(n)


def r2_ml_agrupado(df: pd.DataFrame, semilla: int, features: list[str]) -> float:
    """R² medio de un HistGradientBoosting propio con GroupKFold por anfitrión (preprocesado más simple que el del
    pipeline: categorías como códigos, sin agrupar raras ni imputar). Debe salir parecido, no idéntico."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    d = df[df["precio_valido"]].reset_index(drop=True)
    x = pd.DataFrame(index=d.index)
    categoricas = []
    for f in features:
        if d[f].dtype == object or str(d[f].dtype) in ("string", "str", "category"):
            x[f] = pd.Series(pd.factorize(d[f].astype(str))[0], index=d.index)
            categoricas.append(True)
        else:
            x[f] = d[f].astype(float)
            categoricas.append(False)
    y = d["log_precio"].to_numpy()
    r2 = []
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=semilla).split(x, y, groups=d["host_id"]):
        m = HistGradientBoostingRegressor(random_state=semilla, max_iter=300, early_stopping=True,
                                          categorical_features=categoricas).fit(x.iloc[tr], y[tr])
        r2.append(r2_score(y[te], m.predict(x.iloc[te])))
    return float(np.mean(r2))


def main() -> int:
    """Recalcula, compara y resume."""
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    f = json.loads(FINDINGS.read_text(encoding="utf-8"))
    comparaciones: list[tuple[str, float, float, float]] = []  # (cifra, recalculada, findings, tolerancia)

    def comparar(nombre: str, propio: float, oficial: float, tol: float = 1e-9) -> None:
        comparaciones.append((nombre, float(propio), float(oficial), tol))

    listings = leer_listings()
    tasas = {t["ciudad"]: t for t in f["dataset"]["union"]["tasas"]}
    for ciudad, n in listings.groupby("ciudad").size().items():
        comparar(f"filas {ciudad}", n, tasas[ciudad]["anuncios"])
    comparar("filas total", len(listings), f["dataset"]["filas_procesadas"])

    con_precio = listings[listings["precio"] > 0]
    med = {c["clave"]: c["mediana"] for c in f["descriptivo"]["por_ciudad"]}
    for ciudad, m in con_precio.groupby("ciudad")["precio"].median().items():
        comparar(f"mediana precio {ciudad}", m, med[ciudad])
    comparar("mediana precio total", con_precio["precio"].median(), f["descriptivo"]["precio"]["mediana"])
    comparar("anuncios con precio", len(con_precio), f["descriptivo"]["precio"]["n"])

    asign = pd.read_csv(ASIGNACION, dtype={"codigo_seccion_ine": str, "ciudad": str})
    renta = renta_distritos(int(f["meta"]["anio_ine"]))
    asign["renta"] = asign["codigo_seccion_ine"].str[:7].map(renta)
    comparar("% emparejado con el INE (distrito)", asign["renta"].notna().mean() * 100,
             f["dataset"]["union"]["total"]["pct_distrito"], 0.005)

    unido = con_precio.merge(asign[["ciudad", "id", "renta"]], on=["ciudad", "id"], how="left").dropna()
    comparar("ρ Spearman renta-precio global", stats.spearmanr(unido["renta"], unido["precio"]).statistic,
             f["estadistica"]["renta_precio"]["global_distrito"]["rho"])
    por_ciudad = f["estadistica"]["renta_precio"]["por_ciudad_clave"]
    for ciudad, g in unido.groupby("ciudad"):
        comparar(f"ρ Spearman renta-precio {ciudad}", stats.spearmanr(g["renta"], g["precio"]).statistic,
                 por_ciudad[ciudad]["distrito"]["rho"])

    df = pd.read_parquet(PARQUET)
    comparar("filas del conjunto final (≥ 50.000)", df.shape[0], f["dataset"]["filas_procesadas"])
    comparar("columnas del conjunto final (≥ 20)", df.shape[1], f["dataset"]["columnas_procesadas"])
    r2a, n_ols = r2_ajustado_ols(df)
    comparar("R² ajustado OLS", r2a, f["estadistica"]["ols"]["r2_ajustado"], 1e-6)
    comparar("n del OLS", n_ols, f["estadistica"]["ols"]["n_efectivo"])
    ml = f["estadistica"]["ml"]
    if ml.get("ejecutado"):
        comparar("R² ML agrupado por anfitrión (reimplementación simplificada)",
                 r2_ml_agrupado(df, int(f["meta"]["semilla"]), ml["features"]), ml["cv"]["r2_media"], 0.02)

    fallos = 0
    for nombre, propio, oficial, tol in comparaciones:
        ok = abs(propio - oficial) <= tol
        fallos += not ok
        print(f"{'OK ' if ok else 'DIF'}  {nombre:<62} recalculado {propio:>16.6f}  findings {oficial:>16.6f}")
    print(f"\n{len(comparaciones) - fallos} de {len(comparaciones)} cifras coinciden.")
    print(f"Requisitos del enunciado (recuento propio): {df.shape[0]:,} filas ≥ 50.000: {df.shape[0] >= 50_000}; "
          f"{df.shape[1]} columnas ≥ 20: {df.shape[1] >= 20}")
    return 0 if fallos == 0 and df.shape[0] >= 50_000 and df.shape[1] >= 20 else 5


if __name__ == "__main__":
    sys.exit(main())
