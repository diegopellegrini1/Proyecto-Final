"""Mide y valida la unión Inside Airbnb (España) + INE ADRH sin descargar nada.

Uso:
    python scripts/validar_union_espana.py [--salida DIR]

1. Construye el conjunto unido con ``src.fuentes`` (modo sin descarga) y comprueba los requisitos.
2. Tasas de emparejamiento por ciudad (sección, distrito y municipio con dato del INE).
3. Concordancia anuncio a anuncio con la unidad declarada por Inside Airbnb (``fuentes.concordancia``).
4. Coherencia de coordenadas: punto en polígono contra el GeoJSON de barrios de Inside Airbnb.
5. Muestra aleatoria (semilla 42) de 30 anuncios por ciudad con más de 5.000 anuncios, con la distancia
   de cada discrepancia a la unidad esperada, para revisión manual.
6. Tamaño real del conjunto unido en Parquet y estimado en Excel.
Escribe CSV y un JSON de resumen en ``--salida`` (por defecto ``.agents/validacion``, no versionado).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, fuentes  # noqa: E402

TAM_MUESTRA = 30
MIN_ANUNCIOS_CIUDAD_GRANDE = 5000
# Columnas previstas para el conjunto final (fase 2); solo se usan aquí para estimar tamaños.
COLUMNAS_PREVISTAS_AIRBNB = (
    "ciudad", "id", "fecha_snapshot", "latitude", "longitude", "neighbourhood_cleansed",
    "neighbourhood_group_cleansed", "room_type", "property_type", "accommodates", "bedrooms", "beds",
    "bathrooms", "price", "minimum_nights", "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "review_scores_rating", "host_id", "host_is_superhost", "calculated_host_listings_count",
    "estimated_occupancy_l365d", "estimated_revenue_l365d", "license", "instant_bookable",
)


def coherencia_geojson(unido: pd.DataFrame) -> pd.DataFrame:
    """% de anuncios cuyo punto cae en el polígono de su neighbourhood_cleansed (GeoJSON de Inside Airbnb)."""
    import shapely
    import shapely.geometry

    filas = []
    for ciudad, s in unido.groupby("ciudad"):
        datos = json.loads((config.DIR_RAW_AIRBNB / ciudad / "neighbourhoods.geojson").read_text(encoding="utf-8"))
        geoms = [shapely.geometry.shape(f["geometry"]) for f in datos["features"]]
        nombres = np.array([fuentes.reparar_mojibake(f["properties"]["neighbourhood"]) for f in datos["features"]])
        puntos = shapely.points(s["longitude"].to_numpy(), s["latitude"].to_numpy())
        ip, ig = shapely.STRtree(geoms).query(puntos, predicate="intersects")
        pares = pd.DataFrame({"p": ip, "g": ig}).drop_duplicates("p")
        asignado = pd.Series(pd.NA, index=range(len(s)), dtype="string")
        asignado.iloc[pares["p"].to_numpy()] = nombres[pares["g"].to_numpy()]
        declarado = s["neighbourhood_cleansed"].map(fuentes.reparar_mojibake).astype("string").to_numpy()
        ok = asignado.to_numpy() == declarado
        filas.append({"ciudad": ciudad, "anuncios": len(s), "en_su_barrio": int(np.sum(ok == True)),  # noqa: E712
                      "pct": round(100 * float(np.mean(ok == True)), 2)})  # noqa: E712
    return pd.DataFrame(filas).set_index("ciudad")


def distancias_a_esperado(muestra: pd.DataFrame) -> pd.Series:
    """Metros desde cada discrepancia de la muestra hasta la unidad INE esperada (unión de sus secciones)."""
    import shapely
    from pyproj import Transformer

    malas = muestra[muestra["coincide"] == False]  # noqa: E712
    salida = pd.Series(np.nan, index=muestra.index)
    if malas.empty:
        return salida
    ruta_zip = config.DIR_EXTRA_INE / fuentes.leer_manifiesto(config.DIR_RAW_INE)["cartografia"]["archivo"]
    provincias = sorted({p for c in malas["ciudad"].unique() for p in config.CIUDADES_ESPANA[c].provincias})
    geoms, _, atributos = fuentes.cargar_secciones(ruta_zip, provincias)
    trans = Transformer.from_crs(config.CRS_LISTINGS, config.CRS_SECCIONADO, always_xy=True)
    for idx, fila in malas.iterrows():
        if fila["metodo"].startswith("distrito"):
            sel = atributos["CUDIS"] == fila["esperado_ine"]
        else:
            sel = atributos["NMUN"].map(lambda n: fuentes.coincide_nombre(n, fila["esperado_ine"]))
        if not sel.any():
            continue
        zona = shapely.union_all([geoms[i] for i in np.flatnonzero(sel.to_numpy())])
        x, y = trans.transform(fila["longitude"], fila["latitude"])
        salida[idx] = round(float(zona.distance(shapely.Point(x, y))), 1)
    return salida


def main() -> int:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, default=RAIZ / ".agents" / "validacion")
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    unido, anio = fuentes.construir_conjunto_unido(sin_descarga=True)
    tasas = fuentes.tasas_emparejamiento(unido)
    tabla = fuentes.concordancia(unido)
    concord = fuentes.resumen_concordancia(tabla)
    geo = coherencia_geojson(unido)

    grandes = [c for c, n in unido["ciudad"].value_counts().items() if n > MIN_ANUNCIOS_CIUDAD_GRANDE]
    muestra = (tabla[tabla["ciudad"].isin(grandes)].groupby("ciudad")
               .sample(TAM_MUESTRA, random_state=config.SEMILLA))
    muestra["distancia_m_a_esperado"] = distancias_a_esperado(muestra)
    discrepancias = (tabla[tabla["coincide"] == False]  # noqa: E712
                     .groupby(["ciudad", "declarado_airbnb", "asignado_ine"]).size()
                     .rename("anuncios").reset_index().sort_values(["ciudad", "anuncios"], ascending=[True, False]))

    with tempfile.TemporaryDirectory() as tmp:
        ruta_pq = Path(tmp) / "unido.parquet"
        unido.to_parquet(ruta_pq, index=False)
        bytes_parquet = ruta_pq.stat().st_size
        ruta_xlsx = Path(tmp) / "muestra.xlsx"
        n_xlsx = 5000
        unido.sample(n_xlsx, random_state=config.SEMILLA).to_excel(ruta_xlsx, index=False)
        bytes_xlsx_estimado = int(ruta_xlsx.stat().st_size / n_xlsx * len(unido))
        with zipfile.ZipFile(ruta_xlsx) as z:
            assert z.testzip() is None
        previstas = [c for c in COLUMNAS_PREVISTAS_AIRBNB if c in unido.columns] + [
            c for c in unido.columns if c.startswith("ine_") or c.endswith("_ine")]
        faltan_previstas = sorted(set(COLUMNAS_PREVISTAS_AIRBNB) - set(unido.columns))
        reducido = unido[previstas]
        reducido.to_parquet(ruta_pq, index=False)
        bytes_parquet_previstas = ruta_pq.stat().st_size
        reducido.to_excel(ruta_xlsx, index=False)
        bytes_xlsx_previstas = ruta_xlsx.stat().st_size

    tasas.to_csv(args.salida / "tasas_emparejamiento.csv")
    concord.to_csv(args.salida / "concordancia.csv")
    geo.to_csv(args.salida / "coherencia_geojson.csv")
    muestra.to_csv(args.salida / "muestra_validacion.csv", index=False)
    discrepancias.to_csv(args.salida / "discrepancias.csv", index=False)
    resumen = {
        "filas": int(unido.shape[0]), "columnas": int(unido.shape[1]), "anio_adrh": anio,
        "ids_repetidos_entre_ciudades": int(unido["id"].duplicated().sum()),
        "filas_con_precio": int((pd.to_numeric(unido["price"].astype("string").str.replace(r"[$,]", "", regex=True),
                                               errors="coerce") > 0).sum()),
        "columnas_ine": [c for c in unido.columns if c.startswith("ine_") or c.endswith("_ine")],
        "bytes_parquet": bytes_parquet, "bytes_xlsx_estimado": bytes_xlsx_estimado,
        "columnas_previstas": len(previstas), "faltan_previstas": faltan_previstas,
        "bytes_parquet_previstas": bytes_parquet_previstas, "bytes_xlsx_previstas": bytes_xlsx_previstas,
        "nulos_previstas_pct": reducido.isna().mean().mul(100).round(2).to_dict(),
        "muestra_coinciden": int(muestra["coincide"].sum()), "muestra_validados": int(muestra["coincide"].notna().sum()),
    }
    (args.salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"Conjunto unido: {unido.shape[0]:,} filas x {unido.shape[1]} columnas (ADRH {anio})")
    print("\nTasas de emparejamiento:\n", tasas.to_string())
    print("\nConcordancia con la unidad declarada por Inside Airbnb:\n", concord.to_string())
    print("\nCoherencia con el GeoJSON de Inside Airbnb:\n", geo.to_string())
    print("\nMuestra de validación por ciudad:\n",
          muestra.groupby("ciudad")["coincide"].agg(["size", "sum"]).to_string())
    print("\nDiscrepancias de la muestra:\n", muestra[muestra["coincide"] == False].to_string())  # noqa: E712
    print("\nResumen:", json.dumps({k: v for k, v in resumen.items() if k not in ("columnas_ine", "nulos_previstas_pct")},
                                   ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
