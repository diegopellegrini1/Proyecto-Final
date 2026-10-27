"""Recuento independiente (solo pandas + hashlib, sin importar ``src``) de los datos crudos de España.

Uso:
    python scripts/contar_filas_espana.py

Comprueba el SHA-256 de cada fichero contra los manifiestos de ``data/raw/inside_airbnb`` y
``data/raw/ine`` y cuenta, por ciudad, las filas de ``listings.csv.gz`` y las que tienen precio.
Sirve para contrastar las cifras de la unión (``python -m src.fuentes``) con un código distinto.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_AIRBNB = RAIZ / "data" / "raw" / "inside_airbnb"
DIR_INE = RAIZ / "data" / "raw" / "ine"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def comprobar_hashes(directorio: Path) -> dict:
    manifiesto = json.loads((directorio / "MANIFEST.json").read_text(encoding="utf-8"))
    for entrada in manifiesto["ficheros"]:
        if sha256(directorio / entrada["archivo"]) != entrada["sha256"]:
            raise SystemExit(f"Hash distinto en {entrada['archivo']}")
    return manifiesto


def main() -> int:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    manifiesto = comprobar_hashes(DIR_AIRBNB)
    filas = []
    for entrada in manifiesto["ficheros"]:
        if not entrada["archivo"].endswith("listings.csv.gz"):
            continue
        df = pd.read_csv(DIR_AIRBNB / entrada["archivo"], usecols=["id", "price"], dtype={"price": "string"})
        precio = pd.to_numeric(df["price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
        filas.append({"ciudad": entrada["archivo"].split("/")[0], "fecha_snapshot": entrada["fecha_snapshot"],
                      "filas": len(df), "con_precio": int((precio > 0).sum()), "ids_unicos": df["id"].nunique()})
    tabla = pd.DataFrame(filas).set_index("ciudad").sort_index()
    total = tabla[["filas", "con_precio", "ids_unicos"]].sum()
    print(tabla.to_string())
    print(f"TOTAL filas={total['filas']:,} con_precio={total['con_precio']:,} ciudades={len(tabla)}")
    ine = comprobar_hashes(DIR_INE)
    print(f"INE: {len(ine['ficheros'])} tablas ADRH verificadas por SHA-256 "
          f"({len({e['provincia'] for e in ine['ficheros']})} provincias).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
