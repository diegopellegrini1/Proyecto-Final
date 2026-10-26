"""Convierte las fuentes ``notebooks/fuentes_py/*.py`` (formato percent) en ``.ipynb`` y las ejecuta con salidas.

Uso:  python scripts/ejecutar_notebooks.py
Sale con código 5 si alguna celda falla (indicando notebook y celda).
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402
from nbclient.exceptions import CellExecutionError  # noqa: E402

from src.errores import ErrorVerificacion  # noqa: E402

DIR_FUENTES = RAIZ / "notebooks" / "fuentes_py"
DIR_NOTEBOOKS = RAIZ / "notebooks"
TIMEOUT_CELDA_S = 600


def convertir(fuente: Path) -> nbformat.NotebookNode:
    """Parsea un .py con marcadores ``# %%`` y ``# %% [markdown]`` y devuelve un notebook v4."""
    celdas: list[tuple[str, list[str]]] = []
    tipo, lineas = None, []
    for linea in fuente.read_text(encoding="utf-8").splitlines():
        if linea.startswith("# %%"):
            if tipo is not None:
                celdas.append((tipo, lineas))
            tipo = "markdown" if "[markdown]" in linea else "code"
            lineas = []
            continue
        if tipo is None:
            continue
        if tipo == "markdown":
            lineas.append(linea[2:] if linea.startswith("# ") else linea.lstrip("#"))
        else:
            lineas.append(linea)
    if tipo is not None:
        celdas.append((tipo, lineas))
    nb = nbformat.v4.new_notebook()
    for tipo, lineas in celdas:
        texto = "\n".join(lineas).strip("\n")
        if not texto:
            continue
        nb.cells.append(nbformat.v4.new_markdown_cell(texto) if tipo == "markdown" else nbformat.v4.new_code_cell(texto))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def ejecutar(fuente: Path) -> Path:
    """Convierte, ejecuta y guarda el notebook; elimina metadatos de tiempo para que sea estable."""
    nb = convertir(fuente)
    cliente = NotebookClient(nb, timeout=TIMEOUT_CELDA_S, kernel_name="python3",
                             resources={"metadata": {"path": str(DIR_NOTEBOOKS)}})
    try:
        cliente.execute()
    except CellExecutionError as exc:
        raise ErrorVerificacion(f"Falló una celda de {fuente.name}: {str(exc)[:2000]}") from exc
    except Exception as exc:  # noqa: BLE001 - p. ej. kernel no encontrado
        raise ErrorVerificacion(f"No se pudo ejecutar {fuente.name} ({exc}). Si falta el kernel: "
                                "python -m ipykernel install --user --name python3") from exc
    for celda in nb.cells:
        celda.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    destino = DIR_NOTEBOOKS / f"{fuente.stem}.ipynb"
    with open(destino, "w", encoding="utf-8", newline="\n") as f:
        nbformat.write(nb, f)
    return destino


def main() -> int:
    """Ejecuta todos los notebooks de ``fuentes_py``."""
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    fuentes = sorted(DIR_FUENTES.glob("*.py"))
    if not fuentes:
        print("No hay fuentes en notebooks/fuentes_py.")
        return 5
    try:
        for fuente in fuentes:
            destino = ejecutar(fuente)
            print(f"Notebook ejecutado: {destino.relative_to(RAIZ)}")
    except ErrorVerificacion as exc:
        print(f"ERROR: {exc}")
        return exc.codigo_salida
    return 0


if __name__ == "__main__":
    sys.exit(main())
