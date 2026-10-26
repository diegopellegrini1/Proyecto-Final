"""Comprueba que la carpeta está lista para subir a GitHub arrastrando archivos.

Uso:
    python scripts/verificar_subida.py              # lista de archivos con tamaño; exit 5 si hay problemas
    python scripts/verificar_subida.py --markdown   # la misma lista en tabla Markdown
    python scripts/verificar_subida.py --limpiar    # borra cachés (__pycache__, .pytest_cache) y verifica

Falla si algún archivo supera 25 MB (límite de subida web) o si hay residuos (caches, entornos virtuales,
logs, temporales, data/extra) o patrones de secretos. Excluye .git/ y .agents/ (no se suben).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LIMITE_BYTES = 25_000_000
EXCLUIDOS = {".git", ".agents"}
DIRS_PROHIBIDOS = {"__pycache__", ".venv", "venv", ".ipynb_checkpoints", ".pytest_cache"}
SUFIJOS_PROHIBIDOS = {".pyc", ".log", ".tmp", ".part"}
NOMBRES_PROHIBIDOS = {".env", "secrets.toml"}
SECRETO = re.compile(r"(ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
SUFIJOS_TEXTO = {".py", ".md", ".txt", ".toml", ".ini", ".json", ".j2", ".csv", ".gitignore", ".gitattributes"}


def recorrer() -> list[Path]:
    """Archivos a subir (excluye .git y .agents)."""
    return sorted(p for p in RAIZ.rglob("*") if p.is_file() and not (set(p.relative_to(RAIZ).parts) & EXCLUIDOS))


def problemas(archivos: list[Path]) -> list[str]:
    """Problemas que impiden la subida."""
    salida = []
    for p in archivos:
        rel = p.relative_to(RAIZ)
        partes = set(rel.parts)
        if partes & DIRS_PROHIBIDOS or p.suffix in SUFIJOS_PROHIBIDOS or p.name in NOMBRES_PROHIBIDOS:
            salida.append(f"residuo no permitido: {rel.as_posix()}")
        if rel.parts[:2] == ("data", "extra"):
            salida.append(f"data/extra no debe subirse: {rel.as_posix()}")
        if p.stat().st_size > LIMITE_BYTES:
            salida.append(f"supera 25 MB: {rel.as_posix()} ({p.stat().st_size:,} bytes)")
        if (p.suffix in SUFIJOS_TEXTO or p.name in SUFIJOS_TEXTO) and p.stat().st_size < 5_000_000:
            if SECRETO.search(p.read_text(encoding="utf-8", errors="ignore")):
                salida.append(f"posible secreto en {rel.as_posix()}")
    for d in RAIZ.rglob("*"):
        if d.is_dir() and d.name in DIRS_PROHIBIDOS and not (set(d.relative_to(RAIZ).parts) & EXCLUIDOS):
            salida.append(f"carpeta no permitida: {d.relative_to(RAIZ).as_posix()}")
    return sorted(set(salida))


def limpiar() -> None:
    """Borra cachés regenerables de Python/pytest/Jupyter (nunca .venv ni datos)."""
    import shutil
    for d in sorted(RAIZ.rglob("*"), reverse=True):
        if d.is_dir() and d.name in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"} \
                and not (set(d.relative_to(RAIZ).parts) & EXCLUIDOS):
            shutil.rmtree(d)
            print(f"Borrado: {d.relative_to(RAIZ).as_posix()}")


def main() -> int:
    """Imprime la lista de subida y los problemas."""
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Lista y valida los archivos a subir a GitHub.")
    parser.add_argument("--markdown", action="store_true", help="salida en tabla Markdown")
    parser.add_argument("--limpiar", action="store_true",
                        help="borra antes __pycache__, .pytest_cache y .ipynb_checkpoints (no toca .venv)")
    args = parser.parse_args()
    if args.limpiar:
        limpiar()
    archivos = recorrer()
    total = sum(p.stat().st_size for p in archivos)
    if args.markdown:
        print("| Archivo | Tamaño (bytes) |\n|---|---|")
    for p in archivos:
        rel, tam = p.relative_to(RAIZ).as_posix(), p.stat().st_size
        print(f"| `{rel}` | {tam:,} |".replace(",", ".") if args.markdown else f"{tam:>12,}  {rel}")
    print(f"\n{len(archivos)} archivos, {total:,} bytes en total; mayor: "
          f"{max(archivos, key=lambda p: p.stat().st_size).relative_to(RAIZ).as_posix()} "
          f"({max(p.stat().st_size for p in archivos):,} bytes)")
    errores = problemas(archivos)
    if errores:
        print("PROBLEMAS:")
        for e in errores:
            print(" -", e)
        return 5
    print("Lista de subida correcta: ningún archivo supera 25 MB y no hay residuos ni secretos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
