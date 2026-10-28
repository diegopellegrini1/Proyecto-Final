"""Comprueba que la carpeta está lista para subir a GitHub arrastrando archivos y propone lotes de subida.

Uso:
    python scripts/verificar_subida.py              # lista, problemas y lotes propuestos; exit 5 si hay problemas
    python scripts/verificar_subida.py --markdown   # la lista en tabla Markdown
    python scripts/verificar_subida.py --limpiar    # borra cachés (__pycache__, .pytest_cache) y verifica

No se suben (y no se listan): .git/, .agents/ y data/extra/ (cartografía de más de 25 MB, se descarga sola).
Falla si algún archivo a subir supera 25 MB (límite de subida web), si hay residuos (cachés, entornos
virtuales, logs, temporales) o patrones de secretos.

La subida web de GitHub admite un número limitado de archivos por vez: los lotes agrupan los archivos por
carpeta y no superan ``config.LOTE_SUBIDA_ARCHIVOS`` (100) archivos cada uno.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.dont_write_bytecode = True

from src import config  # noqa: E402

LIMITE_BYTES = config.LIMITE_SUBIDA_BYTES  # fuente de verdad única (límite de subida web de GitHub)
LIMITE_LOTE = config.LOTE_SUBIDA_ARCHIVOS
EXCLUIDOS_DIRS = {".git", ".agents"}
EXCLUIDOS_PREFIJOS = (("data", "extra"),)
DIRS_PROHIBIDOS = {"__pycache__", ".venv", "venv", ".ipynb_checkpoints", ".pytest_cache"}
SUFIJOS_PROHIBIDOS = {".pyc", ".log", ".tmp", ".part"}
NOMBRES_PROHIBIDOS = {".env", "secrets.toml"}
SECRETO = re.compile(r"(ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
SUFIJOS_TEXTO = {".py", ".md", ".txt", ".toml", ".ini", ".json", ".j2", ".csv", ".gitignore", ".gitattributes"}


def excluido(rel: Path) -> bool:
    """True si la ruta relativa no se sube (.git, .agents o data/extra)."""
    return bool(set(rel.parts) & EXCLUIDOS_DIRS) or any(rel.parts[:len(p)] == p for p in EXCLUIDOS_PREFIJOS)


def recorrer() -> list[Path]:
    """Archivos a subir."""
    return sorted(p for p in RAIZ.rglob("*") if p.is_file() and not excluido(p.relative_to(RAIZ)))


def problemas(archivos: list[Path]) -> list[str]:
    """Problemas que impiden la subida."""
    salida = []
    for p in archivos:
        rel = p.relative_to(RAIZ)
        if set(rel.parts) & DIRS_PROHIBIDOS or p.suffix in SUFIJOS_PROHIBIDOS or p.name in NOMBRES_PROHIBIDOS:
            salida.append(f"residuo no permitido: {rel.as_posix()}")
        if p.stat().st_size > LIMITE_BYTES:
            salida.append(f"supera {LIMITE_BYTES / 1e6:.0f} MB: {rel.as_posix()} ({p.stat().st_size:,} bytes)")
        if (p.suffix in SUFIJOS_TEXTO or p.name in SUFIJOS_TEXTO) and p.stat().st_size < 5_000_000:
            if SECRETO.search(p.read_text(encoding="utf-8", errors="ignore")):
                salida.append(f"posible secreto en {rel.as_posix()}")
    for d in RAIZ.rglob("*"):
        if d.is_dir() and d.name in DIRS_PROHIBIDOS and not excluido(d.relative_to(RAIZ)):
            salida.append(f"carpeta no permitida: {d.relative_to(RAIZ).as_posix()}")
    return sorted(set(salida))


def proponer_lotes(archivos: list[Path], limite: int = LIMITE_LOTE) -> list[tuple[str, list[Path]]]:
    """Agrupa los archivos por carpeta (la carpeta de primer nivel o, si no cabe, la de segundo nivel)
    en lotes de como máximo ``limite`` archivos. Las carpetas pequeñas se juntan en un mismo lote."""
    por_carpeta: dict[str, list[Path]] = {}
    for p in archivos:
        rel = p.relative_to(RAIZ)
        clave = rel.parts[0] if len(rel.parts) > 1 else "(raíz)"
        por_carpeta.setdefault(clave, []).append(p)
    bloques: list[tuple[str, list[Path]]] = []
    for carpeta, lista in sorted(por_carpeta.items()):
        if len(lista) <= limite:
            bloques.append((carpeta, lista))
            continue
        sub: dict[str, list[Path]] = {}
        for p in lista:
            partes = p.relative_to(RAIZ).parts
            sub.setdefault("/".join(partes[:2]) if len(partes) > 2 else carpeta, []).append(p)
        for nombre, trozo in sorted(sub.items()):
            for i in range(0, len(trozo), limite):
                bloques.append((nombre, trozo[i:i + limite]))
    lotes: list[tuple[str, list[Path]]] = []
    for nombre, lista in bloques:
        if lotes and len(lotes[-1][1]) + len(lista) <= limite:
            lotes[-1] = (f"{lotes[-1][0]} + {nombre}", lotes[-1][1] + lista)
        else:
            lotes.append((nombre, list(lista)))
    return lotes


def limpiar() -> None:
    """Borra cachés regenerables de Python/pytest/Jupyter (nunca .venv ni datos)."""
    import shutil
    for d in sorted(RAIZ.rglob("*"), reverse=True):
        if d.is_dir() and d.name in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"} \
                and not excluido(d.relative_to(RAIZ)):
            shutil.rmtree(d)
            print(f"Borrado: {d.relative_to(RAIZ).as_posix()}")


def main() -> int:
    """Imprime la lista de subida, los problemas y los lotes propuestos."""
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
    mayor = max(archivos, key=lambda p: p.stat().st_size)
    print(f"\nTotal: {len(archivos)} archivos, {total:,} bytes; mayor: {mayor.relative_to(RAIZ).as_posix()} "
          f"({mayor.stat().st_size:,} bytes). No se suben: .git/, .agents/ y data/extra/.")
    lotes = proponer_lotes(archivos)
    print(f"\nLotes de subida propuestos (máximo {LIMITE_LOTE} archivos cada uno, agrupados por carpeta):")
    for i, (nombre, lista) in enumerate(lotes, 1):
        print(f"  Lote {i}: {nombre} — {len(lista)} archivos, {sum(p.stat().st_size for p in lista):,} bytes")
    assert sum(len(lista) for _, lista in lotes) == len(archivos)
    errores = problemas(archivos)
    if errores:
        print("PROBLEMAS:")
        for e in errores:
            print(" -", e)
        return 5
    print(f"Lista de subida correcta: ningún archivo supera {LIMITE_BYTES / 1e6:.0f} MB y no hay residuos ni secretos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
