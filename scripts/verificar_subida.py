"""Comprueba que la carpeta está lista para subir a GitHub arrastrando archivos y propone lotes de subida.

Uso:
    python scripts/verificar_subida.py              # lista, problemas y lotes propuestos; exit 5 si hay problemas
    python scripts/verificar_subida.py --markdown   # la lista en tabla Markdown
    python scripts/verificar_subida.py --limpiar    # borra cachés (__pycache__, .pytest_cache) y verifica

No se suben (y no se listan): .git/, .agents/ y data/extra/ (cartografía de más de 25 MB, se descarga sola).
Falla si algún archivo a subir supera 25 MB (límite de subida web), si hay residuos (cachés, entornos
virtuales, logs, temporales) o patrones de secretos.

La subida web de GitHub admite un número limitado de archivos por vez: los lotes agrupan los archivos por
carpeta, no superan ``config.LOTE_SUBIDA_ARCHIVOS`` (100) archivos y, si una carpeta pesa más de
``config.LOTE_SUBIDA_BYTES``, se parte por subcarpetas (por ejemplo, cada ciudad de data/raw/inside_airbnb).
También comprueba que están los entregables (README, informe, Parquet, dashboard de Excel...).
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
LIMITE_LOTE_BYTES = config.LOTE_SUBIDA_BYTES
# Archivos que deben estar en la subida (entregables del enunciado que no se ven en una lista larga).
OBLIGATORIOS: tuple[Path, ...] = (
    config.RUTA_README, config.RUTA_INFORME, config.RUTA_FINDINGS, config.RUTA_PARQUET,
    config.RUTA_DASHBOARD_XLSX, config.RUTA_DASHBOARD_README, config.RUTA_VISTA_PREVIA_DASHBOARD,
    RAIZ / ".gitignore", RAIZ / ".gitattributes", RAIZ / "requirements.txt", RAIZ / "run_pipeline.py",
)
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
    presentes = set(archivos)
    for p in OBLIGATORIOS:
        if p not in presentes:
            salida.append(f"falta un entregable: {p.relative_to(RAIZ).as_posix()}")
    return sorted(set(salida))


def _peso(lista: list[Path]) -> int:
    return sum(p.stat().st_size for p in lista)


def _carpeta(p: Path, nivel: int) -> str:
    """Carpeta de la ruta relativa hasta ``nivel`` niveles ("(raíz)" para archivos de la raíz)."""
    partes = p.relative_to(RAIZ).parts[:-1]
    return "/".join(partes[:nivel]) if partes else "(raíz)"


def proponer_lotes(archivos: list[Path], limite: int = LIMITE_LOTE,
                   limite_bytes: int = LIMITE_LOTE_BYTES) -> list[tuple[str, list[Path]]]:
    """Agrupa los archivos por carpeta en lotes de como máximo ``limite`` archivos y, en lo posible,
    ``limite_bytes`` bytes. Si una carpeta no cabe, se parte por subcarpetas (hasta cuatro niveles) y, si aún
    no cabe, en trozos consecutivos. Las carpetas pequeñas se juntan en un mismo lote."""
    def cabe(lista: list[Path]) -> bool:
        return len(lista) <= limite and _peso(lista) <= limite_bytes

    def partir(nombre: str, lista: list[Path], nivel: int) -> list[tuple[str, list[Path]]]:
        if cabe(lista):
            return [(nombre, lista)]
        sub: dict[str, list[Path]] = {}
        for p in lista:
            sub.setdefault(_carpeta(p, nivel), []).append(p)
        if len(sub) > 1 and nivel <= 4:
            return [b for k, v in sorted(sub.items()) for b in partir(k, v, nivel + 1)]
        if nivel <= 4:
            return partir(nombre, lista, nivel + 1)
        trozos, actual = [], []
        for p in lista:
            if actual and not cabe(actual + [p]):
                trozos.append(actual)
                actual = []
            actual.append(p)
        trozos.append(actual)
        return [(f"{nombre} ({i}/{len(trozos)})", t) for i, t in enumerate(trozos, 1)]

    por_carpeta: dict[str, list[Path]] = {}
    for p in archivos:
        por_carpeta.setdefault(_carpeta(p, 1), []).append(p)
    bloques = [b for k, v in sorted(por_carpeta.items()) for b in partir(k, v, 2)]
    lotes: list[tuple[str, list[Path]]] = []
    for nombre, lista in bloques:
        if lotes and cabe(lotes[-1][1] + lista):
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
    print(f"\nLotes de subida propuestos (máximo {LIMITE_LOTE} archivos y, en lo posible, "
          f"{LIMITE_LOTE_BYTES / 1e6:.0f} MB cada uno, agrupados por carpeta):")
    for i, (nombre, lista) in enumerate(lotes, 1):
        peso = _peso(lista)
        aviso = "  [AVISO: lote pesado, la subida web puede tardar]" if peso > LIMITE_LOTE_BYTES / 2 else ""
        print(f"  Lote {i}: {nombre} — {len(lista)} archivos, {peso:,} bytes{aviso}")
        if args.markdown:
            for p in lista:
                print(f"      {p.relative_to(RAIZ).as_posix()}")
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
