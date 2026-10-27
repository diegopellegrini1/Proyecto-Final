"""Verificador independiente: README/informe ↔ findings.json, enlaces, placeholders, figuras y plantillas.

Alcance: comprueba la TRANSCRIPCIÓN (que los documentos reproducen findings.json y sus cifras
recalculadas), no que las interpretaciones sean ciertas frente a los datos.

Uso:
    python scripts/verificar_consistencia.py             # todas las comprobaciones (exit 0 si todo bien, 5 si no)
    python scripts/verificar_consistencia.py --autotest  # además altera una cifra en memoria y exige detectarla

Comprobaciones:
1. Re-render byte a byte de README, informe y diccionario desde findings.json (+ Parquet para el diccionario).
2. Cada cifra marcada ``<!--f:ruta:formato-->valor<!--/f-->`` coincide con findings.json reformateado,
   y las cifras clave (``CIFRAS_CLAVE``) aparecen marcadas en el documento correspondiente.
3. Sin placeholders ([completar], TODO, XXX, lorem, llaves de plantilla) en ningún .md del proyecto.
4. Los enlaces relativos de README e informe apuntan a archivos existentes.
5. Las figuras de findings.json existen, pesan > 5 KB y no están en blanco.
6. (--autotest) Una cifra alterada se detecta.
7. Lint de plantillas: ningún número literal fuera de bloques Jinja, código, URLs, rutas o la lista permitida.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.dont_write_bytecode = True

from src import config  # noqa: E402
from src.errores import ErrorPipeline  # noqa: E402

MARCADOR = re.compile(r"<!--f:([^:>]+):(\w+)-->(.*?)<!--/f-->")
PLACEHOLDERS = re.compile(r"(?i:\[completar\]|lorem)|\bTODO\b|\bXXX\b|\{\{|\}\}|\{%|%\}")
ENLACE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
EXCLUIR_DIRS = {".git", ".agents", ".venv", "venv", "data"}

# Cifras clave que deben aparecer (marcadas) en cada documento.
CIFRAS_CLAVE: dict[str, list[str]] = {
    "README.md": [
        "dataset.filas_raw", "dataset.columnas_raw", "descriptivo.precio.mediana", "descriptivo.precio.media",
        "descriptivo.precio.n", "limpieza.precio.pct_nulo", "estadistica.kruskal_tipo.tamano_efecto.valor",
        "estadistica.mann_whitney_superhost.tamano_efecto.valor", "estadistica.ols.r2_ajustado",
        "estadistica.ols.n_efectivo", "descriptivo.hosts.pct_anuncios_multianuncio", "meta.fecha_snapshot_url",
        "precio_naturaleza.pct_con_descuento", "estadistica.ml.cv.r2_media", "estadistica.ml.aleatorio.r2",
        "dataset.tipos_raw.identificador",
    ],
    "reports/informe.md": [
        "dataset.filas_raw", "dataset.columnas_raw", "dataset.filas_procesadas", "dataset.columnas_procesadas",
        "limpieza.precio.nulo", "limpieza.precio.pct_nulo", "limpieza.outliers.valla_inf", "limpieza.outliers.valla_sup",
        "limpieza.outliers.n_total", "limpieza.minimum_nights.extremo_mayor_365", "limpieza.duplicados.por_id",
        "descriptivo.precio.mediana", "descriptivo.precio.media", "descriptivo.precio.std", "descriptivo.precio.iqr",
        "descriptivo.precio.asimetria", "descriptivo.precio.curtosis_exceso",
        "descriptivo.correlaciones.con_precio.0.spearman", "estadistica.normalidad_precio.p_valor",
        "estadistica.mann_whitney_superhost.p_valor", "estadistica.mann_whitney_superhost.tamano_efecto.valor",
        "estadistica.kruskal_tipo.tamano_efecto.valor", "estadistica.kruskal_distrito.tamano_efecto.valor",
        "estadistica.chi2_tipo_superhost.tamano_efecto.valor", "estadistica.posthoc_distrito.n_pares_significativos",
        "estadistica.bootstrap.tipo.0.ic_inf", "estadistica.ols.r2_ajustado", "estadistica.ols.diagnostico.breusch_pagan_p",
        "estadistica.ols.vif_max.vif", "estadistica.ols.coef_superhost.pct",
        "precio_naturaleza.pct_con_descuento", "precio_naturaleza.pct_noches_igual_minimo",
        "estadistica.ols.f_cluster", "estadistica.ols.vif_completo_distancia", "estadistica.ml.cv.r2_media",
        "estadistica.ml.cv.r2_std", "estadistica.ml.aleatorio.r2", "estadistica.ols.top_distritos.0.mediana_bruta",
    ],
}

# Números permitidos en las plantillas (texto convencional, no resultados).
PERMITIDOS_PLANTILLA: tuple[str, ...] = (
    r"\bH[01]\b", r"IC 95 %", r"CC BY 4\.0", r"Python 3\.13", r"py -3\.13", r"25 MB", r"SHA-256", r"UTF-8",
    r"\bHC3\b", r"\b[QP]\d{1,2}\b", r"VIF > (5|10)", r"^\| \d+ \|", r"^#+ \d+\.", r"^\d+\. ", r"10 criterios",
)


def _leer(ruta: Path) -> str:
    """Lee un archivo de texto UTF-8."""
    return ruta.read_text(encoding="utf-8")


def verificar_marcadores(texto: str, findings: dict, obtener, formatear) -> tuple[list[str], set[str]]:
    """Comprueba cada cifra marcada contra findings. Devuelve (errores, rutas encontradas)."""
    errores, rutas = [], set()
    for m in MARCADOR.finditer(texto):
        ruta, nombre_fmt, mostrado = m.groups()
        rutas.add(ruta)
        try:
            esperado = formatear(obtener(findings, ruta), nombre_fmt)
        except ErrorPipeline as exc:
            errores.append(f"cifra con ruta inválida {ruta}: {exc}")
            continue
        if mostrado != esperado:
            errores.append(f"cifra discrepante en {ruta}: documento «{mostrado}» vs findings «{esperado}»")
    return errores, rutas


def placeholders(texto: str) -> list[str]:
    """Placeholders encontrados en un Markdown."""
    return sorted({m.group(0) for m in PLACEHOLDERS.finditer(texto)})


def enlaces_rotos(texto: str, base: Path) -> list[str]:
    """Enlaces relativos (inline o imagen) cuyo destino no existe, relativos a ``base``."""
    rotos = []
    for destino in ENLACE.findall(texto):
        if re.match(r"^(https?:|mailto:|#)", destino):
            continue
        ruta = destino.split("#", 1)[0]
        if ruta and not (base / ruta).exists():
            rotos.append(destino)
    return rotos


def lint_plantilla(texto: str) -> list[str]:
    """Números literales en una plantilla fuera de bloques permitidos (M6)."""
    t = re.sub(r"\{#.*?#\}", " ", texto, flags=re.S)
    t = re.sub(r"\{\{.*?\}\}", " ", t, flags=re.S)
    t = re.sub(r"\{%.*?%\}", " ", t, flags=re.S)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`\n]*`", " ", t)
    t = re.sub(r"\]\([^)]*\)", "]", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\S+\.(png|py|md|json|csv|gz|parquet|ipynb|txt|toml|geojson|j2)\b", " ", t)
    problemas = []
    for n, linea in enumerate(t.splitlines(), 1):
        limpia = linea
        for patron in PERMITIDOS_PLANTILLA:
            limpia = re.sub(patron, " ", limpia)
        if re.search(r"\d", limpia):
            problemas.append(f"línea {n}: {linea.strip()[:120]}")
    return problemas


def archivos_md() -> list[Path]:
    """Todos los .md del proyecto (excluye .git, .agents, venvs y data)."""
    return [p for p in RAIZ.rglob("*.md") if not (set(p.relative_to(RAIZ).parts) & EXCLUIR_DIRS)]


def verificar(autotest: bool = False) -> list[str]:
    """Ejecuta todas las comprobaciones y devuelve la lista de errores."""
    import pandas as pd

    from src import diccionario
    from src.informe import PLANTILLAS, formatear, obtener, render_todos
    from src.visualizacion import validar_figuras

    errores: list[str] = []
    findings = json.loads(_leer(config.RUTA_FINDINGS))

    # 1. Re-render
    df = pd.read_parquet(config.RUTA_PARQUET)
    for ruta, texto in render_todos(findings, diccionario.tabla(df)).items():
        if not ruta.exists():
            errores.append(f"[1] falta {ruta.relative_to(RAIZ)}")
        elif _leer(ruta) != texto:
            errores.append(f"[1] {ruta.relative_to(RAIZ)} no coincide con el render desde findings.json")

    # 2. Cifras marcadas y cifras clave
    textos = {nombre: _leer(RAIZ / nombre) for nombre in CIFRAS_CLAVE}
    for nombre, texto in textos.items():
        errs, rutas = verificar_marcadores(texto, findings, obtener, formatear)
        errores += [f"[2] {nombre}: {e}" for e in errs]
        if not rutas:
            errores.append(f"[2] {nombre}: no contiene cifras marcadas")
        faltan = [c for c in CIFRAS_CLAVE[nombre] if c not in rutas]
        errores += [f"[2] {nombre}: falta la cifra clave {c}" for c in faltan]

    # 3. Placeholders
    for md in archivos_md():
        for ph in placeholders(MARCADOR.sub("", _leer(md))):
            errores.append(f"[3] placeholder «{ph}» en {md.relative_to(RAIZ)}")

    # 4. Enlaces relativos
    for nombre, texto in textos.items():
        for roto in enlaces_rotos(texto, (RAIZ / nombre).parent):
            errores.append(f"[4] enlace roto en {nombre}: {roto}")

    # 5. Figuras
    errores += [f"[5] {p}" for p in validar_figuras(config.DIR_FIGURAS, findings["figuras"])]

    # 6. Autotest: una cifra alterada debe detectarse
    if autotest:
        texto = textos["README.md"]
        m = MARCADOR.search(texto)
        alterado = texto[:m.start(3)] + m.group(3) + "9" + texto[m.end(3):]
        errs, _ = verificar_marcadores(alterado, findings, obtener, formatear)
        if not errs:
            errores.append("[6] el autotest no detectó una cifra alterada")
        else:
            print(f"[6] autotest correcto: se detectó la alteración ({errs[0]})")

    # 7. Lint de plantillas
    for nombre in PLANTILLAS:
        for p in lint_plantilla(_leer(config.DIR_PLANTILLAS / nombre)):
            errores.append(f"[7] número literal en {nombre}, {p}")
    return errores


def main() -> int:
    """Punto de entrada."""
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Verifica la consistencia de documentos y resultados.")
    parser.add_argument("--autotest", action="store_true", help="comprueba que una cifra alterada se detecta")
    args = parser.parse_args()
    errores = verificar(args.autotest)
    if errores:
        print(f"VERIFICACIÓN FALLIDA: {len(errores)} problema(s)")
        for e in errores:
            print(" -", e)
        return 5
    print("Verificación de consistencia correcta: README, informe y diccionario coinciden con findings.json; "
          "sin placeholders, enlaces rotos ni números literales en plantillas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
