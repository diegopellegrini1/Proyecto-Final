"""Exporta a PNG algunos gráficos REALES del dashboard (no son capturas de pantalla).

Usa las mismas funciones de ``dashboard/graficos.py`` y los mismos filtros por defecto del dashboard
(precios hasta el P99 y sin estancias de 30 noches o más) sobre ``data/processed/listings_limpio.parquet``.
Escribe ``reports/figures/dashboard_*.png`` y un índice ``reports/figures/dashboard_graficos.json`` que
el pipeline lee para enlazarlos en el README.

Requisito opcional (no está en requirements.txt porque el dashboard no lo necesita):
    pip install kaleido==1.5.0
kaleido 1.x usa un navegador Chrome/Chromium instalado en el equipo.

Uso:  python scripts/exportar_graficos_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.dont_write_bytecode = True

import pandas as pd  # noqa: E402

from dashboard import filtros as fl, graficos as gr  # noqa: E402
from src import config  # noqa: E402

ANCHO, ALTO = 1100, 600


def graficos_por_defecto() -> list[tuple[str, str, object]]:
    """(archivo, título, figura) con los filtros por defecto del dashboard."""
    datos = pd.read_parquet(config.RUTA_PARQUET, columns=list(config.COLUMNAS_DASHBOARD))
    base = datos[datos["precio_valido"]]
    _, _, defecto = fl.limites_precio(base)
    filtrado = fl.aplicar_filtros(base, [], [], [], defecto, excluir_largas=True)
    p = config.PREFIJO_GRAFICOS_DASHBOARD
    return [
        (f"{p}01_histograma_precio.png", "Dashboard, pestaña Resumen: distribución del precio",
         gr.fig_histograma_precio(filtrado)),
        (f"{p}02_mediana_distrito.png", "Dashboard, pestaña Resumen: mediana del precio por distrito",
         gr.fig_barras_distrito(filtrado)),
        (f"{p}03_precio_tipo.png", "Dashboard, pestaña Precios: precio por tipo de habitación",
         gr.fig_box_tipo(filtrado)),
        (f"{p}04_precio_distancia.png", "Dashboard, pestaña Precios: precio frente a distancia al centro",
         gr.fig_dispersion_distancia(filtrado, config.MAX_PUNTOS_MAPA, config.SEMILLA)),
        (f"{p}05_amenities.png", "Dashboard, pestaña Anfitriones y calidad: amenities",
         gr.fig_amenities(filtrado)),
    ]


def main() -> int:
    """Exporta los gráficos y escribe el índice. Devuelve 5 si falta kaleido o falla la exportación."""
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        import kaleido  # noqa: F401
    except ImportError:
        print("Falta kaleido: pip install kaleido==1.5.0 (opcional; solo para exportar estos PNG).")
        return 5
    indice = []
    for archivo, titulo, figura in graficos_por_defecto():
        ruta = config.DIR_FIGURAS / archivo
        figura.write_image(ruta, width=ANCHO, height=ALTO, scale=1)
        indice.append({"archivo": archivo, "titulo": titulo})
        print(f"Exportado {ruta.relative_to(RAIZ).as_posix()} ({ruta.stat().st_size:,} bytes)")
    with open(config.DIR_FIGURAS / config.INDICE_GRAFICOS_DASHBOARD, "w", encoding="utf-8", newline="\n") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Vuelve a ejecutar python run_pipeline.py para enlazarlos en el README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
