"""Pipeline end-to-end del proyecto: descarga → limpieza → features → análisis → figuras → findings → documentos.

Uso (PowerShell, desde la raíz del repositorio):

    python run_pipeline.py                      # ejecución completa
    python run_pipeline.py --sin-descarga       # exige data/raw ya presente (modo offline)
    python run_pipeline.py --incluir-reviews    # descarga además reviews.csv.gz a data/extra/

Códigos de salida: 0 correcto; 1 error inesperado; 2 descarga; 3 datos; 4 informe;
5 verificación; 64 uso incorrecto de argumentos; 130 interrumpido.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import os

# Evita __pycache__ en la carpeta (se sube arrastrando archivos y .gitignore no aplica).
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.errores import ErrorPipeline, ErrorVerificacion  # noqa: E402

logger = logging.getLogger("pipeline")
EXIT_USO = 64


class _Parser(argparse.ArgumentParser):
    """ArgumentParser que sale con código 64 ante errores de uso (N6)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_USO, f"Error de uso: {message}\n")


def _rango_n_boot(valor: str) -> int:
    """Valida --n-boot dentro de [N_BOOT_MIN, N_BOOT_MAX]."""
    n = int(valor)
    if not config.N_BOOT_MIN <= n <= config.N_BOOT_MAX:
        raise argparse.ArgumentTypeError(f"debe estar entre {config.N_BOOT_MIN} y {config.N_BOOT_MAX}")
    return n


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de línea de comandos."""
    p = _Parser(description="Pipeline del Proyecto Final Airbnb Madrid (Inside Airbnb).")
    p.add_argument("--ciudad", default=config.CIUDAD_POR_DEFECTO, choices=sorted(config.CIUDADES),
                   help="ciudad a analizar (solo Madrid está validada)")
    p.add_argument("--semilla", type=int, default=config.SEMILLA, help="semilla de aleatoriedad")
    p.add_argument("--n-boot", type=_rango_n_boot, default=config.N_BOOT, help="remuestreos bootstrap")
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument("--forzar-descarga", action="store_true", help="vuelve a descargar data/raw")
    grupo.add_argument("--sin-descarga", action="store_true", help="no usa la red; exige data/raw válido")
    p.add_argument("--incluir-reviews", action="store_true", help="descarga reviews.csv.gz a data/extra/")
    p.add_argument("--incluir-calendar", action="store_true", help="descarga calendar.csv.gz a data/extra/")
    p.add_argument("--sin-ml", action="store_true", help="omite el modelo de machine learning")
    p.add_argument("--sin-notebooks", action="store_true", help="no ejecuta los notebooks")
    p.add_argument("--sin-verificacion", action="store_true", help="no ejecuta la verificación de consistencia")
    return p


@contextmanager
def etapa(nombre: str, tiempos: dict[str, float]) -> Iterator[None]:
    """Registra inicio, fin y duración de una etapa."""
    logger.info("▶ %s", nombre)
    t0 = time.perf_counter()
    yield
    tiempos[nombre] = time.perf_counter() - t0
    logger.info("✔ %s (%.1f s)", nombre, tiempos[nombre])


def a_serializable(obj: Any) -> Any:
    """Convierte recursivamente tipos numpy/pandas a JSON (NaN → null, Timestamp → ISO)."""
    if isinstance(obj, dict):
        return {str(k): a_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_serializable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, np.ndarray):
        return a_serializable(obj.tolist())
    return obj


def escribir_json(datos: dict[str, Any], ruta: Path) -> None:
    """Escribe JSON UTF-8 indentado y estricto (sin NaN)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        json.dump(a_serializable(datos), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def versiones() -> dict[str, str]:
    """Versiones de Python y de las librerías principales."""
    import importlib.metadata as md
    paquetes = ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "scikit-posthocs", "matplotlib",
                "seaborn", "streamlit", "plotly", "pyarrow"]
    v = {"python": platform.python_version()}
    for p in paquetes:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = "no instalado"
    return v


def _ejecutar_script(nombre: str, *args: str) -> None:
    """Ejecuta un script de ``scripts/`` con el mismo intérprete; lanza ErrorVerificacion si falla."""
    ruta = config.RAIZ / "scripts" / nombre
    if not ruta.exists():
        raise ErrorVerificacion(f"No existe scripts/{nombre}.")
    res = subprocess.run([sys.executable, str(ruta), *args], cwd=config.RAIZ)
    if res.returncode != 0:
        raise ErrorVerificacion(f"scripts/{nombre} terminó con código {res.returncode}.")


def ejecutar(args: argparse.Namespace) -> None:
    """Ejecuta todas las etapas del pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    from src import analisis, calidad, carga, diccionario, features, informe, limpieza, visualizacion

    ciudad = config.CIUDADES[args.ciudad]
    tiempos: dict[str, float] = {}
    t_total = time.perf_counter()

    with etapa("1. Datos originales (descarga o verificación offline)", tiempos):
        manifiesto = carga.asegurar_datos_raw(ciudad, forzar=args.forzar_descarga, sin_descarga=args.sin_descarga)
        extras = [c for c, flag in (("reviews", args.incluir_reviews), ("calendar", args.incluir_calendar)) if flag]
        manifiesto = carga.asegurar_extras(ciudad, manifiesto, extras)

    with etapa("2. Lectura y perfil de calidad «antes»", tiempos):
        df_raw = carga.leer_listings()
        barrios = carga.leer_barrios()
        geojson = carga.leer_geojson()
        perfil_antes = calidad.perfilar(df_raw)
        requisitos = calidad.comprobar_requisitos(perfil_antes)
        if not (requisitos["cumple_filas"] and requisitos["cumple_columnas"] and requisitos["cumple_tipos"]):
            logger.warning("El dataset no cumple todos los umbrales mínimos: %s", requisitos)

    with etapa("3. Limpieza", tiempos):
        df, registro = limpieza.limpiar(df_raw, barrios, ciudad)

    with etapa("4. Variables derivadas", tiempos):
        fecha_ref = df["last_scraped"].max()
        df, meta_features = features.agregar_features(df, ciudad, fecha_ref)

    with etapa("5. Validación y guardado del procesado", tiempos):
        sin_doc = limpieza.validar_procesado(df, diccionario.documentadas())
        config.DIR_PROCESADO.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.RUTA_PARQUET, index=False, compression="zstd")
        muestra = df.sample(n=min(config.TAM_MUESTRA_CSV, len(df)), random_state=args.semilla).sort_values("id")
        muestra.to_csv(config.RUTA_MUESTRA_CSV, index=False, encoding="utf-8", lineterminator="\n")
        relectura = pd.read_parquet(config.RUTA_PARQUET)
        if relectura.shape != df.shape:
            raise ErrorPipeline("La relectura del Parquet no coincide con el DataFrame escrito.")

    with etapa("6. Perfil de calidad «después»", tiempos):
        perfil_despues = calidad.perfilar(df)
        informe_calidad = calidad.construir_informe_calidad(perfil_antes, perfil_despues, registro)
        escribir_json(informe_calidad, config.RUTA_CALIDAD)

    with etapa("7. Análisis descriptivo y estadístico", tiempos):
        hallazgos, artefactos = analisis.ejecutar(df, args.semilla, args.n_boot, not args.sin_ml,
                                                  ciudad.distrito_referencia)

    with etapa("8. Figuras", tiempos):
        nombres_fig = visualizacion.generar_todas(
            df, hallazgos, artefactos, geojson, perfil_antes, ciudad.centro, ciudad.centro_nombre,
            config.DIR_FIGURAS, config.MIN_N_GRUPO, config.MIN_N_BARRIO_MAPA)
        problemas = visualizacion.validar_figuras(config.DIR_FIGURAS, nombres_fig)
        if problemas:
            raise ErrorPipeline("Figuras inválidas: " + "; ".join(problemas))
        esperadas = set(nombres_fig)
        for sobrante in config.DIR_FIGURAS.glob("*.png"):
            if sobrante.name not in esperadas:
                sobrante.unlink()

    tiempos_computo = dict(tiempos)
    with etapa("9. findings.json (con afirmaciones verificadas)", tiempos):
        findings = {
            "meta": {
                "proyecto": "Proyecto Final Máster Data Analytics: EDA y dashboard de Airbnb en Madrid",
                "autor": "Diego Hernández Pellegrini", "github": "diegopellegrini1",
                "ciudad": ciudad.nombre, "centro_nombre": ciudad.centro_nombre, "centro": list(ciudad.centro),
                "fecha_snapshot_url": manifiesto["fecha_snapshot"],
                "last_scraped_min": df["last_scraped"].min(), "last_scraped_max": df["last_scraped"].max(),
                "fecha_referencia": fecha_ref, "semilla": args.semilla, "alfa": config.ALFA, "n_boot": args.n_boot,
                "parametros": {
                    "min_n_grupo": config.MIN_N_GRUPO, "min_n_categoria_rara": config.MIN_N_CATEGORIA_RARA,
                    "min_n_barrio_mapa": config.MIN_N_BARRIO_MAPA, "top_n_distritos": config.TOP_N_DISTRITOS,
                    "iqr_k": config.IQR_K, "min_nights_extremo": config.MIN_NIGHTS_EXTREMO,
                    "min_nights_larga_estancia": config.MIN_NIGHTS_LARGA_ESTANCIA, "n_shapiro": config.N_SHAPIRO,
                    "tam_muestra_csv": config.TAM_MUESTRA_CSV, "max_puntos_mapa": config.MAX_PUNTOS_MAPA,
                    "umbral_filas": config.UMBRAL_FILAS, "umbral_columnas": config.UMBRAL_COLUMNAS,
                    "test_size_ml": 0.2, "n_figuras": len(nombres_fig),
                },
                "versiones": versiones(), "generado_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            },
            "dataset": {
                "filas_raw": perfil_antes["filas"], "columnas_raw": perfil_antes["columnas"],
                "filas_procesadas": int(len(df)), "columnas_procesadas": int(df.shape[1]),
                "tipos_raw": perfil_antes["tipos_logicos"], "requisitos": requisitos,
                "fuente": {f["archivo"]: f["url"] for f in manifiesto["ficheros"]},
                "bytes_raw": {f["archivo"]: f["bytes"] for f in manifiesto["ficheros"]},
                "licencia": manifiesto["licencia"], "origen_urls": manifiesto["origen_urls"],
                "pagina_fuente": manifiesto["pagina_fuente"],
                "n_distritos": registro["barrios"]["distritos"], "n_barrios": registro["barrios"]["barrios_con_anuncios"],
                "n_hosts": int(df["host_id"].nunique()),
                "bytes_parquet": config.RUTA_PARQUET.stat().st_size, "filas_muestra_csv": int(len(muestra)),
                "columnas_sin_documentar": sin_doc,
            },
            "moneda": registro.get("moneda", {}),
            "precio_naturaleza": registro.get("precio_naturaleza", {}),
            "calidad": {
                "antes": {k: perfil_antes[k] for k in ("filas", "columnas", "nulos_total", "pct_nulos_total",
                                                       "columnas_con_nulos", "columnas_100_nulas", "top_nulos",
                                                       "dtypes", "memoria_mb")},
                "despues": {k: perfil_despues[k] for k in ("filas", "columnas", "nulos_total", "pct_nulos_total",
                                                           "columnas_con_nulos", "dtypes", "memoria_mb")},
                "n_columnas_100_nulas": len(perfil_antes["columnas_100_nulas"]),
            },
            "limpieza": {
                **{k: registro[k] for k in ("precio", "outliers", "minimum_nights", "coordenadas", "barrios",
                                            "imputados", "duplicados", "columnas_eliminadas",
                                            "columnas_reconstruidas", "reglas", "contabilidad_filas")},
                "n_columnas_eliminadas": len(registro["columnas_eliminadas"]),
                "n_columnas_derivadas": int(df.shape[1] - (perfil_antes["columnas"] - len(registro["columnas_eliminadas"]))),
                "amenities_malformadas": registro["conversiones"]["amenities"]["malformadas_json"],
            },
            "features": {**meta_features, "prevalencia_amenities": registro["prevalencia_amenities"],
                         "n_flags_amenities": len(config.PATRONES_AMENITIES)},
            **hallazgos,
            "figuras": nombres_fig,
            "rendimiento": {"etapas_s": tiempos_computo, "total_computo_s": float(sum(tiempos_computo.values()))},
        }
        findings = a_serializable(findings)
        findings["afirmaciones"] = informe.evaluar_afirmaciones(findings)
        escribir_json(findings, config.RUTA_FINDINGS)

    with etapa("10. Documentos (README, informe, diccionario)", tiempos):
        informe.escribir_documentos(findings, df)

    if not args.sin_notebooks:
        with etapa("11. Notebooks ejecutados", tiempos):
            _ejecutar_script("ejecutar_notebooks.py")
    if not args.sin_verificacion:
        with etapa("12. Verificación de consistencia", tiempos):
            _ejecutar_script("verificar_consistencia.py")
    logger.info("Pipeline completado en %.1f s (cómputo etapas 1-8: %.1f s).",
                time.perf_counter() - t_total, sum(tiempos_computo.values()))


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada; devuelve el código de salida."""
    # En consola interactiva Windows ya escribe Unicode; redirigido, se fuerza UTF-8.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure") and not flujo.isatty():
            flujo.reconfigure(encoding="utf-8")
    args = construir_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        ejecutar(args)
        return 0
    except ErrorPipeline as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return exc.codigo_salida
    except KeyboardInterrupt:
        logger.error("Interrumpido por el usuario.")
        return 130
    except Exception:  # noqa: BLE001
        logger.exception("Error inesperado")
        return 1


if __name__ == "__main__":
    sys.exit(main())
