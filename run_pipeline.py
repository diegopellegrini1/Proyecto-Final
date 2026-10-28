"""Pipeline end-to-end: dos fuentes (Inside Airbnb, 9 ciudades de España + INE) → unión → limpieza →
features → análisis → figuras → findings → documentos → notebooks → verificación.

Uso (PowerShell, desde la raíz del repositorio):

    python run_pipeline.py                      # usa data/raw si sus hashes coinciden; si no, descarga
    python run_pipeline.py --sin-descarga       # exige data/raw ya presente (modo offline, sin red)
    python run_pipeline.py --forzar-descarga    # vuelve a descubrir y descargar las dos fuentes

Códigos de salida: 0 correcto; 1 error inesperado; 2 descarga; 3 datos; 4 informe;
5 verificación; 64 uso incorrecto de argumentos; 130 interrumpido.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Evita __pycache__ en la carpeta (se sube arrastrando archivos y .gitignore no aplica).
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.errores import ErrorDatos, ErrorPipeline, ErrorVerificacion  # noqa: E402

logger = logging.getLogger("pipeline")
EXIT_USO = 64
# Columnas cuyo % de nulos se compara por ciudad en el informe de calidad (antes y después).
COLUMNAS_CALIDAD: tuple[str, ...] = (
    "price", "bedrooms", "beds", "bathrooms", "bathrooms_text", "review_scores_rating", "license",
    "neighbourhood_group_cleansed", "host_is_superhost", "minimum_nights", config.RENTA_PRINCIPAL,
)


class _Parser(argparse.ArgumentParser):
    """ArgumentParser que sale con código 64 ante errores de uso."""

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
    p = _Parser(description="Pipeline del Proyecto Final: Airbnb en España (Inside Airbnb) + renta del INE.")
    p.add_argument("--semilla", type=int, default=config.SEMILLA, help="semilla de aleatoriedad")
    p.add_argument("--n-boot", type=_rango_n_boot, default=config.N_BOOT, help="remuestreos bootstrap")
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument("--forzar-descarga", action="store_true", help="vuelve a descargar data/raw")
    grupo.add_argument("--sin-descarga", action="store_true", help="no usa la red; exige data/raw válido")
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
    if obj is pd.NaT or obj is pd.NA:
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
                "seaborn", "pyarrow", "pyshp", "shapely", "pyproj", "openpyxl"]
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


def datos_locales_validos() -> bool:
    """True si las dos fuentes están en data/raw y sus hashes coinciden con los manifiestos."""
    from src import fuentes
    try:
        fuentes.verificar_manifiesto(config.DIR_RAW_AIRBNB)
        fuentes.verificar_manifiesto(config.DIR_RAW_INE)
    except ErrorDatos:
        return False
    return True


def leer_barrios_oficiales() -> dict[str, set[str]]:
    """Nombres oficiales de barrio o municipio de ``neighbourhoods.csv`` de cada ciudad."""
    salida = {}
    for clave in config.CIUDADES_ESPANA:
        ruta = config.DIR_RAW_AIRBNB / clave / "neighbourhoods.csv"
        salida[clave] = set(pd.read_csv(ruta)["neighbourhood"].dropna().astype(str))
    return salida


def fechas_descarga(manifiesto: dict[str, Any]) -> list[str]:
    """Primera y última fecha (AAAA-MM-DD) de descarga de los ficheros de un manifiesto."""
    fechas = sorted(f["descargado_utc"][:10] for f in manifiesto["ficheros"])
    return [fechas[0], fechas[-1]]


def ejecutar(args: argparse.Namespace) -> None:
    """Ejecuta todas las etapas del pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    from src import analisis, calidad, diccionario, features, fuentes, informe, limpieza, visualizacion

    tiempos: dict[str, float] = {}
    t_total = time.perf_counter()

    with etapa("1. Dos fuentes (Inside Airbnb + INE) y unión por sección censal", tiempos):
        sin_red = args.sin_descarga or (not args.forzar_descarga and datos_locales_validos())
        if sin_red and not args.sin_descarga:
            logger.info("data/raw está completo y sus hashes coinciden: no se usa la red.")
        unido, anio_ine = fuentes.construir_conjunto_unido(sin_descarga=sin_red, forzar=args.forzar_descarga)
        man_airbnb = fuentes.leer_manifiesto(config.DIR_RAW_AIRBNB)
        man_ine = fuentes.leer_manifiesto(config.DIR_RAW_INE)
        tasas = fuentes.tasas_emparejamiento(unido)
        concord = fuentes.resumen_concordancia(fuentes.concordancia(unido))

    with etapa("2. Perfil de calidad «antes» (global y por ciudad)", tiempos):
        perfil_antes = calidad.perfilar(unido)
        por_ciudad_antes = calidad.perfilar_por_ciudad(unido, COLUMNAS_CALIDAD)

    with etapa("3. Limpieza por ciudad", tiempos):
        df, registro = limpieza.limpiar(unido, leer_barrios_oficiales())
        del unido

    with etapa("4. Variables derivadas", tiempos):
        fecha_ref, scrape_min = df["last_scraped"].max(), df["last_scraped"].min()
        df, meta_features = features.agregar_features(df, fecha_ref)

    with etapa("5. Selección de columnas, requisitos (≥ 50.000 × 20) y guardado", tiempos):
        limpieza.validar_procesado(df)
        final = df[list(config.COLUMNAS_FINALES)].copy()
        sin_doc = sorted(set(final.columns) - diccionario.documentadas())
        if sin_doc:
            raise ErrorDatos(f"Columnas finales sin documentar en el diccionario: {sin_doc}")
        requisitos = calidad.comprobar_requisitos(final, perfil_antes)
        config.DIR_PROCESADO.mkdir(parents=True, exist_ok=True)
        final.to_parquet(config.RUTA_PARQUET, index=False, compression="zstd")
        if config.RUTA_PARQUET.stat().st_size >= config.LIMITE_PARQUET_BYTES:
            raise ErrorDatos(f"{config.RUTA_PARQUET.name} supera {config.LIMITE_PARQUET_BYTES} bytes.")
        muestra = final.sample(n=min(config.TAM_MUESTRA_CSV, len(final)), random_state=args.semilla)
        muestra.sort_values(["ciudad", "id"]).to_csv(config.RUTA_MUESTRA_CSV, index=False, encoding="utf-8",
                                                     lineterminator="\n")
        relectura = pd.read_parquet(config.RUTA_PARQUET)
        if relectura.shape != final.shape:
            raise ErrorPipeline("La relectura del Parquet no coincide con el DataFrame escrito.")
        del df

    with etapa("6. Perfil de calidad «después» (global y por ciudad)", tiempos):
        perfil_despues = calidad.perfilar(relectura)
        por_ciudad_despues = calidad.perfilar_por_ciudad(relectura, COLUMNAS_CALIDAD)
        informe_calidad = calidad.construir_informe_calidad(perfil_antes, perfil_despues, registro,
                                                            por_ciudad_antes, por_ciudad_despues)
        escribir_json(informe_calidad, config.RUTA_CALIDAD)

    with etapa("7. Análisis descriptivo y estadístico", tiempos):
        hallazgos, artefactos = analisis.ejecutar(relectura, args.semilla, args.n_boot, not args.sin_ml)

    with etapa("8. Figuras", tiempos):
        nombres_fig = visualizacion.generar_todas(relectura, hallazgos, artefactos, informe_calidad,
                                                  config.DIR_FIGURAS)
        problemas = visualizacion.validar_figuras(config.DIR_FIGURAS, nombres_fig)
        if problemas:
            raise ErrorPipeline("Figuras inválidas: " + "; ".join(problemas))
        for sobrante in config.DIR_FIGURAS.glob("*.png"):
            if sobrante.name not in set(nombres_fig):
                sobrante.unlink()

    tiempos_computo = dict(tiempos)
    with etapa("9. findings.json (con afirmaciones verificadas)", tiempos):
        snapshots = sorted(v["fecha_snapshot"] for v in man_airbnb["ciudades"].values())
        tabla_tasas = tasas.reset_index().rename(columns={"index": "ciudad"}).to_dict("records")
        findings = {
            "meta": {
                "proyecto": "Proyecto Final Máster Data Analytics: alojamientos de Airbnb en España y renta de la zona",
                "autor": "Diego Hernández Pellegrini", "github": "diegopellegrini1",
                "ambito": "España (9 ciudades y regiones de Inside Airbnb)",
                "ciudades": {c.clave: {"nombre": c.nombre, "tipo": c.tipo, "centro_nombre": c.centro_nombre,
                                       "fecha_snapshot": man_airbnb["ciudades"][c.clave]["fecha_snapshot"]}
                             for c in config.CIUDADES_ESPANA.values()},
                "snapshot_min": snapshots[0], "snapshot_max": snapshots[-1],
                "last_scraped_min": scrape_min, "last_scraped_max": fecha_ref, "fecha_referencia": fecha_ref,
                "anio_ine": anio_ine, "semilla": args.semilla, "alfa": config.ALFA, "n_boot": args.n_boot,
                "parametros": {
                    "min_n_grupo": config.MIN_N_GRUPO, "min_n_categoria_rara": config.MIN_N_CATEGORIA_RARA,
                    "min_n_zona_ranking": config.MIN_N_ZONA_RANKING, "iqr_k": config.IQR_K,
                    "min_nights_extremo": config.MIN_NIGHTS_EXTREMO,
                    "min_nights_larga_estancia": config.MIN_NIGHTS_LARGA_ESTANCIA, "n_shapiro": config.N_SHAPIRO,
                    "tam_muestra_csv": config.TAM_MUESTRA_CSV, "min_filas": config.MIN_FILAS_UNION,
                    "min_columnas": config.MIN_COLUMNAS_UNION, "n_folds_ml": config.N_FOLDS_ML,
                    "test_size_aleatorio": config.TEST_SIZE_ALEATORIO,
                    "min_n_categoria_rara_ml": config.MIN_N_CATEGORIA_RARA_ML, "n_figuras": len(nombres_fig),
                    "n_boot_correlacion": config.N_BOOT_CORRELACION,
                    "min_distritos_ecologico": config.MIN_DISTRITOS_ECOLOGICO,
                    "limite_subida_mb": config.LIMITE_SUBIDA_BYTES / 1e6,
                },
                "versiones": versiones(), "generado_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            },
            "dataset": {
                "filas_union": perfil_antes["filas"], "columnas_union": perfil_antes["columnas"],
                "filas_procesadas": int(len(relectura)), "columnas_procesadas": int(relectura.shape[1]),
                "tipos_union": perfil_antes["tipos_logicos"], "requisitos": requisitos,
                "n_columnas_ine_final": int(sum(c.startswith("ine_") for c in relectura.columns)),
                "bytes_parquet": config.RUTA_PARQUET.stat().st_size, "filas_muestra_csv": int(len(muestra)),
                "n_hosts": int(relectura["host_id"].nunique()),
                "fuente_airbnb": {
                    "pagina": man_airbnb["pagina_fuente"], "licencia": man_airbnb["licencia"],
                    "n_ficheros": len(man_airbnb["ficheros"]),
                    "bytes": int(sum(f["bytes"] for f in man_airbnb["ficheros"])),
                    "bytes_max": int(max(f["bytes"] for f in man_airbnb["ficheros"])),
                    "fechas_descarga": fechas_descarga(man_airbnb),
                },
                "fuente_ine": {
                    "pagina": man_ine["pagina_fuente"], "licencia": man_ine["licencia"], "cita": config.CITA_INE,
                    "actualizacion": config.ACTUALIZACION_ADRH,
                    "n_tablas": len(man_ine["ficheros"]),
                    "n_provincias": len({f["provincia"] for f in man_ine["ficheros"]}),
                    "bytes": int(sum(f["bytes"] for f in man_ine["ficheros"])),
                    "bytes_max": int(max(f["bytes"] for f in man_ine["ficheros"])),
                    "fechas_descarga": fechas_descarga(man_ine), "n_indicadores": len(config.INDICADORES_ADRH),
                    "cartografia_bytes": man_ine["cartografia"]["bytes"],
                },
                "union": {"tasas": tabla_tasas, "total": tabla_tasas[-1],
                          "concordancia": concord.reset_index().to_dict("records"),
                          "concordancia_total": concord.reset_index().to_dict("records")[-1],
                          "n_sin_emparejar": int(tasas.loc["TOTAL", "anuncios"] - tasas.loc["TOTAL", "con_distrito"]),
                          "bytes_asignacion": config.RUTA_ASIGNACION_SECCIONES.stat().st_size},
            },
            "moneda": registro.get("moneda", {}),
            "precio_naturaleza": registro.get("precio_naturaleza", {}),
            "calidad": {
                "antes": {k: perfil_antes[k] for k in ("filas", "columnas", "nulos_total", "pct_nulos_total",
                                                       "columnas_con_nulos", "columnas_100_nulas", "memoria_mb")},
                "despues": {k: perfil_despues[k] for k in ("filas", "columnas", "nulos_total", "pct_nulos_total",
                                                           "columnas_con_nulos", "memoria_mb")},
                "n_columnas_100_nulas": len(perfil_antes["columnas_100_nulas"]),
                "por_ciudad": informe_calidad["por_ciudad"],
            },
            "limpieza": {
                **{k: registro[k] for k in ("precio", "outliers", "minimum_nights", "barrios", "imputados",
                                            "duplicados", "reglas", "contabilidad_filas", "mojibake_reparado")},
                "coordenadas": {"fuera_bbox_o_nulas": registro["coordenadas"]["fuera_bbox_o_nulas"]},
                "n_columnas_eliminadas": len(registro["columnas_eliminadas"]),
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
        findings["afirmaciones_invariantes"] = sorted(informe.IDS_INVARIANTES)
        findings["afirmaciones_no_cumplidas"] = sorted(k for k, v in findings["afirmaciones"].items() if not v)
        escribir_json(findings, config.RUTA_FINDINGS)

    with etapa("10. Documentos (README, informe, diccionario)", tiempos):
        informe.escribir_documentos(findings, relectura)

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
