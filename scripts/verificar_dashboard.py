"""Arranque real del dashboard: servidor Streamlit headless, HTTP 200 y log sin excepciones.

Uso:  python scripts/verificar_dashboard.py
Un HTTP 200 solo prueba que el servidor está vivo; la ejecución del script de la app (filtros, KPIs,
comparador, GeoJSON ausente…) se prueba con ``streamlit.testing.v1.AppTest`` en ``tests/test_dashboard.py``.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TIEMPO_MAX_S = 60
PATRONES_ERROR = ("Traceback", "Exception", "Error")


def puerto_libre() -> int:
    """Puerto TCP libre en 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_get(url: str, timeout: float = 5) -> tuple[int, str]:
    """GET sencillo; devuelve (código, cuerpo). (0, '') si no hay conexión."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return 0, ""


def detener(proc: subprocess.Popen) -> None:
    """Termina el proceso y su árbol (taskkill en Windows como último recurso)."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    """Arranca, sondea, revisa el log y detiene el servidor."""
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    puerto = puerto_libre()
    dir_tmp = Path(tempfile.mkdtemp(prefix="dashboard_log_"))
    ruta_log = dir_tmp / "streamlit.log"
    entorno = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    ok = False
    with open(ruta_log, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true",
             "--server.port", str(puerto), "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"],
            cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT, env=entorno)
        try:
            base = f"http://127.0.0.1:{puerto}"
            inicio = time.time()
            estado_salud = (0, "")
            while time.time() - inicio < TIEMPO_MAX_S:
                estado_salud = http_get(f"{base}/_stcore/health")
                if estado_salud[0] == 200:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            estado_raiz = http_get(f"{base}/")
            print(f"/_stcore/health -> HTTP {estado_salud[0]} «{estado_salud[1].strip()}»")
            print(f"/ -> HTTP {estado_raiz[0]}")
            time.sleep(2)
            ok = estado_salud[0] == 200 and estado_salud[1].strip() == "ok" and estado_raiz[0] == 200
        finally:
            detener(proc)
    texto_log = ruta_log.read_text(encoding="utf-8", errors="replace")
    errores_log = [l for l in texto_log.splitlines() if any(p in l for p in PATRONES_ERROR)]
    print("Log del servidor:", "sin excepciones" if not errores_log else f"{len(errores_log)} línea(s) con error")
    for linea in errores_log[:20]:
        print("  ", linea)
    sigue_vivo = http_get(f"http://127.0.0.1:{puerto}/_stcore/health", timeout=2)[0] != 0
    print("Servidor detenido:", not sigue_vivo)
    shutil.rmtree(dir_tmp, ignore_errors=True)
    if ok and not errores_log and not sigue_vivo:
        print("Dashboard verificado: arranca headless, responde HTTP 200 y se detiene limpiamente.")
        return 0
    print("VERIFICACIÓN DEL DASHBOARD FALLIDA")
    return 5


if __name__ == "__main__":
    sys.exit(main())
