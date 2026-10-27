"""Formateo numérico en convención es-ES (miles con punto, decimales con coma).

Es el único punto de redondeo del proyecto: los valores se guardan sin redondear
en ``findings.json`` y solo se redondean al renderizar documentos.
"""

from __future__ import annotations

import math
from typing import Any

_PERMUTAR = str.maketrans({",": ".", ".": ","})
VACIO = "–"


def _es_nulo(x: Any) -> bool:
    """True si ``x`` es None o NaN."""
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def numero(x: Any, decimales: int = 0) -> str:
    """Formatea ``x`` con ``decimales`` decimales en estilo es-ES."""
    if _es_nulo(x):
        return VACIO
    texto = f"{float(x):,.{decimales}f}"
    if texto.startswith("-") and float(texto.replace(",", "")) == 0:
        texto = texto[1:]
    return texto.translate(_PERMUTAR)


def n0(x: Any) -> str:
    """Entero con separador de miles: 1234567 -> '1.234.567'."""
    return numero(x, 0)


def n1(x: Any) -> str:
    """Un decimal: 12.34 -> '12,3'."""
    return numero(x, 1)


def n2(x: Any) -> str:
    """Dos decimales: 1234.5 -> '1.234,50'."""
    return numero(x, 2)


def n3(x: Any) -> str:
    """Tres decimales."""
    return numero(x, 3)


def n4(x: Any) -> str:
    """Cuatro decimales (p. ej. un tamaño de efecto cercano a un umbral: 0,0997)."""
    return numero(x, 4)


def pct1(x: Any) -> str:
    """Porcentaje (ya en escala 0-100) con un decimal: 12.34 -> '12,3 %'."""
    if _es_nulo(x):
        return VACIO
    return f"{numero(x, 1)} %"


def pct2(x: Any) -> str:
    """Porcentaje (ya en escala 0-100) con dos decimales."""
    if _es_nulo(x):
        return VACIO
    return f"{numero(x, 2)} %"


def eur0(x: Any) -> str:
    """Importe en euros sin decimales: 151 -> '151 €'."""
    if _es_nulo(x):
        return VACIO
    return f"{numero(x, 0)} €"


def eur2(x: Any) -> str:
    """Importe en euros con dos decimales: 151 -> '151,00 €'."""
    if _es_nulo(x):
        return VACIO
    return f"{numero(x, 2)} €"


def pvalor(p: Any) -> str:
    """p-valor: '< 0,001' si es menor que 0,001; si no, tres decimales."""
    if _es_nulo(p):
        return VACIO
    if float(p) < 0.001:
        return "< 0,001"
    return numero(p, 3)


def p_texto(p: Any) -> str:
    """p-valor con su signo de relación: 'p < 0,001' o 'p = 0,107'."""
    texto = pvalor(p)
    return f"p {texto}" if texto.startswith("<") else f"p = {texto}"


def signo(x: Any, decimales: int = 1) -> str:
    """Número con signo explícito: 3.2 -> '+3,2'; -1 -> '-1,0'."""
    if _es_nulo(x):
        return VACIO
    texto = numero(x, decimales)
    return texto if texto.startswith("-") or float(x) == 0 else f"+{texto}"


def signo_pct(x: Any) -> str:
    """Porcentaje con signo y un decimal: 12.3 -> '+12,3 %'."""
    if _es_nulo(x):
        return VACIO
    return f"{signo(x, 1)} %"


def fecha(x: Any) -> str:
    """Fecha ISO (``YYYY-MM-DD``) a partir de un texto o Timestamp ISO."""
    if _es_nulo(x):
        return VACIO
    return str(x)[:10]


FILTROS_JINJA = {
    "fecha": fecha,
    "n0": n0,
    "n1": n1,
    "n2": n2,
    "n3": n3,
    "n4": n4,
    "pct1": pct1,
    "pct2": pct2,
    "eur0": eur0,
    "eur2": eur2,
    "pvalor": pvalor,
    "p": p_texto,
    "signo": signo,
    "signo_pct": signo_pct,
}
