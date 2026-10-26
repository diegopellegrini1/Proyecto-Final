"""Jerarquía de errores del proyecto con su código de salida asociado."""

from __future__ import annotations


class ErrorPipeline(RuntimeError):
    """Error base del pipeline. Cada subclase define su código de salida."""

    codigo_salida: int = 1


class ErrorDescarga(ErrorPipeline):
    """Fallo irrecuperable al descargar los datos de Inside Airbnb (exit 2)."""

    codigo_salida = 2


class ErrorDatos(ErrorPipeline):
    """Datos ausentes, corruptos o que violan un invariante (exit 3)."""

    codigo_salida = 3


class ErrorInforme(ErrorPipeline):
    """Fallo al generar los documentos o afirmación cualitativa falsa (exit 4)."""

    codigo_salida = 4


class ErrorVerificacion(ErrorPipeline):
    """Discrepancia detectada por los verificadores (exit 5)."""

    codigo_salida = 5
