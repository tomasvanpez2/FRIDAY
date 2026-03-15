"""
Logger central para VIERNES OS.

Proporciona una configuración unificada de logging con niveles
DEBUG/INFO/ERROR y marcas de tiempo de alta precisión basadas en
time.perf_counter().
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Final

# Momento de referencia para las marcas de tiempo relativas.
_INICIO: Final[float] = time.perf_counter()


class _HighPrecisionFormatter(logging.Formatter):
    """
    Formateador que añade una marca temporal de alta precisión.

    La marca representa el tiempo transcurrido desde el arranque del
    logger, medido con time.perf_counter().
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Inserta el tiempo relativo de alta precisión en el registro.

        :param record: Registro de logging a formatear.
        :return: Cadena formateada lista para salida.
        """
        elapsed = time.perf_counter() - _INICIO
        record.perf_time = f"{elapsed:0.6f}"
        return super().format(record)


def configurar_logger(nivel: int = logging.DEBUG) -> logging.Logger:
    """
    Configura y devuelve el logger central de VIERNES OS.

    Si el logger ya está configurado, simplemente se reutiliza.

    :param nivel: Nivel mínimo de log a registrar.
    :return: Instancia del logger configurado.
    """
    logger = logging.getLogger("viernes")
    if logger.handlers:
        return logger

    logger.setLevel(nivel)

    handler = logging.StreamHandler(sys.stdout)
    formato = "[%(perf_time)s] %(levelname)s - %(message)s"
    handler.setFormatter(_HighPrecisionFormatter(formato))

    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Devuelve el logger central ya configurado.

    Esta función es el punto de entrada recomendado para otros módulos.
    """
    return configurar_logger()

