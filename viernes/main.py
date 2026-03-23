"""
Punto de entrada principal de VIERNES OS.

Este módulo inicializa los componentes núcleo, cognitivos y de
interacción, y ejecuta el bucle principal de renderizado AR basado
en la cámara del Mac mediante OpenCV.
"""

from __future__ import annotations

import sys
from typing import NoReturn

from viernes.core.ar_pipeline import ARPipeline
from viernes.utils.logger import get_logger


def bucle_principal() -> None:
    """
    Ejecuta el bucle principal de captura y renderizado AR.

    Se encarga de:
    - Inicializar la cámara.
    - Enviar los frames al pipeline de fusión / seguimiento / AR.
    - Mostrar la ventana AR con OpenCV.
    - Gestionar la salida limpia al pulsar 'q' o Ctrl+C.
    """
    logger = get_logger()
    logger.info("Arrancando VIERNES OS (solo ARPipeline)")

    pipeline = ARPipeline(logger=logger)
    logger.info("Componentes inicializados; iniciando ARPipeline")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado (Ctrl+C)")
    finally:
        logger.info("VIERNES OS apagado correctamente")


def main() -> NoReturn:
    """
    Función main que delega en el bucle principal.

    Define el punto de entrada del proceso y el código de salida.
    """
    try:
        bucle_principal()
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger = get_logger()
        logger.error(f"Error no controlado en main: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
