"""
Módulo de manejo de gestos para VIERNES OS.

En el futuro utilizará modelos de visión por computador para detectar
gestos de manos y posturas corporales a partir del video.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GestureHandler:
    """
    Componente de alto nivel para interacción por gestos.

    Esta versión inicial sólo define el esqueleto de la interfaz.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inicializa el manejador de gestos.

        :param logger: Instancia del logger central del sistema.
        """
        self._logger = logger
        self._logger.info("Módulo GestureHandler inicializado")

    def actualizar(self, frame: np.ndarray) -> None:
        """
        Procesa el frame actual para detectar gestos.

        :param frame: Imagen en formato BGR procedente de OpenCV.
        """
        if frame is None:
            self._logger.error("Frame recibido es None en GestureHandler")
            return

        # Aquí se integrarían modelos de detección de manos y poses.
        self._logger.debug("GestureHandler.actualizar() llamado (stub).")

