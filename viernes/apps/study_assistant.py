"""
Aplicación de asistente de estudio para VIERNES OS.

Esta aplicación se apoya en la vista AR y en los módulos cognitivos
para ayudar al usuario durante sesiones de estudio.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class StudyAssistant:
    """
    Componente de alto nivel para la lógica del asistente de estudio.

    En esta versión inicial sólo registra la actividad sobre los frames.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inicializa el asistente de estudio.

        :param logger: Instancia del logger central del sistema.
        """
        self._logger = logger
        self._logger.info("Aplicación StudyAssistant inicializada")

    def actualizar(self, frame_ar: np.ndarray) -> None:
        """
        Actualiza la lógica de la aplicación a partir del frame AR.

        :param frame_ar: Imagen ya compuesta con elementos de AR.
        """
        if frame_ar is None:
            self._logger.error("Frame AR recibido es None en StudyAssistant")
            return

        # Aquí se podrían extraer elementos relevantes para el estudio.
        self._logger.debug("StudyAssistant.actualizar() llamado (stub).")

