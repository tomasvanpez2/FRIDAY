"""
Módulo de composición AR para VIERNES OS.

Se encarga de superponer elementos gráficos sobre el video de la cámara
para generar la vista de realidad aumentada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Overlay:
    """
    Representa un elemento gráfico superpuesto en la escena AR.

    Cada overlay define su tipo de contenido (texto, panel, caja o
    etiqueta), su posición, tamaño, color, transparencia y tipo de
    anclaje (fijo en pantalla o anclado al mundo).
    """

    id: str
    content_type: str  # TEXT, PANEL, BOX, LABEL
    position: Tuple[int, int]
    size: Tuple[int, int]
    alpha: float
    color: Tuple[int, int, int]
    text: str = ""
    anchor_type: str = "FIXED"  # FIXED o WORLD
    fade_in_ms: float = 0.0
    created_at: float = field(default_factory=lambda: time.perf_counter())


class ARCompositor:
    """
    Clase encargada de componer la imagen de realidad aumentada.

    Gestiona un conjunto de overlays y los dibuja sobre el frame de la
    cámara aplicando transparencias y animaciones de entrada suaves.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inicializa el compositor AR.

        :param logger: Instancia del logger central del sistema.
        """
        self._logger = logger
        self._logger.info("Módulo ARCompositor inicializado")

        self._overlays: Dict[str, Overlay] = {}

    def add_overlay(self, overlay: Overlay) -> None:
        """
        Registra un overlay activo en el sistema de composición.

        Si ya existe un overlay con el mismo id, se reemplaza.
        """
        self._overlays[overlay.id] = overlay
        self._logger.debug(f"Overlay añadido: {overlay.id} ({overlay.content_type})")

    def remove_overlay(self, id: str) -> None:
        """
        Elimina un overlay activo a partir de su identificador.
        """
        if id in self._overlays:
            del self._overlays[id]
            self._logger.debug(f"Overlay eliminado: {id}")

    def update_anchor(self, id: str, new_position: Tuple[int, int]) -> None:
        """
        Actualiza la posición (ancla) de un overlay en tiempo real.

        Esto permite mover elementos UI o anotaciones en función de la
        interacción o del seguimiento espacial.
        """
        overlay = self._overlays.get(id)
        if overlay is None:
            return

        overlay.position = new_position
        self._logger.debug(f"Overlay {id} movido a {new_position}")

    def _aplicar_fade(self, overlay: Overlay, ahora: float) -> float:
        """
        Calcula el factor de fade-in para un overlay.

        Devuelve un factor entre 0.0 y 1.0 que se multiplica por la
        alpha del overlay para suavizar su aparición.
        """
        if overlay.fade_in_ms <= 0.0:
            return max(0.0, min(1.0, overlay.alpha))

        transcurrido_ms = (ahora - overlay.created_at) * 1000.0
        factor = min(1.0, transcurrido_ms / overlay.fade_in_ms)
        alpha_efectiva = overlay.alpha * factor
        return max(0.0, min(1.0, alpha_efectiva))

    def compose(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja todos los overlays activos sobre el frame recibido.

        Para cada overlay se crea una capa auxiliar, se dibujan las
        geometrías correspondientes y se mezcla con el frame usando
        cv2.addWeighted para aplicar la transparencia deseada.
        """
        if frame is None:
            self._logger.error("Frame recibido es None en ARCompositor")
            return frame

        resultado = frame.copy()
        ahora = time.perf_counter()

        overlays: List[Overlay] = list(self._overlays.values())

        for overlay in overlays:
            alpha = self._aplicar_fade(overlay, ahora)
            if alpha <= 0.0:
                continue

            capa = resultado.copy()
            x, y = overlay.position
            w, h = overlay.size

            if overlay.content_type == "TEXT":
                # Para texto se dibuja primero un panel semitransparente
                # de fondo y luego el texto encima.
                texto = overlay.text
                (tw, th), baseline = cv2.getTextSize(
                    texto,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )
                x1, y1 = x, y - th - baseline
                x2, y2 = x + tw + 10, y + 10
                cv2.rectangle(
                    capa,
                    (x1, y1),
                    (x2, y2),
                    overlay.color,
                    thickness=-1,
                )
                cv2.putText(
                    capa,
                    texto,
                    (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            elif overlay.content_type == "PANEL":
                # Un panel es un rectángulo sólido que se mezcla con el
                # frame base para lograr un panel semitransparente.
                cv2.rectangle(
                    capa,
                    (x, y),
                    (x + w, y + h),
                    overlay.color,
                    thickness=-1,
                )

            elif overlay.content_type == "BOX":
                # BOX representa típicamente un bounding box de detección
                # de objeto, dibujado como contorno.
                cv2.rectangle(
                    capa,
                    (x, y),
                    (x + w, y + h),
                    overlay.color,
                    thickness=2,
                )

            elif overlay.content_type == "LABEL":
                # LABEL dibuja una línea y una etiqueta flotante, útil
                # para anotar puntos de interés en la escena.
                x2, y2 = x + w, y + h
                cv2.line(
                    capa,
                    (x, y),
                    (x2, y2),
                    overlay.color,
                    2,
                )

                texto = overlay.text
                (tw, th), baseline = cv2.getTextSize(
                    texto,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                )
                lx1, ly1 = x2, y2 - th - baseline
                lx2, ly2 = x2 + tw + 10, y2 + 10

                cv2.rectangle(
                    capa,
                    (lx1, ly1),
                    (lx2, ly2),
                    overlay.color,
                    thickness=-1,
                )
                cv2.putText(
                    capa,
                    texto,
                    (lx1 + 5, ly2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Mezcla de la capa con el resultado actual aplicando alpha.
            resultado = cv2.addWeighted(capa, alpha, resultado, 1.0 - alpha, 0)

        return resultado

    def componer(
        self,
        frame: np.ndarray,
        estado_seguimiento: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Método de compatibilidad con la versión anterior.

        Ignora el estado de seguimiento y delega en compose(frame).
        """
        return self.compose(frame)


def _demo_ar_compositor() -> None:
    """
    Demo de composición AR con overlays básicos.

    Utiliza la cámara del sistema para mostrar tres overlays:
    - Un texto con fondo semitransparente.
    - Un panel rectangular semitransparente.
    - Una etiqueta flotante con línea de anclaje.
    """
    from viernes.core.sensor_fusion import CameraCapture
    from viernes.utils.logger import get_logger

    logger = get_logger()
    cam = CameraCapture(logger=logger)
    compositor = ARCompositor(logger=logger)

    # Overlay de texto en la esquina superior izquierda.
    overlay_texto = Overlay(
        id="texto_demo",
        content_type="TEXT",
        position=(40, 80),
        size=(0, 0),
        alpha=0.9,
        color=(0, 0, 0),
        text="VIERNES OS - Overlay de texto",
        anchor_type="FIXED",
        fade_in_ms=500.0,
    )

    # Panel semitransparente en la parte inferior de la pantalla.
    overlay_panel = Overlay(
        id="panel_demo",
        content_type="PANEL",
        position=(40, 300),
        size=(400, 150),
        alpha=0.4,
        color=(0, 255, 0),
        text="",
        anchor_type="FIXED",
        fade_in_ms=800.0,
    )

    # Etiqueta flotante con línea de anclaje cerca del centro.
    overlay_label = Overlay(
        id="label_demo",
        content_type="LABEL",
        position=(300, 200),
        size=(80, 60),
        alpha=0.9,
        color=(0, 255, 255),
        text="Etiqueta flotante",
        anchor_type="FIXED",
        fade_in_ms=1000.0,
    )

    compositor.add_overlay(overlay_texto)
    compositor.add_overlay(overlay_panel)
    compositor.add_overlay(overlay_label)

    cam.start()
    logger.info("Demo de ARCompositor iniciada (pulsa 'q' para salir)")

    try:
        while True:
            frame, _, _ = cam.get_frame()
            if frame is None:
                continue

            frame_ar = compositor.compose(frame)
            cv2.imshow("VIERNES OS - ARCompositor demo", frame_ar)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                logger.info("Salida de demo de ARCompositor solicitada por el usuario")
                break

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado durante la demo de ARCompositor")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo_ar_compositor()

