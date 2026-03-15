"""
Pipeline AR unificado para VIERNES OS.

Integra captura de cámara, seguimiento espacial y composición AR en
tiempo real, con HUD de métricas y controles de interacción básicos.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from viernes.core.ar_compositor import ARCompositor, Overlay
from viernes.core.sensor_fusion import CameraCapture
from viernes.core.spatial_tracker import SpatialTracker, TrackedObject, TRACKER_FRAME_HEIGHT, TRACKER_FRAME_WIDTH
from viernes.utils.logger import get_logger
from viernes.utils.timing import PipelineTimer


class ARPipeline:
    """
    Pipeline de realidad aumentada que combina captura, tracking y composición.
    """

    def __init__(self, logger: Optional[Any] = None) -> None:
        if logger is None:
            logger = get_logger()

        self._logger = logger
        self._camera = CameraCapture(logger=self._logger)
        self._tracker = SpatialTracker(logger=self._logger)
        self._compositor = ARCompositor(logger=self._logger)

        self._timer = PipelineTimer(
            nombre="ARPipeline",
            logger=self._logger.info,
        )

        self._frame_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._running = False
        self._debug_landmarks = False
        self._overlays_enabled = True
        self._tracking_low_res = False
        self._active_overlays: Dict[str, Overlay] = {}
        self.frame_height: int = TRACKER_FRAME_HEIGHT

    def _update_tracking_resolution(self, fps: float) -> None:
        if fps < 20.0 and not self._tracking_low_res:
            self._tracker.set_tracking_resolution(320, 240)
            self._tracking_low_res = True
            self._logger.info("Resolución de tracking reducida a 320x240")
        elif fps > 25.0 and self._tracking_low_res:
            self._tracker.set_tracking_resolution(TRACKER_FRAME_WIDTH, TRACKER_FRAME_HEIGHT)
            self._tracking_low_res = False
            self._logger.info("Resolución de tracking restaurada a 640x480")

    def _update_overlays(self, objetos: List[TrackedObject], frame_shape: Tuple[int, int, int]) -> None:
        _, anchura = frame_shape[:2]

        if not self._overlays_enabled:
            return

        detected_ids = {obj.id for obj in objetos}

        for obj in objetos:
            x, y, w, h = obj.bounding_box
            x = max(0, min(x, anchura - 1))
            y = max(0, y)
            w = max(1, min(w, anchura - x))
            h = max(1, h)

            anchor_x, anchor_y = self._tracker.get_anchor_point(obj)

            if obj.type == "HAND":
                box_id = f"{obj.id}_box"
                label_id = obj.id
                label_text = f"Mano — {obj.distance_estimate}"

                overlay_box = Overlay(
                    id=box_id,
                    content_type="BOX",
                    position=(x, y),
                    size=(w, h),
                    alpha=1.0,
                    color=(0, 255, 0),
                    text="",
                    anchor_type="WORLD",
                    fade_in_ms=0.0,
                )
                self._compositor.add_overlay(overlay_box)

                label_position = (anchor_x, anchor_y)

                if label_id in self._active_overlays:
                    self._compositor.update_anchor(label_id, label_position)
                    overlay = self._active_overlays[label_id]
                    overlay.position = label_position
                    overlay.text = label_text
                else:
                    overlay_label = Overlay(
                        id=label_id,
                        content_type="LABEL",
                        position=label_position,
                        size=(40, -40),
                        alpha=1.0,
                        color=(0, 255, 0),
                        text=label_text,
                        anchor_type="WORLD",
                        fade_in_ms=0.0,
                    )
                    self._compositor.add_overlay(overlay_label)
                    self._active_overlays[label_id] = overlay_label

            elif obj.type == "FACE":
                label_id = obj.id
                label_position = (anchor_x, max(0, anchor_y - int(h * 0.2)))

                if label_id in self._active_overlays:
                    self._compositor.update_anchor(label_id, label_position)
                    overlay = self._active_overlays[label_id]
                    overlay.position = label_position
                else:
                    overlay_label = Overlay(
                        id=label_id,
                        content_type="LABEL",
                        position=label_position,
                        size=(60, -60),
                        alpha=1.0,
                        color=(255, 255, 0),
                        text="VIERNES activo",
                        anchor_type="WORLD",
                        fade_in_ms=0.0,
                    )
                    self._compositor.add_overlay(overlay_label)
                    self._active_overlays[label_id] = overlay_label

            elif obj.type == "SURFACE":
                panel_id = obj.id
                panel_position = (x, y)
                panel_size = (w, h)

                if panel_id in self._active_overlays:
                    self._compositor.update_anchor(panel_id, panel_position)
                    overlay = self._active_overlays[panel_id]
                    overlay.position = panel_position
                    overlay.size = panel_size
                else:
                    overlay_panel = Overlay(
                        id=panel_id,
                        content_type="PANEL",
                        position=panel_position,
                        size=panel_size,
                        alpha=0.3,
                        color=(0, 255, 0),
                        text="",
                        anchor_type="WORLD",
                        fade_in_ms=0.0,
                    )
                    self._compositor.add_overlay(overlay_panel)
                    self._active_overlays[panel_id] = overlay_panel

        for old_id in list(self._active_overlays.keys()):
            if old_id not in detected_ids:
                self._compositor.remove_overlay(old_id)
                del self._active_overlays[old_id]
                if old_id.startswith("hand_"):
                    self._compositor.remove_overlay(f"{old_id}_box")

    def _draw_debug_landmarks(self, frame: np.ndarray, objetos: List[TrackedObject]) -> None:
        if not self._debug_landmarks:
            return

        for obj in objetos:
            if obj.landmarks is None:
                continue

            if obj.type in ("HAND", "FACE"):
                for punto in obj.landmarks:
                    cx = int(punto[0])
                    cy = int(punto[1])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
            elif obj.type == "SURFACE":
                for linea in obj.landmarks:
                    x1, y1, x2, y2 = linea
                    cv2.line(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 255),
                        1,
                    )

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        lat_capture_ms: Optional[float],
        lat_tracking_ms: float,
        lat_comp_ms: float,
        lat_total_ms: float,
        distancia: str,
    ) -> None:
        _, anchura = frame.shape[:2]
        x = anchura - 260
        y = 20
        dy = 20

        texto_fps = f"FPS: {fps:5.1f}"
        texto_cap = "Cap: n/a ms" if lat_capture_ms is None else f"Cap: {lat_capture_ms:5.1f} ms"
        texto_track = f"Track: {lat_tracking_ms:5.1f} ms"
        texto_comp = f"Comp: {lat_comp_ms:5.1f} ms"
        texto_total = f"Total: {lat_total_ms:5.1f} ms"
        texto_dist = f"Dist: {distancia}"

        for idx, texto in enumerate([texto_fps, texto_cap, texto_track, texto_comp, texto_total, texto_dist]):
            cv2.putText(
                frame,
                texto,
                (x, y + idx * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        if key == ord("q"):
            self._logger.info("Salida de ARPipeline solicitada por el usuario")
            return False
        if key == ord("s"):
            self._save_screenshot(frame)
        elif key == ord("d"):
            self._debug_landmarks = not self._debug_landmarks
            self._logger.info(f"Debug landmarks {'activado' if self._debug_landmarks else 'desactivado'}")
        elif key == ord("h"):
            self._overlays_enabled = not self._overlays_enabled
            self._logger.info(f"Overlays {'activados' if self._overlays_enabled else 'desactivados'}")
        return True

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._current_frame is None:
                return None
            return self._current_frame.copy()

    def show_response(self, texto: str) -> None:
        overlay = Overlay(
            id="viernes_response",
            content_type="TEXT",
            position=(20, self.frame_height - 80),
            size=(0, 0),
            alpha=1.0,
            color=(0, 255, 200),
            text=f"VIERNES: {texto}",
            anchor_type="FIXED",
            fade_in_ms=0.0,
        )
        self._compositor.add_overlay(overlay)
        threading.Timer(8.0, lambda: self._compositor.remove_overlay("viernes_response")).start()

    def _save_screenshot(self, frame: np.ndarray) -> None:
        base_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(base_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        nombre = os.path.join(base_dir, f"ar_pipeline_{ts}.png")
        ok = cv2.imwrite(nombre, frame)
        if ok:
            self._logger.info(f"Screenshot guardado en {nombre}")
        else:
            self._logger.error("No se pudo guardar el screenshot")

    def run(self) -> None:
        self._camera.start()
        self._running = True

        ultimo_tiempo = time.perf_counter()
        fps_actual = 0.0

        try:
            while self._running:
                with self._frame_lock:
                    frame, _, lat_cap_ms = self._camera.get_frame()
                    if frame is not None:
                        self._current_frame = frame.copy()
                        self.frame_height = frame.shape[0]

                if frame is None:
                    continue

                inicio_total = time.perf_counter()

                inicio_track = time.perf_counter()
                objetos = self._tracker.track(frame)
                fin_track = time.perf_counter()
                lat_track_ms = (fin_track - inicio_track) * 1000.0
                self._timer.record("tracking", lat_track_ms)

                inicio_comp = time.perf_counter()
                frame_base = frame.copy()
                self._update_overlays(objetos, frame_base.shape)
                output = self._compositor.compose(frame_base) if self._overlays_enabled else frame_base
                fin_comp = time.perf_counter()
                lat_comp_ms = (fin_comp - inicio_comp) * 1000.0
                self._timer.record("composition", lat_comp_ms)

                fin_total = time.perf_counter()
                lat_total_ms = (fin_total - inicio_total) * 1000.0
                self._timer.record("pipeline", lat_total_ms)

                dt = fin_total - ultimo_tiempo
                if dt > 0.0:
                    fps_actual = 1.0 / dt
                ultimo_tiempo = fin_total

                self._update_tracking_resolution(fps_actual)

                if lat_cap_ms is not None:
                    self._timer.record("capture", lat_cap_ms)

                hand_distances = [
                    obj.distance_estimate
                    for obj in objetos
                    if obj.type == "HAND" and obj.distance_estimate
                ]
                if hand_distances:
                    if "CERCA" in hand_distances:
                        distancia_label = "CERCA"
                    elif "MEDIA" in hand_distances:
                        distancia_label = "MEDIA"
                    else:
                        distancia_label = "LEJOS"
                else:
                    distancia_label = "N/A"

                self._draw_debug_landmarks(output, objetos)
                self._draw_hud(
                    output,
                    fps_actual,
                    lat_cap_ms,
                    lat_track_ms,
                    lat_comp_ms,
                    lat_total_ms,
                    distancia_label,
                )

                cv2.imshow("VIERNES OS - ARPipeline", output)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self._handle_key(key, output):
                        break
        except KeyboardInterrupt:
            self._logger.info("Interrupción por teclado durante ARPipeline.run()")
        finally:
            self._running = False
            self._camera.stop()
            cv2.destroyAllWindows()
            self._timer.report()


def _demo_ar_pipeline() -> None:
    logger = get_logger()
    pipeline = ARPipeline(logger=logger)
    pipeline.run()


if __name__ == "__main__":
    _demo_ar_pipeline()
