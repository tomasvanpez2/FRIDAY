"""
Módulo de seguimiento espacial para VIERNES OS.

Este módulo estima la posición y orientación de la cámara en el espacio
utilizando la información visual disponible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


TRACKER_FRAME_WIDTH: int = 640
TRACKER_FRAME_HEIGHT: int = 480


@dataclass
class TrackedObject:
    id: str
    type: str
    bounding_box: Tuple[int, int, int, int]
    landmarks: Optional[np.ndarray]
    confidence: float
    world_position_estimate: Optional[np.ndarray]
    distance_estimate: str = ""


class SpatialTracker:
    """
    Clase responsable del seguimiento de características en la escena.

    Utiliza MediaPipe para detectar manos, rostros y superficies
    aproximadas en la imagen de la cámara.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inicializa el tracker espacial.

        :param logger: Instancia del logger central del sistema.
        """
        self._logger = logger
        self._logger.info("Módulo SpatialTracker inicializado")

        self._width: int = TRACKER_FRAME_WIDTH
        self._height: int = TRACKER_FRAME_HEIGHT
        self._scale_factor: float = 1.0

        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )
        self._mp_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=0,
        )

        self._lat_min_ms: Optional[float] = None
        self._lat_max_ms: Optional[float] = None
        self._lat_sum_ms: float = 0.0
        self._lat_count: int = 0
        self._last_latency_ms: Optional[float] = None

    def set_tracking_resolution(self, width: int, height: int) -> None:
        """
        Ajusta la resolución interna usada para el tracking.
        """
        if width <= 0 or height <= 0:
            return
        self._width = int(width)
        self._height = int(height)

    def track(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Realiza el seguimiento espacial sobre un frame BGR.

        Detecta manos, rostros y superficies planas aproximadas y
        devuelve una lista de objetos rastreados ordenados por
        confianza descendente.

        :param frame: Imagen en formato BGR procedente de OpenCV.
        :return: Lista de TrackedObject detectados.
        """
        if frame is None:
            self._logger.error("Frame recibido es None en SpatialTracker.track()")
            return []

        inicio = time.perf_counter()

        h_src, w_src = frame.shape[:2]
        target_w = int(self._width * self._scale_factor)
        target_h = int(self._height * self._scale_factor)

        if w_src != target_w or h_src != target_h:
            frame_resized = cv2.resize(
                frame,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_resized = frame

        h, w = frame_resized.shape[:2]

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        manos = self._mp_hands.process(frame_rgb)
        caras = self._mp_face.process(frame_rgb)
        segmentacion = self._mp_segmentation.process(frame_rgb)

        objetos: List[TrackedObject] = []

        objetos.extend(self._procesar_manos(manos, w, h))
        objetos.extend(self._procesar_caras(caras, w, h))
        objetos.extend(self._procesar_superficies(frame_resized, segmentacion))

        objetos.sort(key=lambda o: o.confidence, reverse=True)

        fin = time.perf_counter()
        latencia_ms = (fin - inicio) * 1000.0
        self._last_latency_ms = latencia_ms
        self._lat_sum_ms += latencia_ms
        self._lat_count += 1
        if self._lat_min_ms is None or latencia_ms < self._lat_min_ms:
            self._lat_min_ms = latencia_ms
        if self._lat_max_ms is None or latencia_ms > self._lat_max_ms:
            self._lat_max_ms = latencia_ms

        if latencia_ms > 30.0:
            self._logger.warning(f"Latencia de tracking alta: {latencia_ms:.3f} ms")

        return objetos

    def _procesar_manos(
        self,
        resultados: Any,
        width: int,
        height: int,
    ) -> List[TrackedObject]:
        objetos: List[TrackedObject] = []

        if resultados is None or resultados.multi_hand_landmarks is None:
            return objetos

        distancias: List[str] = []

        scores: List[float] = []
        if resultados.multi_handedness:
            for h_info in resultados.multi_handedness:
                if h_info.classification:
                    scores.append(h_info.classification[0].score)

        for idx, landmarks in enumerate(resultados.multi_hand_landmarks):
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            if not xs or not ys:
                continue

            x_min = max(int(min(xs) * width), 0)
            y_min = max(int(min(ys) * height), 0)
            x_max = min(int(max(xs) * width), width - 1)
            y_max = min(int(max(ys) * height), height - 1)

            bbox_w = max(x_max - x_min, 1)
            bbox_h = max(y_max - y_min, 1)

            bbox_w_norm = bbox_w / float(width)
            bbox_h_norm = bbox_h / float(height)
            bbox_area_norm = bbox_w_norm * bbox_h_norm

            if bbox_area_norm > 0.15:
                distancia = "CERCA"
                min_conf = 0.6
            elif bbox_area_norm > 0.05:
                distancia = "MEDIA"
                min_conf = 0.4
            else:
                distancia = "LEJOS"
                min_conf = 0.3

            distancias.append(distancia)

            puntos = np.array(
                [(lm.x * width, lm.y * height) for lm in landmarks.landmark],
                dtype=np.float32,
            )

            if idx < len(scores):
                confianza = float(scores[idx])
            else:
                confianza = 0.7

            if confianza < min_conf:
                confianza = min_conf

            cx_norm = ((x_min + x_max) / 2.0) / float(width)
            cy_norm = ((y_min + y_max) / 2.0) / float(height)
            world_estimate = np.array(
                [cx_norm, cy_norm, 1.0],
                dtype=np.float32,
            )

            objetos.append(
                TrackedObject(
                    id=f"hand_{idx}",
                    type="HAND",
                    bounding_box=(x_min, y_min, bbox_w, bbox_h),
                    landmarks=puntos,
                    confidence=confianza,
                    world_position_estimate=world_estimate,
                    distance_estimate=distancia,
                ),
            )

        if distancias:
            if "LEJOS" in distancias:
                self._scale_factor = 2.0
            elif "MEDIA" in distancias:
                self._scale_factor = 1.5
            else:
                self._scale_factor = 1.0
        else:
            self._scale_factor = 1.0

        return objetos

    def _procesar_caras(
        self,
        resultados: Any,
        width: int,
        height: int,
    ) -> List[TrackedObject]:
        objetos: List[TrackedObject] = []

        if resultados is None or resultados.detections is None:
            return objetos

        for idx, det in enumerate(resultados.detections):
            if not det.location_data:
                continue

            bbox_rel = det.location_data.relative_bounding_box
            x_min = int(bbox_rel.xmin * width)
            y_min = int(bbox_rel.ymin * height)
            bbox_w = int(bbox_rel.width * width)
            bbox_h = int(bbox_rel.height * height)

            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            bbox_w = max(min(bbox_w, width - x_min), 1)
            bbox_h = max(min(bbox_h, height - y_min), 1)

            keypoints = det.location_data.relative_keypoints
            puntos = np.array(
                [(k.x * width, k.y * height) for k in keypoints],
                dtype=np.float32,
            )

            if det.score:
                confianza = float(det.score[0])
            else:
                confianza = 0.6

            cx_norm = (x_min + bbox_w / 2.0) / float(width)
            cy_norm = (y_min + bbox_h / 2.0) / float(height)
            world_estimate = np.array(
                [cx_norm, cy_norm, 1.0],
                dtype=np.float32,
            )

            objetos.append(
                TrackedObject(
                    id=f"face_{idx}",
                    type="FACE",
                    bounding_box=(x_min, y_min, bbox_w, bbox_h),
                    landmarks=puntos,
                    confidence=confianza,
                    world_position_estimate=world_estimate,
                ),
            )

        return objetos

    def _procesar_superficies(
        self,
        frame_bgr: np.ndarray,
        segmentacion: Any,
    ) -> List[TrackedObject]:
        objetos: List[TrackedObject] = []

        if frame_bgr is None:
            return objetos

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if segmentacion is not None and hasattr(segmentacion, "segmentation_mask"):
            mask = segmentacion.segmentation_mask
            if mask is not None:
                if mask.shape[:2] != gray.shape[:2]:
                    mask_resized = cv2.resize(
                        mask,
                        (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    mask_resized = mask
                persona = mask_resized > 0.5
                gray = gray.copy()
                gray[persona] = 0

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=80,
            minLineLength=60,
            maxLineGap=10,
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        height, width = gray.shape[:2]
        img_area = float(width * height)

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < img_area * 0.02:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            x, y, w_box, h_box = cv2.boundingRect(approx)
            if w_box < 40 or h_box < 40:
                continue

            confianza = min(1.0, area / (img_area * 0.5))

            cx_norm = (x + w_box / 2.0) / float(width)
            cy_norm = (y + h_box / 2.0) / float(height)
            world_estimate = np.array(
                [cx_norm, cy_norm, 1.0],
                dtype=np.float32,
            )

            if lines is not None:
                line_segments = np.array(lines.reshape(-1, 4), dtype=np.float32)
            else:
                line_segments = None

            objetos.append(
                TrackedObject(
                    id=f"surface_{idx}",
                    type="SURFACE",
                    bounding_box=(x, y, w_box, h_box),
                    landmarks=line_segments,
                    confidence=confianza,
                    world_position_estimate=world_estimate,
                ),
            )

        return objetos

    def get_anchor_point(self, obj: TrackedObject) -> Tuple[int, int]:
        """
        Devuelve un punto óptimo de anclaje para overlays 2D.

        :param obj: Objeto rastreado.
        :return: Tupla (x, y) en coordenadas de píxel.
        """
        x, y, w, h = obj.bounding_box

        if obj.type == "HAND" and obj.landmarks is not None and obj.landmarks.shape[0] >= 10:
            punto = obj.landmarks[9]
            return int(punto[0]), int(punto[1])

        if obj.type == "FACE":
            return int(x + w / 2.0), int(y + h * 0.3)

        return int(x + w / 2.0), int(y + h / 2.0)

    def obtener_metricas_latencia(self) -> Dict[str, Optional[float]]:
        """
        Devuelve estadísticas simples de latencia del tracking.

        :return: Diccionario con latencias mínima, máxima, media y última.
        """
        media = None
        if self._lat_count > 0:
            media = self._lat_sum_ms / float(self._lat_count)

        return {
            "min_ms": self._lat_min_ms,
            "max_ms": self._lat_max_ms,
            "mean_ms": media,
            "last_ms": self._last_latency_ms,
        }

    def actualizar(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Método de compatibilidad con la versión anterior.

        Envuelve track(frame) y devuelve un diccionario con la pose
        (todavía no estimada) y la lista de objetos rastreados.

        :param frame: Imagen en formato BGR procedente de OpenCV.
        :return: Diccionario con el estado de seguimiento estimado.
        """
        objetos = self.track(frame)
        estado: Dict[str, Any] = {
            "pose": None,
            "objects": objetos,
        }
        return estado


def _demo_spatial_tracker() -> None:
    """
    Demo interactiva del SpatialTracker usando la webcam principal.

    Muestra las manos, rostros y superficies detectadas junto con
    las métricas de FPS y latencia de tracking.
    """
    from viernes.utils.logger import get_logger

    logger = get_logger()
    tracker = SpatialTracker(logger)

    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        logger.error("No se pudo abrir la cámara principal en SpatialTracker demo")
        return

    camara.set(cv2.CAP_PROP_FRAME_WIDTH, TRACKER_FRAME_WIDTH)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, TRACKER_FRAME_HEIGHT)

    ultimo_tiempo = time.perf_counter()

    try:
        while True:
            ok, frame = camara.read()
            if not ok or frame is None:
                logger.error("No se pudo leer un frame de la cámara en SpatialTracker demo")
                break

            inicio = time.perf_counter()
            objetos = tracker.track(frame)
            fin = time.perf_counter()

            latencia_ms = (fin - inicio) * 1000.0
            dt = fin - ultimo_tiempo
            fps = 1.0 / dt if dt > 0.0 else 0.0
            ultimo_tiempo = fin

            if latencia_ms > 30.0:
                logger.warning(f"Latencia de tracking alta: {latencia_ms:.3f} ms")

            for obj in objetos:
                x, y, w, h = obj.bounding_box

                if obj.type == "HAND":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if obj.landmarks is not None:
                        for punto in obj.landmarks:
                            cv2.circle(
                                frame,
                                (int(punto[0]), int(punto[1])),
                                2,
                                (0, 255, 0),
                                -1,
                            )
                elif obj.type == "FACE":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    if obj.landmarks is not None:
                        for punto in obj.landmarks:
                            cv2.circle(
                                frame,
                                (int(punto[0]), int(punto[1])),
                                3,
                                (255, 0, 0),
                                -1,
                            )
                elif obj.type == "SURFACE":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    if obj.landmarks is not None:
                        for linea in obj.landmarks:
                            x1, y1, x2, y2 = linea
                            cv2.line(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 255),
                                1,
                            )

                ax, ay = tracker.get_anchor_point(obj)
                cv2.circle(frame, (ax, ay), 4, (255, 255, 255), -1)

            texto_fps = f"FPS: {fps:5.1f}"
            texto_lat = f"Latencia tracking: {latencia_ms:5.1f} ms"

            cv2.putText(
                frame,
                texto_fps,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                texto_lat,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("SpatialTracker demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo_spatial_tracker()
