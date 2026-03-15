"""
Módulo de fusión de sensores para VIERNES OS.

Este módulo se encarga de combinar datos provenientes de la cámara y,
en el futuro, de otras fuentes de sensores (IMU, profundidad, etc.).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np


class SensorFusion:
    """
    Clase principal para la fusión de sensores.

    Por ahora actúa como un paso de preprocesado ligero del frame que
    se recibe desde la cámara.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inicializa el módulo de fusión de sensores.

        :param logger: Instancia del logger central del sistema.
        """
        self._logger = logger
        self._logger.info("Módulo SensorFusion inicializado")

    def actualizar(self, frame: np.ndarray) -> np.ndarray:
        """
        Actualiza el estado interno con un nuevo frame de entrada.

        Por ahora simplemente devuelve el mismo frame, pero este es el
        punto en el que se podrían combinar datos de múltiples sensores.

        :param frame: Imagen en formato BGR procedente de OpenCV.
        :return: Frame posiblemente enriquecido con información de sensores.
        """
        if frame is None:
            self._logger.error("Frame recibido es None en SensorFusion")
            return frame

        return frame


class CameraCapture:
    """
    Captura de cámara en hilo dedicado con doble buffer y métricas.

    Gestiona un hilo de captura basado en cv2.VideoCapture usando el
    backend CAP_AVFOUNDATION (Mac) optimizado para baja latencia en
    Intel, forzando MJPG como formato de captura y reduciendo la
    resolución interna a 640x480.
    """

    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS: int = 60

    def __init__(
        self,
        logger: Any,
        device_index: int = 0,
        backend: int = cv2.CAP_AVFOUNDATION,
    ) -> None:
        """
        Inicializa la captura de cámara.

        :param logger: Instancia del logger central para registrar eventos.
        :param device_index: Índice de la cámara a utilizar.
        :param backend: Backend de captura de OpenCV (CAP_AVFOUNDATION en Mac).
        """
        self._logger = logger
        self._device_index = device_index
        self._backend = backend

        self._capture: Optional[cv2.VideoCapture] = None
        self._buffer: Deque[Tuple[np.ndarray, float, float]] = deque(maxlen=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()

        self._lat_min_ms: Optional[float] = None
        self._lat_max_ms: Optional[float] = None
        self._lat_sum_ms: float = 0.0
        self._lat_count: int = 0

    def _configurar_camara(self) -> bool:
        """
        Configura los parámetros básicos de la cámara.

        :return: True si la cámara quedó lista, False en caso contrario.
        """
        assert self._capture is not None

        # Forzar formato MJPG para reducir la latencia en Mac Intel.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._capture.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Reducir resolución interna a 640x480 para minimizar el coste
        # de captura; el compositor puede escalar posteriormente.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)

        # Intentar fijar la tasa de FPS deseada.
        self._capture.set(cv2.CAP_PROP_FPS, self.FPS)

        # Minimizar el buffer interno de OpenCV para reducir la cola de
        # frames y acercar el feed lo máximo posible al tiempo real.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ancho = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        alto = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self._capture.get(cv2.CAP_PROP_FPS)

        self._logger.info(
            f"Cámara configurada a {ancho:.0f}x{alto:.0f} @ {fps:.0f} FPS "
            f"(objetivo {self.FRAME_WIDTH}x{self.FRAME_HEIGHT} @ {self.FPS} FPS)",
        )

        return True

    def _bucle_captura(self) -> None:
        """
        Hilo dedicado a la captura de frames y cálculo de latencia.
        """
        assert self._capture is not None

        while self._running:
            # Se mide la latencia únicamente asociada a las llamadas
            # grab() + retrieve(), sin incluir ningún procesamiento
            # adicional sobre el frame.
            inicio = time.perf_counter()

            ok_grab = self._capture.grab()
            if not ok_grab:
                self._logger.error("No se pudo hacer grab() sobre la cámara")
                continue

            ok, frame = self._capture.retrieve()
            fin = time.perf_counter()

            if not ok or frame is None:
                self._logger.error("No se pudo hacer retrieve() del frame")
                continue

            latencia_ms = (fin - inicio) * 1000.0
            timestamp_ms = fin * 1000.0

            self._lat_sum_ms += latencia_ms
            self._lat_count += 1
            if self._lat_min_ms is None or latencia_ms < self._lat_min_ms:
                self._lat_min_ms = latencia_ms
            if self._lat_max_ms is None or latencia_ms > self._lat_max_ms:
                self._lat_max_ms = latencia_ms

            if latencia_ms > 10.0:
                self._logger.warning(
                    f"Latencia de captura alta: {latencia_ms:.3f} ms",
                )

            with self._lock:
                self._buffer.append((frame, timestamp_ms, latencia_ms))

    def start(self) -> None:
        """
        Arranca el hilo de captura si todavía no está en ejecución.
        """
        if self._running:
            return

        self._logger.info("Iniciando captura de cámara en hilo dedicado")
        self._capture = cv2.VideoCapture(self._device_index, self._backend)

        if not self._capture.isOpened():
            self._logger.error(
                f"No se pudo abrir la cámara con backend AVFOUNDATION "
                f"(índice {self._device_index})",
            )
            self._capture.release()
            self._capture = None
            return

        self._configurar_camara()

        self._running = True
        self._thread = threading.Thread(target=self._bucle_captura, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Detiene el hilo de captura y libera la cámara.
        """
        if not self._running:
            return

        self._logger.info("Deteniendo captura de cámara")
        self._running = False

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
        """
        Devuelve el último frame disponible en el buffer.

        :return: Tupla (frame, timestamp_ms, latency_ms). Puede contener None.
        """
        with self._lock:
            if not self._buffer:
                return None, None, None

            frame, ts_ms, lat_ms = self._buffer[-1]
            return frame, ts_ms, lat_ms

    def get_stats(self) -> Dict[str, Optional[float]]:
        """
        Devuelve estadísticas agregadas de latencia de captura.

        :return: Diccionario con min, max y media de latencia en ms.
        """
        media = None
        if self._lat_count > 0:
            media = self._lat_sum_ms / float(self._lat_count)

        return {
            "min_ms": self._lat_min_ms,
            "max_ms": self._lat_max_ms,
            "avg_ms": media,
            "count": float(self._lat_count),
        }


def _demo_camera_capture() -> None:
    """
    Demo de uso de CameraCapture mostrando la latencia en tiempo real.

    Abre una ventana con la imagen de la cámara y la latencia medida en
    la esquina superior izquierda. Al salir muestra estadísticas
    agregadas de latencia.
    """
    from viernes.utils.logger import get_logger

    logger = get_logger()
    cam = CameraCapture(logger=logger)
    cam.start()

    logger.info("Demo de CameraCapture iniciada (pulsa 'q' para salir)")

    try:
        while True:
            frame, _, lat_ms = cam.get_frame()
            if frame is None:
                continue

            vista = frame.copy()

            if lat_ms is not None:
                texto = f"Latencia: {lat_ms:.2f} ms"
                cv2.putText(
                    vista,
                    texto,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("VIERNES OS - CameraCapture demo", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                logger.info("Salida de demo solicitada por el usuario")
                break

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado durante la demo de CameraCapture")
    finally:
        stats = cam.get_stats()

        def _fmt(valor: Optional[float]) -> str:
            """
            Formatea un valor flotante de latencia en ms.
            """
            if valor is None:
                return "n/a"
            return f"{valor:.3f}"

        logger.info(
            "Estadísticas de latencia (ms) - "
            f"min: {_fmt(stats['min_ms'])} | "
            f"max: {_fmt(stats['max_ms'])} | "
            f"avg: {_fmt(stats['avg_ms'])} | "
            f"frames: {stats['count']:.0f}",
        )

        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo_camera_capture()
