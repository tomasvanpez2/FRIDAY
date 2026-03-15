"""
Utilidades de temporización para VIERNES OS.

Este módulo proporciona primitivas ligeras para medir tiempos de
ejecución usando relojes de alta resolución.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Tuple


@contextmanager
def medir_bloque(nombre: str, registrar: Optional[Callable[[str], None]] = None) -> Iterator[None]:
    """
    Context manager para medir la duración de un bloque de código.

    :param nombre: Etiqueta descriptiva para el bloque medido.
    :param registrar: Función de logging a la que enviar el resultado.
    """
    inicio = time.perf_counter()
    try:
        yield
    finally:
        fin = time.perf_counter()
        duracion = fin - inicio
        mensaje = f"{nombre} tardó {duracion:.6f} segundos"
        if registrar is not None:
            registrar(mensaje)


def medir_funcion(funcion: Callable[..., object]) -> Callable[..., object]:
    """
    Decorador sencillo para medir el tiempo de una función.

    El resultado se imprime por stdout. Puede adaptarse para integrar
    con el logger central si se desea.
    """

    def _envoltura(*args: object, **kwargs: object) -> object:
        inicio = time.perf_counter()
        resultado = funcion(*args, **kwargs)
        fin = time.perf_counter()
        duracion = fin - inicio
        print(f"{funcion.__name__} tardó {duracion:.6f} segundos")
        return resultado

    return _envoltura


class PipelineTimer:
    """
    Utilidad para acumular tiempos de etapas de un pipeline.

    Permite registrar muestras en milisegundos por métrica y calcular
    percentiles básicos (P50, P95, P99) al finalizar.
    """

    def __init__(
        self,
        nombre: str = "pipeline",
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._nombre = nombre
        self._logger = logger
        self._samples: Dict[str, List[float]] = {}
        self._last_values: Dict[str, float] = {}

    def record(self, metric: str, value_ms: float) -> None:
        """
        Registra una muestra de latencia para una métrica dada.
        """
        if metric not in self._samples:
            self._samples[metric] = []
        self._samples[metric].append(value_ms)
        self._last_values[metric] = value_ms

    def last(self, metric: str) -> Optional[float]:
        """
        Devuelve la última muestra registrada para una métrica.
        """
        return self._last_values.get(metric)

    def _percentiles(self, values: List[float]) -> Tuple[float, float, float]:
        valores = sorted(values)
        n = len(valores)
        if n == 0:
            return 0.0, 0.0, 0.0

        def _idx(p: float) -> int:
            pos = p * (n - 1)
            return int(pos)

        p50 = valores[_idx(0.50)]
        p95 = valores[_idx(0.95)]
        p99 = valores[_idx(0.99)]
        return p50, p95, p99

    def stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula estadísticas de percentiles para todas las métricas.
        """
        resultado: Dict[str, Dict[str, float]] = {}
        for metric, values in self._samples.items():
            if not values:
                continue
            p50, p95, p99 = self._percentiles(values)
            resultado[metric] = {
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "count": float(len(values)),
            }
        return resultado

    def report(self) -> None:
        """
        Emite un resumen de estadísticas usando el logger, si está presente.
        """
        stats = self.stats()
        if not stats:
            return

        for metric, valores in stats.items():
            mensaje = (
                f"[{self._nombre}] {metric}: "
                f"P50={valores['p50_ms']:.3f} ms, "
                f"P95={valores['p95_ms']:.3f} ms, "
                f"P99={valores['p99_ms']:.3f} ms, "
                f"n={int(valores['count'])}"
            )
            if self._logger is not None:
                self._logger(mensaje)
            else:
                print(mensaje)
