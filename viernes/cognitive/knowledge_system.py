"""
Módulo de sistema de conocimiento para VIERNES OS.

Este módulo gestionará memorias a largo plazo, recuperación de
información y almacenamiento semántico utilizando vectores.
"""

from __future__ import annotations

from typing import Any, List


class MemoryItem:
    def __init__(self, content: str) -> None:
        self.content = content


class KnowledgeSystem:
    """
    Sistema de conocimiento de alto nivel.

    La implementación real integrará bases vectoriales y embeddings,
    pero de momento sólo define la interfaz básica.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inicializa el sistema de conocimiento.

        :param logger: Instancia del logger central del sistema.
        """
        self._logger = logger
        self._logger.info("Módulo KnowledgeSystem inicializado")
        self._memories: List[MemoryItem] = []

    def almacenar(self, texto: str) -> None:
        """
        Almacena una pieza de información en la memoria de largo plazo.

        :param texto: Contenido textual a almacenar.
        """
        self._logger.debug(f"Almacenando conocimiento: {texto!r}")
        self._memories.append(MemoryItem(content=texto))

    def recuperar(self, consulta: str, k: int = 5) -> List[str]:
        """
        Recupera entradas relevantes a partir de una consulta de texto.

        :param consulta: Texto de búsqueda.
        :param k: Número máximo de resultados a devolver.
        :return: Lista de fragmentos de conocimiento relevantes.
        """
        self._logger.debug(f"Recuperando conocimiento para consulta: {consulta!r}")
        resultados = self.retrieve(consulta, top_k=k)
        return [m.content for m in resultados]

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        self._logger.debug(f"Retrieve llamado para consulta: {query!r}")
        if not self._memories:
            return []

        query_lower = query.lower()
        coincidencias: List[MemoryItem] = []
        resto: List[MemoryItem] = []

        for m in reversed(self._memories):
            if query_lower and query_lower in m.content.lower():
                coincidencias.append(m)
            else:
                resto.append(m)

        ordenados: List[MemoryItem] = coincidencias + resto
        return ordenados[:top_k]
