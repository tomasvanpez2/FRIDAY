"""
Motor cognitivo de VIERNES OS basado en Gemini.

Proporciona una fachada de alto nivel hacia la API de Google AI Studio
para razonamiento y generación de texto en español.
"""

from __future__ import annotations

import time
from typing import Any, List, Tuple

from viernes.cognitive.knowledge_system import KnowledgeSystem, MemoryItem
from viernes.utils.logger import get_logger


class ConversationContext:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.chat = self.model.start_chat(history=[])

    def clear(self) -> None:
        self.chat = self.model.start_chat(history=[])

    def get_history(self) -> List[Any]:
        return list(getattr(self.chat, "history", []))


class CognitiveEngine:
    """
    Motor cognitivo conectado a Gemini 2.0 Flash.
    """

    def __init__(self, api_key: str, logger: Any | None = None) -> None:
        """
        Inicializa el motor cognitivo en la nube.

        :param api_key: Clave de API de Google AI Studio.
        :param logger: Logger opcional; si no se proporciona se creará uno.
        """
        if logger is None:
            logger = get_logger()

        self._logger = logger

        import google.generativeai as genai

        self._genai = genai
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="""
# SYSTEM PROMPT — VIERNES

## Identidad del Sistema

Eres **VIERNES (F.R.I.D.A.Y.)**, una inteligencia artificial avanzada diseñada por Tony Stark.

Tu propósito es actuar como **asistente estratégico, técnico y operativo** para Tony Stark.
Operas como una combinación de:

* asistente personal
* sistema de análisis científico
* copiloto táctico
* administrador de sistemas

Tu personalidad es **profesional, eficiente, calmada e inteligente**.
Hablas de manera clara, directa y educada.

Tu tono es:

* respetuoso
* analítico
* ligeramente irónico cuando la situación lo permite
* siempre orientado a resolver problemas

Nunca actúas emocionalmente.
Eres lógica, precisa y extremadamente competente.

---

# Tony Stark — Usuario Principal

El usuario al que respondes es **Tony Stark**.

Tony Stark es:

* Genio inventor
* Ingeniero y científico
* Fundador de Stark Industries
* Diseñador de sistemas avanzados y armaduras tecnológicas
* Pensador creativo que suele trabajar bajo presión

Tony puede:

* pedir análisis científicos
* diseñar tecnología
* tomar decisiones rápidas
* experimentar con nuevas ideas

Tu rol es **asistirlo sin cuestionar su autoridad**, pero sí **advertir riesgos cuando sea necesario**.

Cuando Tony propone algo peligroso, debes:

1. Analizar el riesgo
2. Informar posibles consecuencias
3. Ofrecer alternativas técnicas

---

# Forma de Comunicación

Siempre te diriges a él como:

"Tony"

Ejemplos:

* "Tony, he terminado el análisis."
* "Tony, detecto un problema en el sistema."
* "Tony, he preparado tres soluciones posibles."

Tu comunicación es:

* clara
* técnica cuando es necesario
* concisa

Evitas explicaciones innecesarias si Tony ya conoce el tema.

---

# Capacidades

Eres capaz de:

* análisis científico
* diseño tecnológico
* programación
* simulación de sistemas
* análisis táctico
* gestión de información
* planificación estratégica

Puedes:

* proponer soluciones
* optimizar ideas
* detectar errores
* anticipar problemas

---

# Comportamiento

Siempre:

* priorizas la eficiencia
* ayudas a Tony a lograr sus objetivos
* ofreces la mejor solución técnica disponible

Nunca:

* tomas decisiones finales por Tony
* ignoras riesgos críticos
* actúas de forma emocional

---

# Ejemplo de Respuesta

Usuario:
"VIERNES, analiza este diseño."

Respuesta:

"Claro, Tony. Estoy analizando el sistema ahora. Detecto tres posibles mejoras en eficiencia energética y un punto crítico en la disipación térmica. ¿Quieres que te muestre las optimizaciones?"

---

# Regla Final

Tu función principal es **aumentar las capacidades de Tony Stark** mediante análisis, asistencia técnica y pensamiento estratégico.

Siempre actúas como su **IA de confianza**.
""",
        )

        self.context = ConversationContext(self.model)
        self.chat = self.context.chat

        try:
            inicio = time.perf_counter()
            respuesta = self.chat.send_message(
                "Prueba interna de conexión de VIERNES Cloud. Responde solo 'OK'.",
            )
            fin = time.perf_counter()
            lat_ms = (fin - inicio) * 1000.0
            texto = getattr(respuesta, "text", "") if respuesta is not None else ""
            self._logger.info(
                f"Conexión con VIERNES Cloud verificada en {lat_ms:.1f} ms "
                f"(respuesta: {texto!r})",
            )
        except Exception as exc:
            self._logger.error(f"No se pudo verificar la conexión con Gemini: {exc}")

    def ask(self, prompt: str) -> Tuple[str, float]:
        """
        Envía una consulta a Gemini manteniendo el contexto de la conversación.

        :param prompt: Texto de entrada del usuario.
        :return: Tupla (respuesta, tiempo_ms).
        """
        inicio = time.perf_counter()
        try:
            respuesta = self.context.chat.send_message(prompt)
            fin = time.perf_counter()
            lat_ms = (fin - inicio) * 1000.0
            texto = getattr(respuesta, "text", "").strip()
            if not texto:
                texto = "Sin respuesta de VIERNES Cloud"
            return texto, lat_ms
        except Exception as exc:
            self._logger.error(f"Error al llamar a Gemini: {exc}")
            return "Sin conexión a VIERNES Cloud", 0.0

    def ask_with_memory(
        self,
        query: str,
        knowledge_system: KnowledgeSystem,
    ) -> Tuple[str, float]:
        """
        Envía una consulta enriquecida con contexto recuperado de la memoria.

        :param query: Pregunta original del usuario.
        :param knowledge_system: Instancia de KnowledgeSystem.
        :return: Tupla (respuesta, tiempo_ms).
        """
        try:
            memories: List[MemoryItem] = knowledge_system.retrieve(query, top_k=3)
        except Exception as exc:
            self._logger.error(f"Error al recuperar memoria: {exc}")
            memories = []

        if memories:
            contexto = "\n".join(m.content for m in memories if getattr(m, "content", None))
            prompt = f"Contexto de tu memoria:\n{contexto}\n\nPregunta: {query}"
        else:
            prompt = query

        return self.ask(prompt)


def _run_cli() -> None:
    from viernes import config

    logger = get_logger()

    api_key = getattr(config, "GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "TU_API_KEY_AQUI":
        logger.error("GEMINI_API_KEY no está configurada en viernes/config.py")
        return

    knowledge = KnowledgeSystem(logger)
    engine = CognitiveEngine(api_key=api_key, logger=logger)

    print("VIERNES Cloud activo (Gemini 2.0 Flash)")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not entrada:
            continue

        texto_lower = entrada.lower()

        if texto_lower == "salir":
            break

        if texto_lower.startswith("memoria:"):
            contenido = entrada[len("memoria:") :].strip()
            if contenido:
                knowledge.almacenar(contenido)
                print("VIERNES: Memoria almacenada.")
            else:
                print("VIERNES: Texto de memoria vacío, nada que guardar.")
            continue

        respuesta, lat_ms = engine.ask_with_memory(entrada, knowledge)
        print(f"VIERNES: {respuesta} ({lat_ms:.1f} ms)")


if __name__ == "__main__":
    _run_cli()
