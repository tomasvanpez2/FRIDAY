# VIERNES — Research Prompt
## Investiga todo esto antes de escribir una línea de código

---

## Contexto para el investigador

Estoy construyendo VIERNES: una entidad de inteligencia artificial personal que corre
completamente local en hardware de bajos recursos (Mac Intel con 16GB RAM en desarrollo,
objetivo final: dispositivo ARM con 6GB RAM).

NO es un chatbot. NO usa APIs externas. NO usa Whisper, Gemini, GPT ni ningún modelo
externo. Es una entidad propia que aprende por refuerzo automático con el uso.

Necesito que investigues cada uno de los siguientes temas y respondas con:
- Qué es exactamente
- Por qué es relevante para VIERNES
- La implementación más liviana posible (menos RAM, menos CPU)
- Librerías Python específicas con versiones
- Limitaciones reales que debo conocer antes de implementar
- Alternativas si la opción principal no corre en hardware limitado

---

## BLOQUE 1 — Modelo de lenguaje propio ultraligero

**Pregunta 1:**
¿Cuál es el transformer más pequeño que puedo entrenar desde cero en CPU
con menos de 4GB de RAM, que sea capaz de generar lenguaje técnico coherente
en español? Compara: nanoGPT, GPT-2 small, DistilGPT-2, RWKV-v4 tiny.
Dame parámetros exactos de arquitectura para cada uno y su consumo real de RAM
en entrenamiento e inferencia.

**Pregunta 2:**
¿Cómo funciona el preentrenamiento de un LM desde cero sobre un corpus pequeño
(menos de 100MB de texto)? ¿Cuántas épocas necesito? ¿Qué loss esperar al inicio
y al final? ¿Qué significa que el modelo "aprendió" realmente?

**Pregunta 3:**
¿Qué es el aprendizaje incremental (online learning) en un LM? ¿Cómo actualizo
el modelo con texto nuevo de cada sesión sin olvidar lo anterior (catastrophic
forgetting)? ¿Qué técnica es más liviana: EWC, LoRA incremental, o replay buffer?

**Pregunta 4:**
¿Qué es un tokenizer BPE y cómo lo entreno desde cero sobre texto en español
técnico con la librería `tokenizers` de HuggingFace? Dame el código mínimo
para entrenarlo con vocabulario de 8000 tokens.

---

## BLOQUE 2 — Aprendizaje por refuerzo ultraligero

**Pregunta 5:**
¿Cómo funciona PPO (Proximal Policy Optimization) en términos simples?
¿Qué es la política, el value function, el actor-critic? ¿Por qué PPO
es más estable que otros algoritmos RL para este caso?

**Pregunta 6:**
¿Cómo diseño un entorno Gymnasium personalizado para un agente que observa
un vector de estado de 20 floats y toma 10 acciones discretas?
Dame el template mínimo de código con `observation_space`, `action_space`,
`step()`, `reset()` y cómo conectarlo a Stable-Baselines3.

**Pregunta 7:**
¿Qué es reward shaping implícito? ¿Cómo diseño señales de recompensa que
el agente pueda recibir sin que el usuario presione ningún botón?
Dame 5 ejemplos concretos de señales implícitas observables en comportamiento
humano frente a una pantalla.

**Pregunta 8:**
¿Cuánta RAM y CPU consume un modelo PPO con MlpPolicy sobre un espacio
de observación de 20 floats y 10 acciones discretas en Stable-Baselines3?
¿Cuánto tarda 1000 pasos de entrenamiento en CPU básica?

---

## BLOQUE 3 — Sistema de voz propio (sin Whisper)

**Pregunta 9:**
¿Cómo funciona un modelo de reconocimiento de voz basado en CTC
(Connectionist Temporal Classification)? ¿Qué arquitectura mínima
(capas Conv1D + GRU) puede transcribir español con menos de 5M parámetros?
¿Qué dataset en español puedo usar para preentrenarlo (licencia libre)?

**Pregunta 10:**
¿Cómo funciona un espectrograma mel? ¿Cómo convierto audio crudo (array numpy
16kHz) a espectrograma mel de 80 bandas con librosa o torchaudio?
Dame el código exacto para hacer esa conversión.

**Pregunta 11:**
¿Cómo funciona la detección de actividad de voz (VAD) basada en energía RMS
sin librerías externas? ¿Cómo calculo el umbral adaptativo que se ajusta
al ruido de fondo? Dame el código mínimo en numpy.

**Pregunta 12:**
¿Cómo funciona la detección de wake word por correlación de espectrograma?
¿Es confiable sin modelo entrenado? ¿Qué alternativa liviana existe si
la correlación da demasiados falsos positivos?

**Pregunta 13:**
¿Cómo funciona un vocoder Griffin-Lim para convertir un espectrograma mel
a audio? ¿Qué calidad de audio produce comparado con WaveNet?
¿Es suficiente para una voz de asistente inteligible?

---

## BLOQUE 4 — Memoria vectorial

**Pregunta 14:**
¿Cómo funciona LanceDB para almacenamiento vectorial local?
¿Cómo creo una tabla, inserto vectores, y hago búsqueda ANN (Approximate
Nearest Neighbor)? Dame el código mínimo para: crear tabla, insertar
un vector con metadata, buscar los 5 más similares.

**Pregunta 15:**
¿Cómo funciona el modelo `all-MiniLM-L6-v2` de sentence-transformers?
¿Cuánta RAM consume? ¿Cuánto tarda en vectorizar una frase en CPU?
¿Puedo usarlo sin conexión a internet después de la primera descarga?

**Pregunta 16:**
¿Qué es la curva de olvido de Ebbinghaus y cómo la implemento como
función de decay sobre un score de importancia? Dame la fórmula matemática
exacta y el código Python para aplicarla a una lista de nodos con timestamp.

---

## BLOQUE 5 — Percepción e interpretación de escena

**Pregunta 17:**
¿Cómo clasifico la actividad del usuario (coding, debugging, reading, idle)
usando solo los datos que ya tengo de MediaPipe (landmarks de manos y cara)
y OpenCV (movimiento entre frames)? Sin YOLO, sin modelo adicional.
Dame un enfoque basado en reglas con las métricas exactas a calcular.

**Pregunta 18:**
¿Cómo detecto el estado emocional básico (concentrado, frustrado, neutral)
usando solo los 468 landmarks faciales de MediaPipe FaceMesh?
¿Qué landmarks específicos son relevantes para cada estado?

**Pregunta 19:**
¿Cómo calculo flujo óptico entre frames consecutivos con OpenCV para
detectar nivel de actividad del usuario? ¿Cuál es el método más liviano:
Lucas-Kanade, Farneback, o diferencia de frames? ¿Cuánto CPU consume
a 15 FPS en resolución 640x480?

---

## BLOQUE 6 — Arquitectura de sistema y recursos

**Pregunta 20:**
¿Cómo gestiono múltiples hilos en Python (threading) para que el loop AR
a 15 FPS no se bloquee por el procesamiento cognitivo? ¿Cómo uso
`queue.Queue` para pasar datos entre hilos sin race conditions?
Dame el patrón exacto para productor-consumidor con timeout.

**Pregunta 21:**
¿Cuánta RAM total esperar con todos estos componentes corriendo simultáneamente?
Estima: LM en inferencia (~200MB) + PPO policy (~5MB) + LanceDB (~100MB)
+ MediaPipe (~150MB) + OpenCV pipeline (~200MB) + ASR model (~50MB).
¿Hay algún componente que pueda cargar y descargar bajo demanda para
ahorrar RAM en hardware de 6GB?

**Pregunta 22:**
¿Qué es `torch.no_grad()` y por qué es crítico usarlo en inferencia
para ahorrar memoria? ¿Qué otras optimizaciones de PyTorch son relevantes
para CPU con recursos limitados: `torch.compile()`, cuantización int8,
modo de inferencia `torch.inference_mode()`?

---

## Formato de respuesta esperado

Para cada pregunta responde con esta estructura:

### Pregunta N
**Respuesta directa:** [2-3 líneas con la respuesta concreta]

**Por qué importa para VIERNES:** [1-2 líneas]

**Implementación mínima:** [código o pasos concretos]

**Limitación crítica:** [lo más importante que debo saber antes de implementar]

**Alternativa si falla:** [qué hacer si la opción principal no funciona en el hardware]

---

## Dónde usar este prompt

Pégalo en:
- Claude.ai con modo Research activado (búsqueda web habilitada)
- Perplexity.ai en modo Research
- ChatGPT con búsqueda web activada

El objetivo es tener todas las respuestas documentadas en
`viernes/data/research/RESEARCH_RESULTS.md` antes de escribir código.
Ese archivo se convierte en la referencia técnica del proyecto.
