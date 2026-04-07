# 📁 VIERNES (F.R.I.D.A.Y.) - La Guía Maestra

Este documento representa la ontología definitiva del proyecto. Responde en profundidad qué es VIERNES, por qué existe, cómo funciona su ingeniería subyacente, qué hace el sistema, su futuro programado y un glosario microscópico de su esquema de archivos.

---

## 1. ¿Qué es el Proyecto? (Soberanía y Entidad)

**VIERNES** es un sistema de Inteligencia Artificial de Realidad Mixta (AR) ejecutado de manera completamente **nativa, offline y soberana (Edge AI)**. 
Nació inspirado por la entidad tecnológica F.R.I.D.A.Y. de Iron Man, pero se desmarca de ser un "Voice Assistant" clásico. VIERNES es un **Agente Autónomo Reforzado**. No es un chatbot que espera a que le hables; es un sistema óptico asíncrono que observa tu cara, tus manos y tu escritorio, decidiendo cuándo interactuar contigo de manera activa. Todo el procesamiento está diseñado con una premisa extrema de austeridad de hardware: debe ser capaz de ejecutar la inferencia del lenguaje, la visión computacional y el razonamiento en un **procesador CPU sencillo (ARM o Intel) con menos de 6 GB de memoria RAM**, aboliendo para siempre el envío de datos a las APIs corporativas como OpenAI o Google Gemini.

---

## 2. ¿Cómo Funciona la Arquitectura? (El Workflow)

La arquitectura de VIERNES trabaja asíncronamente en bucles o *Pipelines* que comparten memoria mediante exclusión mutua (*Locks*).

1. **Absorción Sensorial**: Un hilo del Procesador se encarga exclusivamente de secuestrar la cámara web utilizando protocolos en el bajo nivel (`CAP_AVFOUNDATION`), manteniendo un buffer ultracorto (de 2 imágenes máximo). Esto obliga al sistema a nunca tener "imágenes viejas", logrando una latencia biológica casi imperceptible al eludir el *overhead* de OpenCV.
2. **Razonamiento Espacial Categórico**: Los fotogramas pasan inmediatamente a los motores ligeros vectoriales (*MediaPipe*). Se extraen topografías tridimensionales. Si el sistema ve tus manos, un algoritmo métrico basado en las cajas de contorno (*Bounding Box Area*) sabe instintivamente si tu mano está alejándose o interactuando de cerca. Al escanear tu rostro, anota *468 puntos faciales* exactos (útiles para futuras detecciones emocionales) y segmenta tu escritorio plano usando filtros matemáticos (*Canny Edge*).
3. **El Cerebro Causal (La Conciencia)**: Toda esta matemática de OpenCV se compila como el "Espacio de Observación". El cerebro artificial tomará estos datos dictando una respuesta según el entrenamiento que recibió leyendo libros de tecnología técnica en su "infancia", decidiendo recuperar información antigua mediante arreglos si lo juzga pertinente.
4. **Respuesta Aumentada**: Si VIERNES tiene algo que decir, o detecta cansancio y te lanza un aviso, no te abrirá una interfaz web. Renderizará un holograma falso superponiendo matrices (*Capas Alfa*) flotando directamente ancladas a tu monitor o frente a tu rostro en su *Head-Up Display (HUD)* calculado mediante tasas asíncronas para que jamás altere el flujo real de tiempo.

---

## 3. ¿Qué se hace en esta etapa de desarrollo? 

En este preciso instante la red neuro-técnica se está desconectando. Se destruyó intencionalmente la conexión comercial (`llm_engine.py` fue borrado) forzando el salto a una red neuronal local construida a mano.

El esfuerzo actual de código puro es la **Ingeniería de Datos**: Se ha diseñado un potente *Crawler/Scraper* para devorar la red en la carpeta `viernes/data/`. El código raspa recursivamente y limpia textos extrayendo librerías masivas de Python, matemáticas, tensores (PyTorch) y teoría de Aprendizaje por Refuerzo (RL). El ecosistema luego es filtrado extrayendo el unicode residual, eliminando espacios, forzando saltaciones UTF-8 limpias e implementando *Chunking de 400 palabras con *Overlaps* de 50 palabras* (fragmentación semántica óptima). Todo esto para inyectarle un conocimiento duro de `4.4 Megabytes` técnicos sin grasa basura extra a nuestro transformador casero.

---

## 4. ¿Qué se espera del futuro? (La Hoja de Ruta Extrema)

Se espera que VIERNES se incube de la amalgama de los archivos descritos hacia estas **tres grandes fases**:

- **Cognición Offline Absoluta (BPE + nanoGPT)**: VIERNES correrá la clase lógica del Tokenizador (*minbpe*) para entender el español técnico y leer tu código. La inferencia se amarrará en el hardware usando técnicas agresivas de mitigación como `torch.no_grad()` para evitar que el compilador memorice los perfiles de la red ahorrando 50% de la RAM en el Transformer.
- **Aprendizaje Interlocutor Implícito (PPO de Gymnasium)**: La cámara no solo es visualización, **es un mecanismo de castigo/recompensa biológico**. VIERNES se instanciará como un entorno de *Gymnasium*. Actuará mediante Reinforcement Learning: Cuando detecte (gracias a flujos ópticos) que has permanecido quieto leyendo, guardará silencio. Si hablas frente el código en crisis, intentará ayudar en el HUD. Si su ayuda calma tus micro-expresiones medidas, la red *PPO* optimizará su ganancia positivamente sin que toques una tecla.
- **Audífonos Locales Óptimos**: Exterminar al peso pesado de Whisper. Se creará una tubería acústica desde cero usando librerías *Librosa/NumPy*: interceptar energía en frecuencia (VAD/Umbrales RMS adaptativos) esquivando ruidos periféricos, traduciendo eso a *Espectrogramas Mel 80-band* en matrices Numpy alimentándolo en una red convolucional-GRU estricta. Respondiendo mediante audios re-generados por el clásico *Griffin-Lim* vocoder en cero segundos a tu oído físico.

---

## 5. El Glosario del Hardware Arquitectónico (Para Qué Sirve Cada Archivo)

Cada byte y módulo listado desempeña un rol atómico con responsabilidades de hardware profundas:

### Raíz del Proyecto
- **`.gitignore`**: Barrera de sanidad. Aisla el entorno base `venv/`, basuras binarias y esconde llaves `.env` con privilegios o tokens temporales restantes.
- **`What is it.md`**:  El documento de manifiesto que estás leyendo ahora. Un sumario fundacional continuo.

### `viernes/` (El Paquete Madre)
- **`__init__.py`**: Convierte el directorio llanamente en un módulo estándar importable al ambiente virtual Python.
- **`config.py`**: Centro Nervioso Estático (Variables de Entorno). Almacenaba las llaves externas y guarda lo que alguna vez figuró como "El Alma Analítica": los *Prompts Nativos* que forjaban el comportamiento sereno, profesional y en español colombiano de VIERNES simulando al mayordomo cibernético.
- **`main.py`**: El cordón conmutador del sistema local. Su única misión es despertar a `ARPipeline` logrando que el software base fluya sin fracturas terminales (manejando las paradas con excepciones de teclas limpiamente).
- **`requirements.txt`**: Un compendio de módulos duros pesados (`NumPy`, `MediaPipe` `OpenCV-Python`, `LanceDB`, `Torch`, `SciPy`) que se extraen desde el índice PyPI en versiones precompiladas a la arquitectura de W11/Mac.

### `viernes/core/` (La Capa Visual Primordial - Realidad Aumentada)
- **`sensor_fusion.py`**: Reside la clase fundamental `CameraCapture`. Responsable de interceptar la lente web sin interrumpir al hilo base. Genera estadísticas nativas que son enviadas asíncronamente manteniendo a raya las anomalías del sensor (Ej. bajando a 320x240 en caídas violentas de FPS). Expone `SensorFusion` que es hoy un "Stub" listo para conectar mañana cámaras de temperatura (FLIR) o Giras inerciales (IMUs).
- **`spatial_tracker.py`**: Motor topográfico. Extrae por fuerza bruta con base de redes neuronales *MediaPipe* cajas posicionales, mallas de *Selfie-Segmentation*, rastreo craneal de 468 nodos y heurísticas isométricas. 
- **`ar_compositor.py`**: El Gestor de Opacidad. Trabaja controlando diccionarios de estructuras nativas `Overlay`. Este código decide quién pinta píxeles arriba e impone superposiciones alfa mediante pesos y ecuaciones visuales sobre tu cámara sin destrozar la memoria de rasterizado.
- **`ar_pipeline.py`**: El corazón rítmico que lo abraza todo. Contiene el bucle de latido mientras la aplicación corre. Dibuja las etiquetas (*HUD SystemFPS*), detecta si tus manos se muestran cruzadas en cámara, e interactúa con el teclado (ej: la tecla `s` toma un volcado de captura al disco, la `h` desaparece todas las proyecciones).

### `viernes/cognitive/` (El Futuro Lóbulo Lógico)
*(Archivo `llm_engine.py` destituido permanentemente hacia el paradigma Offline).*
- **`knowledge_system.py`**: El núcleo de retención actual de memoria (actualmente en bruto "Stub" almacenando en arreglos planos simulados). A futuro, implementará la tecnología *LanceDB* inyectando memoria semántica convertida a vectores densos mediante *Sentence-Transformers*, programada para aplicar el desgaste de pesos según la Curva biológica del Olvido (Ebbinghaus).

### `viernes/data/` (Neurogénesis: Base del Motor a Escala)
- **`scripts/download_corpus_raw.py`**: El raspador cibernético que viajó por internet decodificando HTML, extrayendo textos precisos evadiendo los bloqueos SSL para recolectar el "Código Técnico".
- **`scripts/prepare_corpus.py`**: El limpiador y fragmentador de contexto. Carga la información en sucio, aplica bloqueos REGEX normalizando codificaciones y espaciados eliminando saltos de memoria infinitos, y dividiendo el texto en los célebres fragmentos matemáticos (*Chunks* de 400 y *Overlaps* de 50) para que tu modelo BPE Tokenizer digiera lógicamente.
- **`corpus_final.txt`**: El cerebro estático. El volcado resultante conteniendo más de 4 Megabytes de los mayores libros modernos de Inteligencia Artificial que será forzado en las venas del modelo naciente.
- **`research/ (Varios .md)`**: Repositorio de filosofía arquitectónica de tu investigación: Explicita los consumos en RAM, justificaciones para matar Whisper y cómo diseñar el VAD de la máquina.

### `viernes/utils/` (Infraestructura de Instrumentación)
- **`logger.py`**: Bitácora quirúrgica. Desecha el loggeado rudimentario y aprovecha micro-latencias usando el ultra fino reloj hardware del Procesador Central de tu placa madre para saber absolutamente cuándo arrancó un archivo en fracciones de nano segundo.
- **`timing.py`**: Context Manager `PipelineTimer`. El contador exacto de percentiles (P50/P95/P99) y varianzas (min/max). Dictamina matemáticamente al momento de cerrar VIERNES cuán castigado fue el sistema en el proceso gráfico (`lat_comp_ms`) vs el lógico.

### `viernes/interaction/` & `viernes/apps/` (Integraciones Superiores)
- **`interaction/gesture_handler.py`**: Esqueleto destinado a tomar la heurística de los nodos del Tracker para traducirlos a eventos lógicos del Sistema Operativo de la PC real. (Activar volumen apretando dedos, scrollear con la mano al aire).
- **`apps/study_assistant.py`**: Modelo arquitectónico para futuras "Aplicaciones de 3eros" instaladas en VIERNES (Plugins), encapsulando visiones modulares controladas enfocadas para rutinas y sub-procesos dedicados sin manchar tu Core AR.
