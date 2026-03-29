## VIERNES OS - Descripción Total del Proyecto

Este proyecto es una base funcional de **VIERNES OS** (inspirado en F.R.I.D.A.Y. de Tony Stark), orientado a una experiencia de **realidad aumentada en tiempo real** con capacidades cognitivas en evolución.

En su estado actual, el sistema está centrado en:

- Captura de video de cámara en baja latencia.
- Detección visual de manos, rostro y superficies.
- Composición de overlays AR sobre el feed de cámara.
- Monitoreo de rendimiento (FPS y latencias).
- Estructura preparada para memoria semántica y LLM (Gemini), aunque no acoplada aún al loop principal AR.

---

## Objetivo General

Construir una arquitectura modular para un asistente avanzado tipo VIERNES que combine:

1. **Percepción visual** (visión por computador).
2. **Interfaz aumentada** (HUD y overlays anclados a objetos).
3. **Cognición** (memoria + razonamiento con modelo de lenguaje).
4. **Interacción multimodal** (gestos/voz, aún en stubs).

---

## Flujo Principal de Ejecución

Punto de entrada: `viernes/main.py`

1. Se configura el logger central.
2. Se crea `ARPipeline`.
3. `ARPipeline` inicia:
   - `CameraCapture` (captura en hilo separado).
   - `SpatialTracker` (detección de objetos en frame).
   - `ARCompositor` (renderizado AR).
4. Loop por frame:
   - Obtiene frame más reciente.
   - Corre tracking de manos/caras/superficies.
   - Actualiza overlays según lo detectado.
   - Dibuja HUD con métricas.
   - Muestra ventana OpenCV.
5. Salida:
   - Tecla `q`: cerrar.
   - Tecla `s`: guardar screenshot.
   - Tecla `d`: toggle de landmarks debug.
   - Tecla `h`: toggle de overlays.

---

## Estructura General del Repositorio

### Archivos de raíz

#### `.gitignore`
Ignora:
- entorno virtual (`venv/`)
- cachés y compilados de Python (`__pycache__/`, `*.pyc`, etc.)
- artefactos de macOS (`.DS_Store`)
- archivo de variables de entorno (`.env`)

#### `What is it?.md`
Documento de descripción del proyecto (este archivo).

---

## Paquete principal: `viernes/`

### `viernes/__init__.py`
Describe el paquete principal y su propósito. No contiene lógica ejecutable.

### `viernes/config.py`
Archivo de configuración global:

- `GEMINI_API_KEY`: clave API para conexión a Gemini.
- `COGNITIVE_SYSTEM_PROMPT`: prompt de sistema extenso para definir personalidad, tono y reglas de respuesta del asistente.

Notas:
- Centraliza parámetros de cognición.
- Actualmente la API key está hardcodeada (en producción debería venir de variables de entorno).

### `viernes/main.py`
Punto de entrada ejecutable del sistema:

- `bucle_principal()`: arranca logger y `ARPipeline`.
- `main()`: envuelve ejecución con control de excepciones y código de salida (`0` éxito, `1` error no controlado).

Es el archivo que se debe ejecutar para iniciar el modo AR principal.

### `viernes/requirements.txt`
Dependencias:

- `opencv-python`: captura y render de video.
- `numpy`: estructuras y operaciones numéricas.
- `mediapipe`: detección de manos/cara/segmentación.
- `lancedb`, `sentence-transformers`: base para memoria vectorial/embeddings.
- `Pillow`: utilidades de imagen.
- `google-generativeai`, `google-genai`: integración con Gemini.
- `scipy`: utilidades matemáticas adicionales.

---

## Subpaquete `viernes/core/` (núcleo de percepción y AR)

### `viernes/core/__init__.py`
Docstring del subpaquete core. No lógica ejecutable.

### `viernes/core/sensor_fusion.py`
Contiene dos clases:

#### `SensorFusion`
- Stub de fusión multisensor.
- Método `actualizar(frame)` actualmente devuelve el frame sin cambios.
- Punto preparado para integrar IMU/profundidad/etc.

#### `CameraCapture`
Captura de cámara optimizada para baja latencia en Mac:

- Usa `cv2.VideoCapture` con backend `CAP_AVFOUNDATION`.
- Fuerza formato MJPG y resolución 640x480.
- Ejecuta captura en **hilo dedicado**.
- Usa buffer de tamaño 2 para mantener el frame más reciente.
- Expone:
  - `start()` / `stop()`
  - `get_frame()` -> `(frame, timestamp_ms, latency_ms)`
  - `get_stats()` -> métricas agregadas de latencia.

Incluye demo ejecutable local (`_demo_camera_capture`).

### `viernes/core/spatial_tracker.py`
Módulo principal de tracking visual.

Elementos clave:

- Constantes:
  - `TRACKER_FRAME_WIDTH = 640`
  - `TRACKER_FRAME_HEIGHT = 480`
- `TrackedObject` (dataclass):
  - `id`, `type`, `bounding_box`, `landmarks`, `confidence`, `world_position_estimate`, `distance_estimate`

#### `SpatialTracker`
Inicializa modelos de MediaPipe:

- Hands
- Face Detection
- Selfie Segmentation

Método principal: `track(frame)`

Procesa el frame para producir lista de objetos detectados:

1. **Manos** (`_procesar_manos`)
   - Extrae bbox y landmarks.
   - Estima distancia cualitativa: `CERCA`, `MEDIA`, `LEJOS` según área relativa.
   - Ajusta confianza mínima según distancia.
   - Ajusta `scale_factor` dinámicamente.

2. **Caras** (`_procesar_caras`)
   - Obtiene bbox, keypoints y score de detección.

3. **Superficies** (`_procesar_superficies`)
   - Convierte a gris, aplica Canny, HoughLinesP y contornos.
   - Busca contornos aproximados de 4 vértices para superficies planas.
   - Usa segmentación para suprimir región de persona y mejorar detección de entorno.

Utilidades:
- `get_anchor_point(obj)`: punto óptimo para anclar overlays.
- `obtener_metricas_latencia()`: min/max/media/última latencia.
- `actualizar(frame)`: wrapper de compatibilidad (retorna dict con objetos).

Incluye demo ejecutable (`_demo_spatial_tracker`).

### `viernes/core/ar_compositor.py`
Sistema de composición de overlays AR.

#### `Overlay` (dataclass)
Define un elemento gráfico:
- `id`
- `content_type` (`TEXT`, `PANEL`, `BOX`, `LABEL`)
- `position`, `size`, `alpha`, `color`, `text`
- `anchor_type` (`FIXED` / `WORLD`)
- `fade_in_ms`, `created_at`

#### `ARCompositor`
Gestiona overlays activos y los dibuja sobre el frame:

- `add_overlay(overlay)`: agrega/reemplaza.
- `remove_overlay(id)`: elimina.
- `update_anchor(id, new_position)`: mueve overlay.
- `_aplicar_fade(...)`: calcula alpha efectiva con fade-in.
- `compose(frame)`: dibuja y mezcla capas (`cv2.addWeighted`).
- `componer(...)`: método legacy compatible con versión previa.

Incluye demo (`_demo_ar_compositor`).

### `viernes/core/ar_pipeline.py`
Orquestador principal de todo el pipeline AR.

Inicializa:
- `CameraCapture`
- `SpatialTracker`
- `ARCompositor`
- `PipelineTimer`

Gestión de estado:
- lock para frame compartido (`_frame_lock`)
- frame actual (`_current_frame`)
- flags: debug landmarks, overlays on/off, tracking low-res
- diccionario de overlays activos por id

Funciones clave:

- `_update_tracking_resolution(fps)`: baja a 320x240 si cae FPS (<20), restaura a 640x480 si sube (>25).
- `_update_overlays(objetos, frame_shape)`:
  - HAND: bbox + label con distancia.
  - FACE: label "VIERNES activo".
  - SURFACE: panel semitransparente.
  - Limpia overlays de objetos ya no detectados.
- `_draw_debug_landmarks(...)`: puntos/líneas de depuración.
- `_draw_hud(...)`: imprime FPS, latencias y distancia.
- `_handle_key(...)`: teclas `q`, `s`, `d`, `h`.
- `get_current_frame()`: copia segura del último frame.
- `show_response(texto)`: overlay temporal de respuesta textual de VIERNES.
- `_save_screenshot(frame)`: guarda PNG en carpeta `screenshots/`.
- `run()`: loop principal completo.

Incluye demo local (`_demo_ar_pipeline`).

---

## Subpaquete `viernes/cognitive/` (capa cognitiva)

### `viernes/cognitive/__init__.py`
Docstring del subpaquete cognitivo.

### `viernes/cognitive/knowledge_system.py`
Sistema de memoria básico (versión inicial no vectorial):

- `MemoryItem`: contenedor de `content`.
- `KnowledgeSystem`:
  - `_memories`: lista interna.
  - `almacenar(texto)`: guarda un memory item.
  - `retrieve(query, top_k)`:
    - prioriza memorias que contengan el texto de consulta.
    - combina coincidencias + resto (recientes primero).
  - `recuperar(consulta, k)`: devuelve solo strings.

Es un stub funcional de memoria textual, pensado para evolucionar a embeddings + DB vectorial.

### `viernes/cognitive/llm_engine.py`
Motor de conversación con Gemini.

Componentes:

- `ConversationContext`:
  - crea chat con historial,
  - permite limpiar y recuperar history.

- `CognitiveEngine`:
  - configura `google.generativeai` con API key.
  - crea `GenerativeModel` con `gemini-2.5-flash`.
  - aplica `COGNITIVE_SYSTEM_PROMPT`.
  - verifica conexión al iniciar.
  - `ask(prompt)`:
    - envía mensaje al chat.
    - retorna `(texto, latencia_ms)`.
  - `ask_with_memory(query, knowledge_system)`:
    - recupera hasta 3 memorias.
    - inyecta contexto en prompt.
    - delega en `ask`.

Incluye CLI en modo texto (`_run_cli`):
- escribir `salir` termina.
- `memoria: ...` guarda conocimiento.

Nota importante:
- Esta capa cognitiva está implementada, pero no está todavía conectada al loop de `ARPipeline` en `main.py`.

---

## Subpaquete `viernes/interaction/` (interacción usuario)

### `viernes/interaction/__init__.py`
Docstring del subpaquete de interacción.

### `viernes/interaction/gesture_handler.py`
Stub para interacción por gestos:

- `GestureHandler.__init__`: inicialización y logging.
- `actualizar(frame)`: validador de entrada + punto de integración futuro.

Actualmente no participa en el flujo principal, pero define la interfaz para integrar comandos gestuales.

---

## Subpaquete `viernes/apps/` (aplicaciones de alto nivel)

### `viernes/apps/__init__.py`
Docstring del subpaquete de aplicaciones.

### `viernes/apps/study_assistant.py`
Aplicación de ejemplo encima de la capa AR:

- Clase `StudyAssistant`:
  - `__init__`: logging de inicialización.
  - `actualizar(frame_ar)`: stub que valida frame y deja punto para lógica de estudio.

No está acoplado aún al loop principal, pero funciona como plantilla de app específica.

---

## Subpaquete `viernes/utils/` (infraestructura transversal)

### `viernes/utils/__init__.py`
Docstring del subpaquete de utilidades.

### `viernes/utils/logger.py`
Logger central del proyecto:

- Formato con tiempo relativo de alta precisión (`time.perf_counter`).
- `configurar_logger(...)`: crea y configura logger `viernes`.
- `get_logger()`: punto recomendado de acceso.

Permite trazabilidad temporal fina en módulos de tiempo real.

### `viernes/utils/timing.py`
Herramientas de medición:

- `medir_bloque(...)`: context manager para medir bloques.
- `medir_funcion`: decorador para medir funciones.
- `PipelineTimer`:
  - guarda muestras por métrica.
  - calcula percentiles P50/P95/P99.
  - emite reporte final por logger o stdout.

Es la base de observabilidad de rendimiento del pipeline.

---

## Qué está funcional hoy vs qué está en construcción

### Funcional
- Captura de cámara en hilo separado.
- Tracking de manos/caras/superficies.
- Renderizado AR de overlays.
- HUD de rendimiento y distancia de manos.
- Controles por teclado y screenshots.

### Parcial / en evolución
- Fusión real de múltiples sensores.
- Memoria vectorial (aún no usa LanceDB/embeddings reales).
- Integración de LLM con experiencia AR en vivo.
- Interacción por gestos avanzada.
- Aplicaciones de alto nivel acopladas al pipeline.

---

## Arquitectura lógica (resumen rápido)

- **Entrada visual**: `CameraCapture`
- **Percepción**: `SpatialTracker`
- **Render AR**: `ARCompositor`
- **Orquestación**: `ARPipeline`
- **Cognición**: `CognitiveEngine` + `KnowledgeSystem`
- **Apps/UX futura**: `StudyAssistant`, `GestureHandler`
- **Infraestructura**: `logger`, `timing`, `config`

---

## Riesgos técnicos observables y recomendaciones

1. **API key en código**  
   Mover `GEMINI_API_KEY` a variable de entorno (`.env`) para seguridad.

2. **Acoplamiento pendiente AR + Cognición**  
   Integrar `CognitiveEngine` con `ARPipeline.show_response()` para respuestas en overlay en tiempo real.

3. **Sistema de conocimiento básico**  
   Evolucionar `KnowledgeSystem` a embeddings + búsqueda vectorial para relevancia semántica real.

4. **Falta de pruebas automáticas**  
   Agregar tests unitarios para tracker, compositor y knowledge retrieval.

5. **Uso parcial de dependencias**  
   Algunas librerías están declaradas pero aún no explotadas en la implementación actual.

---

## Conclusión

Este proyecto es un **núcleo sólido de AR en tiempo real** con arquitectura bien separada para crecer hacia un asistente multimodal completo.  
La base visual está lista y operativa; la capa cognitiva ya tiene componentes funcionales independientes, pero falta conectarla al pipeline central para lograr la experiencia "VIERNES" integral de extremo a extremo.
