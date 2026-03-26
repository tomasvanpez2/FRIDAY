# VIERNES — RESEARCH_RESULTS (Consolidado)

Referencia técnica única del proyecto. Consolidado a partir de 3 fuentes (ChatGPT, Gemini, Perplexity), eligiendo por pregunta la respuesta **más útil y conservadora en recursos** (RAM/CPU) y eliminando redundancias.

---

### Pregunta 1
**Respuesta directa:** En CPU y con <4GB RAM, lo más realista para entrenar *desde cero* es un **GPT “nano”** (tipo `nanoGPT`/GPT-2-like) con arquitectura muy reducida (p.ej. 4 capas, 4 heads, embedding 128). `GPT-2 small` y `distilgpt2` suelen ser **demasiado grandes** para entrenar desde cero en <4GB (y aun fine-tuning puede apretar la RAM). `RWKV` puede ser eficiente en **inferencia** (memoria casi constante por longitud), pero “tiny” varía por checkpoint y no siempre será lo más pequeño en parámetros.

**Por qué importa para VIERNES:** El LM es el componente más caro. Si el modelo no cabe en RAM durante entrenamiento/inferencia, todo el proyecto se cae (OOM/latencia extrema).

**Implementación mínima:**

```python
from transformers import GPT2Config, GPT2LMHeadModel

# “nanoGPT-like” (muy pequeño) — entrenable en CPU con batch/context reducidos
config = GPT2Config(
    vocab_size=8000,
    n_layer=4,
    n_head=4,
    n_embd=128,
    n_positions=256,
)
model = GPT2LMHeadModel(config)
```

- Entrenamiento conservador en CPU: `batch_size` muy pequeño + `gradient_accumulation_steps`, `context` 128–256, `fp32`, checkpointing, evaluación frecuente.

**Librerías (Python) + versiones (referencia):**
- `torch==2.1.0` (o compatible)
- `transformers==4.35.0` (o compatible)
- (Opcional) `tokenizers==0.15.0`

**Limitación crítica:** Con <100MB de texto, un modelo tan pequeño puede producir salida **poco coherente** y/o sobreajustar. Además, entrenar en CPU puede ser **muy lento** (minutos/horas por epoch según tamaño/contexto).

**Alternativa si falla:** Reducir aún más el modelo y aceptar menor calidad (char-level/word-level), o entrenar solo adaptadores (LoRA) sobre un modelo base pequeño *si se relaja* la restricción “desde cero”. Como alternativa “arquitectura”, probar RWKV por su perfil de memoria en inferencia, pero sin asumir que “tiny” será siempre el más liviano en parámetros.

---

### Pregunta 2
**Respuesta directa:** El preentrenamiento de un LM desde cero es **next-token prediction** (entropía cruzada). En un corpus <100MB, suelen bastar **pocas épocas** para bajar fuerte la loss, y luego entra en rendimientos decrecientes; el riesgo principal es **overfitting**. “Aprendió” cuando la **loss de validación** se estabiliza y la generación es coherente (no solo memoriza).

**Por qué importa para VIERNES:** El dataset es pequeño y técnico: necesitas señales claras (loss/validación) para no gastar CPU en entrenamiento inútil o destructivo.

**Implementación mínima:**

```python
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="out",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    learning_rate=1e-4,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

**Librerías (Python) + versiones (referencia):**
- `torch==2.1.0`
- `transformers==4.35.0`
- `datasets` (si lo usas para cargar corpus; versión según entorno)

**Limitación crítica:** En corpus pequeño, la loss de train puede seguir bajando mientras la de validación **sube** (memoriza). Además, “loss objetivo” depende de tokenizador/vocab; no hay un único número mágico.

**Alternativa si falla:** Regularización (dropout/weight decay), early stopping por validación, mezclar datos sintéticos/augmentación ligera, o cambiar a estrategia de fine-tuning (si se permite) en vez de preentrenar desde cero.

---

### Pregunta 3
**Respuesta directa:** Aprendizaje incremental = seguir entrenando con texto nuevo sesión a sesión. El problema es **catastrophic forgetting**. De las opciones, lo más liviano en RAM/CPU suele ser **LoRA incremental** (entrenas pocos parámetros y congelas el modelo base). `Replay buffer` es conceptualmente simple y robusto, pero agrega costo de entrenamiento; `EWC` agrega complejidad/costo por el cálculo de importancias (Fisher).

**Por qué importa para VIERNES:** VIERNES debe mejorar con el uso sin perder lenguaje/hábitos previos.

**Implementación mínima (LoRA conceptual):**

```python
# Esquema (alto nivel):
# 1) congelar modelo base
# 2) inyectar LoRA en proyecciones lineales
# 3) entrenar solo parámetros LoRA con lr bajo por sesión
```

**Librerías (Python) + versiones (referencia):**
- `torch==2.1.0`
- `transformers==4.35.0`
- `peft==0.7.0` (LoRA)

**Limitación crítica:** Si acumulas adaptadores por sesión, la RAM y la complejidad crecen. “Mergear” LoRA al modelo base periódicamente puede degradar si no se controla (y sigue existiendo olvido parcial).

**Alternativa si falla:** Replay buffer pequeño (muestras antiguas) mezclado con datos nuevos; o reentrenamiento offline periódico con el corpus acumulado (más simple que “online puro”).

---

### Pregunta 4
**Respuesta directa:** BPE (Byte-Pair Encoding) aprende subpalabras fusionando pares frecuentes para construir un vocabulario eficiente. Para español técnico y vocab=8000, puedes entrenar un tokenizador BPE desde cero con `tokenizers`.

**Por qué importa para VIERNES:** Un tokenizador propio reduce longitud de secuencia y mejora eficiencia/consistencia en términos técnicos en español.

**Implementación mínima (vocab 8000):**

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus_tecnico.txt"],
    vocab_size=8000,
    min_frequency=2,
)
tokenizer.save_model("viernes_tokenizer", "bpe_8000")
```

**Librerías (Python) + versiones (referencia):**
- `tokenizers==0.15.0`

**Limitación crítica:** Vocab 8000 puede quedarse corto para jerga/abreviaturas nuevas; reentrenar tokenizador cambia el mapeo de tokens (impacta compatibilidad con modelos ya entrenados).

**Alternativa si falla:** Unigram LM / SentencePiece, o subir vocab (p.ej. 12k–16k) si el modelo lo permite.

---

### Pregunta 5
**Respuesta directa:** PPO es un algoritmo RL **actor-critic**: el *actor* (política) decide acciones; el *critic* (value function) estima retorno esperado. PPO es estable porque limita cuánto puede cambiar la política por update con un objetivo “clipped”, evitando saltos grandes.

**Por qué importa para VIERNES:** En señales ruidosas (recompensas implícitas), PPO tiende a ser más estable que policy gradient simple.

**Implementación mínima:**

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, learning_rate=1e-4, clip_range=0.2, verbose=1)
model.learn(total_timesteps=10_000)
```

**Librerías (Python) + versiones (referencia):**
- `gymnasium==0.29.1`
- `stable-baselines3==2.2.0`

**Limitación crítica:** PPO es on-policy: consume muestras frescas y puede ser lento si el entorno es caro. Hiperparámetros/reward shaping importan mucho.

**Alternativa si falla:** A2C (más simple), o DQN si el espacio de acciones es discreto y el problema lo permite.

---

### Pregunta 6
**Respuesta directa:** Define `observation_space=Box(shape=(20,))` y `action_space=Discrete(10)`, e implementa `reset()`/`step()` retornando la 5-tupla de Gymnasium: `(obs, reward, terminated, truncated, info)`.

**Por qué importa para VIERNES:** Necesitas una interfaz RL estándar para aprender políticas sobre el “estado” del usuario/sistema.

**Implementación mínima:**

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ViernesEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(20, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(20, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}
```

**Librerías (Python) + versiones (referencia):**
- `gymnasium==0.29.1`
- `stable-baselines3==2.2.0`

**Limitación crítica:** Si el vector de 20 floats no captura bien el problema, PPO no aprende. Rewards muy esparsos = aprendizaje lento/inestable.

**Alternativa si falla:** Vectorizar entornos (`VecEnv`) si hay CPU suficiente, o simplificar acciones/estado.

---

### Pregunta 7
**Respuesta directa:** Reward shaping implícito es derivar recompensa de señales observables sin que el usuario pulse “bien/mal”. Ejemplos concretos:
- **Tiempo activo** (teclado/ratón) tras una sugerencia (más actividad útil → recompensa +).
- **Corrección inmediata** (borrar/rehacer en <2s) → penalización.
- **Pausas largas / idle** tras sugerencia → penalización (con cuidado).
- **Patrones de scroll/lectura** (leer la respuesta completa vs cerrar rápido).
- **Señales faciales** (frustración/concentración) como modulador (no como verdad absoluta).

**Por qué importa para VIERNES:** Es la base para que aprenda “solo” sin fricción.

**Implementación mínima:**

```python
reward = (
    0.1 * keystroke_rate
    - 0.2 * immediate_undo_rate
    - 0.1 * idle_time_seconds
    - 0.1 * frustration_score
)
```

**Librerías (Python) + versiones (referencia):**
- (Opcional, según señales) `opencv-python==4.9.0`
- (Opcional) MediaPipe (si ya lo usas en visión)

**Limitación crítica:** Shaping mal diseñado puede inducir políticas “tramposas” (optimiza la métrica, no la utilidad). Muchas señales implícitas son ruidosas y personales.

**Alternativa si falla:** Añadir un feedback explícito opcional (1 tecla/atajo), o usar reward más directo por “tarea completada”.

---

### Pregunta 8
**Respuesta directa:** Un PPO con `MlpPolicy` para 20 floats y 10 acciones es **pequeñísimo** (decenas de KB a pocos MB de pesos). El consumo real viene de Python/PyTorch/SB3, no de la red. 1000 pasos en CPU suele estar en el orden de **segundos** (muy dependiente de la dinámica del entorno y logging).

**Por qué importa para VIERNES:** Confirma que RL no debería ser el cuello de botella, si el entorno es barato.

**Implementación mínima:**

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000)
```

**Librerías (Python) + versiones (referencia):**
- `stable-baselines3==2.2.0`
- `gymnasium==0.29.1`

**Limitación crítica:** Si tu `step()` hace visión/IO pesado, el costo por paso domina. Además, PPO hace updates periódicos que pueden “picar” CPU.

**Alternativa si falla:** Reducir frecuencia de updates, simplificar `net_arch`, o cambiar a métodos tabulares si el problema es muy pequeño.

---

### Pregunta 9
**Respuesta directa:** CTC entrena un modelo para mapear audio→secuencia de tokens sin alineación explícita. Una arquitectura mínima típica: **Conv1D + (Bi)GRU + Linear**, <5M parámetros si se dimensiona modestamente. Dataset libre en español: **Mozilla Common Voice (es)** (licencia libre, ampliamente usada).

**Por qué importa para VIERNES:** Permite ASR local sin Whisper, con un modelo propio relativamente pequeño (aunque entrenar bien sigue siendo caro).

**Implementación mínima (arquitectura ejemplo):**

```python
import torch.nn as nn

class ASRModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out)
```

**Librerías (Python) + versiones (referencia):**
- `torch==2.1.0`

**Limitación crítica:** Entrenar ASR “de verdad” requiere **muchos datos y tiempo**; en CPU puede ser impráctico. CTC sin LM externo suele cometer errores en nombres técnicos.

**Alternativa si falla:** En vez de ASR completo, reconocer *palabras clave* / comandos; o usar un ASR offline preentrenado local (si se permite) con footprint pequeño.

---

### Pregunta 10
**Respuesta directa:** Un mel-spectrograma representa energía tiempo-frecuencia en escala mel. Conversión de audio 16kHz a 80 bandas con `librosa` o `torchaudio`:

**Por qué importa para VIERNES:** Es el “frontend” estándar para ASR ligero (CTC).

**Implementación mínima (librosa):**

```python
import numpy as np
import librosa

y = np.array(audio, dtype=np.float32)  # mono, 16kHz
S = librosa.feature.melspectrogram(
    y=y, sr=16000, n_fft=400, hop_length=160, n_mels=80
)
S_db = librosa.power_to_db(S, ref=np.max)
```

**Implementación mínima (torchaudio):**

```python
import torch
import torchaudio

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)
db = torchaudio.transforms.AmplitudeToDB()
S = mel(torch.tensor(y).unsqueeze(0))
S_db = db(S)
```

**Librerías (Python) + versiones (referencia):**
- `librosa==0.10.1` (si usas librosa)
- `torch==2.1.0` y `torchaudio` (si usas torchaudio)

**Limitación crítica:** FFT en tiempo real consume CPU; en hardware limitado conviene bajar resolución (hop más grande) o usar implementación más eficiente (torchaudio suele ganar).

**Alternativa si falla:** Precomputar features offline (si aplica) o reducir `n_mels`/sample_rate.

---

### Pregunta 11
**Respuesta directa:** VAD por energía RMS: divide en frames, calcula RMS y activa voz si RMS supera un umbral adaptativo basado en el ruido de fondo (mediana/media móvil).

**Por qué importa para VIERNES:** Evita correr ASR siempre; ahorra CPU/RAM.

**Implementación mínima (numpy):**

```python
import numpy as np

def vad_rms(y, sr=16000, frame_ms=30, thresh_ratio=1.5):
    frame_len = int(sr * frame_ms / 1000)
    rms = []
    for i in range(0, len(y), frame_len):
        frame = y[i:i+frame_len]
        if len(frame) == 0:
            continue
        rms.append(np.sqrt(np.mean(frame**2) + 1e-9))
    rms = np.array(rms)
    noise_floor = np.median(rms)  # robusto
    threshold = noise_floor * thresh_ratio
    return rms > threshold
```

**Librerías (Python) + versiones (referencia):**
- `numpy` (versión según entorno)

**Limitación crítica:** Confunde voz con ruidos fuertes. Si el ruido cambia, el umbral se descalibra.

**Alternativa si falla:** WebRTC VAD (clásico y ligero) o VAD por bandas de frecuencia.

---

### Pregunta 12
**Respuesta directa:** Wake word por correlación de espectrograma compara un patrón (template) con la ventana actual. Sin modelo entrenado suele ser **poco confiable** (muchos falsos positivos) y sensible a ruido/velocidad/tono.

**Por qué importa para VIERNES:** Necesitas activación manos-libres sin gastar CPU todo el tiempo.

**Implementación mínima:** Recomendación conservadora: usar correlación solo como *prefiltro* tras VAD, no como decisión final.

**Librerías (Python) + versiones (referencia):**
- `numpy`
- (Opcional) `librosa==0.10.1` o `torchaudio` para features

**Limitación crítica:** Alta tasa de falsos positivos/negativos sin un clasificador real.

**Alternativa si falla:** Un clasificador tiny (CNN sobre MFCC/mel) con pocos parámetros (order \(10^5\)) entrenado con positivos/negativos, o una librería de wake-word local ligera si se permite.

---

### Pregunta 13
**Respuesta directa:** Griffin-Lim reconstruye fase a partir de magnitud (iterativo, iSTFT). Produce audio **inteligible pero metálico**; típicamente inferior a vocoders neuronales (WaveNet/WaveGlow) en naturalidad, pero puede ser suficiente para “voz de asistente” si priorizas recursos.

**Por qué importa para VIERNES:** Permite TTS/vocoder sin un modelo grande.

**Implementación mínima (librosa):**

```python
import numpy as np
import librosa

# S_db: mel en dB (80 x T)
S = librosa.db_to_power(S_db)
mel_basis = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
inv_mel = np.dot(np.linalg.pinv(mel_basis), S)
audio = librosa.griffinlim(inv_mel, n_iter=32, hop_length=160, win_length=400)
```

**Librerías (Python) + versiones (referencia):**
- `librosa==0.10.1`
- `numpy`

**Limitación crítica:** Iteraciones = costo CPU; calidad limitada (artefactos).

**Alternativa si falla:** Vocoder ligero (MelGAN/WaveRNN “lite”) si cabe, o síntesis paramétrica/voz pregrabada si solo necesitas prompts básicos.

---

### Pregunta 14
**Respuesta directa:** LanceDB es un vector store local (Arrow/Parquet). Permite crear tablas, insertar embeddings con metadata y hacer búsqueda de vecinos más cercanos.

**Por qué importa para VIERNES:** Memoria semántica persistente sin servidor.

**Implementación mínima:**

```python
from lancedb import connect

db = connect("viernes_lancedb")
table = db.create_table(
    "memories",
    schema={"id": "int64", "vector": ("vector", 384), "text": "string"},
)

table.insert([{"id": 1, "vector": [0.0] * 384, "text": "Ejemplo de memoria"}])

hits = table.search([0.0] * 384, "vector").limit(5).to_list()
```

**Librerías (Python) + versiones (referencia):**
- `lancedb==0.5.0` (verificar en tu entorno)

**Limitación crítica:** Sin índices/optimización, el costo puede crecer con el tamaño (y el IO a disco puede ser cuello). Crear índices también cuesta recursos.

**Alternativa si falla:** FAISS CPU-only, Annoy o incluso búsqueda exacta con NumPy para pocos vectores.

---

### Pregunta 15
**Respuesta directa:** `all-MiniLM-L6-v2` es un encoder de oraciones que produce embeddings de 384 dimensiones. Puede usarse offline después de la primera descarga. En CPU, la latencia depende del hardware y longitud del texto; en footprint, suele estar en el orden de **~0.1GB** para pesos + overhead.

**Por qué importa para VIERNES:** Embeddings para memoria vectorial local.

**Implementación mínima:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
emb = model.encode("Texto técnico en español")
```

**Librerías (Python) + versiones (referencia):**
- `sentence-transformers==2.3.0`
- `torch==2.1.0`

**Limitación crítica:** El modelo no es el más fuerte en español (fue muy optimizado para inglés). Batch=1 puede ser relativamente lento si haces muchas consultas.

**Alternativa si falla:** Un modelo multilingual MiniLM (más pesado) o vectorizadores clásicos (TF-IDF) si priorizas RAM/CPU sobre semántica.

---

### Pregunta 16
**Respuesta directa:** La curva del olvido se modela de forma conservadora con decaimiento exponencial:
\[
R(t)=e^{-t/S}
\]
donde \(t\) es tiempo transcurrido y \(S\) controla la “fuerza” (más grande = olvida más lento). Para un score: `score *= R(t)`.

**Por qué importa para VIERNES:** Permite priorizar recuerdos recientes/relevantes sin crecer memoria indefinidamente.

**Implementación mínima (lista de nodos con timestamp):**

```python
import math
import time

def ebbinghaus_retention(age_seconds: float, strength_seconds: float) -> float:
    return math.exp(-age_seconds / strength_seconds)

def decay_score(score: float, timestamp: float, strength_seconds: float = 5 * 24 * 3600) -> float:
    age = time.time() - timestamp
    return score * ebbinghaus_retention(age, strength_seconds)

for node in nodes:
    node["score"] = decay_score(node["score"], node["timestamp"])
```

**Librerías (Python) + versiones (referencia):**
- stdlib (`math`, `time`)

**Limitación crítica:** Elegir \(S\) es un hiperparámetro del producto (depende del tipo de memoria). Exponencial es simple pero puede “matar” recuerdos útiles si no hay refuerzos.

**Alternativa si falla:** Decay lineal por ventanas, o decay condicionado a “revisiones” (spaced repetition).

---

### Pregunta 17
**Respuesta directa:** Sin modelos extra, puedes clasificar actividad con reglas sobre:
- **manos** (velocidad/varianza de landmarks),
- **cara** (estabilidad, ceño),
- **movimiento global** (frame diff u optical flow ligero).

**Por qué importa para VIERNES:** Permite adaptar interrupciones/ayuda al contexto del usuario con costo bajo.

**Implementación mínima (métricas + reglas):**

```python
# Features (ejemplo):
# hand_speed: media de ||lm_t - lm_{t-1}|| en landmarks de mano
# face_motion: métrica simple de movimiento (frame diff normalizado)
# brow_frown: distancia/relación entre puntos de cejas (menor = más ceño)

if hand_speed > H and brow_frown < F:
    activity = "debugging"
elif hand_speed > H:
    activity = "coding"
elif hand_speed < L and face_motion > M:
    activity = "reading"
elif hand_speed < L and face_motion < M:
    activity = "idle"
else:
    activity = "neutral"
```

**Librerías (Python) + versiones (referencia):**
- `opencv-python==4.9.0`
- MediaPipe (si ya lo usas para landmarks)

**Limitación crítica:** Heurísticas frágiles: cambian por usuario, iluminación, postura, cámara. Requiere calibración/umbrales.

**Alternativa si falla:** Un clasificador ligero (árbol/SVM) entrenado con esas features (sin deep learning pesado), o un HMM para suavizar estados en el tiempo.

---

### Pregunta 18
**Respuesta directa:** Con FaceMesh (468 landmarks), detecta emociones básicas usando **distancias/relaciones**: cejas (frustración), apertura ocular (concentración), tensión de boca. Landmarks exactos pueden variar; lo importante es usar puntos consistentes y normalizar por distancia interocular.

**Por qué importa para VIERNES:** Modula el comportamiento del agente (no interrumpir si frustración alta; tono/ayuda).

**Implementación mínima (idea):**

```python
# Normaliza por distancia entre ojos para robustez a escala
# eye_dist = ||lm_eye_left - lm_eye_right||
# brow_gap = ||lm_brow_left - lm_brow_right|| / eye_dist
# eye_open = ||lm_upper_lid - lm_lower_lid|| / eye_dist

if brow_gap < T_frown and eye_open < T_eye_low:
    emotion = "frustrado"
elif eye_open > T_eye_high and brow_gap >= T_frown:
    emotion = "concentrado"
else:
    emotion = "neutral"
```

**Librerías (Python) + versiones (referencia):**
- MediaPipe (para landmarks)
- `numpy`

**Limitación crítica:** Alta variabilidad individual; oclusiones y mala luz degradan landmarks. No es “verdad”, es una señal ruidosa.

**Alternativa si falla:** Mantener solo “frustración probable” con umbrales conservadores, o pedir confirmación ocasional (muy baja frecuencia).

---

### Pregunta 19
**Respuesta directa:** Para detectar actividad a 15 FPS (640×480), el orden de ligereza típico es:
1) **Diferencia de frames** (más ligero),
2) **Lucas–Kanade** (sparse),
3) **Farneback** (dense, más pesado).
Farneback puede saturar CPU; frame diff suele ser viable incluso en hardware modesto.

**Por qué importa para VIERNES:** Evita que visión se coma el presupuesto CPU y baje de 15 FPS.

**Implementación mínima (frame diff):**

```python
import cv2
import numpy as np

diff = cv2.absdiff(prev_gray, gray)
motion_level = float(np.sum(diff)) / diff.size
```

**Implementación mínima (Lucas–Kanade, sparse):**

```python
import cv2
import numpy as np

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=7)
p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
motion = float(np.mean(np.linalg.norm(p1 - p0, axis=2)))
```

**Librerías (Python) + versiones (referencia):**
- `opencv-python==4.9.0`
- `numpy`

**Limitación crítica:** Cambios de iluminación generan falsos “movimientos”. Flujos ópticos densos (Farneback) suelen ser demasiado caros sin optimización o bajar resolución.

**Alternativa si falla:** Bajar resolución (320×240), usar blur + threshold con frame diff, o correr visión en hilo separado.

---

### Pregunta 20
**Respuesta directa:** Usa patrón productor–consumidor con `queue.Queue(maxsize=N)` y `timeout` para que el loop de AR (15 FPS) nunca se bloquee. Descarta frames si la cola está llena.

**Por qué importa para VIERNES:** Mantiene percepción fluida aunque el “cerebro” (LM/RL) sea lento.

**Implementación mínima:**

```python
import threading
import queue

frame_queue = queue.Queue(maxsize=5)
running = True

def producer():
    while running:
        frame = capture_frame()
        try:
            frame_queue.put(frame, timeout=0.01)
        except queue.Full:
            pass

def consumer():
    while running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        process_frame(frame)
        frame_queue.task_done()

threading.Thread(target=producer, daemon=True).start()
threading.Thread(target=consumer, daemon=True).start()
```

**Librerías (Python) + versiones (referencia):**
- stdlib (`threading`, `queue`)

**Limitación crítica:** El GIL limita paralelismo real para tareas CPU-bound; si `process_frame` es pesado, la cola se llena y se perderán frames (diseño esperado).

**Alternativa si falla:** `multiprocessing` para CPU-bound, o mover el cómputo pesado a extensiones nativas / Torch ops.

---

### Pregunta 21
**Respuesta directa:** Una estimación conservadora para runtime “todo a la vez” suele estar ~**1–1.5GB** (sumas + overhead Python/OS/buffers), lo que cabe en 6GB con margen si controlas buffers. Componentes que conviene **cargar bajo demanda**: LM, ASR, pipelines de visión si no siempre activos; la base vectorial puede vivir en disco.

**Por qué importa para VIERNES:** Define si el sistema es viable en el objetivo ARM 6GB y qué módulos deben ser “lazy-loaded”.

**Implementación mínima (idea):**
- Cargar LM/ASR solo cuando se necesiten, y liberar referencias (`del model`) + `gc.collect()` si corresponde.
- Mantener VAD/colas/buffers pequeños.

**Librerías (Python) + versiones (referencia):**
- stdlib (`gc`)
- `torch==2.1.0`

**Limitación crítica:** Cargar modelos a demanda introduce latencia (segundos) y Python no siempre devuelve memoria al SO inmediatamente (fragmentación/caching).

**Alternativa si falla:** Mantener modelos cargados pero reducir tamaño (cuantización/arquitecturas más pequeñas), o desactivar módulos por “modo” (voz vs visión vs texto).

---

### Pregunta 22
**Respuesta directa:** `torch.no_grad()` desactiva autograd en inferencia, reduciendo memoria temporal (no guarda activaciones para backprop). `torch.inference_mode()` suele ser aún mejor para inferencia pura. Otras optimizaciones relevantes: `model.eval()`, cuantización int8 (reduce RAM), y `torch.compile()` (puede acelerar tras costo inicial).

**Por qué importa para VIERNES:** Sin esto, la inferencia puede duplicar memoria y volverse lenta, especialmente con modelos grandes para el hardware objetivo.

**Implementación mínima:**

```python
import torch

model.eval()
with torch.inference_mode():
    y = model(x)
```

**Cuantización dinámica (idea):**

```python
import torch

model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Librerías (Python) + versiones (referencia):**
- `torch==2.1.0`

**Limitación crítica:** `torch.compile()` puede aumentar el tiempo de arranque; cuantización puede degradar calidad (sobre todo en modelos pequeños) y requiere validar outputs/perf.

**Alternativa si falla:** Exportar a ONNX y usar ONNX Runtime en CPU; o reducir batch/contexto y usar modelos más pequeños.

---
