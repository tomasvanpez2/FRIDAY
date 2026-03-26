
Respuestas detalladas de investigación para VIERNES
Pregunta 1
Respuesta directa: El menor transformer entrenable (~<4GB RAM) es una versión muy reducida tipo nanoGPT con 4 capas, 4 cabezas y dimensión 128 (≈0.5M parámetros)
. GPT-2 small (12 capas, 12 cabezas, 768 dim, ≈117M parámetros
) y su versión distilada DistilGPT-2 (6 capas, 12 cabezas, 768 dim, ≈82M
) son mucho más grandes. El modelo RWKV-v4 tiny usa arquitectura recurrente (transformer-RNN híbrido) con parámetros reducidos; destaca por uso de memoria constante en inferencia
. DistilGPT-2 (82M) ya reduce ~30% de parámetros respecto a GPT-2 original
. En términos de memoria: GPT-2 small usa ~0.5GB para pesos (float32) y varios GB en entrenamiento; DistilGPT-2 ~0.3GB de pesos. NanoGPT mini ocuparía <<1GB total. RWKV (en modo inferencia) mantiene la memoria casi constante
, por lo que incluso “tiny” cabe cómodamente.

Por qué importa para VIERNES: El hardware limitado (6–16 GB RAM CPU) exige modelos muy pequeños. Conocer parámetros y memoria de cada opción permite elegir el que pueda entrenar y correr localmente sin out-of-memory.

Implementación mínima: Por ejemplo, usando HuggingFace / PyTorch:

python
Copiar
from transformers import GPT2Config, GPT2LMHeadModel
# NanoGPT minimal (ejemplo con vocab=8000):
config = GPT2Config(vocab_size=8000, n_layer=4, n_head=4, n_embd=128)
nano_model = GPT2LMHeadModel(config)  # ~0.5M parámetros【2†L381-L389】

# GPT-2 Small (pre-entrenado):
small_model = GPT2LMHeadModel.from_pretrained('gpt2')  # 117M parámetros【25†L187-L195】

# DistilGPT-2 (pre-entrenado):
distil_model = GPT2LMHeadModel.from_pretrained('distilgpt2')  # 82M parámetros【27†L112-L120】

# RWKV (pseudo-código):
# WIP: se entrena con librería oficial RWKV, por ejemplo RWKV-Runner.
Limitación crítica: Los modelos grandes (GPT-2) consumen RAM intensivamente (peso + optimizador + activaciones), lo que puede superar 4 GB al entrenar. Modelos demasiado pequeños (nanoGPT minúsculo) generan texto de muy baja calidad. GPT-2 y Distil están entrenados en inglés, por lo que habría que afinarlos en español técnico (más tiempo). RWKV es novedoso; puede ser eficiente en memoria
, pero es menos probado y carece de ejemplos públicos para español.

Alternativa si falla: Si GPT-2/Distil no caben en memoria, usar cuantización (int8) o modelos aún más pequeños (ej. TinyStories GPT, char-RNN). Si nanoGPT pequeño genera incoherencias, se puede aumentar capas/dimensión moderadamente. Como último recurso, usar un modelo RNN tradicional (LSTM) o técnicas ligeras de seq2seq, aunque sacrifican fluidez. También evaluar ajustar hyperparámetros (batch pequeño, gradientes acumulados) para entrenar en CPU.

Pregunta 2
Respuesta directa: El preentrenamiento de un LM en un corpus pequeño es ajustar pesos por probabilidad de siguiente token. Se entrenan múltiples épocas (a menudo 10–20+) hasta que la pérdida (loss) se estabiliza. Al inicio la loss es alta (orden log(|vocabulario|) – ej. ~6–10) y cae rápidamente en las primeras épocas, luego se nivela
. Por ejemplo, estudios muestran “rapidez de reducción de pérdida al principio, luego convergencia más lenta”
. Un modelo “ha aprendido” cuando la pérdida de validación converge en un nivel bajo y genera texto coherente (perplejidad moderada).

Por qué importa para VIERNES: Se dispone de <100 MB texto técnico; saber cuántas pasadas (epochs) y qué pérdida esperar ayuda a evitar subentrenamiento (texto incomprensible) o sobreajuste (memorizar pocos documentos).

Implementación mínima:

python
Copiar
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
# Configurar tokenizer y modelo
config = GPT2Config(vocab_size=8000)
model = GPT2LMHeadModel(config)
# Preparar Trainer
args = TrainingArguments(output_dir='out', num_train_epochs=15, per_device_train_batch_size=4)
trainer = Trainer(model=model, args=args, train_dataset=mi_dataset)
trainer.train()  # preentrena varias épocas
Monitorizar trainer.train().history para ver la pérdida en entrenamiento/validación. Tip: usar .eval() y calcular loss periódico para comprobar que baja hasta estabilizar.

Limitación crítica: Con <100 MB el modelo tiende a memorizar (overfit) si muchas épocas; además, corpus técnico puede ser sesgado. La pérdida final alta indica que modelo aún no generaliza. No hay garantía de “entender” lenguaje: solo producirá patrones vistos. Muchos ruidos o datos pobres conducirán a baja calidad de generación.

Alternativa si falla: En lugar de entrenar desde cero, podría usarse una estrategia de fine-tuning incremental sobre un modelo más pequeño (si está permitido). O generar datos sintéticos para enriquecer el corpus. Reducir aún más tamaño del modelo o usar regularización (dropout) para combatir el sobreajuste. Otra opción: entrenar primero en corpus general más grande (latente) y luego afinar en el pequeño corpus técnico (transfer learning).

Pregunta 3
Respuesta directa: El aprendizaje incremental (continuo) consiste en seguir ajustando el modelo con nuevos datos por sesión, sin reiniciar desde cero. Sin embargo, esto causa olvido catastrófico: se pierden conocimientos previos al ajustarse a nuevos textos
. Para mitigarlo se usan técnicas como EWC (penalizar cambios en parámetros cruciales), LoRA incremental (añadir/adaptar pequeños módulos entrenables) o replay buffer (reservar algunos ejemplos antiguos). Por ejemplo, EWC añade en la pérdida un término que castiga alejar pesos importantes
. Replay mantiene y reentrena con datos viejos (la estrategia más efectiva para LLMs
). De las opciones, LoRA incremental es más liviano: solo entrena adaptadores pequeños sin modificar todo el modelo (pocos parámetros), mientras EWC requiere calcular matrices de Fisher (costoso) y replay demanda almacenamiento y CPU para reentrenar con ejemplos previos.

Por qué importa para VIERNES: VIERNES debe incorporar cada sesión nueva (feedback, correcciones) sin perder lo aprendido antes. Sin protección, actualizar el modelo directamente puede borrar lenguaje aprendido y habilidades previas
.

Implementación mínima:

EWC (Ejemplo conceptual): tras entrenar, calcular la diagonal de la matriz de Fisher (información de cada peso). Al ajustar con nuevos datos, usar pérdida combinada:
python
Copiar
loss_new = loss_original + (λ/2) * Σ_i F_i * (θ_i - θ_old_i)^2
LoRA incremental: Congelar el modelo base y añadir módulos LoRA a capas Transformer (pr. linear). Al recibir nueva sesión, entrenar solo estos parámetros adicionales.
Replay buffer: Mantener un buffer circular de pares (texto, respuesta) antiguos. En cada actualización de sesión, mezclar muestras nuevas con antiguas al entrenar el modelo.
Limitación crítica:
EWC y similares son costosos y pierden efectividad conforme se acumulan más tareas
. Replay requiere almacenar ejemplos (memoria) y tiempo de entrenamiento extra. Ninguna técnica garantiza retención perfecta; siempre hay compromiso entre estabilidad y plasticidad. Además, en modelo local la RAM limitada impone fronteras fuertes a estos métodos.

Alternativa si falla: Si las anteriores no caben en 6 GB, una alternativa es reentrenar periódicamente el modelo completo usando todo el corpus acumulado (batch offline) en lugar de continuo. También se puede usar técnicas ligeras tipo Elastic Weight Consolidation simplificado o simplemente aprender a una tasa muy baja (mínimo ajuste) para reducir el olvido. Si LoRA genera demasiados módulos, quizá convenga mergearlos y mantener solo los pesos base reducidos.

Pregunta 4
Respuesta directa: Un tokenizer BPE (Byte-Pair Encoding) es un segmentador de subpalabras que fusiona iterativamente los pares de caracteres (o tokens) más frecuentes para formar nuevos tokens
. Así crea un vocabulario ajustado al corpus. Para entrenarlo en texto técnico español con 8000 vocab, se usa tokenizers de HuggingFace, por ejemplo:

python
Copiar
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
# Entrenar en los archivos de texto (listas de oraciones técnicas en español):
tokenizer.train(files=["tech_spanish.txt"], vocab_size=8000, min_frequency=2)
# Guardar vocabulario y merges:
tokenizer.save_model("mi_tokenizer", "bpe_8000")
Esto produce archivos vocab.json y merges.txt con 8000 tokens BPE.

Por qué importa para VIERNES: El tokenizador define cómo el modelo ve el texto. Un vocabulario bien ajustado al lenguaje técnico español evita tokens desconocidos y reduce longitud de secuencias, aprovechando mejor el modelo mínimo.

Implementación mínima:

Recolectar corpus de texto técnico en español (CSV o TXT).
Instalar tokenizers:
bash
Copiar
pip install tokenizers
Código (como arriba) para instanciar ByteLevelBPETokenizer, entrenar con vocab_size=8000.
Cargar el tokenizador para el modelo de lenguaje:
python
Copiar
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("bpe_8000-vocab.json")
O convertir al formato de Transformers para usar con el LM.
Limitación crítica: Un vocabulario fijo de 8000 puede quedarse corto para términos muy específicos o neologismos, generando tokens subóptimos. Además, BPE solo captura frecuencia global, no relaciones semánticas. El entrenamiento de tokenizador no penaliza rarezas ni homógrafos.

Alternativa si falla: En vez de BPE puro, podría usarse Unigram LM (tokenizers) o WordPiece, que ofrecen mecanismos distintos de selección de subpalabras. También se puede entrenar un vocabulario mayor (más tokens) para cubrir técnica, o reducir a 5000 para modelo aún más ligero (aunque con más OOV). Como último recurso, usar tokenización basada en caracteres (uno o dos caracteres) garantiza cobertura total pero alarga secuencias.

Pregunta 5
Respuesta directa: PPO (Proximal Policy Optimization) es un algoritmo actor-critic on-policy. La política (actor) es una red que da una distribución de acciones dada cada estado, y la value function (critic) es otra red que estima la recompensa futura esperada de un estado. Durante cada actualización, el actor propone acciones y el critic calcula su valor; luego se ajusta la política para mejorar la recompensa anticipada. PPO es estable gracias a un mecanismo de clipping: en la función objetivo se recortan las razones de probabilidad para evitar que la nueva política se aleje demasiado de la anterior
, evitando cambios bruscos. En resumen: actor = red de política; critic = red de valor; ambos se entrenan simultáneamente. PPO limita la magnitud de cada actualización (mantiene la nueva política “cerca” de la vieja) lo que lo hace más estable y robusto que un simple policy gradient.

Por qué importa para VIERNES: PPO suele converger sin las inestabilidades de otros algoritmos (como A2C puro)
. En un entorno custom con pocos recursos, queremos aprendizaje gradual sin oscilaciones grandes en la política.

Implementación mínima:
Usando Stable-Baselines3:

python
Copiar
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, learning_rate=1e-4, clip_range=0.2, verbose=1)
model.learn(total_timesteps=10000)
Aquí MlpPolicy crea actor y critic MLP; clip_range es el parámetro de PPO para recorte
. Se podría mostrar ejemplos más básicos con redes PyTorch, pero SB3 facilita.

Limitación crítica: PPO es on-policy, requiere recolectar datos nuevos cada vez y puede ser lento de entrenar en CPU (mucha aleatoriedad y muestras). Además, calcular la función de valor añade overhead. En hardware limitado, cuidado con batch grande. PPO funciona bien para entornos continuos y discretos, pero en espacios muy simples podría ser sobrekill.

Alternativa si falla: Si PPO resulta demasiado pesado, se puede probar A2C (sin clipping) o incluso DQN (para acciones discretas) si el problema lo permite. A2C mantiene actor-critic pero sin clip (menos estable). DQN sería off-policy (requiere menos muestras online). Otra opción ligera: entrenamiento de SARSA/Q-learning clásico usando arreglos simples si el espacio es muy reducido.

Pregunta 6
Respuesta directa: Un entorno Gym personalizado para un vector de 20 floats y 10 acciones sería algo así: definir observation_space = spaces.Box(low=-inf, high=inf, shape=(20,)) y action_space = spaces.Discrete(10)
. Luego implementar reset() que retorna el estado inicial (array de 20 floats) y step(action) que aplica la acción, devuelve (nuevo_estado, recompensa, done, info). Ejemplo mínimo:

python
Copiar
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MiEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)  # 10 acciones posibles

    def reset(self):
        estado = np.zeros(20, dtype=np.float32)
        return estado, {}  # Gymnasium retorna (obs, info)

    def step(self, action):
        # Aquí definir dinámica: por ejemplo:
        nuevo_estado = np.random.rand(20).astype(np.float32)  # placeholder
        recompensa = 0.0
        done = False
        return nuevo_estado, recompensa, done, False, {}
Para conectarlo a Stable-Baselines3 y PPO:

python
Copiar
from stable_baselines3 import PPO
env = MiEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
La clave es usar spaces.Box para el vector continuo (20 floats) y spaces.Discrete(10) para las acciones
.

Por qué importa para VIERNES: VIERNES necesita simular entornos complejos (p.ej. tareas interactivas) en CPU; un custom Env permite definir exactamente la interfase de observaciones y acciones que el agente usa.

Implementación mínima: Ver arriba. Básicamente:

Crear clase con observation_space y action_space apropiados
.
Programar reset() para devolver estado inicial como un vector de 20 floats.
Programar step(action) con la lógica de transición, devolviendo (obs, reward, done, truncated, info). SB3 usa Gymnasium, por lo que devolvemos 5-tupla.
Luego instanciar PPO con este env y llamar model.learn().
Limitación crítica: El loop de Gym y Stable-Baselines3 añade overhead de Python (creación de estados, colas). En CPU limitado, entornos muy rápidos (loops simples) podrían saturar. Además, SB3 PPO consume CPU en cada paso. Para 15 FPS real-time, puede haber cuello de botella si el modelo es complejo.

Alternativa si falla: Si el rendimiento con Gym y SB3 es insuficiente, se puede simplificar más: usar un loop manual sin Gym, o usar Ray RLlib con menos overhead. También se puede vectorizar entornos (hacer varios en paralelo) para aprovechar CPU, o elegir un algoritmo de RL más ligero (DQN para discretos, A2C simple) que use menos muestras por actualización.

Pregunta 7
Respuesta directa: El reward shaping implícito significa derivar recompensas de comportamientos del usuario sin botones. Ejemplos de señales implícitas observables:

Tiempo de interacción activo: mayor tiempo sin pausas podría indicar interés (recompensa positiva), o si larga inactividad (idle) podría penalizar.
Movimiento de cursor/ratón: muchos clics o scrolls relevantes indican engagement (buen score), pocos pueden indicar desinterés.
Velocidad de tipeo o clic: respuestas rápidas pueden sugerir acertadas y satisfacción (mayor recompensa), lento sugiere confusión.
Expresiones faciales: por MediaPipe: sonrisa o asombro (ojos muy abiertos) podrían interpretarse como satisfacción (+), frente fruncida o ceño (frustración) como negativa.
Atención visual: por flujo óptico: mirar la pantalla fijo (poca variación ocular) indica concentrado positivo, desvíos frecuentes (mirar a otro lado) indicar posible frustración.
Por qué importa para VIERNES: VIERNES debe autoevaluarse sin que el usuario presione “bien” o “mal”. Estas señales implícitas permiten asignar recompensa automáticamente según comportamiento natural del usuario.

Implementación mínima: Medir métricas ejemplo:

Calcular porcentaje de tiempo de actividad (mouse/teclado en los últimos segundos).
Usar MediaPipe FaceMesh: detectar sonrisa (curva de labios) o ceño (distancia de cejas) para ajustar recompensa.
Contar clics o cambios de ventana (eventos OpenCV/OS).
Analizar flujo óptico ocular (cámaras): ojos bien abiertos (positivo), parpadeo rápido (negativo).
Limitación crítica: Estas señales son ruido: un usuario concentrado puede no moverse mucho (penalizarlo erróneamente), o un buggy código puede confundir. Interpretar correctamente es difícil y caso-de-uso específico. Se requieren calibración y umbrales empíricos.

Alternativa si falla: Diseñar recompensas más directas (p.ej. tareas cumplidas, valoración cuantitativa). Si implícito da falsos positivos, usar un botón opcional de feedback, o entrenar un pequeño clasificador de estado mental del usuario. Otra opción: usar reconocimiento de voz para confirmar satisfacción.

Pregunta 8
Respuesta directa: Un modelo PPO con MlpPolicy (2 capas de 64 neuronas por defecto) y espacio de 20 floats/10 acciones es muy ligero: el MLP tendrá ~(2064 + 6464 + 6410)*≈14k parámetros, ocupando pocos kilobytes. En RAM total, SB3 y PyTorch toman más, pero seguro <100 MB. En CPU, cada paso (acción->red->retroprop) es rápido (<0.1 ms). Mil pasos de entrenamiento (interacciones + actualización) en CPU básico suele tardar unos segundos (depende del CPU; p.ej. ~1–5 s por 1000 pasos en un procesador moderno de gama media).

Por qué importa para VIERNES: Con hardware limitado, necesitamos saber si este entrenamiento cabe en tiempo razonable. Con ~20 floats/10 acciones, PPO es tan pequeño que su peso en memoria y tiempo es despreciable frente a otros módulos, pero aún así acumula en cada episodio.

Implementación mínima: Usando SB3:

python
Copiar
from stable_baselines3 import PPO
env = MiEnv()  # del ejercicio anterior
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000)
Esto ejecutará ~1000 pasos de entorno+entrenamiento en CPU. Se puede medir con time.time() alrededor de learn() para obtener el tiempo real.

Limitación crítica: Aunque pequeño, PPO en CPU aún es más lento que algoritmos tabulares. Procesamiento de lotes y cálculo de gradientes en cada update consume CPU. Si el entorno es complejo de simular (no es solo vector), puede perder el requerimiento de 15 FPS. Además, SB3 agrega overhead de serialización y Python que aumenta latencia por paso.

Alternativa si falla: Si PPO resulta lento, se podría usar un algoritmo off-policy más ligero (DQN para discretos, que actualiza menos frecuentemente y puede reusar transiciones) o A2C que no guarda buffer grande. Otra opción es reducir la complejidad de la red (menos capas) o usar un paso de entorno más rápido (vectorizar entornos). También se puede interactuar menos frecuentemente y acumular actualizaciones.

Pregunta 9
Respuesta directa: Un modelo CTC (Connectionist Temporal Classification) para ASR es una red (por ejemplo Conv1D + RNN) que emite en cada frame la probabilidad de cada fonema o carácter, utilizando la CTC loss para alinear secuencia de audio con texto sin segmentación previa
. Una arquitectura mínima podría ser, por ejemplo, una capa convolucional 1D seguida de 2 capas bidireccionales GRU y una capa lineal de salida. Ejemplo aproximado (menos de 5M parámetros):

python
Copiar
import torch.nn as nn
class ASRModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)  # 256*2 por bidireccional
    def forward(self, x):
        # x: [batch, 1, time] con espectrograma, por ejemplo
        x = nn.functional.relu(self.conv(x))
        x = x.transpose(1, 2)  # -> [batch, time, features]
        out, _ = self.gru(x)   # -> [batch, time, 512]
        return self.fc(out)    # logits por time-step
Para entrenar usar torch.nn.CTCLoss() entre la secuencia de salidas y la transcripción. Un dataset en español libre es Mozilla Common Voice Español (cientos de horas, CC0), o VoxForge Spanish (CC BY-SA) que pueden usarse para preentrenar CTC.

Por qué importa para VIERNES: El reconocimiento de voz permite entrada natural sin servicios externos. CTC es eficiente para ASR sin alinear cuadros, y una red pequeña (conv+GRU) podría ajustarse a <5M parámetros para CPU.

Implementación mínima:

Conv1D extrae características locales, luego GRU modela dependencias temporales. Se usa CTC loss:
python
Copiar
import torch
ctc_loss = torch.nn.CTCLoss(blank=0)  # 0 token como blank
# Durante entrenamiento:
logits = model(audio_input)  # [time, batch, classes]
loss = ctc_loss(logits.log_softmax(2), targets, logit_lengths, target_lengths)
Obtener dataset: descargar Common Voice Spanish, extraer audio+texto, convertir audio a espectrograma (ver siguiente pregunta).
Limitación crítica: Los modelos CTC pequeños suelen tener errores de reconocimiento (similaridad fonética confusa) y requieren muchos datos transcritos. Además, CTC no modela lenguaje (no usa contexto gramatical). Sin un vocabulario específico, podría deletrear mal nombres técnicos. Se necesita mucha GPU/CPU para entrenar bien; en CPU será muy lento.

Alternativa si falla: Una alternativa liviana es usar un servicio ASR offline pre-entrenado (p.ej. VOSK con modelos pequeños) o un modelo secuencia-a-secuencia pequeño (p.ej. Transformer TinyASR, aún así pesado). Si ningún modelo recorta, usar detección de sonido para reconocer palabras clave en vez de full ASR.

Pregunta 10
Respuesta directa: Un espectrograma mel es una representación de la señal de audio donde los ejes son el tiempo y la frecuencia (en escala mel). Para obtener uno de 80 bandas mel desde un array de audio 16 kHz usando librosa o torchaudio:

python
Copiar
import numpy as np
import librosa

y = np.array(mi_audio, dtype=np.float32)  # señal mono 16 kHz
S = librosa.feature.melspectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=80)
S_db = librosa.power_to_db(S, ref=np.max)  # convierte a escala log (dB)
Con n_fft=400, hop_length=160 obtenemos ~25 ms de ventana y 10 ms salto. Con torchaudio:

python
Copiar
import torch, torchaudio
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)
db = torchaudio.transforms.AmplitudeToDB()
S = mel_spec(torch.tensor(y).unsqueeze(0))  # [1, 80, T]
S_db = db(S)
Ambos métodos devuelven un tensor 80×T con energía mel a lo largo del tiempo.

Por qué importa para VIERNES: Convertir audio en espectrogramas es el primer paso para cualquier ASR o procesamiento de voz (incluido CTC). Los 80 bancos mel son un estándar en ASR que destaca características auditivas importantes.

Implementación mínima: La del código arriba. En resumen:

Leer audio crudo (numpy 16kHz).
Llamar a librosa.feature.melspectrogram(...) con sr=16000, n_mels=80.
Aplicar librosa.power_to_db para escala log (opcional, pero común para entrada de red neuronal).
Esta transformación se puede hacer en cada paso de preprocesamiento o en tiempo real con torchaudio.
Limitación crítica: Calcular espectrogramas en CPU es costoso (FFT). Más filtros (80 bandas) significa más cálculos. En tiempo real 15 FPS, el CPU puede ser sobrecargado si hay muchos canales de audio. Además, librosa hace cálculos en Python que son más lentos que la versión optimizada de torchaudio.

Alternativa: Si librosa es lento, usar torchaudio (PyTorch C++) o implementar FFT vía scipy.signal.stft/openCV. O reducir la resolución (menos bandas mel o mayor hop) para ahorrar CPU. Como otra opción, convertir audio offline y almacenar los espectrogramas pre-calculados.

Pregunta 11
Respuesta directa: Un VAD por energía RMS marca segmentos de voz cuando la energía (RMS) supera un umbral dinámico. El umbral se ajusta al ruido de fondo calculando, por ejemplo, la media o mediana del RMS en segmentos silenciosos y multiplicándolo por un factor (p.ej. 1.5–2). En código (numpy):

python
Copiar
import numpy as np

def voice_activity_rms(y, sr=16000, frame_ms=30, thresh_ratio=1.5):
    frame_len = int(sr * frame_ms / 1000)
    # Calcula RMS por frame
    rms = []
    for i in range(0, len(y), frame_len):
        frame = y[i:i+frame_len]
        rms_val = np.sqrt(np.mean(frame**2) + 1e-9)
        rms.append(rms_val)
    rms = np.array(rms)
    # Umbral adaptativo basado en ruido (mediana de RMS)
    noise_floor = np.median(rms)
    threshold = noise_floor * thresh_ratio
    # Marcamos voz cuando RMS > threshold
    mask = rms > threshold
    return mask  # array booleano por frame
Este código divide el audio en frames de 30ms, calcula RMS de cada uno, estima el ruido de fondo como la mediana de todos los RMS y declara voz si supera mediana * 1.5.

Por qué importa para VIERNES: Detectar cuándo hay voz (wake word) o simplemente hablar ayuda a activar los módulos de ASR solo cuando el usuario habla, ahorrando recursos.

Implementación mínima: Como arriba. Clave: elegir bien la ventana (frame_len) y el factor (1.5–2). Se podría refinar calculando un umbral actualizado (p.ej. media móvil del ruido en silencio). El código calcula RMS manualmente sin bibliotecas externas.

Limitación crítica: RMS no distingue voz de ruidos fuertes (metro, ventilador). Umbral fijo puede fallar si el ruido cambia. Además, el tamaño de ventana afecta sensibilidad (ventanas grandes suavizan, ventanas pequeñas son más ruidosas). 1.5×mediana es heurístico simple; ambientes muy dinámicos requieren ajustes manuales.

Alternativa: Usar VAD basado en energía FFT o un algoritmo clásico (p.ej. WebRTC VAD) que modele mejor el ruido. O implementar detección de silencio basada en cepstrum u otro filtro adaptativo más sofisticado.

Pregunta 12
Respuesta directa: La detección de wake word por correlación de espectrogramas consiste en comparar la forma del espectrograma actual con el espectrograma de la palabra clave (por correlación cruzada). Sin un modelo entrenado, esto suele dar falsos positivos (cualquier sonido con patrón similar puede activarlo) y es sensible a ruido y variabilidad de pronunciación. Es poco confiable porque la correlación no capta variaciones sutiles del habla.

Por qué importa para VIERNES: VIERNES debería activarse por una palabra clave («¡Viernes!») sin procesar todo el tiempo. La correlación es un método muy ligero para wake-word, pero difícilmente suficiente por sí sola.

Implementación mínima: Se haría tomando, por ejemplo, un espectrograma mel de la frase clave pregrabada y correlacionándolo (por FFT) con el espectrograma entrante en ventanas de tiempo. No se muestran pormenores aquí porque no se recomienda.

Limitación crítica: La correlación lineal no es robusta ante variaciones en tono, velocidad o ruido. Muchos sonidos de fondo pueden coincidir con partes del patrón, dando falsos positivos frecuentes.

Alternativa si falla: Una alternativa liviana es un pequeño modelo DSP/ML entrenado (p.ej. un clasificador ligero o Dynamic Time Warping sobre coeficientes cepstrales), o usar detección basada en mel-cepstrum con umbrales adaptativos. Otra opción: libros de wake-words eficaces (como Porcupine de Picovoice) pero pueden no ser open-source. Si la correlación falla, lo más fiable es entrenar un modelo pequeño específico del wake-word (CNN de 1-2 capas) con datos positivos/negativos, que sigue siendo relativamente ligero para CPU.

Pregunta 13
Respuesta directa: Griffin-Lim es un algoritmo iterativo que reconstruye la fase de audio a partir del espectrograma de magnitudes (mel) usando transformadas inversas (iSTFT)
. Suena metálico y con artefactos, pero es inteligible. En estudios (ej. WaveGlow) el MOS de Griffin-Lim fue ~3.82/5 frente a ~3.88/5 de WaveNet
, mostrando calidad ligeramente inferior. Aunque no iguala la claridad y naturalidad de vocoders avanzados (WaveNet, WaveGlow), Griffin-Lim suele ser suficiente para voz de asistente entendible (tiembla un poco el timbre).

Por qué importa para VIERNES: Permite convertir los espectrogramas mel en audio sin red neuronal pesada. Es un compromiso: rápido y librería-completa (librosa implementa griffinlim), con calidad básica pero voz clara.

Implementación mínima: Con librosa:

python
Copiar
import numpy as np, librosa
# S_db: espectrograma mel en escala dB (80xT) obtenido antes
S = librosa.db_to_power(S_db)  # convertir dB a potencia
# Invertir filtro mel usando la pseudo-inversa
mel_basis = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
inv_mel = np.dot(np.linalg.pinv(mel_basis), S)
# Aplicar Griffin-Lim
audio = librosa.griffinlim(inv_mel, n_iter=32, hop_length=160, win_length=400)
Librosa también ofrece librosa.feature.inverse.mel_to_audio() para todo en uno. Con torchaudio, no hay función directa, se haría a mano similar.

Limitación crítica: Griffin-Lim requiere varias iteraciones de IFFT (ej. 32) para calidad decente, lo que consume CPU/GPU. El resultado es cercano al audio original en contenido, pero con artefactos (sonido “aspirado”). No capta matices finos de la voz, y suele sonar menos natural que un vocoder entrenado.

Alternativa si falla: Vocoders más ligeros pero más precisos, como WaveRNN o MelGAN lite, producirían voz más natural. Sin embargo, incluso MelGAN exigirá memoria adicional (PyTorch). Si esos son muy pesados, otro truco es usar codificación LPC o síntesis paramétrica básica, aunque sacrifica calidad. Otra opción: usar un pipeline híbrido, p.ej. generar texto a una base de voz pregrabada (menos flexible).

Pregunta 14
Respuesta directa: LanceDB es una base de datos de vectores local basada en Apache Arrow/Parquet. Para usarla en Python se conecta a un archivo local. Ejemplo mínimo con la SDK:

python
Copiar
from lancedb import connect
db = connect("local_lancedb.db")  # abre/crea la base en disco
# Crear tabla con columnas: id (int) y 'vector' (Vector dim 128)
table = db.create_table("mi_tabla", schema={"id": "int64", "vector": ("vector", 128)})
# Insertar vectores
table.insert([{"id": 1, "vector": [0.1, 0.2, ..., 0.0], "info": "metadato"}])
# Búsqueda ANN: busca 5 vectores más parecidos a un query
query_vec = [0.1, 0.2, ..., 0.0]
results = table.search(query_vec, "vector").limit(5).to_list()
Aquí la columna vector debe declararse con su dimensión. LanceDB almacena internamente estos vectores en un esquema Columnar (ej. en Parquet).

Por qué importa para VIERNES: LanceDB permite guardar localmente embeddings del contexto o historial para búsquedas semánticas rápidas (ANN) en hardware local, sin servidor externo.

Implementación mínima:

Instalar lancedb (pip install lancedb).
Conectar/crear base: db = connect("ruta/a/lancedb.db").
Definir y crear tabla usando el schema (ver ejemplo).
Insertar vectores con .insert([...]).
Consultar similares con .search(v, "vector").limit(5).
Limitación crítica: LanceDB en modo local hace kNN por fuerza bruta por defecto (sin índice), lo cual escala O(n). Con pocas decenas de miles de vectores está bien; si crece mucho puede ser lento. Además, el acceso en disco Parquet añade latencia si no se indexa (para grandes tablas se requiere índice vectorial).

Alternativa si falla: Si LanceDB no cumple en CPU baja, se puede usar bibliotecas más sencillas en memoria, como faiss (puede usar un índice ligero HNSW) o Annoy (basado en archivos). Para búsquedas pequeñas, incluso un simple cálculo de distancias con NumPy puede servir. Otra alternativa: SQLite con función extendida de distancia, pero normalmente es más lento.

Pregunta 15
Respuesta directa: El modelo all-MiniLM-L6-v2 es un encoder de frase que produce embeddings semánticos de 384 dimensiones
. Internamente usa MiniLM (6 capas Transformer distilado). Ocupa poca RAM: los pesos float32 suman ≈87 MB
. En CPU la vectorización (codificar texto) toma ~100–300 ms por frase, dependiendo del largo y del CPU. Tras descargarlo la primera vez, se puede guardar y usar offline sin internet. Por ejemplo, con SentenceTransformer:

python
Copiar
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # descarga una vez
emb = model.encode("Texto de ejemplo", device='cpu')
No requiere conexión repetida.

Por qué importa para VIERNES: Permite convertir consultas o texto breve en vectores para búsquedas semánticas (LanceDB). Su bajo consumo (∼0.1–0.2 GB) es tolerable en 6 GB y no necesita GPU.

Implementación mínima: Como arriba con SentenceTransformer. Se puede usar también directamente HuggingFace Transformers (tokenizador + modelo + pooling) para generar el vector.

Limitación crítica: Aunque eficiente, no es multilingüe por defecto (fue entrenado en inglés). Sus embeddings en español pueden ser menos precisos. Además, en CPU las frases muy largas tardan más, y el modelo entero ocupa ~100 MB, lo que ya es un tercio del margen disponible en 6 GB cuando se suman todos los módulos.

Alternativa: Para cargas más ligeras, se puede usar paraphrase-multilingual-MiniLM-L12-v2 que soporta español (pero es un poco más pesado). O usar vectorizadores no neuronales (fastText, TF-IDF) si la precisión semántica no es crucial. Otra opción es cargar el modelo solo al inicio y reutilizarlo (no recargar repetidamente).

Pregunta 16
Respuesta directa: La curva de olvido de Ebbinghaus se puede modelar, por ejemplo, como retención que decae exponencialmente. Ebbinghaus propuso una fórmula empírica:
[ b = \frac{100,k}{(\log(t))^c + k}, ]
donde (c=1.25), (k=1.84) y (t) es el tiempo desde el aprendizaje (minutos)
. Podemos usar este modelo para decrementar un score: por ejemplo, al aplicar decay: [ \text{score}\text{nuevo} = \text{score}\text{viejo} \times \frac{k}{(\log(\Delta t))^c + k}. ] En código Python, dado un nodo con timestamp de cuando fue relevante:

python
Copiar
import math
import time

c, k = 1.25, 1.84
now = time.time()
def decay_score(old_score, timestamp):
    age_min = (now - timestamp) / 60  # minutos desde que se añadió
    if age_min < 1e-6:
        return old_score
    b = k / ((math.log(age_min) ** c) + k)
    return old_score * b

# Ejemplo de aplicación a lista de nodos
for node in nodos:
    node['score'] = decay_score(node['score'], node['timestamp'])
Este código aplica la fórmula de Ebbinghaus (ajustada a fracción en lugar de %).

Por qué importa para VIERNES: Asignar menor importancia a información antigua es útil para gestionar memoria/atención del agente. Ebbinghaus da un modelo basado en psicología de cómo disminuye la retención con el tiempo.

Implementación mínima: Ver arriba: función decay_score que calcula la decaimiento según la edad (age_min). Se usa math.log para aplicar la ley de Ebbinghaus. La multiplicación k/((log age)^c + k) da la fracción de retención. Se itera sobre todos los nodos con su timestamp e importancia.

Limitación crítica: La fórmula original usa minutos y tiene singularidad en (t=1). En la práctica, para (t<1) se debe manejar aparte (aquí lo evitamos con un check). Además es empírica para memoria humana, no precisamente aplicable a scores de importancia, pero es una aproximación. El modelo decae rápido al inicio, luego se estabiliza, lo que podría dejar scores muy bajos o muy altos inesperadamente.

Alternativa: En vez de Ebbinghaus, puede usarse una decaída exponencial simple: (\text{score} \times e^{-\lambda \Delta t}). Esto es más simple y controlable (λ ajustable). También se podría usar decay lineal o basar el decaimiento en recuento de revisiones (curva del olvido doblado con repasos).

Pregunta 17
Respuesta directa: Sin modelos adicionales, podemos usar métricas de movimiento de manos/rostro con reglas heurísticas. Por ejemplo:

Coding: Mucho movimiento de manos (teclado) y poca expresión facial. Métrica: alta velocidad promedio de landmarks de las manos; cabeza fija mirando al frente.
Debugging: Igual que coding pero con ceño fruncido o movimiento irregular de cabeza. Métrica: manos en alto + distancia entre cejas pequeña (ceño) o movimientos laterales frecuentes del rostro indican frustración.
Reading: Pocas manos o fijas (baja movilidad de manos) y ojos/pupilas activos. Métrica: flujo óptico moderado (páginas/títulos en pantalla) y ojos abiertos; cabeza suele inclinarse a veces (mirar texto).
Idle: Casi sin movimiento de manos ni del rostro; ojos cerrados o borrosos (bloqueo de landmarks), flujo óptico mínimo. Métrica: velocidad de landmarks ~0 y ojos cerrados (distancia entre párpados pequeña).
Por qué importa para VIERNES: Distinguir actividad del usuario permite que VIERNES adapte su atención (p. ej. no interrumpir si el usuario está concentrado o idle). Usamos solo datos ya disponibles (MediaPipe landmarks y movimiento OpenCV), sin modelos extra.

Implementación mínima:
Pseudocódigo basado en mediciones de landmarks de MediaPipe y diferencias entre frames:

python
Copiar
# Ejemplo de métricas:
hand_speed = np.mean(velocity_of_hand_landmarks)  # rapidez de manos
face_move = average_optical_flow()               # cuánto se mueve la imagen general
eyebrow_gap = dist(brow_left, brow_right)        # distancia cejas (ceño cuando baja)
eye_openness = dist(upper_lid, lower_lid)        # pestañas
# Reglas:
if hand_speed > H and eyebrow_gap < E_low:
    actividad = "debugging"
elif hand_speed > H:
    actividad = "coding"
elif hand_speed < L and face_move > M:
    actividad = "reading"
elif face_move < M and eye_openness < E_min:
    actividad = "idle"
else:
    actividad = "neutral"
Donde H/L, E_low/M/E_min son umbrales empíricos basados en datos.

Limitación crítica: Estas reglas son muy heurísticas. Difieren ampliamente por usuario, iluminación, posición de cámara, etc. Sin un modelo, habrá falsos clasificaciones (p.ej. alguien leyendo puede mover las manos si pasa páginas, o pasar mucho tiempo mirando sin teclado). Los umbrales deben calibrarse cuidadosamente.

Alternativa si falla: Lo ideal sería un clasificador aprendido (p.ej. decision tree o SVM simple) entrenado con ejemplos de cada actividad usando estas métricas como features. Si no es posible, quizá usar algún modelo precapacitado ligero de pose/gestos o de análisis de actividad humana, aunque fuera mediante aprendizaje automático ligero.

Pregunta 18
Respuesta directa: Para emociones básicas con FaceMesh:

Concentrado: Cejas neutras o ligeramente alzadas, ojos bien abiertos. Medir: espacio vertical entre párpados normal, distancia intercejas media (sin fruncimiento).
Frustrado: Cejas juntas/bajadas (“ceño fruncido”), labios tensos. Medir: distancia entre puntos medios de las cejas reducida; quizá la línea de la boca más horizontal tirando hacia abajo.
Neutral: Cejas relajadas, boca cerrada neutra. Medir: parámetros intermedios (p.ej. cejas horizontales, labios relajados).
Landmarks relevantes (MediaPipe FaceMesh): por ejemplo, índices 105 y 334 marcan centro de cejas, 13/14 boca. Entonces, un cálculo podría ser:

python
Copiar
# Sea lm el array de 468 landmarks
dist_cejas = np.linalg.norm(lm[105] - lm[334])  # menor si hay ceño
dist_boca = np.linalg.norm(lm[13] - lm[14])    # distancia vertical de boca
eye_open = np.linalg.norm(lm[386] - lm[159])   # distancia ojo abierto
Valores bajos de dist_cejas + poca apertura de ojos pueden indicar frustración; altos de ojos abiertos y cejas neutrales indican concentración.

Por qué importa para VIERNES: Detectar frustración o concentración ayuda a ajustar la respuesta (p.ej. modular entonación, sugerir descanso).

Implementación mínima:
Calcular, por frame:

python
Copiar
# Cejas (landmarks aproximados de frente MediaPipe)
left_brow = lm[55]   # cerca de la ceja izquierda
right_brow = lm[285] # ceja derecha
dist_cejas = np.linalg.norm(left_brow - right_brow)
mouth_left = lm[61]; mouth_right = lm[291]
dist_labios = np.linalg.norm(mouth_left - mouth_right)
# Deduzco estado
if dist_cejas < T1 and dist_labios < T2:
    estado = "frustrado"
elif dist_cejas > T3:
    estado = "concentrado"
else:
    estado = "neutral"
Con umbrales T1,T2,T3 definidos tras calibrar.

Limitación crítica: Aun con buenos landmarks, expresiones faciales son sutiles y muy personales. Detección solo por geometría puede fallar (p.ej. alguien serio no está frustrado). Además, MediaPipe puede fallar con oclusiones o baja luz.

Alternativa: Usar modelos de detección de expresión facial (CNNs con pocas clases) o análisis de acción unit (AU) que pueden ser más robustos. Si se dispone de más recursos, hay bibliotecas de emociones faciales preentrenadas (pero eso añadiría modelo extra).

Pregunta 19
Respuesta directa: Diferencia de frames es el método más ligero (solo resta matrices). Lucas-Kanade (sparse, tracking de puntos) es más liviano que Farneback (flujo denso). Farneback implica muchos cálculos polinomiales. Para 15 FPS a 640×480 en CPU básico: la resta de frames usará muy poco (~<10% CPU) y es casi instantánea. Lucas-Kanade puede rondar 10–30 ms por frame (~30–50% CPU) dependiendo de cuántos puntos siga. Farneback suele ser mucho más lento (>50 ms por frame) y podría saturar la CPU (cercano a 100% en un solo hilo). En términos prácticos, Farneback suele ser inapropiado en tiempo real sin optimización, mientras que frame diff o Lucas-Kanade (OpenCV calcOpticalFlowPyrLK) pueden ejecutarse casi en tiempo real (incluso en un solo núcleo)
.

Por qué importa para VIERNES: Detectar actividad (oagresividad) con óptico consume CPU. Seleccionar el más ligero permite seguir otros módulos sin caer FPS.

Implementación mínima: En OpenCV (Python):

python
Copiar
import cv2
cap = cv2.VideoCapture(0)
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Método ligero (diferencia simple):
    diff = cv2.absdiff(prev_gray, gray)
    motion_level = np.sum(diff) / (640*480)  # métrica de movimiento
    prev_gray = gray.copy()
    # ... usar motion_level como indicador
Lucas-Kanade ejemplo:

python
Copiar
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
motion = np.mean(np.linalg.norm(p1-p0, axis=1))
Farneback (más pesado):

python
Copiar
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
motion = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
Limitación crítica: A 640×480, Farneback puede ser demasiado lento para 15 FPS sin GPU. Incluso Lucas-Kanade puede atascar si hay muchos puntos y una CPU muy limitada. La diferencia de frames es rápida pero solo detecta cambios muy evidentes.

Alternativa: Si CPU está al límite, lo más barato es usar solo diferencia de frames con un umbral dinámico. O reducir resolución (p.ej. a 320×240) antes de calcular flujo
. Otra opción es usar el algoritmo TV-L1 de OpenCV con optimizaciones SIMD, o correr cualquier flujo en un hilo separado (como en Q20).

Pregunta 20
Respuesta directa: Se puede usar hilos Python (threading) y queue.Queue para un patrón productor-consumidor. Por ejemplo, un hilo produce frames AR y los pone en la cola; otro hilo los consume (procesamiento cognitivo). Con timeout manejamos saturación. Ejemplo:

python
Copiar
import threading
import queue
frame_queue = queue.Queue(maxsize=5)

def productor():
    while running:
        frame = capturar_frame_AR()  # función de captura
        try:
            frame_queue.put(frame, timeout=0.5)
        except queue.Full:
            pass  # si la cola está llena, descarta el frame

def consumidor():
    while running:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        procesar_frame_cognitivo(frame)  # tarea intensiva
        frame_queue.task_done()

# Iniciar hilos
t_prod = threading.Thread(target=productor, daemon=True)
t_cons = threading.Thread(target=consumidor, daemon=True)
t_prod.start()
t_cons.start()
Aquí productor() captura a 15 FPS y encola; consumidor() procesa de la cola. El timeout evita que el productor bloquee si la cola está llena, y el consumidor espera si está vacía. La cola con maxsize limita memoria y sincroniza hilos.

Por qué importa para VIERNES: Para no cortar los 15 FPS de realidad aumentada mientras el procesamiento pesado (LM, RL) ocurre, hay que correrlo en paralelo. Este patrón garantiza que las tareas cognitivas no congelan la interfaz de AR.

Implementación mínima: Como arriba:

Crear Queue(maxsize=…).
Hilo productor: bucle de captura (15FPS), put() con timeout para no bloquear.
Hilo consumidor: lee con get(timeout) y procesa.
Controlar daemon=True para que cierren al salir del programa.
Limitación crítica: El GIL de Python limita la concurrencia verdadera en CPU, pero como la captura y cola suelen ser I/O/bloqueantes, el procesamiento (CPU) se ejecuta parcialmente en paralelo. Aún así, el cómputo intensivo del consumidor puede retrasar el productor si la cola se llena continuamente. Además, manejar excepciones (Full/Empty) es crucial para no bloquear nunca el loop principal de AR.

Alternativa si falla: Usar procesos (multiprocessing) en lugar de hilos, para aprovechar múltiples CPUs sin GIL. Por ejemplo, un proceso captura y otro procesa, comunicándose con multiprocessing.Queue. O usar concurrent.futures.ThreadPoolExecutor con max_workers=2. Si el procesamiento es muy pesado, se puede incluso delegar partes a GPU (Torch) con torch.compile() para acelerar inferencias dentro del hilo.

Pregunta 21
Respuesta directa: Sumando las estimaciones: LM ~200 MB + PPO ~5 MB + LanceDB ~100 MB + MediaPipe ~150 MB + OpenCV ~200 MB + ASR ~50 MB = ~705 MB. Añadiendo overhead (Python, SO, buffers) se puede esperar ~1–1.5 GB total. En 6 GB hay mucho margen. Para ahorrar RAM, la pieza más pesada es MediaPipe/OpenCV (~350 MB juntas) y el LM (200 MB). Podríamos cargar/descaragar algunos módulos:

LM: Cargar solo cuando se necesite generación de texto y luego liberar (p.ej. guardar modelo en disco y llamar a gc.collect() tras usarlo).
ASR: Similar, carga el modelo solo al reconocer voz y cierra el modelo al terminar.
MediaPipe/Visión: Iniciar el pipeline de MP sólo mientras haya cámara activa; desactivarlo si el usuario está inactivo.
LanceDB: Mantener la base de datos en disco; abrirla (connect()) al buscar información y luego cerrarla (db.close()) para liberar caches.
PPO y SB3 (solo unos MB) pueden permanecer cargados. El audio/VAD solo requiere buffers pequeños.
Por qué importa para VIERNES: En 6 GB cada MB cuenta. Saber qué componentes no tienen que estar en memoria constantemente puede multiplicar eficientemente la RAM disponible en tiempo de ejecución.

Implementación mínima: Por ejemplo, para no retener el modelo ASR:

python
Copiar
def reconocer_voz(audio):
    import torch
    model = ASRModel(...)  # cargar modelo ASR (~50 MB)
    output = model(audio)
    del model
    torch.cuda.empty_cache()  # si GPU
    return output
Hacer similar al generar con LM. Para MediaPipe/OpenCV, iniciar/destruir los objetos al entrar/salir de escenas intensivas.

Limitación crítica: Cargar y descargar modelos repetidamente es costoso (tiempo de carga). Si se hace muy a menudo podría causar latencia. Además, liberar memoria en Python no siempre la devuelve al SO inmediatamente (fragmentación). Hay que balancear frecuencia de carga con uso continuo.

Alternativa: Si no se puede cargar bajo demanda (p. ej. por latencia), conviene simplificar: usar versiones aún más ligeras (p.ej. TinyMediaPipe en vez de la versión completa, OpenCV optimizado con cv2.ocl.setUseOpenCL(False)), o tiempo compartido (baja resolución / FPS de algunos pipelines). Otra opción es usar swap ligero o compactar datos antiguos a disco temporalmente (si el sistema lo permite).

Pregunta 22
Respuesta directa: torch.no_grad() es un contexto que desactiva el cálculo de gradientes durante la inferencia
. Esto evita almacenar activaciones intermedias para retropropagación, reduciendo significativamente el uso de memoria temporal (GPU/CPU). Otras optimizaciones PyTorch para CPU limitado:

torch.inference_mode(): similar a no_grad pero optimiza internamente y elimina algunas sobrecargas adicionales (es aún más eficiente para solo inferencia).
torch.compile() (PyTorch 2.0+): compila el modelo en una versión optimizada, acelerando la ejecución. Útil si el modelo se usa repetidamente.
Cuantización int8: convertir pesos/activaciones a int8 reduce a ~25% el tamaño del modelo y acelera cálculos entero, con pérdida mínima de precisión.
model.eval(): desactiva capas de dropout/batchnorm, importante para inferencia eficiente (aunque no ahorra memoria como no_grad).
Por qué importa para VIERNES: Al inferir con el modelo de lenguaje (o ASR), no se usan gradientes. Usar torch.no_grad() o mejor torch.inference_mode() ahorra memoria (no se duplican tensores) y acelera. Cuantizar el modelo podría hacerlo caber incluso en CPU limitadas.

Implementación mínima:

python
Copiar
model.eval()
with torch.inference_mode():
    salida = model(input_tensor)
Para quantización dinámica:

python
Copiar
model_int8 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
O con torch.compile() (PyTorch ≥2.0):

python
Copiar
model = torch.compile(model)
Limitación crítica: torch.compile() puede requerir PyTorch muy reciente y en CPU a veces no acelera mucho (depende de código). La cuantización int8 requiere calibración o finetuning (o al menos chequear desempeño) y puede degradar la calidad si el modelo es muy pequeño. inference_mode() es lo mejor cuando solo se infiere, pero hay que asegurarse de no llamar a .backward() dentro de él (no afecta al CPU con no_grad, solo deshabilita grad).

Alternativa si falla: Si PyTorch 2.0 no está disponible para torch.compile(), se puede convertir el modelo a ONNX y usar ONNX Runtime, que a menudo es más rápido en CPU. Para cuantización, se puede usar bibliotecas externas como Intel OpenVINO o TensorRT (si se dispone de GPU). Finalmente, reducir batch de inferencia o usar float16 (en CPU con bfloat16 en PyTorch) podría ahorrar algo de memoria sin pasos extras.

Fuentes:

Documentación PyTorch: torch.no_grad() reduce uso de memoria al desactivar gradientes
.
HuggingFace/In-house benchmarks: all-MiniLM-L6-v2 pesa ~87 MB (float32)
, lo que ayuda a estimar el uso RAM.
Escritos y repositorios (nanoGPT, SB3 docs, LanceDB docs) consultados para arquitectura y ejemplos concretos
.