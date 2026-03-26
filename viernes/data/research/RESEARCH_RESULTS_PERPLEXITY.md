Pregunta 1
Respuesta directa: RWKV-v4 tiny (430M params) es el más liviano y eficiente en CPU/RAM comparado con nanoGPT (10M-124M params configurable), GPT-2 small (117M params), DistilGPT-2 (82M params). nanoGPT permite configs ultra-pequeñas (e.g., 4 layers, 64 embed dim, ~1-10M params) entrenables en <4GB RAM CPU; inferencia ~100-500MB todos. Entrenamiento nanoGPT/RWKV ~2-4GB RAM batch pequeño; DistilGPT-2/GPT-2 small necesitan ~6GB+ FP16 para fine-tune, no from scratch en 4GB.

Por qué importa para VIERNES: Necesitas LM propio ultraligero entrenable local en CPU 16GB dev, 6GB ARM target, generando español técnico coherente sin APIs.
Implementación mínima: Usa nanoGPT repo (v0.1), config: n_layer=4, n_head=4, n_embd=64 (~5M params); PyTorch 2.1+, batch_size=4, context=128; entrena en CPU con torch.float32. Librerías: torch==2.1.0, transformers==4.35.0.
Limitación crítica: Training from scratch en corpus pequeño da coherencia limitada (perplexity >10); RWKV mejor secuencia larga pero más params; todos lentos CPU inferencia (>1s/token sin optim).

Alternativa si falla: TinyLlama 1.1B cuantizado int8 (~500MB) o Phi-1.5 mini destilado, pero viola "propio desde cero".

Pregunta 2
Respuesta directa: Preentrenamiento next-token prediction en corpus tokenizado; con <100MB (~50M tokens), 3-5 épocas suficientes; loss inicial ~4-5 (random), final ~2.5-3.5; "aprendió" si loss valida estabiliza, genera texto coherente >10 tokens, perplexity <20.

Por qué importa para VIERNES: Base LM propio con conocimiento inicial español técnico en datos locales pequeños.
Implementación mínima: nanoGPT train.py, dataset=texto.txt limpio; epochs=5000 iters (batch=32), lr=1e-4, eval cada 500; usa DataLoader PyTorch.
Limitación crítica: Sobreajuste rápido en corpus pequeño; necesita validación cruzada evitar memorización.
​
Alternativa si falla: Continual pretraining en chunks de 10MB, eval manual coherencia.

Pregunta 3
Respuesta directa: Online learning actualiza LM con nuevo texto sesión-sesión; evita catastrophic forgetting con LoRA incremental (más liviana: ~1% params extra, bajo RAM/CPU vs EWC Fisher matrix o replay buffer storage).

Por qué importa para VIERNES: Aprende refuerzo uso sin olvidar conocimiento base.
Implementación mínima: peft==0.7.0, LoRA rank=4, alpha=8; cada sesión: LoRA nuevo merge base con sft_trainer (lr=1e-5, epochs=1).
Limitación crítica: LoRA acumula adapters (RAM crece); merge periódico degrada si >10 sesiones.
​
Alternativa si falla: Replay buffer pequeño (10k samples previos) + SGD simple.

Pregunta 4
Respuesta directa: BPE subpalabra merging frequencies; entrena tokenizer desde cero en español técnico para vocab eficiente. Código mínimo con tokenizers HuggingFace.

Por qué importa para VIERNES: Tokenizer propio optimizado español técnico, bajo RAM sin HF models.
Implementación mínima:

python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=['[UNK]','[PAD]'])
tokenizer.train(['corpus.txt'], trainer)
tokenizer.save('viernes_tokenizer.json')
tokenizers==0.15.0.
​
Limitación crítica: Vocab 8000 bajo OOV en técnico nuevo; necesita reentreno periódico.
Alternativa si falla: SentencePiece unigram, mismo API.

Pregunta 5
Respuesta directa: PPO optimiza policy (actor: selecciona acciones) con critic (value: estima retorno); clips ratio updates para estabilidad vs TRPO/Vanilla PG. Estable por clipped surrogate objective.
​
Por qué importa para VIERNES: RL propio para aprender preferencias usuario implícitas estable en hardware low.
Implementación mínima: Stable-Baselines3 PPO; policy=actor-critic MLP; learn(total_timesteps). sb3==2.2.0.
Limitación crítica: Hiperparams sensibles; necesita buen reward shaping inicial.
​
Alternativa si falla: A2C (más simple, menos estable).

Pregunta 6
Respuesta directa: Env Gymnasium con Box(20,) obs, Discrete(10) actions; template SB3 compatible.
​
Por qué importa para VIERNES: Entorno RL para agente VIERNES actuar en estado usuario.
Implementación mínima:

python
import gymnasium as gym, numpy as np
from gymnasium import spaces
class VierenesEnv(gym.Env):
    def __init__(self): self.observation_space=spaces.Box(-1,1,(20,)); self.action_space=spaces.Discrete(10)
    def reset(self, **kwargs): return np.zeros(20), {}
    def step(self, action): return np.zeros(20), 0.0, False, False, {}
# Uso: model = PPO('MlpPolicy', env); model.learn(1000)
gymnasium==0.29.1.
​
Limitación crítica: Estado 20 floats debe capturar bien dinámica; reward sparse lento converge.
Alternativa si falla: VecEnv parallel si multi-core.

Pregunta 7
Respuesta directa: Reward shaping añade dense rewards guía sin cambiar optima policy; implícitos: 1)tiempo mirada pantalla, 2)keystrokes/min, 3)mouse velocity, 4)scroll rate, 5)pausas cursor.
​
Por qué importa para VIERNES: Recompensas auto sin input usuario para RL continuo.
Implementación mínima: reward = 0.1 * keystroke_rate + 0.05 * (1 - idle_time) - 0.1 * frustration_score.
Limitación crítica: Mal shaping induce suboptimal policy; necesita validación humana.
​
Alternativa si falla: Curiosity intrinsic reward (prediction error).

Pregunta 8
Respuesta directa: PPO MlpPolicy (20 obs,10 actions) ~10-50MB RAM, <10% CPU single-core; 1000 steps ~1-5s CPU básica (e.g., i5).
​
Por qué importa para VIERNES: Verificar RL cabe en 6GB ARM.
Implementación mínima: PPO('MlpPolicy', env, policy_kwargs={'net_arch':}); learn(1000).
Limitación crítica: CPU lento vs GPU; batch grande OOM.
​
Alternativa si falla: SAC continuo actions, similar RAM.

Pregunta 9
Respuesta directa: CTC alinea secuencias audio-text sin phones; arch mínima: 3 Conv1D + 2 GRU (~3-5M params) para español; dataset Common Voice es (CC0 libre).

Por qué importa para VIERNES: ASR propio sin Whisper, local.
Implementación mínima: torch.nn (Conv1d(1,32,k=3)+GRU(32,128)+Linear CTC); train CommonVoice-es subset. torch==2.1.0.
Limitación crítica: WER >20% sin mucho data; noisy real-world peor.
​
Alternativa si falla: Vosk API local models (pero ~50MB preentrenados).

Pregunta 10
Respuesta directa: Mel spectrogram comprime freq humano-perc; librosa/torchaudio convierten 16kHz->80 mel.

Por qué importa para VIERNES: Frontend ASR propio.
Implementación mínima:

python
import librosa
mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80, n_fft=400, hop_length=160, fmax=8000)
librosa==0.10.1.
​
Limitación crítica: Fijo sr=16kHz; realtime necesita stream.
Alternativa si falla: torchaudio.transforms.MelSpectrogram.

Pregunta 11
Respuesta directa: VAD RMS energy > threshold adaptativo (media noise * factor); numpy puro.

Por qué importa para VIERNES: Detecta speech sin libs externas.
Implementación mínima:

python
import numpy as np
rms = np.sqrt(np.mean(audio**2, axis=-1))
noise_floor = np.mean(rms[rms<np.percentile(rms,30)])
is_speech = rms > 3 * noise_floor
Limitación crítica: Sensible ruido no-estacionario; falsos pos/neg.
Alternativa si falla: WebRTC VAD (pip webrtcvad, ~1MB).

Pregunta 12
Respuesta directa: Correlación espectrograma template wake-word; semi-confiable pero falsos pos altos ruido; alt: CNN tiny MFCC.

Por qué importa para VIERNES: Activación manos-libres local.
Implementación mínima: Cross-corr mel vs template_wake; peak>0.8 detect.
Limitación crítica: Falsos pos 20-30% ruido; necesita post-proc.
​
Alternativa si falla: Porcupine lite o CNN 100k params.

Pregunta 13
Respuesta directa: Griffin-Lim iterativo phase reconstruct de mel mag; inteligible pero metálico vs WaveNet natural; suficiente asistente.
​
Por qué importa para VIERNES: TTS propio simple post-LM.
Implementación mínima: librosa.griffinlim(mel, n_iter=32).
Limitación crítica: Artefactos phase; no expresivo.
​
Alternativa si falla: Tortoise-TTS mini o espeak-ng.

Pregunta 14
Respuesta directa: LanceDB embed DB local; create_table, add_vectors, search knn/hnsw.
​
Por qué importa para VIERNES: Memoria vectorial persistente low-resource.
Implementación mínima:

python
import lancedb
db = lancedb.connect('viernes_db')
table = db.create_table('mem', data=[{'vector':v, 'meta':d}])
hits = table.search(qvec, query_type='hybrid').limit(5).to_list()
lancedb==0.5.0.
​
Limitación crítica: Index build lento grandes DB; disk IO.
Alternativa si falla: Faiss CPU-only.

Pregunta 15
Respuesta directa: Sentence embedder ~80M params, ~300MB RAM, ~50ms/frase CPU; sí offline post-download.
​
Por qué importa para VIERNES: Embed memoria sesiones español.
Implementación mínima: sentence_transformers==2.3.0; model.encode("texto").
Limitación crítica: CPU lento batch=1; no cuantizado default.
Alternativa si falla: TF-IDF + PCA 128d.

Pregunta 16
Respuesta directa: Ebbinghaus R(t)=e^{-t/s} retention decay time t desde review, s=strength; aplica score_importancia *= decay.
​
Por qué importa para VIERNES: Olvido natural memoria vectorial.
Implementación mínima:

python
import numpy as np
def ebbinghaus_decay(age_days, strength=5.0):  # s=5 días half-life
    return np.exp(-age_days / strength)
scores = [s * ebbinghaus_decay((now-ts).days) for ts,s in nodes]
Limitación crítica: s hiperparam por tipo memoria.
Alternativa si falla: Linear decay.

Pregunta 17
Respuesta directa: Reglas landmarks: coding=mano cerca teclado + dedos move; debugging=mano cara frustración + scroll; reading=ojos fijos bajo movimiento; idle=bajo todo.
​
Por qué importa para VIERNES: Contexto actividad sin models extra.
Implementación mínima: Métricas: hand_key_dist<0.2, finger_var>0.05, eye_gaze_std<0.1, motion_mag<10px/frame.
Limitación crítica: Reglas heurísticas frágiles poses variadas.
​
Alternativa si falla: HMM states secuencia features.

Pregunta 18
Respuesta directa: AU landmarks: frustrado (frente arrugada 1-4, cejas 2-5 bajos); concentrado (ojos abiertos 62-66, pupila focus); neutral baseline.
​
Por qué importa para VIERNES: Empatía emocional usuario.
Implementación mínima: brow_raise=(lm-lm)/dist_eyes; jaw_clench=lm-lm>thresh.
​
Limitación crítica: Occlusiones/oculares fallan; cultural bias.
Alternativa si falla: OpenFace rules.

Pregunta 19
Respuesta directa: Farneback dense más liviano CPU vs LK sparse; ~20-50% CPU 15FPS 640x480; frame diff simplest <10%.
​
Por qué importa para VIERNES: Actividad sin YOLO low CPU.
Implementación mínima: cv2.calcOpticalFlowFarneback(prev, curr, ...); mag=np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)).
opencv-python==4.9.0.
​
Limitación crítica: Iluminación cambia degrada.
Alternativa si falla: Frame diff abs + blur.

Pregunta 20
Respuesta directa: Threading + queue.Queue productor (AR 15FPS)->consumidor (cognitivo); get(timeout=0.1).
​
Por qué importa para VIERNES: Realtime AR no bloqueado.
Implementación mínima:

python
import threading, queue, time
q = queue.Queue(maxsize=10)
def producer(): while True: q.put(frame, timeout=1); time.sleep(1/15)
def consumer(): while True: frame = q.get(timeout=0.1); process(frame); q.task_done()
threading.Thread(target=producer).start(); threading.Thread(target=consumer).start()
Limitación crítica: GIL Python limita CPU-bound; usa multiprocessing si >4 cores.
Alternativa si falla: asyncio queues.

Pregunta 21
Respuesta directa: Total ~1.2-1.8GB RAM peak; sí, unload/reload LM/ASR/PPO torch.save/load_state_dict bajo demanda.
​
Por qué importa para VIERNES: Verificar 6GB ARM viable.
Implementación mínima: del model; torch.cuda.empty_cache() (CPU similar gc.collect()).
Limitación crítica: Load time 1-5s grande models; peak spikes training.
​
Alternativa si falla: Prioriza: siempre MediaPipe+OpenCV, on-demand resto.

Pregunta 22
Respuesta directa: torch.no_grad() deshabilita grads save RAM 50%; torch.inference_mode() más estricto; torch.compile() speed CPU; int8 quant torch.quantization.
​
Por qué importa para VIERNES: Inferencia CPU low RAM critical.
Implementación mínima: with torch.inference_mode(): output=model(input); model=torch.compile(model).
Limitación crítica: compile() first-run lento; quant pierde precisión ~2-5%.
​
Alternativa si falla: ONNX export CPU runtime.