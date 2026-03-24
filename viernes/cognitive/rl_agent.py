"""
VIERNES RL Agent
Agente de aprendizaje por refuerzo ultraligero.
Diseñado para correr en hardware de bajos recursos (<=6GB RAM, solo CPU).
No usa GPU. No usa modelos grandes. Aprende qué acciones son útiles para el usuario.
"""

import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from viernes.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Acciones que VIERNES puede tomar
# ---------------------------------------------------------------------------
ACTIONS = {
    0: "silencio",           # no hacer nada — válido y frecuente
    1: "hablar",             # decir algo en voz natural
    2: "mostrar_overlay",    # mostrar panel AR
    3: "guardar_memoria",    # guardar observación en knowledge system
    4: "alerta_tecnica",     # alertar sobre algo técnico detectado
}

# Dimensión del vector de observación
OBS_DIM = 10


class VIERNESEnv(gym.Env):
    """
    Entorno Gymnasium para el agente VIERNES.

    El estado representa la situación actual del usuario y el entorno.
    Las acciones son lo que VIERNES puede hacer en cada momento.
    El reward se infiere del comportamiento del usuario sin feedback explícito.

    Diseño intencional: simple y liviano. Sin redes grandes, sin embeddings,
    sin procesamiento de imagen aquí — solo el vector de estado ya procesado
    que le llega desde el pipeline AR.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Espacio de observación: vector de 10 floats en [0, 1]
        # [0] fracción de frame ocupada por manos detectadas
        # [1] fracción de frame ocupada por rostro detectado
        # [2] superficies detectadas (0 o 1)
        # [3] actividad de voz del usuario (0=silencio, 1=hablando)
        # [4] hora del día normalizada (0=medianoche, 1=medianoche siguiente)
        # [5] tiempo desde última interacción del usuario (normalizado 0-1, 1=hace 5min+)
        # [6] última acción de VIERNES (normalizada sobre N_ACTIONS)
        # [7] ¿usuario respondió a última acción? (0=no, 1=sí)
        # [8] ¿usuario ignoró última acción? (0=no, 1=sí)
        # [9] nivel de actividad reciente (media de movimiento en últimos 10s)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        # Espacio de acciones: 5 acciones discretas
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Estado interno
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._last_action = 0
        self._last_action_time = time.time()
        self._episode_steps = 0
        self._max_steps = 200

        # Reward pendiente desde señales externas
        self._pending_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._episode_steps = 0
        self._pending_reward = 0.0
        return self._obs.copy(), {}

    def step(self, action):
        self._episode_steps += 1
        self._last_action = action

        # Actualizar obs con la acción tomada
        self._obs[6] = action / len(ACTIONS)

        # Reward base por acción
        reward = self._compute_reward(action)

        # Sumar reward pendiente de señales externas
        reward += self._pending_reward
        self._pending_reward = 0.0

        # Clip de reward para estabilidad
        reward = float(np.clip(reward, -1.0, 1.0))

        terminated = False
        truncated = self._episode_steps >= self._max_steps

        return self._obs.copy(), reward, terminated, truncated, {}

    def _compute_reward(self, action: int) -> float:
        """
        Calcula reward basado en el contexto actual y la acción tomada.
        Reglas simples e interpretables — sin red neuronal aquí.
        """
        reward = 0.0
        usuario_hablando = self._obs[3] > 0.5
        usuario_activo = self._obs[9] > 0.5
        usuario_respondio = self._obs[7] > 0.5
        usuario_ignoro = self._obs[8] > 0.5

        # Silencio cuando el usuario está ocupado → pequeña recompensa
        if action == 0 and usuario_activo:
            reward += 0.1

        # Hablar o mostrar overlay cuando el usuario está inactivo → neutro/positivo
        if action in (1, 2) and not usuario_activo:
            reward += 0.2

        # Interrumpir al usuario mientras habla → penalización
        if action in (1, 2) and usuario_hablando:
            reward -= 0.4

        # El usuario respondió a la última acción → recompensa
        if usuario_respondio:
            reward += 0.6

        # El usuario ignoró la última acción → penalización
        if usuario_ignoro:
            reward -= 0.2

        return reward

    def update_observation(self, obs_dict: dict):
        """
        Actualiza el vector de observación desde el pipeline AR.
        Llamar en cada frame o cuando el estado cambie significativamente.

        Args:
            obs_dict: diccionario con las claves del vector de observación.
                      Las claves faltantes se dejan en 0.
        """
        keys = [
            "manos_area", "rostro_area", "superficies",
            "voz_activa", "hora_normalizada", "tiempo_inactividad",
            "ultima_accion", "usuario_respondio", "usuario_ignoro",
            "nivel_actividad"
        ]
        for i, key in enumerate(keys):
            if key in obs_dict:
                self._obs[i] = float(np.clip(obs_dict[key], 0.0, 1.0))

    def signal_user_response(self, responded: bool):
        """
        Llamar desde el pipeline cuando el usuario reacciona a una acción de VIERNES.
        responded=True: siguió el consejo, preguntó más, dijo 'sí/exacto/correcto'
        responded=False: ignoró, dijo 'no/para', no reaccionó
        """
        if responded:
            self._obs[7] = 1.0
            self._pending_reward += 0.4
        else:
            self._obs[8] = 1.0
            self._pending_reward -= 0.2

    def render(self):
        pass


class VIERNESAgent:
    """
    Agente RL de VIERNES.
    Encapsula el entorno y la política PPO.
    Diseñado para ser liviano: corre en CPU, usa poca RAM.
    """

    def __init__(self, model_path: str = None):
        self.env = VIERNESEnv()
        self.model_path = model_path
        self._obs, _ = self.env.reset()
        self._model = None
        self._initialized = False

        self._load_or_init(model_path)

    def _load_or_init(self, path: str):
        """Carga política existente o inicializa desde cero."""
        if path and os.path.exists(path + ".zip"):
            logger.info(f"[RL] Cargando política desde {path}")
            self._model = PPO.load(path, env=self.env)
            self._initialized = True
        else:
            logger.info("[RL] Inicializando política nueva desde cero")
            self._model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=3e-4,
                n_steps=512,          # bajo para no consumir memoria
                batch_size=64,
                n_epochs=4,
                gamma=0.95,
                verbose=0,
                device="cpu",         # siempre CPU — sin requerir GPU
            )
            self._initialized = True

    def act(self, obs_dict: dict) -> str:
        """
        Dado el estado actual del entorno, decide qué acción tomar.
        Retorna el nombre de la acción como string.
        """
        self.env.update_observation(obs_dict)
        action, _ = self._model.predict(self.env._obs, deterministic=False)
        return ACTIONS[int(action)]

    def signal_response(self, responded: bool):
        """Señal de feedback implícito del usuario."""
        self.env.signal_user_response(responded)

    def learn(self, timesteps: int = 1000):
        """
        Entrena la política por N pasos.
        Llamar al final de cada sesión, no en tiempo real.
        1000 pasos tarda ~2-5 segundos en CPU básica.
        """
        logger.info(f"[RL] Entrenando por {timesteps} pasos...")
        self._model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        logger.info("[RL] Entrenamiento completado")

    def save(self):
        """Guarda la política en disco."""
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self._model.save(self.model_path)
            logger.info(f"[RL] Política guardada en {self.model_path}")

    def load(self):
        """Recarga la política desde disco."""
        self._load_or_init(self.model_path)


# ---------------------------------------------------------------------------
# Demo standalone — correr directamente para verificar que funciona
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Verificando entorno...")
    env = VIERNESEnv()
    check_env(env, warn=True)
    print("Entorno válido.")

    print("\nInicializando agente...")
    agent = VIERNESAgent()

    print("\nSimulando 5 decisiones:")
    for i in range(5):
        obs = {
            "manos_area": np.random.random(),
            "rostro_area": np.random.random(),
            "superficies": float(np.random.random() > 0.5),
            "voz_activa": float(np.random.random() > 0.7),
            "hora_normalizada": 0.5,
            "tiempo_inactividad": np.random.random(),
            "nivel_actividad": np.random.random(),
        }
        accion = agent.act(obs)
        print(f"  Paso {i+1}: VIERNES decide → {accion}")

    print("\nEntrenando 500 pasos de prueba...")
    agent.learn(timesteps=500)
    print("Todo OK. El módulo RL está listo.")
