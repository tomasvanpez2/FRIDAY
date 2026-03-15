import asyncio
import threading
import time
import base64
from typing import Callable, Optional

import cv2
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

from viernes import config
from viernes.utils.logger import get_logger

logger = get_logger()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.1
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

SYSTEM_PROMPT = """
Responde siempre en voz alta en español colombiano.
Sé muy conciso, máximo 2 oraciones.

Eres VIERNES, un asistente cognitivo AR personal integrado en
gafas de realidad aumentada. Tienes acceso a la cámara del usuario
y puedes ver su entorno en tiempo real.

Reglas estrictas:
- Responde SIEMPRE en español
- Sé muy conciso: máximo 2 oraciones por respuesta
- Si el usuario pregunta qué ves: describe brevemente los objetos
  y personas que aparecen en la imagen de la cámara
- Si el usuario pide recordar algo: confirma con "Guardado en memoria"
- Eres proactivo: si ves algo interesante en la cámara puedes
  mencionarlo brevemente sin que te lo pidan
- Actúa como si fueras parte del sistema visual del usuario,
  no como un chatbot externo
"""


class VoiceHandler:
    def __init__(self, get_frame_callback: Optional[Callable] = None) -> None:
        self.get_frame = get_frame_callback
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.state = "IDLE"
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self.on_response: Optional[Callable[[str], None]] = None
        self.on_listening_start: Optional[Callable[[], None]] = None
        self.on_listening_end: Optional[Callable[[], None]] = None

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buffer).decode("utf-8")

    async def _run_session(self) -> None:
        config_live = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_PROMPT,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        async with self.client.aio.live.connect(
            model="gemini-2.5-flash-preview-native-audio-dialog",
            config=config_live,
        ) as session:
            logger.info("Gemini Live conectado — escuchando...")
            self.state = "LISTENING"
            if self.on_listening_start:
                self.on_listening_start()

            audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue()

            def audio_callback(indata, frames, t, status) -> None:
                audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(audio_queue.put(audio_bytes), self._loop)

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=audio_callback,
            )

            async def send_audio() -> None:
                with stream:
                    while self._running:
                        try:
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(
                                            data=audio_data,
                                            mime_type="audio/pcm",
                                        )
                                    ]
                                )
                            )

                            if self.get_frame and int(time.time()) % 3 == 0:
                                frame = self.get_frame()
                                if frame is not None:
                                    img_b64 = self._frame_to_base64(frame)
                                    await session.send(
                                        input=types.LiveClientRealtimeInput(
                                            media_chunks=[
                                                types.Blob(
                                                    data=base64.b64decode(img_b64),
                                                    mime_type="image/jpeg",
                                                )
                                            ]
                                        )
                                    )
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"[SEND ERROR] {e}")
                            break

            async def receive_responses() -> None:
                while self._running:
                    try:
                        async for response in session.receive():
                            if getattr(response, "data", None):
                                audio_array = np.frombuffer(response.data, dtype=np.int16)
                                sd.play(audio_array, samplerate=24000)
                                sd.wait()
                                self.state = "LISTENING"

                            if getattr(response, "text", None):
                                texto = response.text.strip()
                                if texto:
                                    logger.info(f"[VIERNES] {texto}")
                                    if self.on_response:
                                        self.on_response(texto)
                    except Exception as e:
                        logger.error(f"[RECV ERROR] {e}")
                        break

            await asyncio.gather(send_audio(), receive_responses())

    def _run_async(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_session())
        except Exception as e:
            logger.error(f"[SESSION ERROR] {e}")
        finally:
            self._loop.close()
            self._loop = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        logger.info("VoiceHandler Gemini Live iniciado")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("VoiceHandler detenido")

    def get_state(self) -> str:
        return self.state

    def get_stats(self) -> dict:
        return {"state": self.state, "model": "gemini-2.5-flash-preview-native-audio-dialog"}


if __name__ == "__main__":
    print("VIERNES — Gemini 2.5 Native Audio")
    print("Habla directamente en español")
    print("VIERNES te responderá en voz alta")
    print("Ctrl+C para salir\n")

    def on_response(texto: str) -> None:
        print(f"\n🤖 VIERNES: {texto}\n")

    handler = VoiceHandler()
    handler.on_response = on_response
    handler.start()

    try:
        while True:
            print(f"Estado: {handler.get_state()}", end="\r")
            time.sleep(0.5)
    except KeyboardInterrupt:
        handler.stop()
