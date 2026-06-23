"""Native audio backend via mlx-audio (speech-to-text + text-to-speech).

Two capabilities the agent can use as tools: ``transcribe`` (STT, e.g. for Telegram
voice notes) and ``speak`` (TTS, to voice a reply). Speech-to-speech is left for
later behind the same seam. Optional and defensive; generation runs in a worker
thread because MLX is blocking.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class AudioService(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text (STT)."""

    @abstractmethod
    async def speak(self, text: str) -> Path:
        """Synthesise speech for ``text`` and return the saved audio path (TTS)."""


class MlxAudioBackend(AudioService):
    def __init__(
        self,
        audio_dir: Path,
        stt_model: str = "mlx-community/whisper-tiny",
        tts_model: str = "mlx-community/Kokoro-82M-bf16",
    ):
        self._dir = Path(audio_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._stt_model = stt_model
        self._tts_model = tts_model

    def available(self) -> bool:
        return importlib.util.find_spec("mlx_audio") is not None

    async def transcribe(self, audio_path: str) -> str:
        if not self.available():
            raise RuntimeError(
                'audio requires mlx-audio. Install with: uv pip install -e ".[audio]"'
            )
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    async def speak(self, text: str) -> Path:
        if not self.available():
            raise RuntimeError(
                'audio requires mlx-audio. Install with: uv pip install -e ".[audio]"'
            )
        return await asyncio.to_thread(self._speak_sync, text)

    def _transcribe_sync(self, audio_path: str) -> str:
        # Single integration point with mlx-audio STT.
        from mlx_audio.stt.generate import generate as stt_generate

        result = stt_generate(model_path=self._stt_model, audio_path=audio_path)
        return getattr(result, "text", str(result))

    def _speak_sync(self, text: str) -> Path:
        # Single integration point with mlx-audio TTS. It writes ``<file_prefix>.wav``.
        from mlx_audio.tts.generate import generate_audio

        prefix = self._dir / f"tts_{uuid.uuid4().hex[:8]}"
        generate_audio(
            text=text,
            model_path=self._tts_model,
            file_prefix=str(prefix),
            audio_format="wav",
            verbose=False,
        )
        return prefix.with_suffix(".wav")
