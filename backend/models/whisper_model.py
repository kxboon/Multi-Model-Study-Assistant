"""
Whisper STT wrapper — transcribes audio files to text locally.
Uses openai-whisper which runs entirely on-device (no API key needed).
"""

import os
import time
import whisper


class WhisperModel:
    """Lazy-loading wrapper around openai-whisper.

    The model is not loaded at import time — only when transcribe() is first
    called. This keeps startup time fast when Whisper isn't needed yet.
    """

    def __init__(self, model_size: str = None):
        # Read model size from env so the caller can override via .env
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        self._model = None  # loaded on first use

    def _load(self):
        """Load the Whisper model into memory (called lazily)."""
        if self._model is None:
            t0 = time.perf_counter()
            self._model = whisper.load_model(self.model_size)
            print(f"[TIMER] Whisper model load ({self.model_size}): {time.perf_counter() - t0:.2f}s")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the plain-text transcript.

        Args:
            audio_path: Absolute or relative path to an .mp3 / .wav file.

        Returns:
            The full transcript as a single string.
        """
        self._load()
        t0 = time.perf_counter()
        result = self._model.transcribe(audio_path, fp16=False)
        print(f"[TIMER] Whisper transcribe: {time.perf_counter() - t0:.2f}s")
        return result["text"].strip()
