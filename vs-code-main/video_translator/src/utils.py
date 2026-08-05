"""
Shared helpers: logging, device selection, json io, ffmpeg wrappers,
and time-stretching used to keep dubbed audio perfectly in sync with video.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


class StepTimer:
    """Small context manager to print a clean step banner + timing."""

    def __init__(self, logger: logging.Logger, title: str):
        self.logger = logger
        self.title = title
        self.start = 0.0

    def __enter__(self):
        self.logger.info("=" * 70)
        self.logger.info(self.title)
        self.logger.info("=" * 70)
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            self.logger.info(f"Done in {elapsed:.2f}s")
        else:
            self.logger.error(f"Failed after {elapsed:.2f}s: {exc}")


# ---------------------------------------------------------------------------
# Device selection (CPU-first, GPU used automatically when available)
# ---------------------------------------------------------------------------

def resolve_device(preference: str = "auto") -> str:
    if preference in ("cpu", "cuda"):
        return preference

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def configure_cpu_threads(n: int = 0) -> int:
    """Cap CPU threads sensibly so heavy models don't thrash small machines."""
    import torch
    threads = n if n > 0 else max(1, min(os.cpu_count() or 4, 4))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except RuntimeError:
        pass  # interop threads can only be set once per process
    return threads


def set_blas_thread_env(n: Optional[int] = None) -> int:
    """
    Set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS /
    NUMEXPR_NUM_THREADS *before* numpy/torch/scipy/librosa get imported
    anywhere in the process. These BLAS/OpenMP thread pools are sized once,
    the first time each library touches its native backend - setting the
    env var after that point has no effect, which is why this must run as
    early as possible (run_pipeline.py calls it before importing anything
    else). Left alone if the person already set these themselves, so an
    explicit `OMP_NUM_THREADS=1 python run_pipeline.py ...` still works.
    """
    threads = n if n and n > 0 else (os.cpu_count() or 4)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    return threads


# ---------------------------------------------------------------------------
# Memory hygiene between pipeline steps
# ---------------------------------------------------------------------------

def free_memory() -> None:
    """
    Best-effort release of RAM/VRAM between pipeline steps. Each step loads
    its own (sometimes multi-GB) model(s) into local variables inside its
    own `run(cfg)` function; once that function returns, those objects have
    no more references and are eligible for garbage collection - but:

      - reference cycles (common in torch nn.Module graphs with hooks)
        aren't freed until a GC cycle actually runs, not just on refcount
        drop, so an explicit `gc.collect()` here matters, not just letting
        Python's normal refcounting handle it;
      - CUDA's caching allocator keeps freed GPU memory reserved for reuse
        rather than handing it back to the driver, which is normally what
        you want *within* a step but not *between* two unrelated steps -
        `torch.cuda.empty_cache()` releases it;
      - CPython's own allocator likewise keeps freed heap arenas mapped
        rather than returning them to the OS; on glibc Linux,
        `malloc_trim(0)` asks it to actually give that memory back, which
        is what makes the *next* step's RSS start from a clean baseline
        instead of stacking on top of the previous step's peak usage.

    Every step of this is wrapped in try/except - this is a best-effort
    optimization, never something that should be allowed to fail a step.
    """
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass  # not glibc/Linux - nothing to do, gc.collect() above still ran


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------

def run_ffmpeg(args: list[str], logger: Optional[logging.Logger] = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    if logger:
        logger.debug("ffmpeg " + " ".join(args))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def media_duration_seconds(path: Path) -> float:
    data = ffprobe_json(path)
    return float(data["format"]["duration"])


# ---------------------------------------------------------------------------
# Audio time-stretching (used to lock dubbed segments to the original timing)
# ---------------------------------------------------------------------------

def time_stretch_to_duration(
    audio: np.ndarray,
    sr: int,
    target_duration: float,
    max_rate: float = 2.2,
    min_rate: float = 0.55,
) -> np.ndarray:
    """
    Stretch/compress `audio` (mono float32, range [-1, 1]) so its duration
    matches `target_duration` seconds, preserving pitch. This is what keeps
    the dubbed voice aligned with the original speaker's on-screen timing.
    """
    import librosa

    current_duration = len(audio) / float(sr)
    if current_duration <= 1e-6 or target_duration <= 1e-6:
        return audio

    rate = current_duration / target_duration  # >1 => speed up, <1 => slow down
    rate = float(np.clip(rate, min_rate, max_rate))

    if abs(rate - 1.0) < 1e-3:
        stretched = audio
    else:
        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)

    target_len = int(round(target_duration * sr))

    if len(stretched) > target_len:
        stretched = stretched[:target_len]
    elif len(stretched) < target_len:
        stretched = np.pad(stretched, (0, target_len - len(stretched)))

    return stretched


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def resolve_speech_source(cfg) -> Path:
    """
    Prefer the Demucs-isolated clean-vocals track (step04) for every speech
    model (VAD, diarization, whisper, reference clips); fall back to the
    plain denoised track if step04 was skipped.
    """
    if cfg.audio_clean_vocals.exists():
        return cfg.audio_clean_vocals
    return cfg.audio_enhanced


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def denoise_array(audio: np.ndarray, sr: int, prop_decrease: float = 0.6) -> np.ndarray:
    """Light post-synthesis cleanup pass (used on TTS output, not raw speech)."""
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=prop_decrease)


_SENTENCE_SPLIT_RE = None


def split_into_tts_chunks(text: str, max_chars: int = 220) -> list[str]:
    """
    Split `text` into sentence-sized chunks safe to hand to a TTS engine
    one at a time. Long, unsplit text is the usual cause of a TTS model
    truncating output after the first clause or going silent partway
    through - synthesizing sentence-by-sentence and concatenating avoids
    that entirely, and also gives natural sentence-boundary pauses instead
    of one continuous run-on utterance.

    Works across scripts by splitting on common sentence terminators from
    several languages (., !, ?, the Hindi/Sanskrit danda "।", the CJK "。",
    and the Arabic/Urdu "؟"), then further splits any resulting chunk that
    is still longer than `max_chars` at the nearest comma/space so no
    single TTS call ever gets an overly long string.
    """
    import re

    global _SENTENCE_SPLIT_RE
    if _SENTENCE_SPLIT_RE is None:
        _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।。！？؟])\s+")

    text = " ".join(text.split()).strip()
    if not text:
        return []

    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text]

    chunks: list[str] = []
    buffer = ""

    for sentence in raw_sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence

        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
            buffer = ""

        if len(sentence) <= max_chars:
            buffer = sentence
            continue

        # A single sentence longer than max_chars: break at the nearest
        # comma/space boundaries instead of mid-word.
        words = sentence.split(" ")
        piece = ""
        for word in words:
            candidate_piece = f"{piece} {word}".strip() if piece else word
            if len(candidate_piece) > max_chars and piece:
                chunks.append(piece)
                piece = word
            else:
                piece = candidate_piece
        if piece:
            buffer = piece

    if buffer:
        chunks.append(buffer)

    return chunks


def trim_silence_array(audio: np.ndarray, threshold: float = 0.02, pad_samples: int = 400) -> np.ndarray:
    """Trim leading/trailing near-silence from a synthesized clip (keeps a small pad)."""
    if audio.size == 0:
        return audio
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if non_silent.size == 0:
        return audio
    start = max(0, int(non_silent[0]) - pad_samples)
    end = min(len(audio), int(non_silent[-1]) + pad_samples)
    return audio[start:end]


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
