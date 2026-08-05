"""Step 05 - Voice Activity Detection with Silero VAD."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from config import PipelineConfig
from src.utils import ensure_mono_float32, get_logger, resolve_speech_source, save_json

LOG = get_logger(__name__)


def load_audio_tensor(path, target_sr: int) -> torch.Tensor:
    """
    Load audio as a mono float32 torch tensor at `target_sr`.

    Deliberately uses soundfile instead of silero_vad's own `read_audio()`
    helper: that helper calls torchaudio.load(), and torchaudio>=2.9 raises
    unless the separate `torchcodec` package is also installed. Our audio is
    always already mono at cfg.sample_rate by this point in the pipeline,
    so this stays a simple, dependency-light load+resample.
    """
    audio, sr = sf.read(str(path), always_2d=False)
    audio = ensure_mono_float32(np.asarray(audio))

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return torch.from_numpy(audio)


def run(cfg: PipelineConfig) -> list:
    from silero_vad import get_speech_timestamps, load_silero_vad

    source = resolve_speech_source(cfg)
    if not source.exists():
        raise FileNotFoundError(source)

    cfg.dir_vad.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading Silero VAD...")
    model = load_silero_vad()

    LOG.info(f"Reading audio from {source}...")
    audio = load_audio_tensor(source, cfg.sample_rate)

    LOG.info("Detecting speech...")
    speech = get_speech_timestamps(audio, model, sampling_rate=cfg.sample_rate)

    LOG.info(f"Speech segments: {len(speech)}")

    save_json(cfg.vad_json, speech)

    with open(cfg.dir_vad / "speech_timeline.txt", "w") as f:
        for i, s in enumerate(speech):
            start = round(s["start"] / cfg.sample_rate, 2)
            end = round(s["end"] / cfg.sample_rate, 2)
            f.write(f"{i + 1}. {start:.2f}s --> {end:.2f}s\n")

    return speech


if __name__ == "__main__":
    run(PipelineConfig())

