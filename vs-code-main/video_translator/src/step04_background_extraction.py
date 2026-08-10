"""
Step 04 - Background music / SFX separation (Demucs).

Splits the enhanced audio into:
  - clean_vocals.wav - speech only, no music/SFX bleeding through. Every
    downstream speech step (VAD, diarization, transcription, the per-speaker
    reference clips used for voice cloning) runs on THIS file instead of the
    raw mix. This is the single biggest lever for high-fidelity voice
    cloning: XTTS clones whatever is in the reference clip, so a reference
    clip with music/noise bleeding into it teaches it to reproduce that
    noise too.
  - background.wav - everything that isn't vocals (music, ambience, SFX),
    kept at a higher sample rate. Step 16 mixes this back in under the
    dubbed speech so the final video doesn't go silent on BGM/SFX when you
    switch to the dubbed audio track.
"""

from __future__ import annotations

import time

import numpy as np
import soundfile as sf
import torch

from config import PipelineConfig
from src.utils import ensure_mono_float32, get_logger, resolve_device, save_json

LOG = get_logger(__name__)


def load_stereo_tensor(path, target_channels: int = 2):
    audio, sr = sf.read(str(path), always_2d=True)  # (samples, channels)
    if audio.shape[1] < target_channels:
        audio = np.repeat(audio, target_channels, axis=1)[:, :target_channels]
    wav = torch.from_numpy(audio.T).float()  # (channels, samples)
    return wav, sr


def run(cfg: PipelineConfig) -> dict:
    from demucs.api import Separator

    source_audio = cfg.audio_hq if cfg.audio_hq.exists() else cfg.audio_enhanced
    if not source_audio.exists():
        raise FileNotFoundError(source_audio)

    cfg.dir_background.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    LOG.info(f"Loading Demucs '{cfg.demucs_model}' on {device} (first run downloads the model)...")
    separator = Separator(model=cfg.demucs_model, device=device, progress=True)

    LOG.info(f"Reading {source_audio} for separation...")
    wav, sr = load_stereo_tensor(source_audio)

    LOG.info("Separating vocals from background music/SFX (this is the slowest CPU step)...")
    start = time.time()
    _, stems = separator.separate_tensor(wav, sr)
    elapsed = time.time() - start

    if "vocals" not in stems:
        raise RuntimeError(f"Demucs model '{cfg.demucs_model}' has no 'vocals' stem: {list(stems)}")

    vocals = stems["vocals"]
    background = sum(tensor for name, tensor in stems.items() if name != "vocals")

    # Clean speech-only track, downsampled to the pipeline's speech sample
    # rate for VAD/diarization/whisper/reference-clip selection.
    vocals_mono = ensure_mono_float32(vocals.mean(dim=0).cpu().numpy())
    if separator.samplerate != cfg.sample_rate:
        import librosa
        vocals_mono = librosa.resample(vocals_mono, orig_sr=separator.samplerate, target_sr=cfg.sample_rate)
    sf.write(cfg.audio_clean_vocals, vocals_mono, cfg.sample_rate, subtype="PCM_16")

    # Background bed, kept at the higher separation sample rate/stereo for
    # the final mix in step 17.
    background_np = background.clamp(-1.0, 1.0).cpu().numpy().T  # (samples, channels)
    sf.write(cfg.background_music, background_np, separator.samplerate, subtype="PCM_16")

    metadata = {
        "model": cfg.demucs_model,
        "device": device,
        "source_audio": str(source_audio),
        "separator_samplerate": separator.samplerate,
        "clean_vocals": str(cfg.audio_clean_vocals),
        "background_music": str(cfg.background_music),
        "processing_time_seconds": round(elapsed, 2),
    }
    save_json(cfg.dir_background / "separation_metadata.json", metadata)

    LOG.info(f"Separation done in {elapsed:.2f}s")
    LOG.info(f"  clean speech  -> {cfg.audio_clean_vocals}")
    LOG.info(f"  background    -> {cfg.background_music}")
    return metadata


if __name__ == "__main__":
    run(PipelineConfig())
