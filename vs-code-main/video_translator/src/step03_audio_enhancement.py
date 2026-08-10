"""
Step 03 - Denoise the extracted audio with `noisereduce` (spectral gating).

Originally this step used DeepFilterNet3. It was swapped out because
DeepFilterNet's native extension (deepfilterlib) ships no prebuilt PyPI
wheels - every install compiles it from source, which requires a Rust
toolchain - and it hard-pins numpy<2.0, which conflicts with the numpy>=2
that pyannote.audio 4.x (step06 diarization / step11 speaker embedding)
requires. `noisereduce` needs
neither, installs from a single pip wheel, and runs fast on CPU (it can
also use a torch/GPU backend automatically when available).
"""

from __future__ import annotations

import time

import noisereduce as nr
import soundfile as sf

from config import PipelineConfig
from src.utils import ensure_mono_float32, get_logger, resolve_device, save_json

LOG = get_logger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.audio_raw.exists():
        raise FileNotFoundError(cfg.audio_raw)

    cfg.dir_enhanced.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading audio...")
    audio, sr = sf.read(cfg.audio_raw)
    audio = ensure_mono_float32(audio)

    device = resolve_device(cfg.device)
    use_torch = device == "cuda"

    LOG.info(f"Denoising audio (use_torch={use_torch}, device={device})...")
    start = time.time()

    enhanced = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,           # adapts to changing background noise
        prop_decrease=0.85,         # keep some natural room tone (avoids robotic artifacts)
        use_torch=use_torch,
        device=device if use_torch else "cpu",
    )

    processing_time = round(time.time() - start, 2)

    LOG.info("Saving enhanced audio...")
    sf.write(cfg.audio_enhanced, enhanced, sr, subtype="PCM_16")

    save_json(cfg.dir_enhanced / "enhancement_metadata.json", {
        "method": "noisereduce (spectral gating, non-stationary)",
        "sample_rate": sr,
        "channels": 1,
        "samples": len(enhanced),
        "processing_time_seconds": processing_time,
        "input": str(cfg.audio_raw),
        "output": str(cfg.audio_enhanced),
    })

    LOG.info(f"Enhanced audio saved: {cfg.audio_enhanced} ({processing_time}s)")


if __name__ == "__main__":
    run(PipelineConfig())
