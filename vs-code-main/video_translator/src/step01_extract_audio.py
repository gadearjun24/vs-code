"""
Step 01 - Extract audio from the input video.

Produces two files:
  - audio.wav     mono, 16kHz  - feeds every speech model (VAD, diarization,
                                  whisper, embeddings)
  - audio_hq.wav  stereo, 44.1kHz (or the source rate if lower) - feeds the
                                  background-music separation step (04) so
                                  the preserved BGM/SFX in the final dub
                                  don't sound duller than necessary.
"""

from __future__ import annotations

import ffmpeg

from config import PipelineConfig
from src.utils import ffprobe_json, get_logger

LOG = get_logger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {cfg.input_video}")

    cfg.dir_audio.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Extracting mono {cfg.sample_rate}Hz audio -> {cfg.audio_raw}")
    (
        ffmpeg
        .input(str(cfg.input_video))
        .output(str(cfg.audio_raw), ac=1, ar=cfg.sample_rate, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    LOG.info(f"Saved: {cfg.audio_raw}")

    # Don't upsample past the source's own rate - only cap it.
    try:
        probe = ffprobe_json(cfg.input_video)
        source_rate = int(next(s["sample_rate"] for s in probe["streams"] if s["codec_type"] == "audio"))
    except (StopIteration, KeyError):
        source_rate = cfg.hq_sample_rate

    hq_rate = min(cfg.hq_sample_rate, source_rate) if source_rate else cfg.hq_sample_rate

    LOG.info(f"Extracting stereo {hq_rate}Hz audio -> {cfg.audio_hq}")
    (
        ffmpeg
        .input(str(cfg.input_video))
        .output(str(cfg.audio_hq), ac=2, ar=hq_rate, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    LOG.info(f"Saved: {cfg.audio_hq}")


if __name__ == "__main__":
    run(PipelineConfig())
