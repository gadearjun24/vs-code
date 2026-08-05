"""Step 07 - Split the clean-vocals audio into per-speaker segment clips."""

from __future__ import annotations

import soundfile as sf

from config import PipelineConfig
from src.utils import get_logger, load_json, resolve_speech_source

LOG = get_logger(__name__)


def run(cfg: PipelineConfig) -> None:
    source = resolve_speech_source(cfg)
    if not source.exists():
        raise FileNotFoundError(source)
    if not cfg.diarization_json.exists():
        raise FileNotFoundError(cfg.diarization_json)

    audio, sr = sf.read(source)
    segments = load_json(cfg.diarization_json)

    LOG.info(f"Loaded {len(segments)} diarization segments (source: {source.name})")

    for seg in segments:
        folder = cfg.dir_speakers / seg["speaker_id"]
        folder.mkdir(parents=True, exist_ok=True)

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        clip = audio[start_sample:end_sample]

        sf.write(folder / f"{seg['segment_id']}.wav", clip, sr)

    LOG.info(f"Speaker folders saved in {cfg.dir_speakers}")


if __name__ == "__main__":
    run(PipelineConfig())
