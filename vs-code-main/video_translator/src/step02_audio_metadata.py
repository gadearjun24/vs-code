"""Step 02 - Extract and persist audio metadata via ffprobe."""

from __future__ import annotations

from pathlib import Path

from config import PipelineConfig
from src.utils import ffprobe_json, get_logger, save_json

LOG = get_logger(__name__)


def parse_metadata(data: dict) -> dict:
    stream = next(s for s in data["streams"] if s["codec_type"] == "audio")
    fmt = data["format"]

    return {
        "filename": Path(fmt["filename"]).name,
        "duration_seconds": float(fmt["duration"]),
        "sample_rate": int(stream["sample_rate"]),
        "channels": stream["channels"],
        "codec": stream["codec_name"],
        "bit_rate": int(fmt.get("bit_rate", 0)),
        "format": fmt["format_name"],
        "size_mb": round(int(fmt["size"]) / (1024 * 1024), 2),
    }


def run(cfg: PipelineConfig) -> dict:
    if not cfg.audio_raw.exists():
        raise FileNotFoundError(cfg.audio_raw)

    raw = ffprobe_json(cfg.audio_raw)
    metadata = parse_metadata(raw)
    save_json(cfg.audio_metadata_json, metadata)

    LOG.info("Audio metadata:")
    for key, value in metadata.items():
        LOG.info(f"  {key}: {value}")

    return metadata


if __name__ == "__main__":
    run(PipelineConfig())
