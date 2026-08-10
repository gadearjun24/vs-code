"""Step 09 - Merge every speaker's clips into a single per-speaker audio file."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from config import PipelineConfig
from src.utils import get_logger, save_json

LOG = get_logger(__name__)


def merge_speaker(folder, output_dir):
    files = sorted(folder.glob("*.wav"))
    merged, sample_rate, total_duration = [], None, 0.0

    for wav in files:
        audio, sr = sf.read(wav)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise RuntimeError(f"Sample rate mismatch in {wav}")

        merged.append(audio)
        total_duration += len(audio) / sr
        merged.append(np.zeros(int(0.2 * sr)))  # 200ms gap between clips

    if not merged:
        return None

    merged_audio = np.concatenate(merged)
    output_file = output_dir / f"{folder.name}.wav"
    sf.write(output_file, merged_audio, sample_rate)

    return {
        "speaker_id": folder.name,
        "audio_file": str(output_file),
        "sample_rate": sample_rate,
        "segments": len(files),
        "duration": round(total_duration, 2),
    }


def run(cfg: PipelineConfig) -> list:
    cfg.dir_speaker_profiles.mkdir(parents=True, exist_ok=True)

    speaker_dirs = sorted(d for d in cfg.dir_speakers.iterdir() if d.is_dir())
    profiles = []

    for index, speaker in enumerate(speaker_dirs, start=1):
        LOG.info(f"[{index}/{len(speaker_dirs)}] {speaker.name}")
        profile = merge_speaker(speaker, cfg.dir_speaker_profiles)
        if profile:
            profiles.append(profile)

    save_json(cfg.speaker_profiles_json, profiles)
    LOG.info(f"Speakers merged: {len(profiles)}")
    return profiles


if __name__ == "__main__":
    run(PipelineConfig())
