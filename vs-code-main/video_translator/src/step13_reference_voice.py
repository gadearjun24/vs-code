"""
Step 13 - Reference voice preparation.

Instead of picking one single ~15s window, this picks the best
`cfg.reference_clip_count` *non-overlapping* clean windows spread across
the whole speaker track and concatenates them (with small natural gaps)
into one `reference.wav`. XTTS-v2 conditions its cloning on whatever
phonetic/prosodic variety is present in the reference clip - a single
15s window often only covers a narrow slice of the person's voice (one
pitch range, one energy level), while 3 shorter clips taken from
different parts of the conversation give it a much more representative
sample of the actual voice, which is the single biggest lever for
cloning fidelity.

The concatenated reference then gets a light denoise + peak-normalize
pass as a final safety net (on top of the Demucs vocal isolation already
applied upstream in step04) before being handed to step14.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from config import PipelineConfig
from src.utils import denoise_array, ensure_mono_float32, get_logger, load_json, save_json

LOG = get_logger(__name__)

SILENCE_THRESHOLD = 0.015
TRIM_PADDING_SECONDS = 0.25
WINDOW_HOP_SECONDS = 0.5
GAP_SECONDS = 0.35            # natural pause inserted between concatenated clips
MIN_CLIP_SECONDS = 2.0


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio
    duration = len(audio) / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    audio = ensure_mono_float32(np.asarray(audio))
    audio = resample_linear(audio, sr, target_sr)
    return audio, target_sr


def trim_silence(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int, int]:
    if audio.size == 0:
        return audio, 0, 0
    abs_audio = np.abs(audio)
    non_silent = np.where(abs_audio > SILENCE_THRESHOLD)[0]
    if non_silent.size == 0:
        return audio, 0, len(audio)
    pad = int(TRIM_PADDING_SECONDS * sr)
    start = max(0, int(non_silent[0]) - pad)
    end = min(len(audio), int(non_silent[-1]) + pad)
    return audio[start:end], start, end


def score_window(segment: np.ndarray) -> float:
    if segment.size == 0:
        return -1e9
    rms = float(np.sqrt(np.mean(segment ** 2) + 1e-12))
    silence_ratio = float(np.mean(np.abs(segment) <= SILENCE_THRESHOLD))
    clip_ratio = float(np.mean(np.abs(segment) >= 0.98))
    zero_energy = float(np.mean(np.abs(segment) < 1e-4))
    return rms - 0.45 * silence_ratio - 0.10 * clip_ratio - 0.15 * zero_energy


def select_top_k_windows(trimmed: np.ndarray, sr: int, cfg: PipelineConfig) -> List[Tuple[int, int, float]]:
    """Greedy non-max-suppression: pick the best-scoring windows that don't overlap."""
    n_clips = max(1, cfg.reference_clip_count)
    per_clip_seconds = max(MIN_CLIP_SECONDS, cfg.target_reference_seconds / n_clips)
    window_len = int(per_clip_seconds * sr)
    hop_len = max(1, int(WINDOW_HOP_SECONDS * sr))

    if len(trimmed) <= window_len:
        return [(0, len(trimmed), score_window(trimmed))]

    last_start = max(0, len(trimmed) - window_len)
    candidates = [
        (start, start + window_len, score_window(trimmed[start:start + window_len]))
        for start in range(0, last_start + 1, hop_len)
    ]
    candidates.sort(key=lambda c: c[2], reverse=True)

    chosen: List[Tuple[int, int, float]] = []
    for start, end, score in candidates:
        if len(chosen) >= n_clips:
            break
        overlaps = any(not (end <= cs or start >= ce) for cs, ce, _ in chosen)
        if not overlaps:
            chosen.append((start, end, score))

    if not chosen:
        chosen = [candidates[0]]

    chosen.sort(key=lambda c: c[0])  # chronological order reads/concatenates more naturally
    return chosen


def build_reference_clip(trimmed: np.ndarray, sr: int, windows: List[Tuple[int, int, float]]) -> np.ndarray:
    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    pieces = []
    for i, (start, end, _) in enumerate(windows):
        if i > 0:
            pieces.append(gap)
        pieces.append(trimmed[start:end])
    return np.concatenate(pieces) if pieces else trimmed


def run(cfg: PipelineConfig) -> dict:
    if not cfg.normalized_speakers_json.exists():
        raise FileNotFoundError(cfg.normalized_speakers_json)

    cfg.dir_voice_reference.mkdir(parents=True, exist_ok=True)

    speakers = load_json(cfg.normalized_speakers_json)

    analysis_by_id = {}
    if cfg.speaker_analysis_json.exists():
        for row in load_json(cfg.speaker_analysis_json):
            analysis_by_id[row["speaker_id"]] = row

    results = []
    started = time.time()

    for index, speaker in enumerate(speakers, start=1):
        speaker_id = speaker["speaker_id"]
        audio_file = Path(speaker["audio_file"])
        LOG.info(f"[{index}/{len(speakers)}] {speaker_id}")

        if not audio_file.exists():
            LOG.warning(f"  missing audio: {audio_file}")
            continue

        try:
            audio, sr = load_audio(audio_file, cfg.sample_rate)
            trimmed, trim_start, trim_end = trim_silence(audio, sr)

            if trimmed.size == 0:
                raise RuntimeError("no non-silent audio found for this speaker")

            windows = select_top_k_windows(trimmed, sr, cfg)
            ref_audio = build_reference_clip(trimmed, sr, windows)

            # Safety-net cleanup on top of the upstream Demucs vocal isolation.
            ref_audio = denoise_array(ref_audio, sr, prop_decrease=0.5)
            peak = float(np.max(np.abs(ref_audio))) if ref_audio.size else 0.0
            if peak > 1e-6:
                ref_audio = (ref_audio / peak) * 0.95

            speaker_dir = cfg.dir_voice_reference / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            reference_wav = speaker_dir / "reference.wav"
            sf.write(reference_wav, ref_audio, sr, subtype="PCM_16")

            analysis = analysis_by_id.get(speaker_id, {})
            avg_score = round(float(np.mean([w[2] for w in windows])), 4)

            metadata = {
                "speaker_id": speaker_id,
                "source_audio": str(audio_file),
                "reference_audio": str(reference_wav),
                "clip_count": len(windows),
                "clip_windows_seconds": [
                    [round((trim_start + s) / sr, 2), round((trim_start + e) / sr, 2)] for s, e, _ in windows
                ],
                "reference_duration_seconds": round(len(ref_audio) / sr, 3),
                "sample_rate": sr,
                "score": avg_score,
                "gender": analysis.get("gender"),
                "gender_confidence": analysis.get("gender_confidence"),
                "age_range": analysis.get("age_range"),
                "emotion": analysis.get("emotion"),
            }

            results.append(metadata)
            LOG.info(
                f"  reference.wav saved: {len(windows)} clip(s), "
                f"{metadata['reference_duration_seconds']}s total (score={avg_score}, gender={metadata['gender']})"
            )

        except Exception as e:
            LOG.error(f"  FAILED for {speaker_id}: {e}")

    output = {
        "created_at_seconds": round(time.time() - started, 2),
        "speaker_count": len(results),
        "speakers": results,
    }
    save_json(cfg.voice_reference_json, output)

    LOG.info(f"Step 13 done in {round(time.time() - started, 2)}s for {len(results)} speakers")
    return output


if __name__ == "__main__":
    run(PipelineConfig())
