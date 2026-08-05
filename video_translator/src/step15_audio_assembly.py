"""
Step 15 - Audio assembly.

Two things happen here:

1. Every generated segment (step13) is placed at its original timestamp -
   at its NATURAL, unmodified speaking speed. Earlier versions of this
   step time-stretched each line to force it into exactly the original
   slot's duration, which is what caused the "sped up / chipmunk" effect
   when a translated line ran longer than the source line: compressing
   audio in time to make it fit is a real, audible speed change, and it
   never truncates a word either way - the whole line is always played in
   full.

   Instead: lines are placed in chronological order. Each line starts at
   `max(its own original timestamp, the moment the previous line actually
   finished + a small natural gap)`. A line is never pulled earlier than
   its own timestamp, and never sped up/slowed down/cut - if a translation
   runs long, only the *start* of later lines drifts forward to avoid
   overlapping it, never the audio itself. This keeps sync tight in the
   normal case (most lines have some silence after them) and degrades
   gracefully (a little later, never distorted or clipped) in the rare
   case a translation is much longer than the original line.

   Set `cfg.stretch_audio_to_fit = True` if you'd rather have strict
   timestamp alignment at the cost of occasional pitch-preserved
   speed-up/down - off by default.

2. The background bed extracted in step04 (music/ambience/SFX, with the
   original speech removed) is mixed back in underneath the dubbed speech,
   ducked a little during active speech (like a professional voiceover
   mix) and left at full volume during silence.

The final mix is written at the higher `cfg.hq_sample_rate` (stereo)
instead of the 16kHz mono speech-model rate, since this file is a
deliverable, not an intermediate feature for a speech model.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from config import PipelineConfig
from src.utils import (
    db_to_linear, ensure_mono_float32, get_logger, load_json,
    media_duration_seconds, time_stretch_to_duration,
)

LOG = get_logger(__name__)

DUCK_FADE_SECONDS = 0.20
DRIFT_WARNING_SECONDS = 0.75  # log a warning if a line has to drift later than this


def load_resampled(path, target_sr: int, mono: bool = False) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=not mono)
    if mono:
        audio = ensure_mono_float32(np.asarray(audio))
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio

    audio = np.asarray(audio, dtype=np.float32)  # (samples, channels)
    if sr != target_sr:
        import librosa
        audio = np.stack(
            [librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr) for ch in range(audio.shape[1])],
            axis=1,
        )
    return audio


def build_duck_envelope(length: int, sr: int, windows: list, duck_level: float) -> np.ndarray:
    """1.0 = full background volume, `duck_level` = volume during active speech."""
    envelope = np.ones(length, dtype=np.float32)
    fade = max(1, int(DUCK_FADE_SECONDS * sr))

    for start, end in windows:
        s = max(0, int(round(start * sr)))
        e = min(length, int(round(end * sr)))
        if e <= s:
            continue

        segment_env = np.full(e - s, duck_level, dtype=np.float32)

        fade_in_len = min(fade, len(segment_env))
        segment_env[:fade_in_len] = np.linspace(1.0, duck_level, fade_in_len, dtype=np.float32)
        fade_out_len = min(fade, len(segment_env))
        segment_env[-fade_out_len:] = np.minimum(
            segment_env[-fade_out_len:],
            np.linspace(duck_level, 1.0, fade_out_len, dtype=np.float32),
        )

        envelope[s:e] = np.minimum(envelope[s:e], segment_env)

    return envelope


def place_segments(records: list, mix_sr: int, cfg: PipelineConfig, total_samples: int):
    """
    Returns (speech_track, active_windows_seconds, stats).

    Chronological, non-overlapping placement at natural speed. This is the
    core fix for the "audio speeds up" bug: no resampling/time-stretching
    happens here unless `cfg.stretch_audio_to_fit` is explicitly turned on.
    """
    speech_track = np.zeros(total_samples, dtype=np.float32)
    active_windows = []
    placed = skipped = drifted = 0
    max_drift = 0.0

    ordered = sorted(
        (r for r in records if r.get("status") == "ok" and r.get("original_start") is not None),
        key=lambda r: r["original_start"],
    )

    last_end_sample = 0
    min_gap_samples = max(0, int(cfg.min_gap_between_lines_seconds * mix_sr))

    for record in ordered:
        start = float(record["original_start"])
        end = float(record.get("original_end", start))
        target_duration = max(0.05, end - start)

        try:
            audio = load_resampled(record["final_audio"], mix_sr, mono=True)

            if cfg.stretch_audio_to_fit:
                # Opt-in legacy behavior: force-fit the original slot,
                # which can change speed. Off by default.
                audio = time_stretch_to_duration(audio, mix_sr, target_duration)

            intended_start_sample = int(round(start * mix_sr))
            actual_start_sample = max(intended_start_sample, last_end_sample)

            drift_seconds = (actual_start_sample - intended_start_sample) / mix_sr
            if drift_seconds > DRIFT_WARNING_SECONDS:
                drifted += 1
                max_drift = max(max_drift, drift_seconds)
                LOG.warning(
                    f"  {record.get('segment_id')}: previous line(s) ran long, this line starts "
                    f"{drift_seconds:.2f}s after its original timestamp (kept at natural speed, "
                    f"not sped up)"
                )

            actual_end_sample = actual_start_sample + len(audio)

            if actual_end_sample > len(speech_track):
                speech_track = np.pad(speech_track, (0, actual_end_sample - len(speech_track)))

            speech_track[actual_start_sample:actual_end_sample] += audio
            active_windows.append((actual_start_sample / mix_sr, actual_end_sample / mix_sr))

            last_end_sample = actual_end_sample + min_gap_samples
            placed += 1

        except Exception as e:
            LOG.error(f"Failed to place segment {record.get('segment_id')}: {e}")
            skipped += 1

    stats = {"placed": placed, "skipped": skipped, "drifted_lines": drifted, "max_drift_seconds": round(max_drift, 2)}
    return speech_track, active_windows, stats


def run(cfg: PipelineConfig) -> None:
    if not cfg.generated_audio_json.exists():
        raise FileNotFoundError(cfg.generated_audio_json)

    cfg.dir_final_audio.mkdir(parents=True, exist_ok=True)

    records = load_json(cfg.generated_audio_json)

    # Mix at whatever rate the preserved background track actually is
    # (falls back to cfg.hq_sample_rate if step04 was skipped).
    if cfg.background_music.exists():
        mix_sr = sf.info(str(cfg.background_music)).samplerate
    else:
        mix_sr = cfg.hq_sample_rate

    total_duration = media_duration_seconds(cfg.input_video)
    total_samples = int(round(total_duration * mix_sr)) + mix_sr  # 1s safety pad

    speech_track, active_windows, stats = place_segments(records, mix_sr, cfg, total_samples)

    LOG.info(
        f"Speech placement: placed={stats['placed']} skipped={stats['skipped']} "
        f"(natural speed - no time-stretch{' (except stretch_audio_to_fit lines)' if cfg.stretch_audio_to_fit else ''})"
    )
    if stats["drifted_lines"]:
        LOG.warning(
            f"{stats['drifted_lines']} line(s) drifted later than their original timestamp "
            f"(up to {stats['max_drift_seconds']}s) because earlier translated lines ran long. "
            f"This keeps natural speech pace instead of speeding audio up - set "
            f"cfg.stretch_audio_to_fit=True if you'd rather force-fit timestamps instead."
        )

    # Prevent clipping from overlapping speech before mixing with BGM.
    peak = float(np.max(np.abs(speech_track))) if speech_track.size else 0.0
    if peak > 0.98:
        speech_track = speech_track / peak * 0.98

    # Stereo-ize the mono speech track.
    speech_stereo = np.stack([speech_track, speech_track], axis=1)

    if cfg.background_music.exists():
        LOG.info(f"Mixing in preserved background bed: {cfg.background_music}")
        background = load_resampled(cfg.background_music, mix_sr, mono=False)

        # Match lengths (the speech track may now be longer than the
        # original video if lines drifted past the end - pad BGM with
        # silence rather than looping/truncating speech to fit).
        target_len = len(speech_stereo)
        if len(background) < target_len:
            background = np.pad(background, ((0, target_len - len(background)), (0, 0)))
        else:
            background = background[:target_len]

        duck_level = db_to_linear(cfg.background_duck_db)
        envelope = build_duck_envelope(target_len, mix_sr, active_windows, duck_level)
        background = background * envelope[:, None] * db_to_linear(cfg.background_gain_db)

        final_mix = speech_stereo + background
    else:
        LOG.warning("No background.wav found (step04 skipped?) - dub will have no BGM/SFX bed.")
        final_mix = speech_stereo

    peak = float(np.max(np.abs(final_mix))) if final_mix.size else 0.0
    if peak > 0.98:
        final_mix = final_mix / peak * 0.98

    sf.write(cfg.final_dubbed_audio, final_mix, mix_sr, subtype="PCM_16")

    LOG.info(f"Assembled dubbed audio: {cfg.final_dubbed_audio} (sr={mix_sr}, stereo, BGM preserved)")


if __name__ == "__main__":
    run(PipelineConfig())
