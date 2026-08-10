"""
Step 14 - Voice generation (translated text -> cloned speech per segment).

Fixes five real production issues:

1. "Namaskar dosto" plays then the line just stops. Long, multi-sentence
   translated text handed to a TTS model in a single call is a common
   cause of truncation/early-stop - and XTTS's actual per-language limit
   varies a lot (Hindi is 150 characters, English is 250, Japanese is only
   71), so a single flat chunk size for every language either truncates
   some languages or over-splits others. Fix: split each segment's text
   into sentence-sized chunks sized to XTTS's *real* per-language limit
   (`xtts_char_limit` in `config.py`, taken directly from XTTS's own
   tokenizer), synthesize each chunk separately, and concatenate them -
   every sentence gets its own full synthesis pass instead of a single
   call trying to carry the whole paragraph past its language's limit.

2. Unnatural/too-short pauses between sentences. Fix: insert an explicit
   `cfg.tts_sentence_pause_ms` gap of true silence between chunks (instead
   of relying on whatever gap the model happens to leave), and trim each
   chunk's own leading/trailing silence FIRST, conservatively (a low
   amplitude threshold + generous padding - see `assemble_chunks` - biased
   toward keeping a little too much silence rather than risking clipping
   the tail end of a word) so gaps don't compound.

3. Background noise / unclear voice in the generated audio. Fix: a light
   `noisereduce` pass + peak normalization on the final concatenated
   segment before it's written out (toggle with `cfg.tts_post_denoise`).

4. Sync drift against the source video, WITHOUT ever cropping audio or
   letting a segment run longer than its original slot. Right after a
   segment's audio is fully synthesized and cleaned up - BEFORE it's
   written to disk - its duration is compared against that segment's own
   original (diarized) start/end and pitch-preserving time-stretched
   (`librosa.effects.time_stretch` - speeds up or slows down, never cuts
   a single sample) to match it exactly (`cfg.sync_segment_duration`, on
   by default). The stretch-rate ceiling for SPEEDING UP is deliberately
   generous (`cfg.tts_stretch_max_rate`, default 3.0x) specifically so a
   segment essentially never ends up longer than its original slot - the
   one outcome that could push into, and audibly overlap/"crop", the next
   line once everything is combined in step16. Slowing down is capped
   tighter (`cfg.tts_stretch_min_rate`) purely for naturalness, since a
   short line stretched longer can never overflow into the next one
   regardless of how far it's stretched. This step and step16 both NEVER
   truncate synthesized speech to force a duration - only this pitch-
   preserving stretch is ever used, so no word is ever cut short.

5. Flat, uniform-sounding delivery regardless of what's actually being
   said. Gemma 4 (step12) detects an `emotion` label for each segment's
   original delivery; this step nudges `tts_speed`/`tts_temperature` a
   little per segment based on that label (`cfg.tts_emotion_aware_delivery`).
   XTTS-v2 has no direct "emotion" input, so this is a deliberately modest
   proxy through its two real prosody knobs, not a claim of true emotional
   TTS - see `EMOTION_SPEED_ADJUST` / `EMOTION_TEMPERATURE_ADJUST` below.

6. Songs getting mistranslated/redubbed as if they were spoken dialogue.
   Gemma 4 (step12) flags each segment `is_song` (sung, as opposed to
   spoken) and `contains_dialogue` (has genuine spoken dialogue). When
   `cfg.keep_song_audio_original` is on (default):
   - a PURE song segment (`is_song` and not `contains_dialogue`) skips
     synthesis entirely - the original recording (step08's split clip) is
     used verbatim, at its own natural length, no denoise/normalize pass
     that might alter its fidelity;
   - a MIXED segment (both true) uses the original recording as its base
     and splices in a synthesized, translated dub ONLY at Gemma's
     estimated `dialogue_windows` (see step12) - everywhere else in the
     clip stays the untouched original. Because those window boundaries
     are an LLM's best-effort estimate rather than frame-accurate cut
     points, each splice is blended with a short crossfade
     (`cfg.song_splice_crossfade_ms`) rather than a hard cut, and if
     Gemma didn't return usable dialogue timing for a mixed clip, this
     step falls back to the safe choice (original audio, untouched)
     rather than guessing at a cut point;
   - if `is_song` is false (normal dialogue), nothing changes from before.
   See `load_original_segment_audio` / `split_text_proportional` /
   `crossfade_splice` below for the implementation.

7. Every dubbed line coming out at the same flattened loudness regardless
   of whether the original speaker was shouting or whispering - which
   quietly undoes the point of the emotion-aware delivery in item 5 above.
   Gemma doesn't estimate loudness (that's an objective signal property,
   not something to ask a language model to guess); step12 measures each
   segment's real RMS loudness directly from its audio
   (`original_volume_dbfs`). Here, instead of always peak-normalizing to a
   flat 0.95, `compute_target_peak()` offsets that baseline up or down by
   however far this segment's original volume sits from the conversation's
   average - clamped to `cfg.volume_dynamic_range_db` so one extreme
   outlier can't be normalized into clipping or near-silence, then to
   `cfg.volume_min_peak`/`cfg.volume_max_peak` as an absolute safety
   floor/ceiling. Toggle with `cfg.volume_aware_normalization`.

Two backends, chosen automatically per target language:

* XTTS-v2 (coqui-tts) - true zero-shot voice cloning from `reference.wav`
  (now a multi-clip reference from step13). Used whenever `cfg.target_lang`
  is one of XTTS's 17 supported languages. Because it clones the *actual*
  speaker sample, gender/timbre is preserved automatically: male -> male,
  female -> female, other -> other.

* MMS-TTS (facebook/mms-tts-<lang>) - single-voice fallback covering
  ~1100 languages, for targets XTTS doesn't speak. A small pitch shift is
  applied toward the source speaker's estimated register (plus a smaller
  emotion-based nudge on top) so the result still leans toward the right
  gender and isn't completely flat.

CPU performance
----------------
This is the single heaviest CPU step in the pipeline (autoregressive
synthesis, one full model forward pass per sentence), so unlike lighter
steps it's allowed to use every core on the machine rather than the
conservative shared cap the rest of the pipeline uses
(`cfg.tts_cpu_threads`, default 0 = all cores) - nothing else runs
concurrently with it, so there's no contention to protect against. All
synthesis calls also run inside `torch.inference_mode()` to skip
autograd bookkeeping entirely, and the model + reference audio are loaded
exactly once and reused for every segment/chunk rather than being
recreated. See `free_memory()` in `src/utils.py` for the RAM-hygiene
pass this step (and every other step) runs on exit.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from config import PipelineConfig, XTTS_SUPPORTED_LANGUAGES, mms_tts_repo, xtts_char_limit
from src.utils import (
    configure_cpu_threads, db_to_linear, denoise_array, free_memory, get_logger, load_json,
    resample_audio, resolve_device, save_json, split_into_tts_chunks, time_stretch_to_duration,
    trim_silence_array,
)

LOG = get_logger(__name__)

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

GENDER_PITCH_SHIFT = {"female": 3.0, "male": -2.0, "child": 5.0}

# Modest prosody nudges applied per Gemma-4-detected emotion (step12). Kept
# small deliberately - these ride on top of `cfg.tts_speed`/
# `cfg.tts_temperature` as multipliers/offsets, not replacements, and are
# applied BEFORE the duration sync (item 4 above), so they never fight
# with the exact-duration guarantee - sync always has the final word on
# actual output length.
EMOTION_SPEED_ADJUST = {
    "happy": 1.05, "excited": 1.10, "angry": 1.08, "surprised": 1.08,
    "sad": 0.92, "calm": 0.95, "fearful": 1.05, "neutral": 1.0,
}
EMOTION_TEMPERATURE_ADJUST = {
    "happy": 0.05, "excited": 0.08, "angry": 0.05, "surprised": 0.06,
    "sad": -0.05, "calm": -0.05, "fearful": 0.03, "neutral": 0.0,
}
# Small extra pitch nudge for the MMS fallback (added to GENDER_PITCH_SHIFT).
EMOTION_PITCH_ADJUST = {
    "happy": 1.0, "excited": 1.5, "angry": 0.5, "surprised": 1.0,
    "sad": -1.0, "calm": -0.5, "fearful": 0.5, "neutral": 0.0,
}

# Free memory every N segments during the main loop, in addition to the
# guaranteed cleanup when the step finishes - keeps peak RAM flat across a
# very long video instead of only reclaiming it once at the very end.
MEMORY_SWEEP_EVERY = 25


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def normalize_emotion(value: Optional[str]) -> str:
    v = (value or "neutral").strip().lower()
    return v if v in EMOTION_SPEED_ADJUST else "neutral"


def load_voice_reference_index(cfg: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    if not cfg.voice_reference_json.exists():
        raise FileNotFoundError(cfg.voice_reference_json)
    data = load_json(cfg.voice_reference_json)
    speakers = data["speakers"] if isinstance(data, dict) else data
    return {row["speaker_id"]: row for row in speakers if row.get("speaker_id")}


def assemble_chunks(chunk_arrays: List[np.ndarray], sr: int, pause_ms: int) -> np.ndarray:
    if not chunk_arrays:
        return np.zeros(0, dtype=np.float32)

    gap = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
    pieces: List[np.ndarray] = []
    for i, chunk in enumerate(chunk_arrays):
        # Deliberately conservative: a low amplitude threshold + generous
        # padding, biased toward keeping a touch too much silence rather
        # than risking trimming into an actual quiet trailing consonant -
        # trimming real speech here is exactly the kind of "sounds cropped"
        # artifact this step is meant to avoid.
        trimmed = trim_silence_array(chunk, threshold=0.015, pad_samples=int(0.04 * sr))
        if i > 0:
            pieces.append(gap)
        pieces.append(trimmed)
    return np.concatenate(pieces)


def post_process(audio: np.ndarray, sr: int, denoise: bool = True, target_peak: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio
    cleaned = denoise_array(audio, sr, prop_decrease=0.5) if denoise else audio
    peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    if peak > 1e-6:
        cleaned = (cleaned / peak) * target_peak
    return cleaned.astype(np.float32)


# ---------------------------------------------------------------------------
# Volume-aware normalization (preserve the original speaker's relative
# loudness - a shouted line stays louder than a whispered one in the dub -
# instead of flattening every segment's peak to the same fixed level).
# ---------------------------------------------------------------------------

def compute_reference_volume_dbfs(conversation: list) -> Optional[float]:
    values = [s.get("original_volume_dbfs") for s in conversation if s.get("original_volume_dbfs") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compute_target_peak(
    original_volume_dbfs: Optional[float], reference_dbfs: Optional[float],
    cfg: PipelineConfig, base_peak: float = 0.95,
) -> float:
    """
    The flat baseline (`base_peak`) shifted up/down by however far this
    segment's original loudness was from the conversation's average -
    clamped to `cfg.volume_dynamic_range_db` so one extreme outlier can't
    push a line to clipping or near-silence, and then to
    `cfg.volume_min_peak`/`cfg.volume_max_peak` as a final safety clamp.
    """
    if not cfg.volume_aware_normalization or original_volume_dbfs is None or reference_dbfs is None:
        return base_peak

    offset_db = original_volume_dbfs - reference_dbfs
    offset_db = max(-cfg.volume_dynamic_range_db, min(cfg.volume_dynamic_range_db, offset_db))
    target = base_peak * db_to_linear(offset_db)
    return max(cfg.volume_min_peak, min(cfg.volume_max_peak, target))


# ---------------------------------------------------------------------------
# Song vs. dialogue splicing (cfg.keep_song_audio_original)
# ---------------------------------------------------------------------------

def load_original_segment_audio(audio_file: str, target_sr: int) -> np.ndarray:
    """Load the ORIGINAL (un-translated) segment clip step08 extracted, resampled to target_sr."""
    audio, sr = sf.read(audio_file, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    return resample_audio(audio, sr, target_sr)


def split_text_proportional(text: str, windows: List[dict]) -> List[str]:
    """
    Gemma gives ONE translation string for all dialogue in a segment, but a
    mixed song+dialogue clip can have multiple separate dialogue_windows.
    Split the translated text across them by each window's share of the
    total dialogue duration (word count, not character count, so the split
    falls on natural word boundaries) - approximate by nature, same as the
    window timings themselves.
    """
    if len(windows) <= 1:
        return [text]

    words = text.split()
    if not words:
        return ["" for _ in windows]

    total_duration = sum(max(0.0, w["end"] - w["start"]) for w in windows) or 1.0
    pieces, used = [], 0
    for i, w in enumerate(windows):
        if i == len(windows) - 1:
            count = len(words) - used
        else:
            share = max(0.0, w["end"] - w["start"]) / total_duration
            count = min(len(words) - used, max(1, round(len(words) * share)))
        pieces.append(" ".join(words[used:used + count]))
        used += count
    return pieces


def crossfade_splice(base: np.ndarray, sr: int, insert: np.ndarray, start_sample: int, crossfade_ms: int) -> np.ndarray:
    """
    Replace base[start_sample : start_sample+len(insert)] with `insert`,
    blending a short (`crossfade_ms`) ramp at each edge against whatever
    was already in `base` right there, so an approximately-timed splice
    point (dialogue_windows are an LLM's best-effort estimate, not
    frame-accurate) doesn't produce an audible click/pop. This is a
    boundary blend, not a full overlap-add crossfade - sufficient to
    smooth the cut without needing sample-accurate alignment data that
    doesn't exist here.
    """
    result = base.copy()
    end_sample = start_sample + len(insert)
    if end_sample > len(result):
        result = np.pad(result, (0, end_sample - len(result)))

    insert = insert.copy()
    cf = max(0, min(int(sr * crossfade_ms / 1000), len(insert) // 2, start_sample, len(result) - end_sample))

    if cf > 0:
        fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        fade_out = 1.0 - fade_in
        insert[:cf] = insert[:cf] * fade_in + result[start_sample:start_sample + cf] * fade_out
        insert[-cf:] = insert[-cf:] * fade_out + result[end_sample - cf:end_sample] * fade_in

    result[start_sample:end_sample] = insert
    return result


def sync_to_original_duration(
    audio: np.ndarray, sr: int, target_duration: Optional[float], cfg: PipelineConfig,
) -> tuple[np.ndarray, Optional[float]]:
    """
    Time-stretch `audio` (pitch-preserved, full content kept - never a
    crop/truncation) so it matches the segment's own original spoken
    duration exactly. Returns (audio, applied_rate) - applied_rate is None
    when no stretch was applied (disabled, or no timing data available).
    """
    if not cfg.sync_segment_duration or not target_duration or target_duration <= 0 or audio.size == 0:
        return audio, None

    current_duration = len(audio) / float(sr)
    if current_duration <= 1e-6:
        return audio, None

    rate = current_duration / target_duration
    stretched = time_stretch_to_duration(
        audio, sr, target_duration,
        max_rate=cfg.tts_stretch_max_rate, min_rate=cfg.tts_stretch_min_rate,
    )
    return stretched, rate


class XTTSBackend:
    """Zero-shot voice cloning backend, one sentence-chunk at a time."""

    def __init__(self, device: str):
        from TTS.api import TTS

        LOG.info(f"Loading XTTS-v2 on {device} (first run downloads ~2GB of weights)...")
        self.tts = TTS(XTTS_MODEL_NAME).to(device)
        self.sr = self.tts.synthesizer.output_sample_rate
        # Some coqui-tts releases' high-level TTS.api.tts() don't forward a
        # `temperature` kwarg to the underlying model (it's always a valid
        # low-level model.inference() param, but the wrapper's kwarg
        # passthrough has varied across versions). Detected lazily on first
        # use rather than assumed, so this never crashes a run outright.
        self._temperature_supported: Optional[bool] = None
        LOG.info(f"XTTS-v2 loaded (output sample rate: {self.sr}).")

    def synthesize(
        self, text: str, speaker_wav: Path, language: str,
        speed: float = 1.0, temperature: Optional[float] = None,
    ) -> list:
        max_chars = xtts_char_limit(language)
        chunks = split_into_tts_chunks(text, max_chars=max_chars)
        if not chunks:
            return []

        chunk_arrays = []
        with torch.inference_mode():
            for chunk in chunks:
                kwargs = dict(
                    text=chunk,
                    speaker_wav=str(speaker_wav),
                    language=language,
                    speed=speed,
                    split_sentences=False,  # we already split sentence-by-sentence ourselves
                )
                if temperature is not None and self._temperature_supported is not False:
                    try:
                        wav = self.tts.tts(temperature=temperature, **kwargs)
                        self._temperature_supported = True
                    except TypeError:
                        self._temperature_supported = False
                        LOG.warning(
                            "  This coqui-tts version's TTS.tts() doesn't accept `temperature` - "
                            "continuing with speed-only emotion adjustment for the rest of the run."
                        )
                        wav = self.tts.tts(**kwargs)
                else:
                    wav = self.tts.tts(**kwargs)
                chunk_arrays.append(np.asarray(wav, dtype=np.float32))

        return chunk_arrays


class MMSFallbackBackend:
    """Single-voice fallback for languages XTTS doesn't cover."""

    def __init__(self, lang: str, device: str):
        from transformers import AutoTokenizer, VitsModel

        repo = mms_tts_repo(lang)
        LOG.info(f"Loading MMS-TTS fallback model {repo} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = VitsModel.from_pretrained(repo).to(device)
        self.model.eval()
        self.device = device
        self.sr = self.model.config.sampling_rate
        LOG.info("MMS-TTS fallback model loaded.")

    def synthesize(
        self, text: str, gender_hint: Optional[str], emotion: str = "neutral",
    ) -> List[np.ndarray]:
        import librosa

        chunks = split_into_tts_chunks(text, max_chars=300)
        if not chunks:
            return []

        shift = GENDER_PITCH_SHIFT.get((gender_hint or "").lower(), 0.0) + EMOTION_PITCH_ADJUST.get(emotion, 0.0)
        chunk_arrays = []

        with torch.inference_mode():
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                waveform = self.model(**inputs).waveform
                wav = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
                if abs(shift) > 1e-6:
                    wav = librosa.effects.pitch_shift(wav, sr=self.sr, n_steps=shift)
                chunk_arrays.append(wav)

        return chunk_arrays


def run(cfg: PipelineConfig) -> list:
    if not cfg.translated_json.exists():
        raise FileNotFoundError(cfg.translated_json)

    cfg.dir_generated_audio.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)

    if device == "cpu":
        # This is the heaviest CPU step in the pipeline and nothing else
        # runs concurrently with it, so - unlike lighter steps sharing
        # `cfg.cpu_threads` - it's allowed to claim every core available.
        threads = configure_cpu_threads(cfg.tts_cpu_threads if cfg.tts_cpu_threads > 0 else (os.cpu_count() or 4))
        LOG.info(f"CPU synthesis: using {threads} torch threads for step14.")

    use_xtts = cfg.target_lang in XTTS_SUPPORTED_LANGUAGES

    if use_xtts:
        LOG.info(f"Target language '{cfg.target_lang}' is supported by XTTS-v2 -> true voice cloning.")
        backend = XTTSBackend(device)
    else:
        LOG.warning(
            f"Target language '{cfg.target_lang}' is not in XTTS-v2's 17 languages "
            f"-> falling back to MMS-TTS (single voice per language, gender-shifted)."
        )
        backend = MMSFallbackBackend(cfg.target_lang, device)

    conversation = load_json(cfg.translated_json)
    voice_refs = load_voice_reference_index(cfg)
    reference_dbfs = compute_reference_volume_dbfs(conversation)
    if cfg.volume_aware_normalization:
        LOG.info(
            f"Volume-aware normalization on: conversation average is "
            f"{reference_dbfs:.1f} dBFS ({'no per-segment volume data found - falling back to flat normalization' if reference_dbfs is None else f'+/-{cfg.volume_dynamic_range_db}dB dynamic range'})"
            if reference_dbfs is not None else
            "Volume-aware normalization on, but no segments have original_volume_dbfs - falling back to flat normalization."
        )

    generated = []
    started = time.time()
    synced_count = clamped_count = song_passthrough_count = song_spliced_count = 0

    for idx, seg in enumerate(conversation, start=1):
        segment_id = seg.get("segment_id")
        speaker_id = seg.get("speaker_id")
        is_song = bool(seg.get("is_song", False))
        contains_dialogue = bool(seg.get("contains_dialogue", True))
        text = normalize_text(seg.get("translation") or seg.get("text"))
        emotion = normalize_emotion(seg.get("emotion"))
        target_peak = compute_target_peak(seg.get("original_volume_dbfs"), reference_dbfs, cfg)

        if not segment_id or not speaker_id:
            LOG.info(f"[{idx}/{len(conversation)}] skipping (missing id)")
            continue

        out_path = cfg.dir_generated_audio / speaker_id / f"{segment_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Song handling: keep the ORIGINAL recording for any part of this
        # segment that's sung. See this module's docstring, item 6.
        # ------------------------------------------------------------
        if cfg.keep_song_audio_original and is_song:
            try:
                original_audio = load_original_segment_audio(seg["audio_file"], backend.sr)
                windows = seg.get("dialogue_windows") or []

                if contains_dialogue and text and windows:
                    merged = original_audio
                    pieces = split_text_proportional(text, windows)
                    spliced_any = False

                    for window, sub_text in zip(windows, pieces):
                        sub_text = normalize_text(sub_text)
                        if not sub_text:
                            continue
                        window_duration = max(0.05, window["end"] - window["start"])

                        if use_xtts:
                            chunk_arrays = backend.synthesize(
                                text=sub_text,
                                speaker_wav=Path(voice_refs[speaker_id]["reference_audio"]),
                                language=cfg.target_lang,
                                speed=cfg.tts_speed * EMOTION_SPEED_ADJUST.get(emotion, 1.0),
                                temperature=cfg.tts_temperature,
                            )
                        else:
                            chunk_arrays = backend.synthesize(
                                text=sub_text, gender_hint=voice_refs.get(speaker_id, {}).get("gender"),
                                emotion=emotion,
                            )
                        if not chunk_arrays:
                            continue

                        piece = assemble_chunks(chunk_arrays, backend.sr, cfg.tts_sentence_pause_ms)
                        piece = post_process(piece, backend.sr, denoise=cfg.tts_post_denoise, target_peak=target_peak)
                        piece, _ = sync_to_original_duration(piece, backend.sr, window_duration, cfg)

                        start_sample = int(window["start"] * backend.sr)
                        merged = crossfade_splice(merged, backend.sr, piece, start_sample, cfg.song_splice_crossfade_ms)
                        spliced_any = True

                    if not spliced_any:
                        merged = original_audio  # every window's text was empty after splitting - fall back cleanly
                    backend_name = "song_with_spliced_dialogue" if spliced_any else "original_passthrough"
                    song_spliced_count += 1 if spliced_any else 0
                else:
                    # Pure song, or a mixed clip Gemma couldn't localize
                    # dialogue timing for - the honest, safe fallback is
                    # the original recording untouched rather than risking
                    # a badly-placed dub over part of a song.
                    merged = original_audio
                    backend_name = "original_passthrough"

                if backend_name == "original_passthrough":
                    song_passthrough_count += 1

                final_duration = len(merged) / backend.sr
                sf.write(str(out_path), merged, backend.sr, subtype="PCM_16")

                LOG.info(
                    f"[{idx}/{len(conversation)}] {speaker_id} | {segment_id} | SONG "
                    f"({backend_name}, {final_duration:.2f}s kept as original)"
                )

                generated.append({
                    "segment_id": segment_id,
                    "speaker_id": speaker_id,
                    "source_text": seg.get("text"),
                    "translated_text": text,
                    "emotion": emotion,
                    "is_song": True,
                    "contains_dialogue": contains_dialogue,
                    "sentence_chunks": 0,
                    "source_language": seg.get("language"),
                    "target_language": cfg.target_lang,
                    "backend": backend_name,
                    "gender": voice_refs.get(speaker_id, {}).get("gender"),
                    "final_audio": str(out_path),
                    "original_start": seg.get("start"),
                    "original_end": seg.get("end"),
                    "original_duration": seg.get("duration"),
                    "natural_duration_seconds": final_duration,
                    "generated_duration_seconds": round(final_duration, 3),
                    "synced_to_original_duration": True,  # trivially true: it IS the original
                    "duration_stretch_rate": None,
                    "original_volume_dbfs": seg.get("original_volume_dbfs"),
                    "target_peak": round(target_peak, 3),
                    "status": "ok",
                })
                save_json(cfg.generated_audio_json, generated)

                if idx % MEMORY_SWEEP_EVERY == 0:
                    free_memory()
                continue

            except Exception as e:
                LOG.error(f"  {segment_id}: song passthrough FAILED, falling back to normal dub: {e}")
                # fall through to the normal dialogue path below rather than losing the segment

        # ------------------------------------------------------------
        # Normal dialogue path (unchanged from before, plus target_peak)
        # ------------------------------------------------------------
        if not text:
            LOG.info(f"[{idx}/{len(conversation)}] skipping (missing text)")
            continue

        speaker_meta = voice_refs.get(speaker_id)
        if not speaker_meta:
            LOG.warning(f"[{idx}/{len(conversation)}] {segment_id}: no voice reference for {speaker_id}")
            continue

        if cfg.tts_emotion_aware_delivery:
            effective_speed = cfg.tts_speed * EMOTION_SPEED_ADJUST.get(emotion, 1.0)
            effective_temperature = max(0.1, min(1.0, cfg.tts_temperature + EMOTION_TEMPERATURE_ADJUST.get(emotion, 0.0)))
        else:
            effective_speed = cfg.tts_speed
            effective_temperature = cfg.tts_temperature

        LOG.info(f"[{idx}/{len(conversation)}] {speaker_id} | {segment_id} | emotion={emotion}")

        try:
            if use_xtts:
                chunk_arrays = backend.synthesize(
                    text=text,
                    speaker_wav=Path(speaker_meta["reference_audio"]),
                    language=cfg.target_lang,
                    speed=effective_speed,
                    temperature=effective_temperature,
                )
            else:
                chunk_arrays = backend.synthesize(
                    text=text, gender_hint=speaker_meta.get("gender"), emotion=emotion,
                )

            if not chunk_arrays:
                raise RuntimeError("no audio produced (empty text after normalization)")

            merged = assemble_chunks(chunk_arrays, backend.sr, cfg.tts_sentence_pause_ms)
            merged = post_process(merged, backend.sr, denoise=cfg.tts_post_denoise, target_peak=target_peak)
            natural_duration = round(len(merged) / backend.sr, 3)

            original_start = seg.get("start")
            original_end = seg.get("end")
            target_duration = seg.get("duration")
            if not target_duration and original_start is not None and original_end is not None:
                target_duration = max(0.0, float(original_end) - float(original_start))

            merged, stretch_rate = sync_to_original_duration(merged, backend.sr, target_duration, cfg)
            final_duration = len(merged) / backend.sr
            if stretch_rate is not None:
                synced_count += 1
                if not (cfg.tts_stretch_min_rate <= stretch_rate <= cfg.tts_stretch_max_rate):
                    clamped_count += 1
                    LOG.warning(
                        f"  {segment_id}: required stretch rate {stretch_rate:.2f}x exceeds the "
                        f"configured [{cfg.tts_stretch_min_rate}, {cfg.tts_stretch_max_rate}] safety "
                        f"range and was clamped."
                    )
                if target_duration and final_duration > target_duration + 0.02:
                    LOG.warning(
                        f"  {segment_id}: even after clamped stretching, this line is "
                        f"{final_duration - target_duration:.2f}s longer than its original slot "
                        f"({final_duration:.2f}s vs {target_duration:.2f}s) - step16 will push later "
                        f"lines forward rather than overlap or cut this one."
                    )

            sf.write(str(out_path), merged, backend.sr, subtype="PCM_16")

            record = {
                "segment_id": segment_id,
                "speaker_id": speaker_id,
                "source_text": seg.get("text"),
                "translated_text": text,
                "emotion": emotion,
                "is_song": False,
                "contains_dialogue": True,
                "effective_speed": round(effective_speed, 3),
                "effective_temperature": round(effective_temperature, 3),
                "sentence_chunks": len(chunk_arrays),
                "source_language": seg.get("language"),
                "target_language": cfg.target_lang,
                "backend": "xtts_v2" if use_xtts else "mms_tts_fallback",
                "gender": speaker_meta.get("gender"),
                "final_audio": str(out_path),
                "original_start": original_start,
                "original_end": original_end,
                "original_duration": target_duration,
                "natural_duration_seconds": natural_duration,
                "generated_duration_seconds": round(final_duration, 3),
                "synced_to_original_duration": stretch_rate is not None,
                "duration_stretch_rate": round(stretch_rate, 4) if stretch_rate is not None else None,
                "original_volume_dbfs": seg.get("original_volume_dbfs"),
                "target_peak": round(target_peak, 3),
                "status": "ok",
            }

        except Exception as e:
            LOG.error(f"  FAILED: {e}")
            record = {
                "segment_id": segment_id,
                "speaker_id": speaker_id,
                "translated_text": text,
                "emotion": emotion,
                "is_song": is_song,
                "status": "failed",
                "error": str(e),
            }

        generated.append(record)
        save_json(cfg.generated_audio_json, generated)

        if idx % MEMORY_SWEEP_EVERY == 0:
            free_memory()

    if cfg.sync_segment_duration:
        LOG.info(
            f"Duration sync: {synced_count}/{len(generated)} segment(s) time-stretched to their "
            f"original slot ({clamped_count} hit the stretch-rate safety clamp; audio content is "
            f"never cropped/truncated, only sped up or slowed down)."
        )
    if song_passthrough_count or song_spliced_count:
        LOG.info(
            f"Song handling: {song_passthrough_count} segment(s) kept as pure original audio, "
            f"{song_spliced_count} segment(s) had translated dialogue spliced into an otherwise-"
            f"original recording."
        )

    LOG.info(f"Voice generation done in {time.time() - started:.2f}s for {len(generated)} segments")

    del backend
    free_memory()

    return generated


if __name__ == "__main__":
    run(PipelineConfig())
