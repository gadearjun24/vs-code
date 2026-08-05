"""
Step 13 - Voice generation (translated text -> cloned speech per segment).

Rewritten to fix three real production issues:

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
   chunk's own leading/trailing silence first so gaps don't compound.

3. Background noise / unclear voice in the generated audio. Fix: a light
   `noisereduce` pass + peak normalization on the final concatenated
   segment before it's written out.

Two backends, chosen automatically per target language:

* XTTS-v2 (coqui-tts) - true zero-shot voice cloning from `reference.wav`
  (now a multi-clip reference from step12). Used whenever `cfg.target_lang`
  is one of XTTS's 17 supported languages. Because it clones the *actual*
  speaker sample, gender/timbre is preserved automatically: male -> male,
  female -> female, other -> other.

* MMS-TTS (facebook/mms-tts-<lang>) - single-voice fallback covering
  ~1100 languages, for targets XTTS doesn't speak. A small pitch shift is
  applied toward the source speaker's estimated register so the result
  still leans toward the right gender.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from config import PipelineConfig, XTTS_SUPPORTED_LANGUAGES, mms_tts_repo, xtts_char_limit
from src.utils import (
    denoise_array, get_logger, load_json, resolve_device, save_json,
    split_into_tts_chunks, trim_silence_array,
)

LOG = get_logger(__name__)

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

GENDER_PITCH_SHIFT = {"female": 3.0, "male": -2.0, "child": 5.0}


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


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
        trimmed = trim_silence_array(chunk, threshold=0.02, pad_samples=int(0.03 * sr))
        if i > 0:
            pieces.append(gap)
        pieces.append(trimmed)
    return np.concatenate(pieces)


def post_process(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    cleaned = denoise_array(audio, sr, prop_decrease=0.5)
    peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    if peak > 1e-6:
        cleaned = (cleaned / peak) * 0.95
    return cleaned.astype(np.float32)


class XTTSBackend:
    """Zero-shot voice cloning backend, one sentence-chunk at a time."""

    def __init__(self, device: str):
        from TTS.api import TTS

        LOG.info(f"Loading XTTS-v2 on {device} (first run downloads ~2GB of weights)...")
        self.tts = TTS(XTTS_MODEL_NAME).to(device)
        self.sr = self.tts.synthesizer.output_sample_rate
        LOG.info(f"XTTS-v2 loaded (output sample rate: {self.sr}).")

    def synthesize(self, text: str, speaker_wav: Path, language: str, speed: float = 1.0) -> np.ndarray:
        max_chars = xtts_char_limit(language)
        chunks = split_into_tts_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        chunk_arrays = []
        for chunk in chunks:
            wav = self.tts.tts(
                text=chunk,
                speaker_wav=str(speaker_wav),
                language=language,
                speed=speed,
                split_sentences=False,  # we already split sentence-by-sentence ourselves
            )
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

    def synthesize(self, text: str, gender_hint: Optional[str]) -> List[np.ndarray]:
        import librosa

        chunks = split_into_tts_chunks(text, max_chars=300)
        if not chunks:
            return []

        shift = GENDER_PITCH_SHIFT.get((gender_hint or "").lower())
        chunk_arrays = []

        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                waveform = self.model(**inputs).waveform
            wav = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
            if shift:
                wav = librosa.effects.pitch_shift(wav, sr=self.sr, n_steps=shift)
            chunk_arrays.append(wav)

        return chunk_arrays


def run(cfg: PipelineConfig) -> list:
    if not cfg.translated_json.exists():
        raise FileNotFoundError(cfg.translated_json)

    cfg.dir_generated_audio.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
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

    generated = []
    started = time.time()

    for idx, seg in enumerate(conversation, start=1):
        segment_id = seg.get("segment_id")
        speaker_id = seg.get("speaker_id")
        text = normalize_text(seg.get("translation") or seg.get("text"))

        if not segment_id or not speaker_id or not text:
            LOG.info(f"[{idx}/{len(conversation)}] skipping (missing id/text)")
            continue

        speaker_meta = voice_refs.get(speaker_id)
        if not speaker_meta:
            LOG.warning(f"[{idx}/{len(conversation)}] {segment_id}: no voice reference for {speaker_id}")
            continue

        out_path = cfg.dir_generated_audio / speaker_id / f"{segment_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        LOG.info(f"[{idx}/{len(conversation)}] {speaker_id} | {segment_id}")

        try:
            if use_xtts:
                chunk_arrays = backend.synthesize(
                    text=text,
                    speaker_wav=Path(speaker_meta["reference_audio"]),
                    language=cfg.target_lang,
                    speed=cfg.tts_speed,
                )
            else:
                chunk_arrays = backend.synthesize(text=text, gender_hint=speaker_meta.get("gender"))

            if not chunk_arrays:
                raise RuntimeError("no audio produced (empty text after normalization)")

            merged = assemble_chunks(chunk_arrays, backend.sr, cfg.tts_sentence_pause_ms)
            merged = post_process(merged, backend.sr)
            sf.write(str(out_path), merged, backend.sr, subtype="PCM_16")

            record = {
                "segment_id": segment_id,
                "speaker_id": speaker_id,
                "source_text": seg.get("text"),
                "translated_text": text,
                "sentence_chunks": len(chunk_arrays),
                "source_language": seg.get("language"),
                "target_language": cfg.target_lang,
                "backend": "xtts_v2" if use_xtts else "mms_tts_fallback",
                "gender": speaker_meta.get("gender"),
                "final_audio": str(out_path),
                "original_start": seg.get("start"),
                "original_end": seg.get("end"),
                "original_duration": seg.get("duration"),
                "generated_duration_seconds": round(len(merged) / backend.sr, 3),
                "status": "ok",
            }

        except Exception as e:
            LOG.error(f"  FAILED: {e}")
            record = {
                "segment_id": segment_id,
                "speaker_id": speaker_id,
                "translated_text": text,
                "status": "failed",
                "error": str(e),
            }

        generated.append(record)
        save_json(cfg.generated_audio_json, generated)

    LOG.info(f"Voice generation done in {time.time() - started:.2f}s for {len(generated)} segments")
    return generated


if __name__ == "__main__":
    run(PipelineConfig())
