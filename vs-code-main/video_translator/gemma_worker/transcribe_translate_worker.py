#!/usr/bin/env python3
"""
Gemma 4 (E2B) transcription + translation worker.

This script is intentionally standalone: it only imports the packages
installed in the SEPARATE `.venv-gemma` virtual environment
(torch / transformers>=5.10.1 / accelerate / soundfile / scipy / numpy),
never anything from the main project's `src`/`config` modules. That keeps
the two dependency stacks fully isolated - the main pipeline venv pins
`transformers<5.0` (required by coqui-tts for voice cloning), while Gemma 4
requires `transformers>=5.10.1`. The two can't live in the same
interpreter, so `src/step11_transcription_translation.py` (main venv)
invokes this file as a subprocess using `.venv-gemma`'s python instead.

Usage
-----
    <.venv-gemma>/bin/python gemma_worker/transcribe_translate_worker.py \
        --jobs  path/to/jobs.json \
        --output path/to/results.json \
        --model-id google/gemma-4-E2B-it \
        --device auto \
        --max-new-tokens 256 \
        --max-audio-seconds 28

Input `jobs.json` is a list of objects:
    {
        "segment_id": "segment_000012",
        "speaker_id": "speaker_001",
        "audio_file": "/abs/path/to/segment_000012.wav",
        "start": 12.34, "end": 15.02, "duration": 2.68,
        "gender": "female" | "male" | "child" | null,
        "source_lang_code": "en" | null,        (null = auto-detect)
        "source_lang_name": "English" | null,
        "target_lang_code": "hi",
        "target_lang_name": "Hindi"
    }

Output `results.json` is a list of objects, one per input job, in the same
order, each shaped:
    {
        "segment_id": "segment_000012",
        "original_text": "...",
        "detected_language": "en",
        "translation": "...",
        "ok": true,
        "error": null
    }

Every processed job is also echoed to stdout as a single-line JSON progress
record (`{"progress": i, "total": n, "segment_id": ...}`) so the parent
process (running in the main venv) can stream real-time progress into its
own logger without needing to share any Python objects across the venv
boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional


def log(msg: str) -> None:
    print(f"[gemma-worker] {msg}", flush=True)


def progress(index: int, total: int, segment_id: str) -> None:
    print(json.dumps({"progress": index, "total": total, "segment_id": segment_id}), flush=True)


# ---------------------------------------------------------------------------
# Audio loading - resample to Gemma 4's required 16kHz mono float32 [-1, 1]
# ---------------------------------------------------------------------------

def load_audio_16k_mono(path: Path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)

    if sr != 16000:
        # scipy.signal.resample (Fourier method) - Google's own guidance for
        # resampling audio destined for Gemma 4's audio encoder.
        target_len = max(1, int(round(len(data) * 16000.0 / sr)))
        data = resample(data, target_len).astype(np.float32)
        sr = 16000

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    return data, sr


def chunk_audio(data, sr: int, max_seconds: float):
    """Split into <= max_seconds windows (Gemma 4 hard-caps a clip at 30s)."""
    max_samples = int(max_seconds * sr)
    if max_samples <= 0 or len(data) <= max_samples:
        return [(data, len(data) / float(sr))]

    chunks = []
    for start in range(0, len(data), max_samples):
        piece = data[start:start + max_samples]
        if piece.size == 0:
            continue
        chunks.append((piece, len(piece) / float(sr)))
    return chunks


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

GENDER_PHRASES = {
    "male": "male",
    "female": "female",
    "child": "a child (young speaker)",
}

JSON_SHAPE_HINT = (
    '{"original_text": "<verbatim transcription in the original spoken language>", '
    '"detected_language": "<ISO 639-1 code of the spoken language, e.g. en, hi, es>", '
    '"translation": "<the translation>"}'
)


def build_prompt_text(job: dict, duration: float, retry: bool = False) -> str:
    source_lang_name = job.get("source_lang_name")
    target_lang_name = job["target_lang_name"]
    gender = (job.get("gender") or "").lower()
    gender_phrase = GENDER_PHRASES.get(gender, "unspecified - use natural, neutral phrasing")

    if source_lang_name:
        transcribe_instruction = (
            f"Transcribe the following speech clip verbatim in {source_lang_name}, exactly as spoken."
        )
    else:
        transcribe_instruction = (
            "Transcribe the following speech clip verbatim, exactly as spoken in its original "
            "language, and identify that language."
        )

    # --- UPDATED PROMPT SECTION ---
    text = (
        f"{transcribe_instruction}\n"
        f"Then translate the transcription into {target_lang_name}.\n\n"
        "Context to make the translation sound natural when a voice actor speaks it aloud for dubbing:\n"
        f"- Speaker gender: {gender_phrase}. Where {target_lang_name} grammar marks gender "
        f"(pronouns, verb agreement, adjective agreement), phrase the translation the way that "
        f"speaker would naturally say it.\n"
        f"- Spoken duration budget (CRITICAL): This line takes exactly {duration:.2f} seconds to say "
        f"in the original audio. You are a professional video dubbing translator. The {target_lang_name} "
        f"translation MUST be concise and take the exact same amount of time to speak. Do NOT provide "
        f"a wordy, literal translation. You must aggressively shorten the phrasing, using fewer words "
        f"and matching the syllable count of the original audio, so the TTS model does not have to rush.\n\n"
        "Respond with ONLY one compact JSON object and nothing else - no markdown code fences, no "
        f"explanation, no extra text before or after it. Use exactly this shape:\n{JSON_SHAPE_HINT}"
    )
    # ------------------------------

    if retry:
        text = (
            "Your previous response was not valid JSON. Respond with ONLY the JSON object below, "
            "no other text.\n\n" + text
        )

    return text


def build_messages(job: dict, audio_array, duration: float, retry: bool = False) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt_text(job, duration, retry=retry)},
                {"type": "audio", "audio": audio_array},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|?\s*(end_of_turn|turn|eos)\s*\|?>", re.IGNORECASE)


def parse_json_response(raw_text: str) -> Optional[dict]:
    text = _SPECIAL_TOKEN_RE.sub("", raw_text).strip()
    text = _CODE_FENCE_RE.sub("", text).strip()

    # Fast path: the whole thing is valid JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "translation" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: find the first '{' and let json.JSONDecoder consume just the
    # object starting there, ignoring any trailing junk the model added.
    start = text.find("{")
    if start != -1:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and "translation" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GemmaEngine:
    def __init__(self, model_id: str, device_pref: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, GenerationConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        if device_pref == "cpu":
            device = "cpu"
        elif device_pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Requested --device cuda but CUDA is not available inside the Gemma worker venv."
                )
            device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        device_map = "auto" if device == "cuda" else {"": "cpu"}

        log(f"Loading {model_id} on {device} ({dtype})... (first run downloads the weights, ~10GB)")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype="auto", device_map=device_map
        ).eval()

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_id)
        except Exception:
            self.generation_config = GenerationConfig()
        self.generation_config.max_new_tokens = max_new_tokens
        log("Model loaded.")

    def generate(self, messages: list) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                do_sample=False,
            )

        new_tokens = generated[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    def transcribe_translate_chunk(self, job: dict, audio_array, duration: float) -> dict:
        messages = build_messages(job, audio_array, duration, retry=False)
        raw = self.generate(messages)
        parsed = parse_json_response(raw)

        if parsed is None:
            # One retry with a stricter reminder before giving up.
            messages = build_messages(job, audio_array, duration, retry=True)
            raw_retry = self.generate(messages)
            parsed = parse_json_response(raw_retry)
            raw = raw_retry if parsed is not None else raw

        if parsed is None:
            return {
                "original_text": "",
                "detected_language": job.get("source_lang_code") or "",
                "translation": raw.strip(),
                "ok": False,
                "error": "Model did not return parseable JSON; raw text used as translation.",
            }

        return {
            "original_text": str(parsed.get("original_text", "")).strip(),
            "detected_language": str(parsed.get("detected_language", "") or job.get("source_lang_code") or ""),
            "translation": str(parsed.get("translation", "")).strip(),
            "ok": True,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_job(engine: GemmaEngine, job: dict, max_audio_seconds: float) -> dict:
    audio_path = Path(job["audio_file"])
    if not audio_path.exists():
        return {
            "segment_id": job["segment_id"],
            "original_text": "",
            "detected_language": job.get("source_lang_code") or "",
            "translation": "",
            "ok": False,
            "error": f"audio file not found: {audio_path}",
        }

    data, sr = load_audio_16k_mono(audio_path)
    chunks = chunk_audio(data, sr, max_audio_seconds)

    originals, translations, detected_langs, errors = [], [], [], []
    for chunk_array, chunk_duration in chunks:
        if chunk_duration <= 0.02:
            continue
        result = engine.transcribe_translate_chunk(job, chunk_array, chunk_duration)
        if result["original_text"]:
            originals.append(result["original_text"])
        if result["translation"]:
            translations.append(result["translation"])
        if result["detected_language"]:
            detected_langs.append(result["detected_language"])
        if not result["ok"]:
            errors.append(result["error"])

    detected_language = detected_langs[0] if detected_langs else (job.get("source_lang_code") or "")

    return {
        "segment_id": job["segment_id"],
        "original_text": " ".join(originals).strip(),
        "detected_language": detected_language,
        "translation": " ".join(translations).strip(),
        "ok": len(errors) == 0,
        "error": "; ".join(errors) if errors else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma 4 E2B transcription + translation worker")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-audio-seconds", type=float, default=28.0)
    args = parser.parse_args()

    with open(args.jobs, "r", encoding="utf8") as f:
        jobs = json.load(f)

    log(f"{len(jobs)} segment(s) to process.")

    if not jobs:
        with open(args.output, "w", encoding="utf8") as f:
            json.dump([], f)
        return 0

    engine = GemmaEngine(args.model_id, args.device, args.max_new_tokens)

    results = []
    start_time = time.time()
    ok_count = fail_count = 0

    for i, job in enumerate(jobs, start=1):
        try:
            result = process_job(engine, job, args.max_audio_seconds)
        except Exception as e:  # keep going - one bad segment shouldn't kill the whole run
            result = {
                "segment_id": job.get("segment_id", f"unknown_{i}"),
                "original_text": "",
                "detected_language": job.get("source_lang_code") or "",
                "translation": "",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }

        results.append(result)
        if result["ok"]:
            ok_count += 1
        else:
            fail_count += 1

        progress(i, len(jobs), result["segment_id"])

        # Persist incrementally so a crash mid-run doesn't lose completed work.
        with open(args.output, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    log(f"Done in {elapsed:.2f}s | ok={ok_count} failed={fail_count}")
    # Partial per-segment failures are recorded in results.json (ok=false,
    # error=<reason>) but are not fatal to the worker process itself - the
    # orchestrator (step11, main venv) decides what to do with them.
    return 0


if __name__ == "__main__":
    sys.exit(main())
