"""
Gemma 4 (E2B) transcription + translation + emotion + song-detection
engine, via Ollama.

This replaced an earlier version of this step that loaded Gemma 4 directly
through `transformers` in a separate venv (transformers>=5.10.1 conflicted
with coqui-tts's transformers<5.0 pin used by step14). Serving the model
through Ollama instead removes that conflict entirely: Ollama is a
standalone local server process (its own install, not a Python
dependency), so the client side here only needs the tiny `ollama` PyPI
package - a thin HTTP wrapper - which lives happily in the same venv as
everything else. No second venv, no subprocess, no job/result JSON files
crossing a process boundary - `step12_transcription_translation.py`
imports this module directly and calls it in a normal loop.

One-time setup (see README section 1b):
    1. Install Ollama:      https://ollama.com/download
    2. Pull the model:      ollama pull gemma4:e2b
    3. Make sure the server is running: `ollama serve` (the installer
       usually sets this up as a background service already - `ollama
       list` succeeding is enough evidence it's up).

Per segment, this asks Gemma 4 to listen to that segment's own audio clip
and return ONE JSON object, in a single non-streaming call using Ollama's
JSON mode (`format="json"`):

  - `is_song` / `contains_dialogue` - is this clip (or part of it) sung,
    as opposed to normally spoken? A clip can be pure song, pure dialogue,
    or both at once (e.g. dialogue over background music, or a scene that
    cuts from talking to singing). Song lyrics are never transcribed or
    translated - only genuine spoken dialogue is. See step14's docstring
    for what happens to the audio in each case (song audio is always kept
    as the original recording, never re-synthesized).
  - `dialogue_segments` - when a clip mixes song and dialogue, Gemma's
    best-effort estimate of WHERE in the clip (as a 0.0-1.0 fraction of
    its duration) the spoken parts fall, so step14 can splice translated
    dubbed dialogue into just those windows while leaving the sung/musical
    parts of the same clip untouched. This is inherently approximate (an
    LLM is not a forced-aligner) - step14 crossfades at the splice points
    to keep any misalignment from being an audible click.
  - `original_text` / `translation` / `detected_language` - transcription
    and translation of the spoken dialogue only (empty if the clip is pure
    song with no dialogue at all).
  - `emotion` - the speaker's emotional delivery in the dialogue portion.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Optional

LOG_SEPARATOR = "-" * 88

GENDER_PHRASES = {
    "male": "male",
    "female": "female",
    "child": "a child (young speaker)",
}

EMOTION_LABELS = [
    "neutral", "happy", "sad", "angry", "excited",
    "calm", "fearful", "surprised",
]

JSON_SHAPE_HINT = (
    '{"is_song": <true or false>, '
    '"contains_dialogue": <true or false>, '
    '"dialogue_segments": [{"start_fraction": <0.0-1.0>, "end_fraction": <0.0-1.0>}, ...], '
    '"original_text": "<verbatim transcription of the SPOKEN dialogue only, never sung lyrics; '
    'empty string if none>", '
    '"detected_language": "<ISO 639-1 code of the spoken language, e.g. en, hi, es>", '
    '"translation": "<translation of original_text; empty string if original_text is empty>", '
    f'"emotion": "<one of: {", ".join(EMOTION_LABELS)}>"}}'
)


# ---------------------------------------------------------------------------
# Audio loading -> 16kHz mono WAV bytes (what actually gets base64-encoded
# and sent to Ollama - a real WAV file in memory, not a raw sample array)
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
        target_len = max(1, int(round(len(data) * 16000.0 / sr)))
        data = resample(data, target_len).astype(np.float32)
        sr = 16000

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    return np.clip(data, -1.0, 1.0).astype(np.float32), sr


def chunk_audio(data, sr: int, max_seconds: float):
    """
    Split into <= max_seconds windows (Gemma 4 hard-caps a clip at 30s).
    In practice this almost never triggers anymore: step07 caps merged
    blocks at `cfg.merge_max_duration_seconds` (18s by default), well
    under this limit - it's kept as a defensive fallback, not the normal
    path.
    """
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


def array_to_wav_bytes(array, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_prompt_text(job: dict, duration: float, retry: bool = False) -> str:
    source_lang_name = job.get("source_lang_name")
    target_lang_name = job["target_lang_name"]
    gender = (job.get("gender") or "").lower()
    gender_phrase = GENDER_PHRASES.get(gender, "unspecified - use natural, neutral phrasing")

    if source_lang_name:
        transcribe_instruction = (
            f"Transcribe the SPOKEN dialogue in this clip verbatim in {source_lang_name}, "
            f"exactly as spoken."
        )
    else:
        transcribe_instruction = (
            "Transcribe the SPOKEN dialogue in this clip verbatim, exactly as spoken in its "
            "original language, and identify that language."
        )

    text = (
        "Listen to the ENTIRE audio clip carefully, start to finish, before answering - pay close "
        "attention to pronunciation, pacing, tone of voice, and emotional delivery, not just the "
        "words. Analyze it as accurately as you can.\n\n"
        "STEP 1 - Song vs. dialogue: Decide whether any part of this clip is SUNG (singing, a "
        "song, music with vocals) as opposed to normally spoken. Set \"is_song\" to true if any "
        "part is sung. Separately, decide whether any part is normal SPOKEN dialogue (not sung) - "
        "set \"contains_dialogue\" to true if so. A clip can be pure song (is_song=true, "
        "contains_dialogue=false), pure dialogue (is_song=false, contains_dialogue=true), both at "
        "once (dialogue spoken over background music, or a cut from talking to singing "
        "mid-clip), or occasionally neither (silence/noise only - both false).\n\n"
        "STEP 2 - Transcription/translation: NEVER transcribe or translate sung lyrics - "
        "\"original_text\" and \"translation\" must cover ONLY genuinely spoken dialogue. If "
        "contains_dialogue is false, leave both as empty strings. Otherwise:\n"
        f"{transcribe_instruction}\n"
        f"Then translate that transcription into {target_lang_name}.\n\n"
        "STEP 3 - If is_song AND contains_dialogue are BOTH true (mixed clip), also estimate "
        "\"dialogue_segments\": for each separate stretch of spoken dialogue, your best-effort "
        "estimate of where it falls in the clip as a fraction of the clip's total duration (0.0 = "
        "very start of the clip, 1.0 = very end) - e.g. [{\"start_fraction\": 0.1, "
        "\"end_fraction\": 0.35}] means the dialogue happens roughly from 10% to 35% of the way "
        "through the clip. This only needs to be reasonably close, not frame-accurate. If "
        "contains_dialogue is false, or is_song is false, leave dialogue_segments as an empty "
        "list.\n\n"
        "STEP 4 - Emotion: identify the speaker's emotional tone in the DIALOGUE portion (or the "
        "overall vibe if the clip is pure song) - choose exactly one label from: "
        f"{', '.join(EMOTION_LABELS)}.\n\n"
        "Context to make the translation sound natural when a voice actor speaks it aloud for "
        "dubbing (applies only if contains_dialogue is true):\n"
        f"- Speaker gender: {gender_phrase}. Where {target_lang_name} grammar marks gender "
        f"(pronouns, verb agreement, adjective agreement), phrase the translation the way that "
        f"speaker would naturally say it.\n"
        f"- Spoken duration budget: the dialogue takes about {duration:.2f} seconds total in the "
        f"original audio. Word the {target_lang_name} translation so it can be spoken naturally in "
        "roughly that amount of time - prefer a concise, natural phrasing over a long literal one "
        "if a literal translation would clearly take much longer to speak. Never drop meaning or "
        "pad the sentence purely to hit the timing; timing is a soft preference, meaning is not.\n\n"
        "Respond with ONLY one compact JSON object and nothing else - no markdown code fences, no "
        f"explanation, no extra text before or after it. Use exactly this shape:\n{JSON_SHAPE_HINT}"
    )

    if retry:
        text = (
            "Your previous response was not valid JSON. Respond with ONLY the JSON object below, "
            "no other text.\n\n" + text
        )

    return text


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|?\s*(end_of_turn|turn|eos)\s*\|?>", re.IGNORECASE)


def parse_json_response(raw_text: str) -> Optional[dict]:
    text = _SPECIAL_TOKEN_RE.sub("", raw_text or "").strip()
    text = _CODE_FENCE_RE.sub("", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "translation" in obj:
            return obj
    except json.JSONDecodeError:
        pass

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


def normalize_emotion(value: Optional[str]) -> str:
    if not value:
        return "neutral"
    v = str(value).strip().lower()
    return v if v in EMOTION_LABELS else "neutral"


def normalize_dialogue_segments(value) -> list:
    """Best-effort cleanup of the dialogue_segments field - never raises."""
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start_f = float(item.get("start_fraction", 0.0))
            end_f = float(item.get("end_fraction", 0.0))
        except (TypeError, ValueError):
            continue
        start_f = max(0.0, min(1.0, start_f))
        end_f = max(0.0, min(1.0, end_f))
        if end_f > start_f:
            out.append({"start_fraction": round(start_f, 4), "end_fraction": round(end_f, 4)})
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OllamaGemmaEngine:
    def __init__(
        self, model_id: str, host: str, request_timeout: int,
        num_predict: int, log_path: Optional[Path] = None,
    ):
        import ollama

        self._ollama = ollama
        self.model_id = model_id
        self.host = host
        self.num_predict = num_predict
        self.client = ollama.Client(host=host, timeout=request_timeout)
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- pre-flight -----------------------------------------------------

    def check_available(self) -> None:
        """Fail fast with an actionable message if Ollama/the model isn't ready."""
        try:
            models = self.client.list()
        except Exception as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.host}: {e}\n"
                f"Start it with `ollama serve` (or check it's running as a background "
                f"service), then re-run. See README.md section 1b."
            ) from e

        tags = [m.get("model") or m.get("name") for m in models.get("models", models if isinstance(models, list) else [])]
        # Ollama tags a base name like "gemma4:e2b" as-is in `ollama list` -
        # compare loosely (some client versions omit the ":latest" suffix
        # differently) rather than requiring an exact string match.
        if not any(self.model_id == t or self.model_id.split(":")[0] == (t or "").split(":")[0] for t in tags):
            raise LookupError(
                f"Model '{self.model_id}' is not pulled in Ollama (found: {tags}).\n"
                f"Run: ollama pull {self.model_id}"
            )

    # -- logging ----------------------------------------------------------

    def _log(self, segment_id: str, prompt_len: int, raw_response: str, parsed: Optional[dict]) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf8") as f:
                f.write(f"{LOG_SEPARATOR}\n")
                f.write(f"segment_id={segment_id}  model={self.model_id}  prompt_chars={prompt_len}\n")
                f.write(f"raw_response:\n{raw_response}\n")
                f.write(f"parsed_json: {json.dumps(parsed, ensure_ascii=False) if parsed else 'PARSE_FAILED'}\n")
        except Exception:
            pass  # logging must never break the pipeline

    # -- core call --------------------------------------------------------

    def _call_once(self, prompt_text: str, wav_b64: str) -> str:
        response = self.client.chat(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": prompt_text,
                "images": [wav_b64],  # Ollama accepts Gemma 4's audio input base64-encoded here
            }],
            format="json",   # Ollama's JSON mode - guarantees syntactically valid JSON back
            stream=False,     # one full response, not a token stream
            options={"num_predict": self.num_predict},
        )
        return response["message"]["content"]

    def transcribe_translate_chunk(self, job: dict, wav_bytes: bytes, duration: float, segment_id: str) -> dict:
        import base64
        wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

        prompt = build_prompt_text(job, duration, retry=False)
        raw = self._call_once(prompt, wav_b64)
        parsed = parse_json_response(raw)

        if parsed is None:
            prompt_retry = build_prompt_text(job, duration, retry=True)
            raw_retry = self._call_once(prompt_retry, wav_b64)
            parsed = parse_json_response(raw_retry)
            raw = raw_retry if parsed is not None else raw

        self._log(segment_id, len(prompt), raw, parsed)

        if parsed is None:
            # Safe fallback: treat as normal dialogue (previous behavior)
            # rather than silently dropping the segment's audio as "song".
            return {
                "is_song": False,
                "contains_dialogue": True,
                "dialogue_segments": [],
                "original_text": "",
                "detected_language": job.get("source_lang_code") or "",
                "translation": raw.strip(),
                "emotion": "neutral",
                "ok": False,
                "error": "Model did not return parseable JSON; raw text used as translation.",
            }

        return {
            "is_song": bool(parsed.get("is_song", False)),
            "contains_dialogue": bool(parsed.get("contains_dialogue", True)),
            "dialogue_segments": normalize_dialogue_segments(parsed.get("dialogue_segments")),
            "original_text": str(parsed.get("original_text", "")).strip(),
            "detected_language": str(parsed.get("detected_language", "") or job.get("source_lang_code") or ""),
            "translation": str(parsed.get("translation", "")).strip(),
            "emotion": normalize_emotion(parsed.get("emotion")),
            "ok": True,
            "error": None,
        }

    def transcribe_translate_segment(self, job: dict, max_audio_seconds: float) -> dict:
        audio_path = Path(job["audio_file"])
        segment_id = job["segment_id"]

        if not audio_path.exists():
            return {
                "segment_id": segment_id,
                "is_song": False, "contains_dialogue": True, "dialogue_segments": [],
                "original_text": "", "detected_language": job.get("source_lang_code") or "",
                "translation": "", "emotion": "neutral",
                "ok": False, "error": f"audio file not found: {audio_path}",
            }

        data, sr = load_audio_16k_mono(audio_path)
        chunks = chunk_audio(data, sr, max_audio_seconds)
        total_duration = len(data) / float(sr) or 1.0

        originals, translations, detected_langs, emotions, errors = [], [], [], [], []
        is_song_votes, contains_dialogue_votes = [], []
        dialogue_segments: list = []
        elapsed_before_chunk = 0.0

        for i, (chunk_array, chunk_duration) in enumerate(chunks):
            if chunk_duration <= 0.02:
                elapsed_before_chunk += chunk_duration
                continue
            wav_bytes = array_to_wav_bytes(chunk_array, sr)
            chunk_id = segment_id if len(chunks) == 1 else f"{segment_id}_chunk{i}"
            result = self.transcribe_translate_chunk(job, wav_bytes, chunk_duration, chunk_id)

            if result["original_text"]:
                originals.append(result["original_text"])
            if result["translation"]:
                translations.append(result["translation"])
            if result["detected_language"]:
                detected_langs.append(result["detected_language"])
            emotions.append(result["emotion"])
            is_song_votes.append(result["is_song"])
            contains_dialogue_votes.append(result["contains_dialogue"])
            if not result["ok"]:
                errors.append(result["error"])

            # Remap this chunk's own 0.0-1.0 fractions onto the whole
            # segment's timeline (matters only in the rare case a segment
            # needed multiple chunks - see chunk_audio()'s docstring).
            for seg_window in result["dialogue_segments"]:
                chunk_start_frac = elapsed_before_chunk / total_duration
                chunk_span_frac = chunk_duration / total_duration
                dialogue_segments.append({
                    "start_fraction": round(chunk_start_frac + seg_window["start_fraction"] * chunk_span_frac, 4),
                    "end_fraction": round(chunk_start_frac + seg_window["end_fraction"] * chunk_span_frac, 4),
                })

            elapsed_before_chunk += chunk_duration

        detected_language = detected_langs[0] if detected_langs else (job.get("source_lang_code") or "")
        # One emotion label per segment: the most frequent one across its
        # chunks (almost always just one chunk in practice).
        emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
        is_song = any(is_song_votes)
        contains_dialogue = any(contains_dialogue_votes) if contains_dialogue_votes else True

        return {
            "segment_id": segment_id,
            "is_song": is_song,
            "contains_dialogue": contains_dialogue,
            "dialogue_segments": dialogue_segments,
            "original_text": " ".join(originals).strip(),
            "detected_language": detected_language,
            "translation": " ".join(translations).strip(),
            "emotion": emotion,
            "ok": len(errors) == 0,
            "error": "; ".join(errors) if errors else None,
        }

    def unload(self) -> None:
        """Ask Ollama to drop this model from RAM/VRAM immediately."""
        try:
            self.client.chat(model=self.model_id, messages=[], keep_alive=0)
        except Exception:
            pass  # best-effort - never fail the step over this

