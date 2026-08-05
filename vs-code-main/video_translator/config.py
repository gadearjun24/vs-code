"""
Central configuration for the video dubbing / translation pipeline.

Every step (src/step01_*.py ... src/step16_*.py) receives a single
`PipelineConfig` instance instead of hard-coding its own paths. This is what
lets the whole project live in one virtualenv / one requirements.txt and be
driven by a single `run_pipeline.py` entry point, while every step can also
still be run standalone for debugging.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

# ISO 639-1 -> human-readable English language name. Used to build the
# Gemma 4 (step11) transcription/translation prompts, e.g.
# "translate into Hindi". Extend freely.
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
    "te": "Telugu", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi",
    "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu", "ne": "Nepali",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian",
    "uk": "Ukrainian", "cs": "Czech", "sv": "Swedish", "tr": "Turkish",
    "ar": "Arabic", "he": "Hebrew", "fa": "Persian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "ms": "Malay", "sw": "Swahili", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "fi": "Finnish", "da": "Danish",
    "no": "Norwegian", "sk": "Slovak", "bg": "Bulgarian", "hr": "Croatian",
    "sr": "Serbian", "af": "Afrikaans",
}


def language_name(iso_code: Optional[str]) -> str:
    """Human-readable name for an ISO 639-1 code, for use in Gemma prompts."""
    if not iso_code:
        return "the source language"
    return LANGUAGE_NAMES.get(iso_code, iso_code)

# ISO 639-1 -> Coqui XTTS-v2 language code. XTTS only speaks these 17
# languages natively with true zero-shot voice cloning.
XTTS_SUPPORTED_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
    "ar", "zh", "ja", "hu", "ko", "hi",
}

# XTTS-v2's own per-language character limit per synthesis call (from
# TTS/tts/layers/xtts/tokenizer.py's `char_limits` dict). Exceeding this is
# what causes a line to just stop partway through ("truncated audio") -
# Hindi's limit (150) is much lower than English's (250), and CJK languages
# are lower still, so a single flat max_chars for every language is exactly
# what causes that bug. step14 chunks text per-sentence using these limits.
XTTS_CHAR_LIMITS = {
    "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203,
    "pl": 224, "zh": 82, "ar": 166, "cs": 186, "ru": 182, "nl": 251,
    "tr": 226, "ja": 71, "hu": 224, "ko": 95, "hi": 150,
}
DEFAULT_XTTS_CHAR_LIMIT = 150  # conservative fallback for any language missing above


def xtts_char_limit(iso_code: str, safety_margin: float = 0.9) -> int:
    """Real per-language limit, with a small safety margin so we chunk
    comfortably under it rather than right up against it."""
    limit = XTTS_CHAR_LIMITS.get(iso_code, DEFAULT_XTTS_CHAR_LIMIT)
    return max(20, int(limit * safety_margin))

# ISO 639-1 -> Meta MMS-TTS model repo suffix, used as the CPU-friendly
# fallback base voice for any language XTTS does not cover (~1100 languages).
# See https://huggingface.co/facebook/mms-tts for the full list of codes.
MMS_TTS_LANGUAGE_OVERRIDES = {
    "zh": "cmn",  # MMS uses "cmn" (Mandarin) rather than "zh"
}


def mms_tts_repo(iso_code: str) -> str:
    code = MMS_TTS_LANGUAGE_OVERRIDES.get(iso_code, iso_code)
    return f"facebook/mms-tts-{code}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # I/O
    input_video: Path = Path("input/video.mp4")
    work_dir: Path = Path("output")

    # Languages (ISO 639-1). source_lang="auto" lets Whisper detect it.
    source_lang: str = "auto"
    target_lang: str = "hi"

    # Compute
    device: str = "auto"              # auto | cpu | cuda
    cpu_threads: int = 0              # 0 = auto (min(cpu_count, 4))

    # Diarization (pyannote) - needs a HuggingFace token with access to
    # pyannote/speaker-diarization-3.1 + pyannote/segmentation-3.0
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)

    # ------------------------------------------------------------------
    # Transcription + translation (step11) - Gemma 4 E2B, audio-native.
    #
    # Gemma 4 (E2B/E4B) accepts audio directly and is used for BOTH
    # transcription (ASR) and translation (AST) in a single model call,
    # replacing the old faster-whisper + NLLB-200 two-step pipeline.
    #
    # Gemma 4 requires transformers>=5.10.1, which conflicts with
    # coqui-tts's transformers<5.0 pin (used by step13 voice generation)
    # in THIS project's main venv. So this step runs as a subprocess in
    # its own separate virtual environment (see requirements-gemma.txt
    # and scripts/setup_gemma_venv.sh) instead of importing transformers
    # directly here - the two transformers versions never have to
    # coexist in the same interpreter.
    # ------------------------------------------------------------------
    gemma_model_id: str = "google/gemma-4-E2B-it"
    # Path to the separate venv (relative to the project root, or absolute).
    gemma_venv_dir: Path = Path(".venv-gemma")
    # Gemma 4 hard-limits a single audio clip to 30s; we chunk anything
    # longer than this into sequential sub-clips inside the worker.
    gemma_max_audio_seconds: float = 28.0
    gemma_max_new_tokens: int = 256
    # "auto" resolves to cuda if available inside the worker venv, same as
    # the main `device` setting.
    gemma_device: str = "auto"
    # Kill the worker subprocess if a single run takes longer than this.
    gemma_worker_timeout_seconds: int = 10000 

    # Audio
    sample_rate: int = 16000
    hq_sample_rate: int = 44100        # kept for background music / final mix quality
    min_reference_seconds: float = 6.0
    max_reference_seconds: float = 20.0
    target_reference_seconds: float = 15.0
    reference_clip_count: int = 3       # concatenate this many distinct clean windows per speaker

    # Background music / SFX preservation
    demucs_model: str = "htdemucs"      # htdemucs_ft = higher quality, ~4x slower
    background_duck_db: float = -9.0    # how much to duck BGM under active speech
    background_gain_db: float = 0.0     # overall BGM gain in the final mix

    # Audio assembly / timing
    # As of this version, each segment is already time-stretched to match
    # its own original diarized duration right when it's generated (step13,
    # see `sync_segment_duration` below) - so by the time step15 places
    # segments on the timeline they should already fit their slot almost
    # exactly. `stretch_audio_to_fit` below is now a step15-level SAFETY
    # NET only (covers e.g. a segment generated with `sync_segment_duration
    # =False`, or missing timing data) - leave it off; the real sync now
    # happens once, at the source, in step13.
    stretch_audio_to_fit: bool = False
    min_gap_between_lines_seconds: float = 0.12

    # Voice generation
    tts_temperature: float = 0.65
    tts_speed: float = 1.15
    tts_sentence_pause_ms: int = 220    # natural gap inserted between synthesized sentences

    # Per-segment duration sync (step13): right after a segment's audio is
    # synthesized - BEFORE it's written to disk - its duration is compared
    # against that segment's own original (diarized) start/end, and sped up
    # or slowed down (pitch-preserving time-stretch) just enough to match
    # it exactly. This replaces the old approach of only fixing sync at the
    # very end (step15) across the whole track at once: fixing it per
    # segment, at the source, means every stored .wav is already
    # frame-accurate to its own slot, and step15 just has to place already-
    # correct clips rather than reconcile drift after the fact. Gemma 4
    # (step11) is also told each segment's target duration when it
    # translates, so in practice the required stretch is usually small.
    sync_segment_duration: bool = True
    tts_stretch_min_rate: float = 0.55   # don't speed up more than ~1.8x
    tts_stretch_max_rate: float = 1.05    # don't slow down more than ~1.05x
    tts_post_denoise: bool = True        # light noisereduce pass on synthesized audio (quality vs. speed)
    # CPU threads used specifically during step13 synthesis (the heaviest
    # CPU step in the pipeline). 0 = use every core available - unlike the
    # conservative shared `cpu_threads` cap above (meant for lighter steps
    # running back-to-back), this step benefits from using the whole
    # machine since nothing else runs at the same time.
    tts_cpu_threads: int = 0

    # Output muxing
    # Keep original audio as a second track and burn or embed subtitles.
    keep_original_audio_track: bool = True
    embed_soft_subtitles: bool = True
    burn_in_subtitles: bool = False
    subtitle_mode: str = "translated"  # translated | original | bilingual | all

    # Misc
    keep_temp_files: bool = True
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.input_video = Path(self.input_video)
        self.work_dir = Path(self.work_dir)
        self.gemma_venv_dir = Path(self.gemma_venv_dir)

    # ------------------------------------------------------------------
    # Resolved path to the python executable inside the Gemma worker venv
    # (cross-platform: .venv-gemma/bin/python on Linux/macOS,
    # .venv-gemma/Scripts/python.exe on Windows).
    # ------------------------------------------------------------------
    @property
    def gemma_python(self) -> Path:
        import sys as _sys

        venv_dir = self.gemma_venv_dir
        if not venv_dir.is_absolute():
            venv_dir = Path(__file__).resolve().parent / venv_dir
        if _sys.platform.startswith("win"):
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    # ------------------------------------------------------------------
    # Resolved output paths (single source of truth for every step)
    # ------------------------------------------------------------------
    @property
    def dir_audio(self) -> Path: return self.work_dir / "audio"
    @property
    def dir_metadata(self) -> Path: return self.work_dir / "metadata"
    @property
    def dir_enhanced(self) -> Path: return self.work_dir / "enhanced"
    @property
    def dir_background(self) -> Path: return self.work_dir / "background"
    @property
    def dir_vad(self) -> Path: return self.work_dir / "vad"
    @property
    def dir_diarization(self) -> Path: return self.work_dir / "diarization"
    @property
    def dir_speakers(self) -> Path: return self.work_dir / "speakers"
    @property
    def dir_conversation(self) -> Path: return self.work_dir / "conversation"
    @property
    def dir_speaker_profiles(self) -> Path: return self.work_dir / "speaker_profiles"
    @property
    def dir_normalized_speakers(self) -> Path: return self.work_dir / "normalized_speakers"
    @property
    def dir_speaker_analysis(self) -> Path: return self.work_dir / "speaker_analysis"
    @property
    def dir_translations(self) -> Path: return self.work_dir / "translations"
    @property
    def dir_voice_reference(self) -> Path: return self.work_dir / "voice_reference"
    @property
    def dir_generated_audio(self) -> Path: return self.work_dir / "generated_audio"
    @property
    def dir_subtitles(self) -> Path: return self.work_dir / "subtitles"
    @property
    def dir_final_audio(self) -> Path: return self.work_dir / "final_audio"
    @property
    def dir_final(self) -> Path: return self.work_dir / "final"

    # Key files
    @property
    def audio_raw(self) -> Path: return self.dir_audio / "audio.wav"
    @property
    def audio_hq(self) -> Path: return self.dir_audio / "audio_hq.wav"
    @property
    def audio_metadata_json(self) -> Path: return self.dir_metadata / "audio_metadata.json"
    @property
    def audio_enhanced(self) -> Path: return self.dir_enhanced / "enhanced_audio.wav"
    @property
    def audio_clean_vocals(self) -> Path: return self.dir_background / "clean_vocals.wav"
    @property
    def background_music(self) -> Path: return self.dir_background / "background.wav"
    @property
    def vad_json(self) -> Path: return self.dir_vad / "speech_segments.json"
    @property
    def diarization_json(self) -> Path: return self.dir_diarization / "diarization.json"
    @property
    def conversation_json(self) -> Path: return self.dir_conversation / "conversation.json"
    @property
    def speaker_profiles_json(self) -> Path: return self.dir_speaker_profiles / "speaker_profiles.json"
    @property
    def normalized_speakers_json(self) -> Path: return self.dir_normalized_speakers / "speaker_profiles.json"
    @property
    def speaker_analysis_json(self) -> Path: return self.dir_speaker_analysis / "speaker_analysis.json"
    @property
    def translated_json(self) -> Path: return self.dir_translations / "conversation_translated.json"
    @property
    def voice_reference_json(self) -> Path: return self.dir_voice_reference / "voice_reference.json"
    @property
    def generated_audio_json(self) -> Path: return self.dir_generated_audio / "generated_audio.json"
    @property
    def srt_original(self) -> Path: return self.dir_subtitles / "original.srt"
    @property
    def srt_translated(self) -> Path: return self.dir_subtitles / "translated.srt"
    @property
    def srt_bilingual(self) -> Path: return self.dir_subtitles / "bilingual.srt"
    @property
    def final_dubbed_audio(self) -> Path: return self.dir_final_audio / "dubbed_audio.wav"
    @property
    def final_video(self) -> Path: return self.dir_final / f"{self.input_video.stem}_{self.target_lang}_dubbed.mkv"

    def all_dirs(self):
        return [
            self.dir_audio, self.dir_metadata, self.dir_enhanced, self.dir_background,
            self.dir_vad,
            self.dir_diarization, self.dir_speakers, self.dir_conversation,
            self.dir_speaker_profiles, self.dir_normalized_speakers,
            self.dir_speaker_analysis, self.dir_translations,
            self.dir_voice_reference, self.dir_generated_audio,
            self.dir_subtitles, self.dir_final_audio, self.dir_final,
        ]

    def make_dirs(self):
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
