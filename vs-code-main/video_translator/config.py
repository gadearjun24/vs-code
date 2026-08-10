"""
Central configuration for the video dubbing / translation pipeline.

Every step (src/step01_*.py ... src/step17_*.py) receives a single
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
# Gemma 4 (step12) transcription/translation prompts, e.g.
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
# what causes that bug. step15 chunks text per-sentence using these limits.
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
    # Segment merging (step07) - runs right after diarization, before
    # anything else touches segment audio. Raw diarization/VAD output is
    # frequently full of very short fragments (backchannels, a sentence
    # split across a breath pause, sub-second interjections) - handing one
    # of those to step12's Gemma 4 call in isolation gives it too little
    # audio to transcribe or read emotional tone from reliably. This step
    # merges consecutive same-speaker segments separated by a small gap
    # into one logical block BEFORE step08 ever extracts per-segment
    # audio, so every step downstream (split, aggregation, normalization,
    # embedding, transcription/translation/emotion, voice generation,
    # subtitles, assembly) operates on the merged, more-natural-length
    # unit instead of the raw fragment. See step07's docstring for the
    # full merge algorithm and why nothing needs to be "un-merged" later
    # for audio sync to stay correct.
    # ------------------------------------------------------------------
    # Merge two consecutive same-speaker segments if the silence gap
    # between them is <= this many seconds (a natural mid-sentence/
    # between-sentence pause). Segments separated by more than this are
    # left as separate blocks - that gap is treated as a real pause, not
    # something to paper over.
    merge_max_gap_seconds: float = 1.5
    # Never merge a block past this total duration (kept comfortably under
    # Gemma 4's 30s hard clip limit, with headroom for step12's own <=28s
    # chunking margin).
    merge_max_duration_seconds: float = 18.0
    # A block still shorter than this after merging (typically an isolated
    # segment with a different speaker on both sides, so it had no
    # same-speaker neighbor to merge with - e.g. a one-word backchannel)
    # gets its EXTRACTION window padded with a little surrounding audio
    # context purely so step12's Gemma call has enough signal to work
    # with. The padding never crosses into a neighboring segment's actual
    # speech - see `clip_start`/`clip_end` vs `start`/`end` in step07's
    # docstring - and the padding is stripped back out again by step12
    # when it tells Gemma the segment's real spoken-duration budget, so it
    # never affects placement/sync.
    merge_min_clip_seconds: float = 0.6

    # ------------------------------------------------------------------
    # Transcription + translation (step12) - Gemma 4 E2B, audio-native,
    # served locally through Ollama.
    #
    # Gemma 4 (E2B/E4B) accepts audio directly and is used for BOTH
    # transcription (ASR) and translation (AST) - plus a detected
    # per-segment `emotion` label used by step14 to nudge delivery - in a
    # single model call, replacing the old faster-whisper + NLLB-200
    # two-step pipeline.
    #
    # This runs through Ollama (a local model server you install once and
    # leave running - see README section 1b) rather than loading
    # `transformers` directly in this process. That means:
    #   - no separate venv/transformers version to manage - the only new
    #     dependency in requirements.txt is the tiny `ollama` client
    #     library, which just speaks HTTP to the Ollama server;
    #   - Ollama manages the model's own memory/device placement
    #     (CPU/GPU) itself, including unloading it when idle.
    # ------------------------------------------------------------------
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    # Ollama model tag - "gemma4:e2b" (fast, ~5GB RAM at 4-bit) or
    # "gemma4:e4b" (higher quality, more RAM) - pull it once with
    # `ollama pull gemma4:e2b` before running the pipeline.
    gemma_model_id: str = "gemma4:e2b"
    # Gemma 4's audio encoder hard-limits a single clip to 30s; anything
    # longer is chunked into sequential sub-clips before being sent.
    gemma_max_audio_seconds: float = 28.0
    # Ollama's equivalent of `max_new_tokens` (passed as `options.num_predict`).
    ollama_num_predict: int = 2048
    # Per-request timeout to the Ollama server (CPU inference on a long
    # clip can be slow - see README section 6 for speed guidance).
    ollama_request_timeout_seconds: int = 100000
    # Ask Ollama to unload the model from RAM/VRAM the moment step12
    # finishes (rather than keeping it warm for a few minutes, Ollama's
    # normal default) so step14's XTTS load starts from a clean baseline.
    # Pairs with the general `free_memory()` cleanup run_pipeline.py does
    # between every step.
    ollama_unload_after_step: bool = True

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
    # its own original diarized duration right when it's generated (step14,
    # see `sync_segment_duration` below) - so by the time step16 places
    # segments on the timeline they should already fit their slot almost
    # exactly. `stretch_audio_to_fit` below is now a step16-level SAFETY
    # NET only (covers e.g. a segment generated with `sync_segment_duration
    # =False`, or missing timing data) - leave it off; the real sync now
    # happens once, at the source, in step14.
    stretch_audio_to_fit: bool = False
    min_gap_between_lines_seconds: float = 0.12

    # Voice generation
    tts_temperature: float = 0.65
    # >1.0 = talks slightly faster than the reference voice's natural pace.
    # 1.15 by default: a mild, generally-natural bump that also reduces how
    # often step14 needs to lean on time-stretching to hit a segment's
    # original duration (a line that's already a bit brisk needs less
    # after-the-fact compression to fit).
    tts_speed: float = 1.15
    tts_sentence_pause_ms: int = 220    # natural gap inserted between synthesized sentences
    # Nudge `tts_speed`/`tts_temperature` per segment based on Gemma 4's
    # detected `emotion` for that speaker's delivery (step12) - XTTS-v2 has
    # no direct "emotion" input, so this is a deliberately modest proxy via
    # its two real prosody knobs (a few % pace change, a small temperature
    # shift), not a claim of true emotional TTS. See EMOTION_SPEED_ADJUST /
    # EMOTION_TEMPERATURE_ADJUST in step14. Off switch if you'd rather every
    # line use the flat `tts_speed`/`tts_temperature` above unmodified.
    tts_emotion_aware_delivery: bool = True

    # Per-segment duration sync (step14): right after a segment's audio is
    # synthesized - BEFORE it's written to disk - its duration is compared
    # against that segment's own original (diarized) start/end, and sped up
    # or slowed down (pitch-preserving time-stretch, never truncation - no
    # audio content is ever cut) just enough to match it exactly. This
    # replaces the old approach of only fixing sync at the very end
    # (step16) across the whole track at once: fixing it per segment, at
    # the source, means every stored .wav is already frame-accurate to its
    # own slot, and step16 just has to place already-correct clips rather
    # than reconcile drift after the fact. Gemma 4 (step12) is also told
    # each segment's target duration when it translates, so in practice
    # the required stretch is usually small.
    sync_segment_duration: bool = True
    # These two bounds are intentionally asymmetric:
    #   - tts_stretch_max_rate governs SPEED-UP (audio came out longer than
    #     its slot). Kept generous (3.0x) because a segment ending up
    #     LONGER than its original slot is the one outcome that risks
    #     pushing into the next line - so this step would rather speak a
    #     rare outlier line faster than let it overflow. In the normal
    #     case (Gemma 4 already phrases translations to roughly fit the
    #     target duration) the actual rate needed is nowhere near this
    #     ceiling.
    #   - tts_stretch_min_rate governs SLOW-DOWN (audio came out shorter).
    #     Kept tighter (0.7x, i.e. stretched at most ~1.4x longer) purely
    #     for naturalness - slowing down too much sounds muffled/padded -
    #     since a short line slowed down can never overflow into the next
    #     one regardless of how far it's stretched.
    # In either case this is ALWAYS a time-stretch (every sample of speech
    # preserved, just faster/slower) and NEVER truncation - no word is ever
    # cut short to force a duration. See step14's docstring for the full
    # placement guarantee this feeds into.
    tts_stretch_min_rate: float = 0.70
    tts_stretch_max_rate: float = 3.00
    tts_post_denoise: bool = True        # light noisereduce pass on synthesized audio (quality vs. speed)
    # CPU threads used specifically during step14 synthesis (the heaviest
    # CPU step in the pipeline). 0 = use every core available - unlike the
    # conservative shared `cpu_threads` cap above (meant for lighter steps
    # running back-to-back), this step benefits from using the whole
    # machine since nothing else runs at the same time.
    tts_cpu_threads: int = 0

    # ------------------------------------------------------------------
    # Song vs. dialogue handling (step12 detects, step14 acts on it).
    #
    # Gemma 4 (step12) flags each segment `is_song` / `contains_dialogue`.
    # Song lyrics are never transcribed/translated/dubbed - step14 always
    # keeps the ORIGINAL recording for any portion of a segment that's
    # sung, and only ever generates/splices in a synthesized dub for
    # genuinely spoken dialogue:
    #   - pure song (is_song, no dialogue): the whole segment's original
    #     audio is used untouched - no TTS call happens at all.
    #   - pure dialogue (no song): unchanged from before - full TTS dub.
    #   - mixed (both true): the ORIGINAL audio is the base, and translated
    #     dubbed dialogue is spliced in only at Gemma's estimated
    #     `dialogue_segments` windows (crossfaded at the splice points,
    #     since those windows are an LLM's best-effort estimate, not
    #     frame-accurate) - everywhere else in the clip keeps the original
    #     recording exactly as-is.
    # See step14's docstring for the full splicing implementation.
    # ------------------------------------------------------------------
    keep_song_audio_original: bool = True
    # Crossfade applied at each splice point when stitching translated
    # dialogue into an otherwise-original (song) clip, to smooth over the
    # fact that dialogue_segments' boundaries are only approximate.
    song_splice_crossfade_ms: int = 40

    # ------------------------------------------------------------------
    # Volume/energy preservation (step12 measures, step14 applies).
    #
    # step12 measures each segment's ORIGINAL loudness (RMS, in dBFS) from
    # its source audio and stores it as `original_volume_dbfs`. Without
    # this, step14's flat peak-normalization would flatten every line to
    # the same loudness, erasing the difference between a shouted, angry
    # line and a quiet, calm one - exactly the kind of delivery `emotion`
    # (tts_emotion_aware_delivery above) is trying to preserve in the
    # first place. Instead, step14 normalizes each generated line's peak
    # to a level that's offset from the baseline in the same direction and
    # (clamped) proportion as that segment's original volume was offset
    # from the conversation's average - so a shouted line still comes out
    # louder than a whispered one in the final dub, not identical to it.
    # ------------------------------------------------------------------
    volume_aware_normalization: bool = True
    # How far the final peak is allowed to move (in dB) from the flat
    # baseline in either direction, regardless of how extreme the original
    # segment's volume was relative to the conversation average - keeps a
    # single outlier line from being normalized so loud it clips/distorts
    # or so quiet it's inaudible under the background mix.
    volume_dynamic_range_db: float = 8.0
    # Absolute floor/ceiling on the final peak amplitude (0.0-1.0), applied
    # after the dynamic-range offset above - a last-resort safety clamp.
    volume_min_peak: float = 0.15
    volume_max_peak: float = 0.98

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
    def merged_diarization_json(self) -> Path: return self.dir_diarization / "merged_diarization.json"
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
