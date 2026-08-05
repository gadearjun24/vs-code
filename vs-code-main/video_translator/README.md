# Video Translation & Dubbing Pipeline

Translate a video from one language to another, keeping each speaker's own
voice (male stays male, female stays female, child/other stays as they are),
with background music/SFX preserved under the dub, perfectly time-aligned
audio, and both original + translated subtitles.

The pipeline lives in **two virtual environments**: the main one
(`requirements.txt`) runs every step, and a second, small one
(`requirements-gemma.txt`, `.venv-gemma`) runs only step11's Gemma 4
transcription/translation model, because Gemma 4 needs a newer
`transformers` than the rest of the pipeline can use (see step11's
docstring and section 1b below for exactly why).

```
input video
   │
   ▼
01 extract audio           -> mono 16kHz wav (speech) + stereo hq wav (music)
02 audio metadata           -> ffprobe stats
03 audio enhancement        -> noisereduce denoise
04 background separation    -> Demucs: clean_vocals.wav + background.wav
05 voice activity detect    -> Silero VAD (on clean vocals)
06 speaker diarization      -> pyannote (who spoke when, on clean vocals)
07 split audio              -> per speaker / per segment clips (from clean vocals)
08 speaker aggregation      -> merge each speaker's clips
09 audio normalization      -> loudness normalize (EBU R128)
10 speaker embedding        -> gender / age / emotion per speaker
11 transcription+translation -> Gemma 4 E2B (audio-native, gender+timing aware) [separate venv]
12 reference voice          -> 3 clean, non-overlapping clips per speaker
13 voice generation         -> XTTS-v2 clone (or MMS-TTS fallback), chunked
14 subtitles                -> original.srt / translated.srt / bilingual.srt
15 audio assembly           -> time-align dub + mix back preserved BGM/SFX
16 video mux                -> final .mkv: video + dubbed + original audio + subs + BGM
```

Steps 08-10 (aggregation/normalization/gender detection) intentionally run
*before* step11 now, rather than after transcription like the old
faster-whisper-based layout - step11 needs each speaker's gender to make
the translation grammatically correct, and gender is only known once
step10 has run.

## 1a. One-time setup - main venv (one requirements file)

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip

# CPU-only (recommended default - works on any machine):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR, if you have an NVIDIA GPU and want it used automatically:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

You also need **ffmpeg** on your PATH (`ffmpeg -version` should work):
Ubuntu/Debian: `sudo apt install ffmpeg` · macOS: `brew install ffmpeg` ·
Windows: download from ffmpeg.org and add it to PATH.

## 1b. One-time setup - second venv, for Gemma 4 (step11 only)

Step11 (transcription + translation) uses **Gemma 4 E2B**
(`google/gemma-4-E2B-it`), which listens to each segment's own audio and
returns the original-language transcription and the target-language
translation in one call - conditioned on that speaker's detected gender
(from step10) and the segment's spoken duration (from diarization), so
the wording naturally fits both the speaker's voice and the time slot a
dubbing voice actor has to say it in.

Gemma 4 requires `transformers>=5.10.1`. The main venv above is pinned to
`transformers<5.0` because step13's `coqui-tts` (XTTS-v2 voice cloning)
imports `transformers.pytorch_utils.isin_mps_friendly`, which was removed
in the 5.x line - the two requirements can't be satisfied in one venv.
Rather than downgrade Gemma 4 or fork the cloning stack, step11 shells out
to a second, independent venv that only step11 uses:

```bash
bash scripts/setup_gemma_venv.sh          # Linux/macOS, CPU build
# bash scripts/setup_gemma_venv.sh cuda   # or the CUDA 12.4 build

scripts\setup_gemma_venv.bat              # Windows, CPU build
REM scripts\setup_gemma_venv.bat cuda     # or the CUDA 12.4 build
```

This creates `.venv-gemma/` from `requirements-gemma.txt`. You don't
activate it yourself - `src/step11_transcription_translation.py` (in the
*main* venv) finds it automatically at `.venv-gemma/bin/python` (or
`.venv-gemma\Scripts\python.exe` on Windows) and runs
`gemma_worker/transcribe_translate_worker.py` there as a subprocess,
passing job data in and getting transcript+translation JSON back - no
Python objects or imports ever cross between the two venvs, only files.

`google/gemma-4-E2B-it` is a gated model, same as pyannote below:

1. Accept its license at https://huggingface.co/google/gemma-4-E2B-it
   (while logged in).
2. Export the same `HF_TOKEN` used for pyannote (step06/step10):
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

First run downloads the E2B weights (~10GB). If you have more VRAM/RAM and
want higher quality at the cost of speed, pass
`--gemma-model-id google/gemma-4-E4B-it` to `run_pipeline.py`.

### If you see torchcodec / libnvrtc warnings or errors

Steps 5, 6 and 10 already avoid torchcodec entirely (they load audio
themselves and hand pyannote/silero a preloaded waveform instead of a file
path - see the troubleshooting section below). If you still hit a
torchcodec error from some other tool in your environment, reinstalling
its CPU build directly resolves it:

```bash
pip uninstall -y torchcodec
pip install --force-reinstall torchcodec --index-url https://download.pytorch.org/whl/cpu
```

### HuggingFace access (required for diarization + embedding models)

1. Create a free account at huggingface.co and generate a token
   (Settings -> Access Tokens).
2. Accept the model terms on these two pages (one click each, while logged in):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Export it before running the pipeline:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
   (or pass `--hf-token hf_xxx` on the command line)

### Voice cloning & separation model licenses

- **XTTS-v2** (step 13, `coqui-tts` package): weights under the **Coqui
  Public Model License** (free for personal/research use; read before
  commercial use): https://coqui.ai/cpml
- **Demucs** (step 4, `demucs` package): MIT-licensed, Meta AI Research.
  First run downloads the `htdemucs` model (~80MB).
- **Gemma 4 E2B** (step 11, `google/gemma-4-E2B-it`): weights under
  **Google's Gemma license** - accept it at
  https://huggingface.co/google/gemma-4-E2B-it before first use (see
  section 1b).

## 2. Run it

```bash
python run_pipeline.py --input input/video.mp4 --target-lang hi
```

Common flags:

| Flag | Meaning |
|---|---|
| `--input` | path to source video (default `input/video.mp4`) |
| `--target-lang` | ISO 639-1 target language, e.g. `hi`, `es`, `fr`, `de`, `ja`, `ta`, `bn` ... |
| `--source-lang` | ISO 639-1 source language, or `auto` (default, Gemma 4 detects it) |
| `--device` | `auto` (default) / `cpu` / `cuda` - used by the main venv's steps |
| `--gemma-model-id` | `google/gemma-4-E2B-it` (default) / `google/gemma-4-E4B-it` (bigger, higher quality) |
| `--gemma-device` | `auto` (default) / `cpu` / `cuda` - used inside the separate `.venv-gemma` worker |
| `--subtitle-mode` | which subtitle track(s) to embed: `translated` (default) / `original` / `bilingual` / `all` |
| `--no-original-audio-track` | drop the original-language audio track from the final file |
| `--no-soft-subtitles` | don't embed any subtitle tracks |
| `--burn-subtitles` | also export an extra MP4 with subtitles burned into the picture |
| `--from-step` / `--to-step` | run only a subrange of the 16 steps (great for resuming after a crash) |

Output lands in `output/final/<name>_<lang>_dubbed.mkv` - open it in VLC/MPV
and use the audio-track and subtitle-track menus to switch between the
dubbed/original audio and the different subtitle options. Background
music/SFX play under both audio tracks.

### Resuming after a failure

Every step writes its result to disk under `output/<stage>/` before the next
step runs, so if step 13 (voice generation) fails halfway through you don't
need to redo transcription/translation/diarization/separation:

```bash
python run_pipeline.py --input input/video.mp4 --target-lang hi \
    --from-step 13 --to-step 16
```

## 3. How voice cloning stays gender-consistent and high-fidelity

Step 10 measures each speaker's gender/age/emotion. Step 11 (Gemma 4) uses
that same gender when transcribing/translating each segment, so pronouns
and verb/adjective agreement in the translation are natural for that
speaker in languages that mark gender - not just the cloned *voice*, but
the *words themselves*, stay speaker-consistent. Step 12 then picks 3
clean, non-overlapping ~5s windows of *that speaker's own voice* (spread
across the conversation for phonetic variety, not just one narrow slice)
and concatenates them into a single reference clip. Step 13 feeds that
straight into XTTS-v2's zero-shot cloning, so the generated dub is
literally that speaker's voice speaking the new language - gender, pitch
and timbre carry over automatically (male -> male, female -> female,
other -> other), rather than being reassigned to a generic narrator voice.

Reference quality is the single biggest lever for cloning fidelity, which
is why step 4 (Demucs) isolates speech from background music/noise
*before* any reference clip is ever selected - a reference clip with music
bleeding into it teaches the clone to reproduce that music too.

XTTS-v2 natively speaks 17 languages (en, es, fr, de, it, pt, pl, tr, ru, nl,
cs, ar, zh, ja, hu, ko, hi). If `--target-lang` isn't one of these, step 13
automatically falls back to Meta's MMS-TTS (~1100 languages, one voice per
language) and applies a small pitch shift toward the detected gender.
Gemma 4 (step 11) itself supports far more languages than either TTS
engine, so translation quality/coverage is never the bottleneck.

## 3a. How step 11 (Gemma 4) actually works

Old layout: `faster-whisper` transcribed each segment's audio to text,
then `NLLB-200` translated that text - two separate model calls, and the
translator never heard the audio or knew who was speaking. New layout:
Gemma 4 E2B *listens to the segment's own audio clip directly* and does
both in one call, given:

- **the audio itself** (resampled to 16kHz mono, chunked into <=28s
  windows if a segment happens to exceed Gemma 4's 30s clip limit - rare,
  since segments are already per-speaker-turn from diarization),
- **the speaker's gender** from step10's `speaker_analysis.json`, so it
  can choose gender-correct pronouns/agreement in the target language,
- **the segment's spoken duration** from diarization, so it's told
  roughly how long the line has to fit when spoken aloud and can prefer a
  concise, natural phrasing over an unnecessarily long literal one
  (a soft preference - meaning is never sacrificed to hit a timing).

It's asked to return one compact JSON object -
`{"original_text": ..., "detected_language": ..., "translation": ...}` -
which `gemma_worker/transcribe_translate_worker.py` parses (with one
automatic retry if the model's first response isn't valid JSON before
falling back to treating the raw text as the translation, so one odd
segment never crashes the whole run). Both fields land in
`output/translations/translated.json` exactly like the old two-step
pipeline's output did, so step12/13/14/15 downstream needed no changes.

## 4. Why sentences no longer cut off or have odd pauses

Handing a TTS model a whole multi-sentence paragraph in one call is a
common cause of it truncating after the first clause, or leaving
inconsistent gaps between sentences. XTTS also has a hard per-language
character limit per synthesis call that's easy to blow past without
noticing - Hindi's limit is 150 characters, English's is 250, Japanese's is
only 71 (`XTTS_CHAR_LIMITS` in `config.py`, taken directly from XTTS's own
tokenizer) - so a single flat chunk size for every language either
truncates some languages or over-splits others. Step 13 now:

1. Splits each segment's translated text into sentence-sized chunks sized
   to that specific language's real limit (`split_into_tts_chunks` +
   `xtts_char_limit()`) before synthesis - every sentence gets its own
   full synthesis pass instead of competing for space in one call and
   risking truncation.
2. Trims each chunk's own leading/trailing silence, then joins chunks with
   an explicit, configurable pause (`tts_sentence_pause_ms` in
   `config.py`, default 220ms) instead of whatever gap the model happens
   to leave.
3. Runs a light denoise + peak-normalize pass on the finished segment
   before writing it out, cleaning up synthesis artifacts (toggle with
   `cfg.tts_post_denoise`, on by default).

## 5. How every segment stays perfectly in sync with the video

Each segment's audio is synced to its own original timing **right when
it's generated, in step13** - not left until the final mix like earlier
versions of this pipeline did:

1. The chunked, cleaned-up audio for a segment (section 4 above) is
   measured against that same segment's own original diarized duration
   (`end - start` from step06/step11).
2. If it doesn't match, `time_stretch_to_duration()` (pitch-preserving,
   via `librosa.effects.time_stretch`) speeds it up or slows it down just
   enough to fit that exact duration - clamped to a safety range
   (`cfg.tts_stretch_min_rate` / `cfg.tts_stretch_max_rate`, default
   0.55x-2.2x) so an unusually long or short translation is nudged toward
   the target rather than distorted into something unnatural.
3. *Then* the segment is written to disk - so `output/generated_audio/`
   already contains frame-accurate clips, and `generated_audio.json`
   records `natural_duration_seconds` (before stretch),
   `generated_duration_seconds` (after), and `duration_stretch_rate` for
   every segment so you can see exactly what happened.

This is on by default (`cfg.sync_segment_duration = True`) and is why
sync issues are rare in practice: Gemma 4 (step11) is already told each
segment's target duration when it translates and tends to phrase things
to roughly fit, so the stretch step13 ends up applying is usually small
and doesn't sound obviously sped up or slowed down.

Step15 (audio assembly) still keeps a chronological-placement safety net
on top of this for anything step13's sync didn't fully resolve (disabled,
missing timing data, or a stretch that hit the safety clamp): a line
starts at `max(its own original timestamp, the moment the previous line
actually finished + a small gap)`, so nothing overlaps or gets cut, and
`cfg.stretch_audio_to_fit` (off by default) can force-fit any such
leftover-drift line there too if you want stricter timestamp alignment
than that safety net alone provides.

## 5a. CPU speed & memory optimizations

Two changes make full CPU runs meaningfully faster and lighter on RAM,
without touching output quality:

- **Thread tuning.** `run_pipeline.py` sets `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to
  the machine's full core count *before* numpy/torch/scipy/librosa get
  imported anywhere (these libraries size their thread pools once, at
  first use - setting the env var later has no effect), and calls
  `configure_cpu_threads(cfg.cpu_threads)` for torch's own thread pool on
  top of that. Step13 (voice generation) - the single heaviest CPU step,
  and the only one running at that point - additionally claims every core
  for itself via `cfg.tts_cpu_threads` (default 0 = all cores) instead of
  the lighter shared cap (`cfg.cpu_threads`, default `min(cpu_count, 4)`)
  the rest of the pipeline uses to stay considerate of smaller machines.
  All TTS synthesis calls also run inside `torch.inference_mode()`, which
  skips autograd bookkeeping entirely since nothing here is ever trained.
- **RAM hygiene between steps.** Every pipeline step loads its own model(s)
  into local variables that normally get garbage-collected once that
  step's `run(cfg)` returns - but reference cycles (common in torch
  `nn.Module` graphs), CUDA's caching allocator, and CPython's own memory
  allocator all tend to hold on to freed memory rather than returning it
  immediately. `free_memory()` (`src/utils.py`) explicitly runs
  `gc.collect()`, `torch.cuda.empty_cache()` (if a GPU was used), and
  `malloc_trim(0)` (glibc/Linux) after every step in `run_pipeline.py`,
  and again every 25 segments inside step13's own loop for long videos -
  so each step, and each batch of segments, starts from a clean RAM
  baseline instead of stacking on top of whatever the previous one left
  cached.

## 6. CPU vs GPU

Every step runs on CPU by default and only uses a GPU automatically when
`torch.cuda.is_available()` is true (or you pass `--device cuda` /
`--gemma-device cuda`). See section 5a above for the thread-count and
memory-hygiene tuning that applies regardless of device. CPU guidance:

- Gemma 4 E2B (step 11, its own `.venv-gemma`) is the more capable option
  by default; it's a ~2B-parameter multimodal model and runs adequately on
  CPU for short clips, but a GPU (`--gemma-device cuda`) speeds it up
  substantially if you have one available - especially with the larger
  `--gemma-model-id google/gemma-4-E4B-it`.
- noisereduce, Silero VAD and pyannote diarization all run fine on CPU.
- **Demucs (step 4) is the slowest CPU step** after XTTS - separating a
  typical video can take longer than its own runtime on CPU. `htdemucs`
  (default) is the faster model; `htdemucs_ft` (set `demucs_model` in
  `config.py`) is higher quality but ~4x slower.
- XTTS-v2 also works on CPU but is slow (a few seconds of compute per
  second of speech, now called once per sentence rather than once per
  segment) - a GPU speeds both this and Demucs up substantially.

## 7. Troubleshooting / why these specific package choices

This exact package set (`requirements.txt`) was installed together in a
clean venv and verified with `pip check` (no broken requirements) and a
full import test of every model library the pipeline uses. Real conflicts
came up during that testing and are already fixed in this repo - noted
here in case you customize versions later:

- **`No module named 'torchaudio.backend'`** - hit if you swap step 3 back
  to DeepFilterNet. Current `torchaudio` releases removed the
  `torchaudio.backend` module DeepFilterNet imports at load time, and
  there's no newer DeepFilterNet release to fix it (last published Aug
  2023). Its native extension also has no prebuilt PyPI wheels (needs a
  Rust toolchain) and hard-pins `numpy<2.0`, conflicting with
  `pyannote.audio` 4.x. `noisereduce` avoids all of this.
- **`AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'`**
  - `pyannote.audio` 3.x calls a `torchaudio` function current `torchaudio`
  removed. `pyannote.audio>=4.0.1` fixed this with `torchcodec`, which is
  why `requirements.txt` requires 4.x (steps 6/10 use `token=`, not the
  removed `use_auth_token=`).
- **`ImportError: cannot import name 'isin_mps_friendly' from
  'transformers.pytorch_utils'`** - `coqui-tts` hasn't been updated for
  `transformers` 5.x yet. `requirements.txt` pins `transformers<5.0`. This
  is also why Gemma 4 (step 11, needs `transformers>=5.10.1`) lives in its
  own `.venv-gemma` instead - see section 1b.
- **`torchcodec is not available. Cannot read audio file`** /
  **`OSError: libnvrtc.so.13: cannot open shared object file`** -
  torchcodec's prebuilt shared libraries try to `dlopen` a CUDA runtime
  library even on CPU-only machines. Steps 5, 6 and 10 avoid this by
  loading audio themselves with `soundfile`/`librosa` and passing
  pyannote/silero a preloaded `{'waveform': tensor, 'sample_rate': int}`
  in memory instead of a file path - an officially supported input format,
  not a workaround. If some other tool in your stack still needs
  torchcodec itself, see the CPU-wheel reinstall command in section 1a.
- **`No module named 'omegaconf'`** - a transitive dependency some
  pyannote model configs pull in isn't always resolved automatically by
  every pip/environment combination. `pip install omegaconf` if you hit
  this (it's a small, dependency-light package; safe to add directly).
- **Gemma worker can't find `.venv-gemma`** - step11 raises a clear
  `FileNotFoundError` telling you to run `scripts/setup_gemma_venv.sh` /
  `.bat` (section 1b) if that venv doesn't exist yet.
- **`401 Unauthorized` / gated repo error loading `google/gemma-4-E2B-it`**
  - accept the license at https://huggingface.co/google/gemma-4-E2B-it
  while logged in, then make sure `HF_TOKEN` is exported before you run
  the pipeline (same token pyannote uses).

If a fresh `pip install -r requirements.txt` ever fails with an error
that isn't one of these, run `pip check` afterward - it will point at
exactly which two packages disagree on a version.

## 8. Project layout

```
video_translator/
  config.py             <- single source of truth for all paths/settings
  run_pipeline.py       <- CLI entry point, runs steps 1-16 (main venv)
  requirements.txt      <- main venv (steps 1-10, 12-16)
  requirements-gemma.txt <- .venv-gemma (step 11 only - see section 1b)
  scripts/
    setup_gemma_venv.sh  <- one-time .venv-gemma setup (Linux/macOS)
    setup_gemma_venv.bat <- one-time .venv-gemma setup (Windows)
  gemma_worker/
    transcribe_translate_worker.py  <- runs inside .venv-gemma, called by step11
  src/
    utils.py            <- logging, device selection, ffmpeg helpers,
                            time-stretch, sentence chunking, silence trim
    step01_extract_audio.py ... step16_video_mux.py
  input/                <- put your source video here
  output/                <- every step's intermediate + final files
```

Each step also has a `run(cfg)` function and can be executed standalone for
debugging, e.g.:
```bash
python -m src.step11_transcription_translation
```
(uses the default `PipelineConfig()` paths in that case - step11 still
needs `.venv-gemma` set up, since it shells out to it regardless of how
it's invoked).
