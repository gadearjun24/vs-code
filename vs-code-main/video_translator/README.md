# Video Translation & Dubbing Pipeline

Translate a video from one language to another, keeping each speaker's own
voice (male stays male, female stays female, child/other stays as they are),
with background music/SFX preserved under the dub, perfectly time-aligned
audio (never longer than the original line, never cropped mid-word), and
both original + translated subtitles.

The pipeline needs **one Python venv** (`requirements.txt`) plus **one
separate local service, Ollama**, which serves step12's Gemma 4 model.
Ollama is not a venv or a Python dependency - it's a standalone server you
install once (like ffmpeg) and leave running; the venv only needs the tiny
`ollama` client library to talk to it over HTTP. See section 1b.

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
07 segment merge             -> merge tiny same-speaker fragments into natural-length blocks
08 split audio              -> per speaker / per block clips (from clean vocals)
09 speaker aggregation      -> merge each speaker's clips
10 audio normalization      -> loudness normalize (EBU R128)
11 speaker embedding        -> gender / age / emotion per speaker
12 transcription+translation -> Gemma 4, via Ollama (audio-native, gender+timing+emotion aware)
13 reference voice          -> 3 clean, non-overlapping clips per speaker
14 voice generation         -> XTTS-v2 clone (or MMS-TTS fallback), chunked, emotion-nudged
15 subtitles                -> original.srt / translated.srt / bilingual.srt
16 audio assembly           -> time-align dub + mix back preserved BGM/SFX
17 video mux                -> final .mkv: video + dubbed + original audio + subs + BGM
```

Step 07 (segment merge) runs right after diarization: raw diarization/VAD
output is often full of very short fragments (a sentence split across a
breath pause, a one-word backchannel), and handing one of those to step12
in isolation gives Gemma 4 too little audio to transcribe or read
emotional tone from reliably. It merges consecutive same-speaker segments
separated by a small gap into one natural-length block *before* step08
ever extracts per-segment audio - see section 3b below for the full
algorithm and why nothing needs to be "un-merged" later for sync to stay
correct.

Steps 09-11 (aggregation/normalization/gender detection) intentionally run
*before* step12 now, rather than after transcription like the old
faster-whisper-based layout - step12 needs each speaker's gender to make
the translation grammatically correct, and gender is only known once
step11 has run.

## 1a. One-time setup - the venv (one requirements file)

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

## 1b. One-time setup - Ollama, for Gemma 4 (step12 only)

Step11 (transcription + translation + emotion) uses **Gemma 4**, served
locally by **Ollama** rather than loaded directly in this project's venv.
Gemma 4's E2B/E4B sizes listen to each segment's own audio and return the
original-language transcription, the target-language translation, AND a
detected emotion label for that speaker's delivery, all in one call -
conditioned on that speaker's detected gender (from step11) and the
segment's spoken duration (from diarization), so the wording naturally
fits both the speaker's voice and the time slot a dubbing voice actor has
to say it in.

```bash
# 1. Install Ollama (one-time, like installing ffmpeg - not a pip package):
curl -fsSL https://ollama.com/install.sh | sh    # Linux
# macOS: download the app from https://ollama.com/download, or `brew install ollama`
# Windows: download the installer from https://ollama.com/download

# 2. Pull the model (one-time, downloads ~5GB for E2B):
ollama pull gemma4:e2b

# 3. Make sure the server is running (the installer usually sets this up
#    as a background service already - if `ollama list` works, it's up):
ollama serve
```

A convenience script does all three (installs Ollama if missing, starts
the server if needed, pulls the model):

```bash
bash scripts/setup_ollama.sh              # Linux/macOS, pulls gemma4:e2b
scripts\setup_ollama.bat                  # Windows, pulls gemma4:e2b
```

step12 talks to Ollama over plain HTTP (default `http://localhost:11434`,
override with `--ollama-host` or the `OLLAMA_HOST` env var) using the
`ollama` Python client already in `requirements.txt` - no subprocess, no
second venv, no job files. If Ollama isn't reachable or the model isn't
pulled yet, step12 fails fast with a message telling you exactly which of
the three commands above to run.

If you have more VRAM/RAM and want higher quality at the cost of speed,
pull `gemma4:e4b` instead and pass `--gemma-model-id gemma4:e4b` to
`run_pipeline.py`.

### If you see torchcodec / libnvrtc warnings or errors
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

- **XTTS-v2** (step 14, `coqui-tts` package): weights under the **Coqui
  Public Model License** (free for personal/research use; read before
  commercial use): https://coqui.ai/cpml
- **Demucs** (step 4, `demucs` package): MIT-licensed, Meta AI Research.
  First run downloads the `htdemucs` model (~80MB).
- **Gemma 4** (step12, served via Ollama, model tag `gemma4:e2b`): weights
  under **Google's Gemma license** - see
  https://ollama.com/library/gemma4 for the model card and license terms
  before first use.

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
| `--gemma-model-id` | `gemma4:e2b` (default) / `gemma4:e4b` (bigger, higher quality) - Ollama model tag |
| `--ollama-host` | Ollama server URL (default: `$OLLAMA_HOST` or `http://localhost:11434`) |
| `--subtitle-mode` | which subtitle track(s) to embed: `translated` (default) / `original` / `bilingual` / `all` |
| `--no-original-audio-track` | drop the original-language audio track from the final file |
| `--no-soft-subtitles` | don't embed any subtitle tracks |
| `--burn-subtitles` | also export an extra MP4 with subtitles burned into the picture |
| `--from-step` / `--to-step` | run only a subrange of the 17 steps (great for resuming after a crash) |

Output lands in `output/final/<name>_<lang>_dubbed.mkv` - open it in VLC/MPV
and use the audio-track and subtitle-track menus to switch between the
dubbed/original audio and the different subtitle options. Background
music/SFX play under both audio tracks.

### Resuming after a failure

Every step writes its result to disk under `output/<stage>/` before the next
step runs, so if step 14 (voice generation) fails halfway through you don't
need to redo transcription/translation/diarization/separation:

```bash
python run_pipeline.py --input input/video.mp4 --target-lang hi \
    --from-step 14 --to-step 17
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
cs, ar, zh, ja, hu, ko, hi). If `--target-lang` isn't one of these, step 14
automatically falls back to Meta's MMS-TTS (~1100 languages, one voice per
language) and applies a small pitch shift toward the detected gender.
Gemma 4 (step 12) itself supports far more languages than either TTS
engine, so translation quality/coverage is never the bottleneck.

## 3a. How step 12 (Gemma 4 via Ollama) actually works

Old layout: `faster-whisper` transcribed each segment's audio to text,
then `NLLB-200` translated that text - two separate model calls, and the
translator never heard the audio or knew who was speaking. Current layout:
Gemma 4 *listens to the segment's own audio clip directly* and does
transcription, translation, AND emotion detection in one call, given:

- **the audio itself** - loaded, resampled to 16kHz mono, and re-encoded
  as an in-memory WAV (chunked into <=28s windows if a segment happens to
  exceed Gemma 4's 30s clip limit - rare, since segments are already
  per-speaker-turn from diarization), base64-encoded and sent to Ollama,
- **the speaker's gender** from step11's `speaker_analysis.json`, so it
  can choose gender-correct pronouns/agreement in the target language,
- **the segment's spoken duration** from diarization, so it's told
  roughly how long the line has to fit when spoken aloud and can prefer a
  concise, natural phrasing over an unnecessarily long literal one
  (a soft preference - meaning is never sacrificed to hit a timing),
- an explicit instruction to listen to the whole clip carefully before
  answering and analyze pronunciation, pacing, and emotional delivery, not
  just transcribe the words.

It's asked to return one compact JSON object:
```json
{"is_song": false, "contains_dialogue": true, "dialogue_segments": [],
 "original_text": "...", "detected_language": "...",
 "translation": "...", "emotion": "..."}
```
- `emotion` is one of `neutral/happy/sad/angry/excited/calm/fearful/surprised`.
- `is_song`/`contains_dialogue`/`dialogue_segments` (see section 5b below
  for what step14 does with these) - song lyrics are explicitly excluded
  from `original_text`/`translation`, which cover spoken dialogue only.

Two things make this reliable: Ollama's **JSON mode** (`format="json"` in
the request) guarantees the response is syntactically valid JSON, and
`gemma_worker/ollama_engine.py` additionally validates it has the fields
this pipeline actually needs, retrying once with a stricter reminder if
not (falling back to treating the raw text as the translation only if
that retry also fails, so one odd segment never crashes the whole run).
Every call is non-streaming (`stream=False`) - the full JSON object comes
back in one response, nothing to reassemble from partial tokens.

Every field lands in `output/translations/translated.json` (plus
`original_volume_dbfs`, measured directly from the audio rather than
asked of Gemma - see section 5b), and every segment's full prompt + raw
model response + parsed JSON is appended to
`output/translations/ollama_gemma.log` for debugging. `emotion`,
`is_song`/`contains_dialogue`/`dialogue_windows`, and
`original_volume_dbfs` all flow into step14 - see sections 5 and 5b for
what it does with each.

## 3b. How step 07 merges tiny segments (and why nothing needs "un-merging")

Raw diarization output for real conversational audio looks like this far
more often than you'd expect (a real example, one speaker's turn):

```
segment_000001  speaker_001  6.275 - 7.861   (1.586s)
segment_000002  speaker_001  12.670 - 14.577  (1.907s)
segment_000003  speaker_001  14.898 - 15.472  (0.574s)   <- 0.32s gap before this
segment_000004  speaker_001  17.024 - 18.104  (1.080s)   <- 1.55s gap before this
segment_000005  speaker_001  18.577 - 18.948  (0.371s)   <- 0.47s gap before this
...
segment_000013  speaker_001  45.712 - 45.762  (0.051s)   <- a single backchannel,
                                                             sandwiched between two
                                                             speaker_002 segments
```

Handing `segment_000013` (51 milliseconds - barely one phoneme) to Gemma 4
in isolation gives it essentially nothing to transcribe or read emotional
tone from. Step07 fixes this once, at the source, before step08 ever
extracts per-segment audio:

1. **Merge.** Consecutive segments from the *same speaker*, separated by a
   gap of at most `cfg.merge_max_gap_seconds` (default 1.5s), are combined
   into one block - capped at `cfg.merge_max_duration_seconds` (default
   18s, comfortably under Gemma 4's 30s clip limit). On the example above:
   `segment_000002`+`segment_000003` merge (0.32s gap); `segment_000004`
   through `segment_000007` all chain-merge into one ~6.6s block (each gap
   between them is under 1.5s). A gap bigger than that, or a change of
   speaker, always starts a new block - that's a real pause or turn-taking,
   not noise to smooth over. On this real sample: **25 raw segments become
   17 blocks.**
2. **Pad, only when still tiny.** `segment_000013` has no same-speaker
   neighbor close enough to merge with (both its neighbors are
   `speaker_002`), so it stays its own block - but at 51ms it's still under
   `cfg.merge_min_clip_seconds` (default 0.6s), so step07 widens its
   *audio extraction window only* with a little silence from both sides,
   clamped so it can never cross into either neighboring block's actual
   speech (never even by a full millisecond, plus a small safety buffer).
   This gives Gemma 4 enough signal to work with without ever bleeding
   another speaker's words into the clip.

**Why nothing needs to be "reversed" for sync to stay correct:** every
block keeps its **true** `start`/`end`/`duration` (first raw segment's
start to last raw segment's end) completely separate from the padded
`clip_start`/`clip_end` used only for extraction. Every step downstream -
Gemma's "how long does this line have to fit when spoken" budget (step12),
duration sync (step14, section 5), and final placement (step16) - always
uses the block's true `start`/`end`, never the padded clip window. So a
merged block occupies exactly the same span on the final timeline that its
original fragments did; there's no separate "un-merge" step because
nothing about placement ever depended on the merge in the first place. The
original raw fragments (`segment_id`/`start`/`end`/`duration` for each) are
also preserved verbatim under `original_segments` in
`merged_diarization.json` and carried through into `translated.json`, so
they're always there if you want them for something else later.

One side effect worth knowing: since transcription/translation/subtitles
all now operate on merged blocks, a subtitle entry covers a whole block
(e.g. one entry for the ~6.6s `segment_000004`-`000007` block) rather than
one entry per tiny raw fragment. This is normal, even preferable,
subtitling practice - grouping by utterance rather than flickering a new
line every 0.3-1s - and it's a direct benefit of Gemma 4 now getting full
sentences with real context instead of fragments.

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

## 5. How every segment stays perfectly in sync with the video - without ever cropping audio

Each segment's audio is synced to its own original timing **right when
it's generated, in step14** - not left until the final mix like earlier
versions of this pipeline did - and the only tool ever used to do it is
pitch-preserving time-stretch, which keeps 100% of the synthesized speech
just faster or slower; **audio content is never truncated/cropped** to
force a duration, so no word is ever cut short:

1. The chunked, cleaned-up audio for a segment (section 4 above) is
   measured against that same segment's own original diarized duration
   (`end - start` from step06/step12).
2. If it doesn't match, `time_stretch_to_duration()` (pitch-preserving,
   via `librosa.effects.time_stretch`) speeds it up or slows it down just
   enough to fit that exact duration - with deliberately **asymmetric**
   safety bounds:
   - speeding up (audio came out longer than its slot) can go up to
     `cfg.tts_stretch_max_rate` (default **3.0x**) - kept generous
     specifically so a segment essentially never ends up longer than its
     original slot, since that's the one outcome that could push into,
     and audibly overlap, the next line once everything is combined;
   - slowing down (audio came out shorter) is capped tighter at
     `cfg.tts_stretch_min_rate` (default **0.7x**, i.e. stretched at most
     ~1.4x longer) purely for naturalness - a short line slowed down can
     never overflow into the next one no matter how far it's stretched,
     so this bound only exists to avoid sounding muffled/padded.
3. *Then* the segment is written to disk - so `output/generated_audio/`
   already contains frame-accurate clips, and `generated_audio.json`
   records `natural_duration_seconds` (before stretch),
   `generated_duration_seconds` (after), and `duration_stretch_rate` for
   every segment so you can see exactly what happened.

This is on by default (`cfg.sync_segment_duration = True`) and is why
sync issues are rare in practice: Gemma 4 (step12) is already told each
segment's target duration when it translates and tends to phrase things
to roughly fit, so the stretch step14 ends up applying is usually small
and doesn't sound obviously sped up or slowed down. In the rare case even
the 3.0x ceiling can't fully close the gap, step14 logs a clear warning
naming the segment and the overflow amount - and still never crops it.

Step15 (audio assembly) keeps a chronological-placement safety net on top
of this for that rare leftover case (or if `sync_segment_duration` is
turned off, or timing data is missing): a line starts at `max(its own
original timestamp, the moment the previous line actually finished + a
small gap)`, so nothing overlaps or gets cut, and `cfg.stretch_audio_to_fit`
(off by default) can force-fit any such leftover-drift line there too if
you want stricter timestamp alignment than that safety net alone provides.

### Emotion-aware delivery

Gemma 4 (step12) also detects an `emotion` label for each segment's
original delivery (`neutral/happy/sad/angry/excited/calm/fearful/surprised`).
Step13 uses it to nudge `tts_speed` (default **1.15** - a mild, generally
natural pace bump on top of the reference voice's own speed) and
`tts_temperature` a little per segment - e.g. `excited` speaks slightly
faster and with a touch more expressive variance, `sad` speaks slightly
slower and flatter. XTTS-v2 has no direct "emotion" input, so this is a
deliberately modest proxy through its two real prosody knobs
(`EMOTION_SPEED_ADJUST` / `EMOTION_TEMPERATURE_ADJUST` in step14), not a
claim of true emotional TTS - and it's applied *before* the duration sync
above, so it never fights with the exact-duration guarantee: sync always
has the final word on actual output length. Toggle with
`cfg.tts_emotion_aware_delivery` (on by default). The MMS-TTS fallback
backend gets a smaller version of the same idea as an extra pitch nudge
on top of its existing gender-based pitch shift.

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
  and again every 25 segments inside step14's own loop for long videos -
  so each step, and each batch of segments, starts from a clean RAM
  baseline instead of stacking on top of whatever the previous one left
  cached.

## 5b. Songs stay original, and loudness/energy is preserved per line

**Songs.** Gemma 4 (step12) listens to every segment and flags `is_song`
(any part is sung) and `contains_dialogue` (any part is genuinely spoken)
separately - a clip can be pure song, pure dialogue, or both at once (e.g.
dialogue spoken over background music). Song lyrics are never
transcribed, translated, or redubbed. When `cfg.keep_song_audio_original`
is on (default), step14 acts on these flags:

- **Pure song** (`is_song`, no dialogue): synthesis is skipped completely
  - the segment's original recording is used verbatim, at its own natural
    length, with no denoise/normalize pass that could alter its fidelity.
- **Mixed** (both true): the original recording is the base, and a
  synthesized, translated dub is spliced in *only* at the dialogue
  timing Gemma estimated (`dialogue_windows`, step12) - everywhere else
  in the clip keeps the untouched original audio. Since an LLM's window
  estimate is a best-effort guess rather than a frame-accurate cut point,
  each splice is blended with a short crossfade
  (`cfg.song_splice_crossfade_ms`, default 40ms) instead of a hard cut,
  and if Gemma didn't return usable timing for a mixed clip, this step
  falls back to the original recording untouched rather than guessing at
  a cut point and risking a badly-placed dub over part of a song.
- **Not a song**: nothing changes - the full dub pipeline from sections
  3-5 above runs exactly as before.

Every generated-audio record notes which of these happened
(`is_song`, `contains_dialogue`, and `backend`: `original_passthrough` /
`song_with_spliced_dialogue` / `xtts_v2` / `mms_tts_fallback`), so you can
audit exactly what was kept original vs. dubbed after a run.

**Loudness/energy.** A shouted, angry line and a quiet, calm one
shouldn't come out of step14 at the same volume just because TTS output
happens to get peak-normalized - that would quietly undo the whole point
of the emotion-aware delivery in section 3. step12 measures each
segment's real loudness directly from its own audio (RMS,
`original_volume_dbfs` - a signal measurement, not something asked of
Gemma). step14 uses it here: instead of always normalizing every line's
peak to a flat 0.95, the target peak is that baseline shifted up or down
by however far this segment's original volume sits from the
conversation's average - clamped to `cfg.volume_dynamic_range_db`
(default ±8dB) so one extreme outlier line can't be normalized into
clipping or near-silence, then to `cfg.volume_min_peak`/
`cfg.volume_max_peak` (0.15/0.98) as an absolute safety floor/ceiling.
Toggle with `cfg.volume_aware_normalization`. Every generated-audio record
also carries `original_volume_dbfs` and the `target_peak` actually used,
for the same auditability.

## 6. CPU vs GPU

Every step in the venv runs on CPU by default and only uses a GPU
automatically when `torch.cuda.is_available()` is true (or you pass
`--device cuda`). Ollama (step12) manages its own device placement
independently - it uses your GPU automatically if it finds one, no flag
needed on the pipeline side. See section 5a above for the thread-count and
memory-hygiene tuning that applies regardless of device. CPU guidance:

- Gemma 4 (step12, served by Ollama) runs adequately on CPU for short
  clips; Ollama will use a GPU automatically if it detects one, which
  speeds it up substantially - especially with the larger
  `--gemma-model-id gemma4:e4b`.
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
  why `requirements.txt` requires 4.x (steps 6/11 use `token=`, not the
  removed `use_auth_token=`).
- **`ImportError: cannot import name 'isin_mps_friendly' from
  'transformers.pytorch_utils'`** - `coqui-tts` hasn't been updated for
  `transformers` 5.x yet. `requirements.txt` pins `transformers<5.0`. This
  no longer affects Gemma 4 (step12) at all since it's served by Ollama, a
  separate process with no `transformers` dependency of its own - see
  section 1b.
- **`torchcodec is not available. Cannot read audio file`** /
  **`OSError: libnvrtc.so.13: cannot open shared object file`** -
  torchcodec's prebuilt shared libraries try to `dlopen` a CUDA runtime
  library even on CPU-only machines. Steps 5, 6 and 11 avoid this by
  loading audio themselves with `soundfile`/`librosa` and passing
  pyannote/silero a preloaded `{'waveform': tensor, 'sample_rate': int}`
  in memory instead of a file path - an officially supported input format,
  not a workaround. If some other tool in your stack still needs
  torchcodec itself, see the CPU-wheel reinstall command in section 1a.
- **`No module named 'omegaconf'`** - a transitive dependency some
  pyannote model configs pull in isn't always resolved automatically by
  every pip/environment combination. `pip install omegaconf` if you hit
  this (it's a small, dependency-light package; safe to add directly).
- **step12 fails with "Could not reach Ollama"** - start the server with
  `ollama serve` (or check it's running as a background service - `ollama
  list` succeeding is enough evidence), then re-run. See section 1b.
- **step12 fails with "Model 'gemma4:e2b' is not pulled in Ollama"** - run
  `ollama pull gemma4:e2b` (or `bash scripts/setup_ollama.sh`), then re-run.

If a fresh `pip install -r requirements.txt` ever fails with an error
that isn't one of these, run `pip check` afterward - it will point at
exactly which two packages disagree on a version.

## 8. Project layout

```
video_translator/
  config.py             <- single source of truth for all paths/settings
  run_pipeline.py       <- CLI entry point, runs steps 1-17
  requirements.txt      <- the one venv this pipeline needs
  scripts/
    setup_ollama.sh      <- one-time Ollama install + model pull (Linux/macOS)
    setup_ollama.bat     <- one-time Ollama install + model pull (Windows)
  gemma_worker/
    ollama_engine.py     <- Gemma 4 client logic (talks to the Ollama server), used by step12
  src/
    utils.py            <- logging, device selection, ffmpeg helpers,
                            time-stretch, sentence chunking, silence trim
    step01_extract_audio.py ... step17_video_mux.py
  input/                <- put your source video here
  output/                <- every step's intermediate + final files
```

Each step also has a `run(cfg)` function and can be executed standalone for
debugging, e.g.:
```bash
python -m src.step12_transcription_translation
```
(uses the default `PipelineConfig()` paths in that case - step12 still
needs Ollama reachable with the model pulled, since it talks to it
regardless of how it's invoked).
