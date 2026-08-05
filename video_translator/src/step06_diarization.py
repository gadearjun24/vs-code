"""Step 06 - Speaker diarization with pyannote.audio 3.1 (CPU/GPU)."""

from __future__ import annotations

import time

import numpy as np
import soundfile as sf
import torch

from config import PipelineConfig
from src.utils import (
    configure_cpu_threads, ensure_mono_float32, get_logger, resolve_device,
    resolve_speech_source, save_json,
)

LOG = get_logger(__name__)

MODEL_NAME = "pyannote/speaker-diarization-3.1"


def load_waveform_dict(path, target_sr: int) -> dict:
    """
    Load audio ourselves and hand pyannote a preloaded
    {'waveform': tensor, 'sample_rate': int} dict instead of a file path.

    pyannote.audio 4.x decodes file paths through `torchcodec`, whose
    prebuilt shared libraries try to dlopen a CUDA runtime lib
    (libnvrtc.so.13) even on CPU-only installs and fail if it's missing.
    Preloading the waveform ourselves (via soundfile, no torchcodec
    involved) sidesteps that entirely - it's an officially supported input
    format for `Pipeline.__call__`, not a hack.
    """
    audio, sr = sf.read(str(path), always_2d=False)
    audio = ensure_mono_float32(np.asarray(audio))

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel=1, time)
    return {"waveform": waveform, "sample_rate": sr}


def run(cfg: PipelineConfig) -> list:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    source = resolve_speech_source(cfg)
    if not source.exists():
        raise FileNotFoundError(source)

    cfg.dir_diarization.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    if device == "cpu":
        threads = configure_cpu_threads(cfg.cpu_threads)
        LOG.info(f"Using {threads} CPU threads")

    LOG.info("Loading diarization model...")
    # pyannote.audio 4.x renamed use_auth_token= to token=.
    pipeline = Pipeline.from_pretrained(MODEL_NAME, token=cfg.hf_token)
    pipeline.to(torch.device(device))

    LOG.info(f"Reading audio from {source}...")
    audio_input = load_waveform_dict(source, cfg.sample_rate)

    LOG.info("Running speaker diarization...")
    start = time.time()
    with ProgressHook() as hook:
        diarization = pipeline(audio_input, hook=hook)
    elapsed = time.time() - start
    LOG.info(f"Completed in {elapsed:.2f}s")

    # pyannote/speaker-diarization-3.1 returns a plain `Annotation` (has
    # .itertracks()). Some newer pyannote.audio 4.x pipelines instead return
    # a result object exposing `.speaker_diarization` as an iterable of
    # (turn, speaker) pairs. Support both so this doesn't silently break if
    # someone swaps MODEL_NAME for a newer pipeline.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    with open(cfg.dir_diarization / "diarization.rttm", "w") as f:
        if hasattr(annotation, "write_rttm"):
            annotation.write_rttm(f)

    if hasattr(annotation, "itertracks"):
        turns = ((turn, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True))
    else:
        turns = iter(annotation)  # (turn, speaker) pairs

    results = []
    speakers = set()
    counter = {}

    for turn, speaker in turns:
        speakers.add(speaker)
        if speaker not in counter:
            counter[speaker] = len(counter) + 1
        speaker_id = f"speaker_{counter[speaker]:03d}"

        results.append({
            "segment_id": f"segment_{len(results) + 1:06d}",
            "speaker_id": speaker_id,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
        })

    save_json(cfg.diarization_json, results)

    with open(cfg.dir_diarization / "timeline.txt", "w") as f:
        for row in results:
            f.write(f"{row['speaker_id']} : {row['start']:.2f}s -> {row['end']:.2f}s\n")

    save_json(cfg.dir_diarization / "metadata.json", {
        "model": MODEL_NAME,
        "device": device,
        "total_speakers": len(speakers),
        "total_segments": len(results),
        "processing_time_seconds": round(elapsed, 2),
    })

    LOG.info(f"Speakers: {len(speakers)} | Segments: {len(results)}")
    return results


if __name__ == "__main__":
    run(PipelineConfig())
