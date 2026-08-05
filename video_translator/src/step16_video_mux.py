"""
Step 16 - Final video assembly.

Produces a single .mkv containing:
  - the original video stream (copied, no re-encode -> no quality loss)
  - the dubbed/translated audio track (default track)
  - the original audio track, kept as a second selectable track
  - soft subtitle tracks (original / translated / bilingual, selectable)

MKV is used because, unlike MP4, it natively supports multiple audio and
.srt subtitle tracks without lossy conversion. An optional MP4 with
burned-in subtitles can also be produced for platforms that need a single
flattened file (e.g. social media uploads).
"""

from __future__ import annotations

from pathlib import Path

from config import PipelineConfig
from src.utils import get_logger, run_ffmpeg

LOG = get_logger(__name__)


def build_mkv(cfg: PipelineConfig) -> Path:
    if not cfg.input_video.exists():
        raise FileNotFoundError(cfg.input_video)
    if not cfg.final_dubbed_audio.exists():
        raise FileNotFoundError(cfg.final_dubbed_audio)

    cfg.dir_final.mkdir(parents=True, exist_ok=True)

    inputs = ["-i", str(cfg.input_video), "-i", str(cfg.final_dubbed_audio)]
    maps = ["-map", "0:v", "-map", "1:a"]

    if cfg.keep_original_audio_track:
        maps += ["-map", "0:a?"]

    subtitle_files = []
    if cfg.embed_soft_subtitles:
        mode = cfg.subtitle_mode
        candidates = []
        if mode in ("translated", "all"):
            candidates.append(("Translated", cfg.srt_translated))
        if mode in ("original", "all"):
            candidates.append(("Original", cfg.srt_original))
        if mode in ("bilingual", "all"):
            candidates.append(("Bilingual", cfg.srt_bilingual))
        subtitle_files = [(title, path) for title, path in candidates if path.exists()]

        for _, path in subtitle_files:
            inputs += ["-i", str(path)]

    sub_input_offset = 2  # inputs[0]=video, [1]=dubbed audio
    for i, _ in enumerate(subtitle_files):
        maps += ["-map", f"{sub_input_offset + i}:0"]

    codec_args = ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if subtitle_files:
        codec_args += ["-c:s", "srt"]

    metadata_args = [
        "-metadata:s:a:0", "title=Dubbed",
    ]
    if cfg.keep_original_audio_track:
        metadata_args += ["-metadata:s:a:1", "title=Original", "-disposition:a:1", "0"]

    for i, (title, _) in enumerate(subtitle_files):
        metadata_args += [f"-metadata:s:s:{i}", f"title={title}"]

    output_path = cfg.final_video
    args = inputs + maps + codec_args + metadata_args + [str(output_path)]

    LOG.info(f"Muxing final video -> {output_path}")
    run_ffmpeg(args, logger=LOG)

    return output_path


def build_burned_mp4(cfg: PipelineConfig) -> Path:
    """Optional: flatten to MP4 with subtitles burned into the picture."""
    if not cfg.srt_translated.exists():
        raise FileNotFoundError(cfg.srt_translated)

    output_path = cfg.dir_final / f"{cfg.input_video.stem}_{cfg.target_lang}_burned.mp4"
    escaped_srt = str(cfg.srt_translated).replace("\\", "/").replace(":", "\\:")

    args = [
        "-i", str(cfg.input_video),
        "-i", str(cfg.final_dubbed_audio),
        "-map", "0:v", "-map", "1:a",
        "-vf", f"subtitles='{escaped_srt}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]

    LOG.info(f"Rendering burned-subtitle MP4 -> {output_path}")
    run_ffmpeg(args, logger=LOG)
    return output_path


def run(cfg: PipelineConfig) -> Path:
    final_path = build_mkv(cfg)

    if cfg.burn_in_subtitles:
        try:
            build_burned_mp4(cfg)
        except Exception as e:
            LOG.warning(f"Burned-subtitle MP4 export skipped: {e}")

    LOG.info(f"Final video ready: {final_path}")
    return final_path


if __name__ == "__main__":
    run(PipelineConfig())
