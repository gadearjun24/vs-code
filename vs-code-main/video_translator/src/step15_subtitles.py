"""Step 15 - Generate subtitle files (original text, translated text, bilingual)."""

from __future__ import annotations

from config import PipelineConfig
from src.utils import format_srt_timestamp, get_logger, load_json

LOG = get_logger(__name__)


def write_srt(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for i, (start, end, text) in enumerate(entries, start=1):
            if not text:
                continue
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(f"{text.strip()}\n\n")


def run(cfg: PipelineConfig) -> None:
    if not cfg.translated_json.exists():
        raise FileNotFoundError(cfg.translated_json)

    cfg.dir_subtitles.mkdir(parents=True, exist_ok=True)

    conversation = sorted(load_json(cfg.translated_json), key=lambda s: s.get("start", 0.0))

    original_entries = [(s["start"], s["end"], s.get("text", "")) for s in conversation]
    translated_entries = [(s["start"], s["end"], s.get("translation", "")) for s in conversation]
    bilingual_entries = [
        (s["start"], s["end"], f"{(s.get('translation') or '').strip()}\n{(s.get('text') or '').strip()}")
        for s in conversation
    ]

    write_srt(cfg.srt_original, original_entries)
    write_srt(cfg.srt_translated, translated_entries)
    write_srt(cfg.srt_bilingual, bilingual_entries)

    LOG.info(f"Subtitles written: {cfg.srt_original.name}, {cfg.srt_translated.name}, {cfg.srt_bilingual.name}")


if __name__ == "__main__":
    run(PipelineConfig())
