"""
Step 07 - Merge tiny diarization segments into natural-length blocks.

Raw diarization/VAD output is frequently full of very short fragments:
a sentence split across a breath pause, a one-word backchannel ("mhm",
"haan") sandwiched between another speaker's turn, or a sub-second blip.
Handed to step12's Gemma 4 call in isolation, a segment like that gives
the model too little audio to transcribe reliably or read emotional tone
from at all - a 50-150ms clip is barely one phoneme.

This step runs right after diarization (step06) and before anything else
ever touches segment-level audio (step08 onward), so the fix happens once,
at the source, rather than needing to be special-cased in every later
step:

1. Walk the diarization segments in chronological order. Consecutive
   segments from the SAME speaker, separated by a silence gap of at most
   `cfg.merge_max_gap_seconds` (default 1.5s - a natural mid-conversation
   pause), are merged into one block - capped at
   `cfg.merge_max_duration_seconds` (default 18s, comfortably under
   Gemma 4's 30s hard clip limit) so a long monologue doesn't turn into
   one giant block. A gap bigger than that, or a change of speaker, always
   starts a new block - that's a real pause or turn-taking, not noise to
   paper over.

2. Every merged block keeps BOTH:
   - `start`/`end`/`duration`: the TRUE first-segment-start to
     last-segment-end span. This is what every downstream step uses for
     placement/sync (step16) and for the "spoken duration budget" Gemma
     is told when translating (step12) - so merging never changes where
     or how long a block is supposed to occupy on the final timeline.
   - `original_segments`: the full list of raw sub-segments that were
     merged (each with its own original start/end/duration/segment_id) -
     nothing is discarded, so the original per-fragment boundaries are
     always recoverable later even though this pipeline uses the merged
     block as its working unit from here on (see the docstring note on
     subtitles in step15 for why that's a deliberate choice, not a
     limitation).

3. A block that's STILL shorter than `cfg.merge_min_clip_seconds` (default
   0.6s) after merging - typically an isolated segment with a different
   speaker on both sides, so it had no same-speaker neighbor to merge
   with - gets its AUDIO EXTRACTION window (`clip_start`/`clip_end`)
   padded with a little surrounding silence/context purely so step12's
   Gemma call has enough signal to work with. This padding is clamped so
   it never crosses into a neighboring block's true speech (`start`/`end`)
   - not even by a millisecond - so it can never bleed another speaker's
     words into this clip. `start`/`end` themselves are never touched by
     padding, only `clip_start`/`clip_end`, which is why placement/sync
     downstream is unaffected by it.

Nothing here is a substitute for reasonable diarization/VAD settings -
it's a targeted fix for the fragments that will always slip through no
matter how those are tuned.
"""

from __future__ import annotations

from config import PipelineConfig
from src.utils import get_logger, load_json, save_json

LOG = get_logger(__name__)


def _finalize_block(sub_segments: list) -> dict:
    start = sub_segments[0]["start"]
    end = sub_segments[-1]["end"]
    return {
        "speaker_id": sub_segments[0]["speaker_id"],
        "start": start,
        "end": end,
        "duration": round(end - start, 4),
        "clip_start": start,
        "clip_end": end,
        "original_segments": [
            {
                "segment_id": s["segment_id"],
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
            }
            for s in sub_segments
        ],
        "merged_from_count": len(sub_segments),
    }


def merge_segments(segments: list, cfg: PipelineConfig) -> list:
    segments = sorted(segments, key=lambda s: s["start"])
    if not segments:
        return []

    blocks = []
    current = [segments[0]]

    for seg in segments[1:]:
        prev = current[-1]
        gap = seg["start"] - prev["end"]
        block_span_if_added = seg["end"] - current[0]["start"]

        same_speaker = seg["speaker_id"] == current[0]["speaker_id"]
        gap_ok = 0 <= gap <= cfg.merge_max_gap_seconds
        duration_ok = block_span_if_added <= cfg.merge_max_duration_seconds

        if same_speaker and gap_ok and duration_ok:
            current.append(seg)
        else:
            blocks.append(_finalize_block(current))
            current = [seg]

    blocks.append(_finalize_block(current))
    return blocks


def apply_padding(blocks: list, cfg: PipelineConfig) -> list:
    """
    For blocks still shorter than `cfg.merge_min_clip_seconds`, widen
    `clip_start`/`clip_end` (never `start`/`end`) with context from the
    surrounding silence - clamped so it can never reach into a
    neighboring block's true speech, with a small safety buffer.
    """
    SAFETY_BUFFER = 0.03  # seconds - never pad right up to the very edge of a neighbor

    for i, block in enumerate(blocks):
        duration = block["duration"]
        if duration >= cfg.merge_min_clip_seconds:
            continue

        needed = cfg.merge_min_clip_seconds - duration
        pad_each_side = needed / 2.0

        prev_end = blocks[i - 1]["end"] if i > 0 else 0.0
        next_start = blocks[i + 1]["start"] if i < len(blocks) - 1 else block["end"] + 1e9

        max_left_pad = max(0.0, block["start"] - prev_end - SAFETY_BUFFER)
        max_right_pad = max(0.0, next_start - block["end"] - SAFETY_BUFFER)

        left_pad = min(pad_each_side, max_left_pad)
        right_pad = min(pad_each_side, max_right_pad)

        # If one side couldn't take its full share (crowded by a neighbor),
        # let the other side take a bit more, still within its own limit.
        shortfall = (pad_each_side - left_pad) + (pad_each_side - right_pad)
        if shortfall > 1e-6:
            extra_left = min(shortfall, max_left_pad - left_pad)
            left_pad += extra_left
            shortfall -= extra_left
            extra_right = min(shortfall, max_right_pad - right_pad)
            right_pad += extra_right

        block["clip_start"] = round(block["start"] - left_pad, 4)
        block["clip_end"] = round(block["end"] + right_pad, 4)

        if left_pad > 1e-4 or right_pad > 1e-4:
            LOG.info(
                f"  padded short block ({duration:.3f}s, speaker={block['speaker_id']}, "
                f"start={block['start']:.3f}): clip window widened by "
                f"-{left_pad:.3f}s/+{right_pad:.3f}s for Gemma's benefit "
                f"(placement timing unchanged)"
            )

    return blocks


def assign_ids(blocks: list) -> list:
    for i, block in enumerate(blocks, start=1):
        block["segment_id"] = f"segment_{i:06d}"
    # segment_id first, for readability alongside the other step's schema
    return [
        {"segment_id": b["segment_id"], **{k: v for k, v in b.items() if k != "segment_id"}}
        for b in blocks
    ]


def run(cfg: PipelineConfig) -> list:
    if not cfg.diarization_json.exists():
        raise FileNotFoundError(cfg.diarization_json)

    raw_segments = load_json(cfg.diarization_json)
    LOG.info(f"Loaded {len(raw_segments)} raw diarization segment(s).")

    blocks = merge_segments(raw_segments, cfg)
    blocks = apply_padding(blocks, cfg)
    blocks = assign_ids(blocks)

    cfg.dir_diarization.mkdir(parents=True, exist_ok=True)
    save_json(cfg.merged_diarization_json, blocks)

    merged_count = sum(1 for b in blocks if b["merged_from_count"] > 1)
    padded_count = sum(1 for b in blocks if abs(b["clip_start"] - b["start"]) > 1e-4 or abs(b["clip_end"] - b["end"]) > 1e-4)
    tiny_unpadded = sum(
        1 for b in blocks
        if b["duration"] < cfg.merge_min_clip_seconds and abs(b["clip_start"] - b["start"]) < 1e-4 and abs(b["clip_end"] - b["end"]) < 1e-4
    )

    LOG.info(
        f"{len(raw_segments)} raw segment(s) -> {len(blocks)} block(s) "
        f"({merged_count} block(s) are merges of 2+ raw segments, "
        f"{padded_count} short block(s) got a padded extraction window)."
    )
    if tiny_unpadded:
        LOG.warning(
            f"{tiny_unpadded} block(s) are still under {cfg.merge_min_clip_seconds}s and couldn't "
            f"be padded (crowded by neighbors on both sides) - Gemma may still struggle with these; "
            f"they're most likely non-verbal blips (breaths, laughs) rather than real speech."
        )

    return blocks


if __name__ == "__main__":
    run(PipelineConfig())
