"""
wishcribe.output
----------------
Write transcripts as TXT, SRT, or JSON.
"""
from __future__ import annotations

import json
from .models import Segment
from .utils import fmt_time, fmt_time_srt

# Speaker labels in this set are hidden from output (used in --no-diarize mode)
_HIDDEN_LABELS = {""}


def _speaker_label(speaker: str) -> str:
    """Return '[LABEL] ' prefix, or empty string if label should be hidden."""
    if speaker in _HIDDEN_LABELS:
        return ""
    return f"[{speaker}] "


def write_txt(segments: list[Segment], path: str) -> None:
    """
    Write human-readable transcript grouped by speaker turns.

    Format:
        [SPEAKER_00] 00:00:01
          Hello, good morning.
          This is the second line from the same speaker.

        [SPEAKER_01] 00:00:10
          Thanks for joining.
    """
    lines = []
    prev_speaker = None

    for seg in segments:
        label = _speaker_label(seg.speaker)
        header = f"{label}{fmt_time(seg.start)}"

        if seg.speaker != prev_speaker:
            if lines:
                lines.append("")   # blank line between speaker turns
            lines.append(header)
            prev_speaker = seg.speaker

        lines.append(f"  {seg.text}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n" if lines else "")

    print(f"📄 Text transcript → {path}")


def write_srt(segments: list[Segment], path: str, word_timestamps: bool = False) -> None:
    """
    Write SRT subtitle file (spec-compliant: index, timing, text, blank line).

    When word_timestamps=True and segments carry .words data, emits one SRT
    block per word for precise karaoke-style subtitles.  Falls back to
    segment-level blocks when a segment has no word data.

    Segment-level format:
        1
        00:00:01,000 --> 00:00:04,500
        [SPEAKER_00] Hello, good morning.

    Word-level format:
        1
        00:00:01,000 --> 00:00:01,420
        Hello
    """
    blocks = []
    idx = 1  # running SRT index (not enumerate — word mode emits multiple blocks per segment)

    for seg in segments:
        label = _speaker_label(seg.speaker)

        if word_timestamps and seg.words:
            # One SRT block per word.  Speaker label only on the first word of each segment
            # so it doesn't repeat on every word (reads more naturally).
            for w_i, w in enumerate(seg.words):
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                prefix = label if w_i == 0 else ""
                blocks.append(
                    f"{idx}\n"
                    f"{fmt_time_srt(w['start'])} --> {fmt_time_srt(w['end'])}\n"
                    f"{prefix}{word_text}"
                )
                idx += 1
        else:
            # Segment-level fallback (backward compatible)
            blocks.append(
                f"{idx}\n"
                f"{fmt_time_srt(seg.start)} --> {fmt_time_srt(seg.end)}\n"
                f"{label}{seg.text}"
            )
            idx += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n" if blocks else "")

    print(f"🎞️  SRT subtitle    → {path}")


def write_json(segments: list[Segment], path: str) -> None:
    """
    Write machine-readable JSON transcript.
    Each entry: start, end, duration, speaker, text  (via Segment.to_dict())
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)

    print(f"📦 JSON data       → {path}")
