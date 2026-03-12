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
        f.write("\n".join(lines).strip() + "\n")

    print(f"📄 Text transcript → {path}")


def write_srt(segments: list[Segment], path: str) -> None:
    """
    Write SRT subtitle file (spec-compliant: index, timing, text, blank line).

    Format:
        1
        00:00:01,000 --> 00:00:04,500
        [SPEAKER_00] Hello, good morning.

        2
        ...
    """
    blocks = []
    for i, seg in enumerate(segments, 1):
        label = _speaker_label(seg.speaker)
        blocks.append(
            f"{i}\n"
            f"{fmt_time_srt(seg.start)} --> {fmt_time_srt(seg.end)}\n"
            f"{label}{seg.text}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")

    print(f"🎞️  SRT subtitle    → {path}")


def write_json(segments: list[Segment], path: str) -> None:
    """
    Write machine-readable JSON transcript.
    Each entry: start, end, duration, speaker, text  (via Segment.to_dict())
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)

    print(f"📦 JSON data       → {path}")
