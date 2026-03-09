"""Write transcripts as TXT, SRT, or JSON."""
from __future__ import annotations

import json
from .models import Segment
from .utils import fmt_time, fmt_time_srt


def write_txt(segments: list[Segment], path: str) -> None:
    lines, prev = [], None
    for seg in segments:
        if seg.speaker != prev:
            lines.append(f"\n[{seg.speaker}] {fmt_time(seg.start)}")
            prev = seg.speaker
        lines.append(f"  {seg.text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    print(f"📄 Text transcript → {path}")


def write_srt(segments: list[Segment], path: str) -> None:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n"
            f"{fmt_time_srt(seg.start)} --> {fmt_time_srt(seg.end)}\n"
            f"[{seg.speaker}] {seg.text}\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print(f"🎞️  SRT subtitle    → {path}")


def write_json(segments: list[Segment], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)
    print(f"📦 JSON data       → {path}")
