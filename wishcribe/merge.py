"""Merge Whisper segments with diarization speaker labels by time overlap."""
from __future__ import annotations

from .models import Segment


def merge_segments(
    whisper_segs: list,
    diarization: list[tuple[float, float, str]],
) -> list[Segment]:
    merged = []
    for seg in whisper_segs:
        if isinstance(seg, dict):
            w_start = float(seg["start"])
            w_end   = float(seg["end"])
            w_text  = str(seg["text"]).strip()
        else:
            w_start = float(seg.start)
            w_end   = float(seg.end)
            w_text  = str(seg.text).strip()

        if not w_text:
            continue

        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for d_start, d_end, speaker in diarization:
            overlap = min(w_end, d_end) - max(w_start, d_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        merged.append(Segment(
            start=w_start, end=w_end,
            speaker=best_speaker, text=w_text,
        ))
    return merged
