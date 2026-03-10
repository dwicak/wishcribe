"""Merge Whisper segments with diarization speaker labels by time overlap."""
from __future__ import annotations

from .models import Segment


def merge_segments(
    whisper_segs: list,
    diarization: list[tuple[float, float, str]] | None,
) -> list[Segment]:
    """
    Merge Whisper segments with speaker labels.
    If diarization is None, segments are returned without speaker labels
    (speaker field will be empty string).
    """
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

        if diarization is None:
            # No diarization — store without speaker label
            merged.append(Segment(
                start=w_start, end=w_end,
                speaker="", text=w_text,
            ))
        else:
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
