"""
wishcribe.merge
---------------
Merge Whisper segments with diarization speaker labels by maximum time overlap.

Fixes over v1:
  - Default speaker was "SPEAKER_00" (wrong) — now "SPEAKER_??" for no-overlap case
  - Early-exit on non-overlapping diarization segments (faster)
  - "UNKNOWN" label renamed to "SPEAKER_??" (cleaner output)
  - Handles both dict-style and object-style Whisper segments
"""
from __future__ import annotations

from .models import Segment

_NO_OVERLAP_LABEL = "SPEAKER_??"


def merge_segments(
    whisper_segs: list,
    diarization: list[tuple[float, float, str]] | None,
) -> list[Segment]:
    """
    Merge Whisper transcript segments with pyannote speaker labels.

    Strategy: for each whisper segment, find the diarization turn with the
    greatest time overlap and assign its speaker label.

    Parameters
    ----------
    whisper_segs  : List of dicts or objects with start/end/text fields.
                    Optional 'words' key carries word-level timestamps (item 7).
    diarization   : List of (start, end, speaker) tuples from pyannote.
                    None = --no-diarize mode; segments returned without speaker labels.

    Returns
    -------
    List of Segment(start, end, speaker, text, words=...)
    """
    merged = []

    for seg in whisper_segs:
        # Support both dict and object-style segments
        if isinstance(seg, dict):
            w_start = float(seg["start"])
            w_end   = float(seg["end"])
            w_text  = str(seg.get("text", "")).strip()
            w_words = seg.get("words")  # item 7: carry word timestamps through
        else:
            w_start = float(seg.start)
            w_end   = float(seg.end)
            w_text  = str(seg.text).strip()
            w_words = getattr(seg, "words", None)

        if not w_text:
            continue  # skip empty segments (VAD artifact or silence)

        if diarization is None:
            # --no-diarize mode: empty speaker = hidden in output
            merged.append(Segment(start=w_start, end=w_end, speaker="", text=w_text,
                                  words=w_words))
            continue

        # Find speaker with maximum overlap
        best_speaker = _NO_OVERLAP_LABEL
        best_overlap = 0.0

        for d_start, d_end, speaker in diarization:
            # Fast reject: no overlap at all
            if d_end <= w_start or d_start >= w_end:
                continue
            overlap = min(w_end, d_end) - max(w_start, d_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        merged.append(Segment(
            start=w_start, end=w_end,
            speaker=best_speaker, text=w_text,
            words=w_words,
        ))

    return merged
