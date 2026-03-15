"""Data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Segment:
    """A single transcribed segment with speaker label and timestamps."""
    start:   float
    end:     float
    speaker: str
    text:    str
    # Item 7: word-level timestamps.  Each entry is a dict:
    #   {"word": str, "start": float, "end": float, "probability": float}
    # None means word timestamps were not requested or not supported by the
    # active backend.  Must be the last field (dataclass default-value ordering).
    words: Optional[list[dict]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """
        Return segment as a plain dict.
        Includes 'duration' for convenience (end - start, rounded to 3 decimal places).
        'words' is included only when word timestamps are present.
        """
        d = {
            "start":    self.start,
            "end":      self.end,
            "duration": round(self.end - self.start, 3),
            "speaker":  self.speaker,
            "text":     self.text,
        }
        if self.words is not None:
            d["words"] = self.words
        return d
