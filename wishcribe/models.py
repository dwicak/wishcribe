"""Data models."""
from dataclasses import dataclass


@dataclass
class Segment:
    """A single transcribed segment with speaker label and timestamps."""
    start:   float
    end:     float
    speaker: str
    text:    str

    def to_dict(self) -> dict:
        """
        Return segment as a plain dict.
        Includes 'duration' for convenience (end - start, rounded to 3 decimal places).
        """
        return {
            "start":    self.start,
            "end":      self.end,
            "duration": round(self.end - self.start, 3),
            "speaker":  self.speaker,
            "text":     self.text,
        }
