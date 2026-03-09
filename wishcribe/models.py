"""Data models."""
from dataclasses import dataclass


@dataclass
class Segment:
    """A single transcribed segment with speaker label and timestamps."""
    start:   float
    end:     float
    speaker: str
    text:    str

    def to_dict(self):
        return {
            "start":   self.start,
            "end":     self.end,
            "speaker": self.speaker,
            "text":    self.text,
        }
