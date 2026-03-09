"""
wishcribe
=========
Multi-speaker audio/video transcription — Whisper + pyannote.audio (fully offline).
Default Whisper model: large (best accuracy).

Quick start
-----------
    # Step 1 — download all models once
    from wishcribe import download
    download(hf_token="hf_xxx")

    # Step 2 — transcribe (fully offline forever after)
    from wishcribe import transcribe
    segments = transcribe("meeting.mp4")

Each Segment has: .start  .end  .speaker  .text
"""

from .core import transcribe
from .download import download_all as download
from .models import Segment

__all__ = ["transcribe", "download", "Segment"]
__version__ = "1.0.0"
