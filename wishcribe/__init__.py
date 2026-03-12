"""
wishcribe
=========
Fast multi-speaker audio/video transcription.
Backend: faster-whisper (4-8x faster) + pyannote.audio (offline after first run).

Quick start
-----------
    # Step 1 — download all models once
    from wishcribe import download
    download(hf_token="hf_xxx")

    # Step 2 — transcribe (fully offline forever after)
    from wishcribe import transcribe
    segments = transcribe("meeting.mp4")

    # Without speaker labels (no HuggingFace token needed)
    segments = transcribe("meeting.mp4", diarize=False)

    # Speed options
    segments = transcribe("meeting.mp4", batch_size=16, compute_type="float16")

Each Segment has: .start  .end  .speaker  .text
"""

from .core import transcribe
from .download import download_all as download
from .models import Segment

__all__ = ["transcribe", "download", "Segment"]
__version__ = "1.1.0"
