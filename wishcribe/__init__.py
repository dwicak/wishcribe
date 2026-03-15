"""
wishcribe
=========
Fast multi-speaker audio/video transcription.
Backend: faster-whisper (4-8x faster) + pyannote.audio (offline after first run).
Apple Silicon: MLX-Whisper auto-selected on M1/M2/M3/M4 (Neural Engine / GPU).

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

    # Accuracy options (v1.2.0)
    segments = transcribe("meeting.mp4", initial_prompt="Medical: hypertension.")
    segments = transcribe("meeting.mp4", temperature=0.2, beam_size=10)

    # Word-level timestamps in SRT/JSON (v1.3.0)
    segments = transcribe("meeting.mp4", word_timestamps=True)

    # Disable VAD if it trims real speech (v1.3.0)
    segments = transcribe("meeting.mp4", vad_filter=False)

    # Tune silence suppression (v1.3.0)
    segments = transcribe("meeting.mp4", no_speech_threshold=0.8)

Each Segment has: .start  .end  .speaker  .text  .words (when word_timestamps=True)
"""

from .core import transcribe
from .download import download_all as download
from .models import Segment

__all__ = ["transcribe", "download", "Segment"]
__version__ = "1.3.0"
