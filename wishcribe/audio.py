"""Extract a 16 kHz mono WAV from any video or audio file."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}


def extract_audio(input_path: str, tmp_dir: str, verbose: bool = True) -> str:
    from moviepy import AudioFileClip, VideoFileClip

    out = os.path.join(tmp_dir, "audio.wav")
    ext = Path(input_path).suffix.lower()

    if verbose:
        print(f"🎬 Extracting audio from: {Path(input_path).name}")
    try:
        clip = (
            AudioFileClip(input_path)
            if ext in _AUDIO_EXTS
            else VideoFileClip(input_path).audio
        )
        clip.write_audiofile(
            out, fps=16000, nbytes=2, codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"], logger=None,
        )
        clip.close()
    except Exception as exc:
        print(f"❌ Audio extraction failed: {exc}")
        sys.exit(1)

    if verbose:
        print("✅ Audio extracted")
    return out
