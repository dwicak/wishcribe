"""
wishcribe.audio
---------------
Extract a 16 kHz mono WAV from any video or audio file using moviepy + ffmpeg.

Fixes:
  - VideoFileClip memory leak: video clip properly closed in finally block
  - sys.exit() replaced with RuntimeError (library-safe, catchable by CLI)
  - Validates file extension; raises clear error for unsupported formats
  - Detects missing audio track on video files
  - Supports wide range of formats including .webm, .ts
"""
from __future__ import annotations

import os
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".wmv", ".flv"}
_ALL_EXTS   = _AUDIO_EXTS | _VIDEO_EXTS


def extract_audio(input_path: str, tmp_dir: str, verbose: bool = True) -> str:
    """
    Extract 16 kHz mono WAV from video or audio file.

    Returns path to the extracted WAV file.
    Raises RuntimeError on failure (catchable by callers, not sys.exit).
    """
    out = os.path.join(tmp_dir, "audio.wav")
    ext = Path(input_path).suffix.lower()

    if verbose:
        print(f"🎬 Extracting audio from: {Path(input_path).name}")

    if ext not in _ALL_EXTS:
        raise RuntimeError(
            f"Unsupported file format: '{ext}'\n"
            f"   Supported audio: {sorted(_AUDIO_EXTS)}\n"
            f"   Supported video: {sorted(_VIDEO_EXTS)}"
        )

    _extract_with_moviepy(input_path, out, ext)

    if verbose:
        size_mb = os.path.getsize(out) / 1_048_576
        print(f"✅ Audio extracted ({size_mb:.1f} MB)")

    return out


def _extract_with_moviepy(input_path: str, out: str, ext: str) -> None:
    """
    Extract audio using moviepy. Raises RuntimeError on failure.
    Properly closes all clips in a finally block to prevent memory leaks.
    """
    from moviepy import AudioFileClip, VideoFileClip

    video_clip = None
    audio_clip = None

    try:
        if ext in _AUDIO_EXTS:
            audio_clip = AudioFileClip(input_path)
        else:
            video_clip = VideoFileClip(input_path)
            if video_clip.audio is None:
                raise RuntimeError(
                    f"No audio track found in: {Path(input_path).name}"
                )
            audio_clip = video_clip.audio

        audio_clip.write_audiofile(
            out,
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )
    except RuntimeError:
        raise  # re-raise our own errors as-is
    except Exception as exc:
        raise RuntimeError(f"Audio extraction failed: {exc}") from exc
    finally:
        # Close in order: audio first, then video
        if audio_clip is not None:
            try:
                audio_clip.close()
            except Exception:
                pass
        if video_clip is not None:
            try:
                video_clip.close()
            except Exception:
                pass
