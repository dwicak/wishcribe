"""
wishcribe.core
--------------
Main pipeline: extract audio → diarize → transcribe → merge → write outputs.
Default Whisper model: large (best accuracy, 2.9 GB).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from .deps import ensure_dependencies
from .audio import extract_audio
from .diarize import run_diarization
from .transcribe import transcribe_local, transcribe_api, DEFAULT_WHISPER_MODEL
from .merge import merge_segments
from .output import write_txt, write_srt, write_json
from .models import Segment

# Whisper model sizes ranked by accuracy (for display in banner)
_MODEL_INFO = {
    "tiny":   "75 MB  — fastest, fair accuracy",
    "base":   "139 MB — fast, good accuracy",
    "small":  "461 MB — moderate speed, better accuracy",
    "medium": "1.4 GB — slow, very good accuracy",
    "large":  "2.9 GB — slowest, BEST accuracy ⭐ (default)",
}


def transcribe(
    input_path: str,
    *,
    hf_token: Optional[str] = None,
    model_path: Optional[str] = None,
    model: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    output_dir: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    save_txt: bool = True,
    save_srt: bool = True,
    save_json: bool = False,
    verbose: bool = True,
) -> list[Segment]:
    """
    Transcribe an audio/video file with per-speaker labels.

    Parameters
    ----------
    input_path   : Path to video or audio file (mp4, mkv, mp3, wav, m4a …)
    hf_token     : HuggingFace token — only needed ONCE to download the
                   diarization model. Cached forever after first run.
    model_path   : Path to a manually downloaded local pyannote model folder.
    model        : Whisper model size. Default: 'large' (best accuracy).
                   Options: 'tiny' | 'base' | 'small' | 'medium' | 'large'
    language     : BCP-47 code e.g. 'id', 'en'. None = auto-detect.
    num_speakers : Exact speaker count (optional, improves diarization accuracy)
    output_dir   : Where to save output files. Default: same folder as input.
    use_api      : Use OpenAI Whisper API instead of local model
    api_key      : OpenAI API key (required when use_api=True)
    save_txt     : Write <stem>_transcript.txt
    save_srt     : Write <stem>_transcript.srt
    save_json    : Write <stem>_transcript.json
    verbose      : Print progress to stdout

    Returns
    -------
    List of Segment(start, end, speaker, text)
    """
    ensure_dependencies(use_api=use_api)

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if use_api and not api_key:
        raise ValueError("api_key is required when use_api=True")

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    if verbose:
        _banner(input_path.name, model, language, use_api, num_speakers, hf_token, model_path)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = extract_audio(str(input_path), tmp, verbose=verbose)
        diarization = run_diarization(
            audio_path,
            hf_token=hf_token,
            num_speakers=num_speakers,
            model_path=model_path,
            verbose=verbose,
        )
        if use_api:
            whisper_segs = transcribe_api(audio_path, api_key, language, verbose=verbose)
        else:
            whisper_segs = transcribe_local(audio_path, model, language, verbose=verbose)

    if verbose:
        print("\n🔗 Merging transcription with speaker labels...")

    segments = merge_segments(whisper_segs, diarization)

    if verbose:
        print(f"✅ Merged {len(segments)} segments")

    if save_txt:
        write_txt(segments, str(out_dir / f"{stem}_transcript.txt"))
    if save_srt:
        write_srt(segments, str(out_dir / f"{stem}_transcript.srt"))
    if save_json:
        write_json(segments, str(out_dir / f"{stem}_transcript.json"))

    if verbose:
        _print_summary(segments)

    return segments


# ── Internal helpers ──────────────────────────────────────────────────────────

def _banner(name, model, language, use_api, num_speakers, hf_token, model_path):
    if model_path:
        diarize_str = f"Local path: {model_path}"
    elif hf_token:
        diarize_str = "HuggingFace download (first time) → cached offline after"
    else:
        diarize_str = "Offline (local cache)"

    model_desc = _MODEL_INFO.get(model, model)

    print("\n" + "═" * 62)
    print("✍️   WISHCRIBE — MULTI-SPEAKER TRANSCRIBER")
    print("═" * 62)
    print(f"  File      : {name}")
    print(f"  Whisper   : {model}  ({model_desc})")
    print(f"  Language  : {language or 'auto-detect'}")
    print(f"  Transcribe: {'OpenAI API' if use_api else 'Local'}")
    print(f"  Diarize   : {diarize_str}")
    if num_speakers:
        print(f"  Speakers  : {num_speakers} (specified)")
    print("═" * 62 + "\n")


def _print_summary(segments: list[Segment]):
    from collections import Counter
    from .utils import fmt_time

    print("\n" + "═" * 62)
    print("📋  TRANSCRIPT PREVIEW")
    print("═" * 62)
    prev = None
    for seg in segments[:25]:
        if seg.speaker != prev:
            print(f"\n\033[1m[{seg.speaker}]\033[0m {fmt_time(seg.start)}")
            prev = seg.speaker
        print(f"  {seg.text}")
    if len(segments) > 25:
        print(f"\n  … ({len(segments) - 25} more segments in output files)")
    print("═" * 62)

    counts = Counter(s.speaker for s in segments)
    print("\n📊  SPEAKER STATS")
    for speaker, count in sorted(counts.items()):
        total = sum(s.end - s.start for s in segments if s.speaker == speaker)
        print(f"  {speaker}: {count} segments, ~{fmt_time(total)} of speech")
    print()
