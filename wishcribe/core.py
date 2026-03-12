"""
wishcribe.core
--------------
Main pipeline: extract audio → transcribe → diarize → merge → write outputs.

Speed improvements over v1 (openai-whisper):
  - faster-whisper backend (4-8x faster via CTranslate2)
  - Batched inference (multiple audio chunks in parallel)
  - VAD pre-filtering (skips silence, reduces hallucination)
  - float16 on GPU / int8 on CPU (automatic)
  - GPU memory freed after transcription before diarization runs

Robustness improvements:
  - All errors raise exceptions (no sys.exit in library code)
  - CLI wraps calls in try/except for clean error messages
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from .deps import ensure_dependencies
from .audio import extract_audio
from .diarize import run_diarization
from .transcribe import transcribe_local, transcribe_api, DEFAULT_WHISPER_MODEL, _MODEL_INFO
from .merge import merge_segments
from .output import write_txt, write_srt, write_json
from .models import Segment


def _resolve_token(hf_token: Optional[str]) -> Optional[str]:
    """Accept HF token from argument or environment variable."""
    return hf_token or os.environ.get("WISHCRIBE_HF_TOKEN") or os.environ.get("HF_TOKEN")


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
    diarize: bool = True,
    # Speed controls (faster-whisper)
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    device: Optional[str] = None,
) -> list[Segment]:
    """
    Transcribe an audio/video file with per-speaker labels.

    Parameters
    ----------
    input_path   : Path to video or audio file (mp4, mkv, mp3, wav, m4a …)
    hf_token     : HuggingFace token. Also reads WISHCRIBE_HF_TOKEN / HF_TOKEN env.
    model        : Whisper model. Default: 'large-v2' (best accuracy).
                   Options: tiny | base | small | medium | large-v2 | large-v3
                            | turbo | distil-large-v3
    language     : BCP-47 code e.g. 'id', 'en'. None = auto-detect.
    num_speakers : Exact speaker count (improves diarization when known).
    diarize      : False = skip speaker diarization (faster, no token needed).
    batch_size   : faster-whisper batch size. Higher = faster on GPU. Default 16.
    compute_type : 'float16' (GPU), 'int8' (CPU/low-mem). Auto-detected.
    device       : 'cuda' or 'cpu'. Auto-detected.
    output_dir   : Where to save files. Default: same folder as input.
    use_api      : Use OpenAI Whisper API instead of local model.
    api_key      : OpenAI API key (required when use_api=True).
    save_txt     : Write <stem>_transcript.txt
    save_srt     : Write <stem>_transcript.srt
    save_json    : Write <stem>_transcript.json
    verbose      : Print progress to stdout.

    Returns
    -------
    List of Segment(start, end, speaker, text)

    Raises
    ------
    FileNotFoundError  : if input_path does not exist
    ValueError         : if use_api=True but no api_key given
    RuntimeError       : on audio extraction, transcription, or diarization failure
    """
    ensure_dependencies(use_api=use_api)

    hf_token = _resolve_token(hf_token)

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if use_api and not api_key:
        raise ValueError("api_key is required when use_api=True")

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    if verbose:
        _banner(input_path.name, model, language, use_api, num_speakers,
                hf_token, model_path, diarize, batch_size, compute_type, device)

    with tempfile.TemporaryDirectory() as tmp:
        # Step 1: Extract audio
        audio_path = extract_audio(str(input_path), tmp, verbose=verbose)

        # Step 2: Transcribe
        # Run FIRST so GPU memory is freed before diarization loads.
        if use_api:
            whisper_segs = transcribe_api(audio_path, api_key, language, verbose=verbose)
        else:
            whisper_segs = transcribe_local(
                audio_path, model, language, verbose=verbose,
                batch_size=batch_size, compute_type=compute_type, device=device,
            )

        # Step 3: Diarize (after transcription — GPU memory now freed)
        diarization = None
        if diarize:
            diarization = run_diarization(
                audio_path,
                hf_token=hf_token,
                num_speakers=num_speakers,
                model_path=model_path,
                verbose=verbose,
            )

    # Step 4: Merge (audio tempdir now cleaned up — merge uses in-memory data only)
    if verbose:
        print("\n🔗 Merging transcription with speaker labels...")

    segments = merge_segments(whisper_segs, diarization)

    if verbose:
        print(f"✅ Merged {len(segments)} segments")

    # Step 5: Write outputs
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

def _banner(name, model, language, use_api, num_speakers, hf_token, model_path,
            diarize, batch_size, compute_type, device):
    if not diarize:
        diarize_str = "disabled (--no-diarize)"
    elif model_path:
        diarize_str = f"Local path: {model_path}"
    elif hf_token:
        diarize_str = "HuggingFace (token provided)"
    else:
        diarize_str = "Local cache (offline)"

    model_desc = _MODEL_INFO.get(model, model)
    _device  = device or "auto (cuda if available, else cpu)"
    _compute = compute_type or "auto (float16/GPU, int8/CPU)"

    print("\n" + "═" * 64)
    print("⚡  WISHCRIBE — FAST MULTI-SPEAKER TRANSCRIBER")
    print("═" * 64)
    print(f"  File       : {name}")
    print(f"  Whisper    : {model}  ({model_desc})")
    print(f"  Backend    : faster-whisper + batched inference + VAD")
    print(f"  Device     : {_device}")
    print(f"  Compute    : {_compute}  |  Batch size: {batch_size}")
    print(f"  Language   : {language or 'auto-detect'}")
    print(f"  Transcribe : {'OpenAI API' if use_api else 'Local'}")
    print(f"  Diarize    : {diarize_str}")
    if num_speakers:
        print(f"  Speakers   : {num_speakers} (specified)")
    print("═" * 64 + "\n")


def _print_summary(segments: list[Segment]):
    from collections import Counter
    from .utils import fmt_time

    print("\n" + "═" * 64)
    print("📋  TRANSCRIPT PREVIEW")
    print("═" * 64)
    prev = None
    for seg in segments[:25]:
        if seg.speaker != prev:
            label = f"[{seg.speaker}] " if seg.speaker else ""
            print(f"\n\033[1m{label}\033[0m{fmt_time(seg.start)}")
            prev = seg.speaker
        print(f"  {seg.text}")
    if len(segments) > 25:
        print(f"\n  … ({len(segments) - 25} more segments in output files)")
    print("═" * 64)

    # Speaker stats (only for diarized output)
    labeled = [s for s in segments if s.speaker]
    if labeled:
        counts = Counter(s.speaker for s in labeled)
        print("\n📊  SPEAKER STATS")
        for speaker, count in sorted(counts.items()):
            total = sum(s.end - s.start for s in labeled if s.speaker == speaker)
            print(f"  {speaker}: {count} segments, ~{fmt_time(total)} of speech")
    print()
