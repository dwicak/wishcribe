"""
wishcribe.diarize
-----------------
Speaker diarization — fully offline after first run.

Uses pyannote/speaker-diarization-community-1 (better accuracy than 3.1,
only requires 1 license acceptance).

Model loading priority:
  1. Custom local path  (model_path argument)
  2. HuggingFace cache  (~/.cache/huggingface/hub/models--pyannote--...)
  3. HuggingFace download via token (first-time only, then cached forever)

Fixes over v1:
  - sys.exit() replaced with RuntimeError (library-safe, catchable)
  - Handles both community-1 (DiarizeOutput) and 3.1 (Annotation) APIs
"""
from __future__ import annotations

import contextlib
import gc
import os
from typing import Optional


@contextlib.contextmanager
def _suppress_fd2():
    """
    Redirect OS-level stderr (fd 2) to /dev/null for the duration of the block.

    On macOS, onnxruntime (loaded by pyannote) and the AV framework print
    objc[] / AVFFrameReceiver duplicate-dylib warnings on every runSession call.
    These come from the ObjC/C runtime and write directly to file descriptor 2,
    bypassing Python's sys.stderr entirely.  Redirecting fd 2 is the only
    reliable way to silence them.

    Degrades gracefully — if /dev/null cannot be opened, yields without
    suppression rather than crashing the pipeline.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        saved_fd2 = os.dup(2)
    except OSError:
        os.close(devnull_fd)
        yield
        return
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd2, 2)
        os.close(saved_fd2)
        os.close(devnull_fd)


_MODEL_ID = "pyannote/speaker-diarization-community-1"
_HF_CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1"
)
_SNAPSHOT_DIR = "snapshots"


def _find_cached_model() -> Optional[str]:
    """Return path to the newest cached model snapshot, or None."""
    snapshots_dir = os.path.join(_HF_CACHE_PATH, _SNAPSHOT_DIR)
    if not os.path.isdir(snapshots_dir):
        return None
    snapshots = [
        os.path.join(snapshots_dir, d)
        for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    return max(snapshots, key=os.path.getmtime) if snapshots else None


def _extract_segments(diarization) -> list[tuple[float, float, str]]:
    """
    Extract (start, end, speaker) tuples from pyannote diarization output.

    Supports:
      - community-1: DiarizeOutput with .speaker_diarization (segment, speaker) pairs
      - legacy 3.1:  Annotation with .itertracks(yield_label=True)
    """
    if hasattr(diarization, "speaker_diarization"):
        return [
            (turn.start, turn.end, speaker)
            for turn, speaker in diarization.speaker_diarization
        ]
    if hasattr(diarization, "itertracks"):
        return [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    raise RuntimeError(
        f"Cannot parse diarization output of type {type(diarization)}.\n"
        "   Please open an issue: https://github.com/dwicak/wishcribe/issues\n"
        f"   Available attributes: {[a for a in dir(diarization) if not a.startswith('__')]}"
    )


def run_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    model_path: Optional[str] = None,
    verbose: bool = True,
) -> list[tuple[float, float, str]]:
    """
    Run speaker diarization on a WAV file.
    Returns list of (start, end, speaker_label) tuples.
    Raises RuntimeError on failure (instead of sys.exit).
    """
    import torch

    if verbose:
        print("🔍 Running speaker diarization...")

    pipeline = _load_pipeline(hf_token=hf_token, model_path=model_path, verbose=verbose)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU (Metal Performance Shaders)
    pipeline.to(torch.device(device))
    if verbose:
        print(f"   Device: {device.upper()}")

    params = {}
    if num_speakers:
        params["num_speakers"] = num_speakers

    # Suppress macOS objc[] / AVFFrameReceiver duplicate-dylib warnings.
    # onnxruntime (loaded by pyannote) fires these on every runSession call
    # by writing directly to OS fd 2 — not catchable via Python's sys.stderr.
    with _suppress_fd2():
        diarization = pipeline(audio_path, **params)

    # Extract segments BEFORE deleting pipeline/diarization so that if
    # _extract_segments raises (unknown output format), the finally block
    # still frees GPU/MPS/unified memory — no leak on error paths.
    try:
        segments = _extract_segments(diarization)
    finally:
        del pipeline, diarization
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    if verbose:
        n_speakers = len(set(s[2] for s in segments))
        print(f"✅ Diarization done — {n_speakers} speaker(s), {len(segments)} segments")

    return segments


def _pipeline_from_pretrained(path_or_id: str, token: Optional[str]):
    """Load pyannote Pipeline — tries modern token= kwarg, falls back to use_auth_token=."""
    from pyannote.audio import Pipeline
    if token:
        try:
            return Pipeline.from_pretrained(path_or_id, token=token)
        except TypeError:
            return Pipeline.from_pretrained(path_or_id, use_auth_token=token)
    return Pipeline.from_pretrained(path_or_id)


def _load_pipeline(hf_token: Optional[str], model_path: Optional[str], verbose: bool):
    """Load diarization pipeline — local path → cache → download."""

    # 1. Explicit local path
    if model_path:
        if not os.path.isdir(model_path):
            raise RuntimeError(f"model_path not found: {model_path}")
        if verbose:
            print(f"   Loading model from: {model_path}")
        try:
            return _pipeline_from_pretrained(model_path, hf_token)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc

    # 2. HuggingFace local cache (offline)
    cached = _find_cached_model()
    if cached:
        if verbose:
            print("   Loading model from local cache (offline)")
        try:
            return _pipeline_from_pretrained(cached, hf_token)
        except Exception as exc:
            if verbose:
                print(f"   Cache load failed ({exc}), trying download...")

    # 3. Download via HuggingFace token (first time only)
    if not hf_token:
        raise RuntimeError(
            "Diarization model not found in local cache.\n"
            "   Run once with --hf-token to download and cache it:\n"
            "   wishcribe download --hf-token hf_xxxxxxxxxx\n\n"
            "   Setup checklist:\n"
            "   1. Sign up at             → https://huggingface.co/join\n"
            "   2. Accept license (model) → https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "   3. Create Read token      → https://huggingface.co/settings/tokens"
        )

    if verbose:
        print("   Downloading model from HuggingFace (one-time, cached forever after)...")

    try:
        pipeline = _pipeline_from_pretrained(_MODEL_ID, hf_token)
        if verbose:
            print("   ✅ Model downloaded and cached — future runs work offline")
        return pipeline
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download diarization model: {exc}\n"
            "   Checklist:\n"
            "   1. Valid HuggingFace token → https://huggingface.co/settings/tokens\n"
            "   2. License accepted        → https://huggingface.co/pyannote/speaker-diarization-community-1"
        ) from exc
