"""
Speaker diarization — fully offline after first run.

Model loading priority:
  1. Custom local path  (model_path argument / --model-path)
  2. HuggingFace cache  (~/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1)
  3. HuggingFace download via token (first-time only, then cached forever)
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_HF_CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1"
)
_SNAPSHOT_DIR = "snapshots"


def _find_cached_model() -> Optional[str]:
    """
    Look inside the HuggingFace cache and return the path to the
    latest snapshot directory, or None if not cached yet.
    """
    snapshots_dir = os.path.join(_HF_CACHE_PATH, _SNAPSHOT_DIR)
    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = [
        os.path.join(snapshots_dir, d)
        for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not snapshots:
        return None

    return max(snapshots, key=os.path.getmtime)


def run_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    model_path: Optional[str] = None,
    verbose: bool = True,
) -> list[tuple[float, float, str]]:
    """
    Run speaker diarization on a WAV file.

    Parameters
    ----------
    audio_path   : Path to 16 kHz mono WAV file
    hf_token     : HuggingFace token — only needed for first-time download
    num_speakers : Exact speaker count (optional, improves accuracy)
    model_path   : Override path to a local pyannote model folder
    verbose      : Print progress messages

    Returns
    -------
    List of (start_sec, end_sec, speaker_label) tuples
    """
    import torch
    from pyannote.audio import Pipeline

    if verbose:
        print("🔍 Running speaker diarization...")

    pipeline = _load_pipeline(
        hf_token=hf_token,
        model_path=model_path,
        verbose=verbose,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    if verbose:
        print(f"   Device: {device.upper()}")

    params = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = pipeline(audio_path, **params)

    segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    if verbose:
        n = len(set(s[2] for s in segments))
        print(f"✅ Diarization done — {n} speaker(s) detected")

    return segments


def _pipeline_from_pretrained(path_or_id: str, token: Optional[str]):
    """
    Call Pipeline.from_pretrained with token, handling API differences
    between pyannote versions gracefully.
    """
    from pyannote.audio import Pipeline
    if token:
        try:
            return Pipeline.from_pretrained(path_or_id, token=token)
        except TypeError:
            return Pipeline.from_pretrained(path_or_id, use_auth_token=token)
    return Pipeline.from_pretrained(path_or_id)


def _load_pipeline(
    hf_token: Optional[str],
    model_path: Optional[str],
    verbose: bool,
):
    """
    Load the pyannote Pipeline from the best available source.

    Priority:
      1. Explicit model_path argument   (fully offline)
      2. HuggingFace local cache        (token still passed — pyannote checks
                                         segmentation-3.0 access even for cached loads)
      3. HuggingFace download via token (online, one-time only)
    """
    from pyannote.audio import Pipeline

    # ── 1. Explicit local path ────────────────────────────────────────────
    if model_path:
        if not os.path.isdir(model_path):
            print(f"❌ model_path not found: {model_path}")
            sys.exit(1)
        if verbose:
            print(f"   Loading model from: {model_path}")
        try:
            return _pipeline_from_pretrained(model_path, hf_token)
        except Exception as exc:
            print(f"❌ Failed to load model from {model_path}: {exc}")
            sys.exit(1)

    # ── 2. HuggingFace local cache ────────────────────────────────────────
    # NOTE: Token must be passed even for cached loads — pyannote verifies
    # access to pyannote/segmentation-3.0 (a sub-model) on every load.
    # Without the token it raises a 401 error even if files are on disk.
    cached = _find_cached_model()
    if cached:
        if verbose:
            print("   Loading model from local cache (offline)")
        try:
            return _pipeline_from_pretrained(cached, hf_token)
        except Exception as exc:
            if verbose:
                print(f"   Cache load failed ({exc}), trying download...")

    # ── 3. Download via HuggingFace token (first time only) ───────────────
    if not hf_token:
        print("❌ Diarization model not found in local cache.")
        print("   Run once with --hf-token to download and cache it:")
        print("   wishcribe download --hf-token hf_xxxxxxxxxx")
        print()
        print("   Setup checklist (all required):")
        print("   1. Sign up at              → https://huggingface.co/join")
        print("   2. Accept license (model)  → https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   3. Accept license (segm.)  → https://huggingface.co/pyannote/segmentation-3.0")
        print("   4. Request access (comm.)  → https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("   4. Create Read token       → https://huggingface.co/settings/tokens")
        sys.exit(1)

    if verbose:
        print("   Downloading model from HuggingFace (one-time, cached forever after)...")

    try:
        pipeline = _pipeline_from_pretrained(
            "pyannote/speaker-diarization-3.1", hf_token
        )
        if verbose:
            print("   ✅ Model downloaded and cached — future runs work offline")
        return pipeline
    except Exception as exc:
        print(f"❌ Failed to download diarization model: {exc}")
        print("   Checklist:")
        print("   1. Valid HuggingFace token  → https://huggingface.co/settings/tokens")
        print("   2. License accepted (model) → https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   3. License accepted (segm.) → https://huggingface.co/pyannote/segmentation-3.0")
        print("   4. Request access (comm.)  → https://huggingface.co/pyannote/speaker-diarization-community-1")
        sys.exit(1)
