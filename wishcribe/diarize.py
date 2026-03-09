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


def _extract_segments(diarization) -> list[tuple[float, float, str]]:
    """
    Extract (start, end, speaker) tuples from pyannote diarization output.
    Handles both old API (.itertracks) and new API (DiarizeOutput / iterable).
    """
    segments = []

    # ── New pyannote API: DiarizeOutput (iterable of named tuples) ────────
    if hasattr(diarization, '__iter__') and not hasattr(diarization, 'itertracks'):
        try:
            for item in diarization:
                # Each item may be a named tuple: (segment, track, label)
                # or just (segment, label) depending on version
                if hasattr(item, 'segment') and hasattr(item, 'label'):
                    segments.append((item.segment.start, item.segment.end, item.label))
                elif len(item) == 3:
                    turn, _, speaker = item
                    segments.append((turn.start, turn.end, speaker))
                elif len(item) == 2:
                    turn, speaker = item
                    segments.append((turn.start, turn.end, speaker))
            if segments:
                return segments
        except Exception:
            pass

    # ── Old pyannote API: Annotation with .itertracks() ───────────────────
    if hasattr(diarization, 'itertracks'):
        return [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    # ── Fallback: try iterating as (segment, track, label) ────────────────
    try:
        for turn, _, speaker in diarization:
            segments.append((turn.start, turn.end, speaker))
        if segments:
            return segments
    except Exception:
        pass

    # ── Last resort: check for .labels() and .get_timeline() ─────────────
    try:
        for speaker in diarization.labels():
            for segment in diarization.label_timeline(speaker):
                segments.append((segment.start, segment.end, speaker))
        return segments
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot parse diarization output of type {type(diarization)}. "
        "Please open an issue at https://github.com/dwicak/wishcribe/issues"
    )


def run_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    model_path: Optional[str] = None,
    verbose: bool = True,
) -> list[tuple[float, float, str]]:
    import torch

    if verbose:
        print("🔍 Running speaker diarization...")

    pipeline = _load_pipeline(hf_token=hf_token, model_path=model_path, verbose=verbose)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    if verbose:
        print(f"   Device: {device.upper()}")

    params = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = pipeline(audio_path, **params)

    segments = _extract_segments(diarization)

    if verbose:
        n = len(set(s[2] for s in segments))
        print(f"✅ Diarization done — {n} speaker(s) detected")

    return segments


def _pipeline_from_pretrained(path_or_id: str, token: Optional[str]):
    from pyannote.audio import Pipeline
    if token:
        try:
            return Pipeline.from_pretrained(path_or_id, token=token)
        except TypeError:
            return Pipeline.from_pretrained(path_or_id, use_auth_token=token)
    return Pipeline.from_pretrained(path_or_id)


def _load_pipeline(hf_token: Optional[str], model_path: Optional[str], verbose: bool):
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
    # Token must be passed even for cached loads — pyannote verifies access
    # to segmentation-3.0 sub-model on every load.
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
        print("   5. Create Read token       → https://huggingface.co/settings/tokens")
        sys.exit(1)

    if verbose:
        print("   Downloading model from HuggingFace (one-time, cached forever after)...")

    try:
        pipeline = _pipeline_from_pretrained("pyannote/speaker-diarization-3.1", hf_token)
        if verbose:
            print("   ✅ Model downloaded and cached — future runs work offline")
        return pipeline
    except Exception as exc:
        print(f"❌ Failed to download diarization model: {exc}")
        print("   Checklist:")
        print("   1. Valid HuggingFace token  → https://huggingface.co/settings/tokens")
        print("   2. License accepted (model) → https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   3. License accepted (segm.) → https://huggingface.co/pyannote/segmentation-3.0")
        print("   4. Request access (comm.)   → https://huggingface.co/pyannote/speaker-diarization-community-1")
        sys.exit(1)
