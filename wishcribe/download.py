"""
wishcribe download
------------------
Pre-download and cache all required model weights before transcribing.

Downloads:
  1. Whisper model weights   → ~/.cache/whisper/<model>.pt
  2. pyannote diarization    → ~/.cache/huggingface/hub/models--pyannote--...

After this runs, wishcribe works fully offline forever.
"""
from __future__ import annotations

import os
import sys
import shutil
from typing import Optional

from .transcribe import DEFAULT_WHISPER_MODEL
from .core import _resolve_token
from .diarize import _find_cached_model, _HF_CACHE_PATH


def download_all(
    hf_token: Optional[str] = None,
    model: str = DEFAULT_WHISPER_MODEL,
    model_path: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """
    Download and cache all model weights.

    Parameters
    ----------
    hf_token   : HuggingFace token — required to download diarization model
    model      : Whisper model to download. Default: 'large'
    model_path : Path to a local pyannote model (skips HuggingFace download)
    force      : Re-download even if models are already cached
    verbose    : Print progress messages
    """
    from .deps import ensure_dependencies
    ensure_dependencies(use_api=False)

    hf_token = _resolve_token(hf_token)

    if verbose:
        _banner(model, hf_token, model_path, force)

    whisper_ok = _download_whisper(model, force=force, verbose=verbose)
    diarize_ok = _download_diarization(hf_token, model_path, force=force, verbose=verbose)

    if verbose:
        _summary(whisper_ok, diarize_ok, model)


# ── Whisper ───────────────────────────────────────────────────────────────────

def _whisper_cache_path(model: str) -> str:
    return os.path.expanduser(f"~/.cache/whisper/{model}.pt")


def _whisper_is_cached(model: str) -> bool:
    return os.path.isfile(_whisper_cache_path(model))


def _download_whisper(model: str, force: bool = False, verbose: bool = True) -> bool:
    """Download Whisper model weights. Skips if already cached unless force=True."""
    cache = _whisper_cache_path(model)

    if _whisper_is_cached(model) and not force:
        if verbose:
            size_gb = os.path.getsize(cache) / 1_073_741_824
            print(f"✅ Whisper '{model}' already cached  ({size_gb:.1f} GB)")
            print(f"   Path: {cache}")
            print(f"   Tip: use --force to re-download")
        return True

    if verbose:
        _SIZES = {
            "tiny": "75 MB", "base": "139 MB", "small": "461 MB",
            "medium": "1.4 GB", "large": "2.9 GB",
        }
        size_str = _SIZES.get(model, "unknown size")
        action = "Re-downloading" if (force and _whisper_is_cached(model)) else "Downloading"
        print(f"\n📥 {action} Whisper '{model}' model ({size_str})...")
        print("   This may take a few minutes depending on your connection.")

    try:
        import whisper
        whisper.load_model(model)
        if verbose:
            cache_dir = os.path.expanduser("~/.cache/whisper")
            actual = next(
                (os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
                 if f.startswith(model)),
                cache,
            ) if os.path.isdir(cache_dir) else cache
            size_str = ""
            if os.path.isfile(actual):
                size_gb = os.path.getsize(actual) / 1_073_741_824
                size_str = f"  ({size_gb:.1f} GB)"
            print(f"✅ Whisper '{model}' downloaded and cached{size_str}")
            print(f"   Cache: {os.path.expanduser('~/.cache/whisper/')}")
        return True
    except Exception as exc:
        print(f"❌ Failed to download Whisper '{model}': {exc}")
        return False


# ── pyannote diarization ──────────────────────────────────────────────────────

def _download_diarization(
    hf_token: Optional[str],
    model_path: Optional[str],
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """Download pyannote diarization model. Skips if already cached unless force=True."""

    # Custom local path — no download needed
    if model_path:
        if os.path.isdir(model_path):
            if verbose:
                print(f"\n✅ Diarization model using local path: {model_path}")
            return True
        else:
            print(f"❌ model_path not found: {model_path}")
            return False

    # Already cached — skip unless force
    cached = _find_cached_model()
    if cached and not force:
        if verbose:
            print(f"\n✅ Diarization model already cached (offline)")
            print(f"   Path: {cached}")
            print(f"   Tip: use --force to re-download")
        return True

    # force=True — delete old cache first so it re-downloads fresh
    if force and os.path.isdir(_HF_CACHE_PATH):
        if verbose:
            print(f"\n🗑️  Removing old cached model: {_HF_CACHE_PATH}")
        shutil.rmtree(_HF_CACHE_PATH)

    # Need token to download
    if not hf_token:
        print("\n❌ Diarization model not cached — --hf-token required to download.")
        print()
        print("   Setup checklist:")
        print("   1. Sign up at             → https://huggingface.co/join")
        print("   2. Accept license (model) → https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("   3. Create Read token      → https://huggingface.co/settings/tokens")
        return False

    if verbose:
        print("\n📥 Downloading pyannote diarization model (~1 GB)...")
        print("   This may take a few minutes depending on your connection.")

    try:
        from pyannote.audio import Pipeline
        try:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token,
            )
        except TypeError:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                use_auth_token=hf_token,
            )
        cached = _find_cached_model()
        if verbose:
            print(f"✅ Diarization model downloaded and cached")
            if cached:
                print(f"   Path: {cached}")
        return True
    except Exception as exc:
        print(f"❌ Failed to download diarization model: {exc}")
        print("   Checklist:")
        print("   1. Valid HuggingFace token → https://huggingface.co/settings/tokens")
        print("   2. License accepted        → https://huggingface.co/pyannote/speaker-diarization-community-1")
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(model, hf_token, model_path, force=False):
    print("\n" + "═" * 62)
    print("📦  WISHCRIBE — MODEL DOWNLOADER")
    print("═" * 62)
    print(f"  Whisper model : {model}")
    if model_path:
        print(f"  Diarization   : local path → {model_path}")
    elif hf_token:
        print(f"  Diarization   : HuggingFace download (token provided)")
    else:
        print(f"  Diarization   : checking local cache...")
    if force:
        print(f"  Mode          : ⚠️  Force re-download enabled")
    print("═" * 62)


def _summary(whisper_ok: bool, diarize_ok: bool, model: str):
    print("\n" + "═" * 62)
    print("📋  DOWNLOAD SUMMARY")
    print("═" * 62)
    print(f"  Whisper '{model}'    : {'✅ Ready' if whisper_ok else '❌ Failed'}")
    print(f"  pyannote diarization: {'✅ Ready' if diarize_ok else '❌ Failed'}")
    print("═" * 62)

    if whisper_ok and diarize_ok:
        print("\n🎉 All models cached! wishcribe now works fully offline.")
        print("   Run transcription with:")
        print("   wishcribe --video meeting.mp4\n")
    else:
        print("\n⚠️  Some models failed to download. See errors above.\n")
        sys.exit(1)
