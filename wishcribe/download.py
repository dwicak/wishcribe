"""
wishcribe.download
------------------
Pre-download and cache all required model weights before transcribing.

Downloads:
  1. faster-whisper model  → ~/.cache/huggingface/hub/models--Systran--faster-whisper-*/
  2. pyannote diarization  → ~/.cache/huggingface/hub/models--pyannote--...

After this runs, wishcribe works fully offline forever.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

from .transcribe import DEFAULT_WHISPER_MODEL, _FW_MODEL_MAP
from .diarize import _find_cached_model, _HF_CACHE_PATH

_FW_CACHE_BASE = os.path.expanduser("~/.cache/huggingface/hub")


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
    hf_token   : HuggingFace token — required if diarization model not cached yet
    model      : Whisper model to download. Default: 'large-v2'
    model_path : Path to a local pyannote model (skips HuggingFace download)
    force      : Delete existing cache and re-download fresh
    verbose    : Print progress messages

    Raises
    ------
    RuntimeError if any required download fails
    """
    from .deps import ensure_dependencies
    ensure_dependencies(use_api=False)

    if verbose:
        _banner(model, hf_token, model_path, force)

    if force:
        _purge_caches(model, verbose=verbose)

    whisper_ok = _download_whisper(model, verbose=verbose)
    diarize_ok = _download_diarization(hf_token, model_path, verbose=verbose)

    if verbose:
        _summary(whisper_ok, diarize_ok, model)

    # Raise instead of sys.exit so CLI can catch and handle cleanly
    if not (whisper_ok and diarize_ok):
        raise RuntimeError("One or more model downloads failed. See errors above.")


# ── faster-whisper model cache ────────────────────────────────────────────────

def _fw_cache_dir(model: str) -> str:
    """HuggingFace hub cache directory for a faster-whisper model."""
    fw_model = _FW_MODEL_MAP.get(model, model)
    safe = fw_model.replace("/", "--")
    return os.path.join(_FW_CACHE_BASE, f"models--Systran--faster-whisper-{safe}")


def _openai_whisper_cache(model: str) -> str:
    return os.path.expanduser(f"~/.cache/whisper/{model}.pt")


def _whisper_is_cached(model: str) -> bool:
    fw_dir = _fw_cache_dir(model)
    ow_path = _openai_whisper_cache(model)
    ow_alias = _openai_whisper_cache(_FW_MODEL_MAP.get(model, model))

    # Check faster-whisper cache — use context manager to properly close scandir
    if os.path.isdir(fw_dir):
        with os.scandir(fw_dir) as it:
            if any(it):
                return True

    return os.path.isfile(ow_path) or os.path.isfile(ow_alias)


def _download_whisper(model: str, verbose: bool = True) -> bool:
    if _whisper_is_cached(model):
        if verbose:
            print(f"✅ Whisper '{model}' already cached")
        return True

    _SIZES = {
        "tiny": "75 MB", "base": "139 MB", "small": "461 MB",
        "medium": "1.4 GB", "large-v2": "2.9 GB", "large-v3": "3.1 GB",
        "turbo": "1.6 GB", "distil-large-v3": "1.5 GB", "large": "2.9 GB",
    }
    if verbose:
        print(f"\n📥 Downloading faster-whisper '{model}' ({_SIZES.get(model, '~2-3 GB')})...")

    try:
        from faster_whisper import WhisperModel
        fw_model = _FW_MODEL_MAP.get(model, model)
        WhisperModel(fw_model, device="cpu", compute_type="int8")
        if verbose:
            print(f"✅ faster-whisper '{model}' downloaded and cached")
        return True
    except ImportError:
        # Fall back to openai-whisper
        try:
            import whisper
            ow = {"large-v2": "large", "large-v3": "large", "turbo": "large"}.get(model, model)
            whisper.load_model(ow)
            if verbose:
                print(f"✅ openai-whisper '{ow}' downloaded and cached")
            return True
        except Exception as exc:
            print(f"❌ Failed to download Whisper '{model}': {exc}")
            return False
    except Exception as exc:
        print(f"❌ Failed to download faster-whisper '{model}': {exc}")
        return False


# ── pyannote diarization ──────────────────────────────────────────────────────

def _download_diarization(
    hf_token: Optional[str],
    model_path: Optional[str],
    verbose: bool = True,
) -> bool:
    if model_path:
        if os.path.isdir(model_path):
            if verbose:
                print(f"\n✅ Diarization using local path: {model_path}")
            return True
        print(f"❌ model_path not found: {model_path}")
        return False

    cached = _find_cached_model()
    if cached:
        if verbose:
            print(f"\n✅ Diarization model already cached (offline)")
            print(f"   Path: {cached}")
        return True

    if not hf_token:
        print("\n❌ Diarization model not cached — --hf-token required.")
        print()
        print("   Setup checklist:")
        print("   1. Sign up at             → https://huggingface.co/join")
        print("   2. Accept license         → https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("   3. Create Read token      → https://huggingface.co/settings/tokens")
        return False

    if verbose:
        print("\n📥 Downloading pyannote diarization model (~1 GB)...")

    try:
        from pyannote.audio import Pipeline
        try:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=hf_token
            )
        except TypeError:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", use_auth_token=hf_token
            )
        cached = _find_cached_model()
        if verbose:
            print("✅ Diarization model downloaded and cached")
            if cached:
                print(f"   Path: {cached}")
        return True
    except Exception as exc:
        print(f"❌ Failed to download diarization model: {exc}")
        return False


# ── Force re-download ─────────────────────────────────────────────────────────

def _purge_caches(model: str, verbose: bool = True) -> None:
    for path in [_fw_cache_dir(model), _openai_whisper_cache(model), str(_HF_CACHE_PATH)]:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            if verbose:
                print(f"🗑️  Deleted: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            if verbose:
                print(f"🗑️  Deleted: {path}")


# ── Display helpers ───────────────────────────────────────────────────────────

def _banner(model, hf_token, model_path, force):
    print("\n" + "═" * 64)
    print("📦  WISHCRIBE — MODEL DOWNLOADER")
    print("═" * 64)
    print(f"  Whisper model : {model} (faster-whisper)")
    if model_path:
        print(f"  Diarization   : local path → {model_path}")
    elif hf_token:
        print(f"  Diarization   : HuggingFace download (token provided)")
    else:
        print(f"  Diarization   : checking local cache...")
    if force:
        print(f"  Force         : ⚠️  existing cache will be deleted first")
    print("═" * 64)


def _summary(whisper_ok: bool, diarize_ok: bool, model: str):
    print("\n" + "═" * 64)
    print("📋  DOWNLOAD SUMMARY")
    print("═" * 64)
    print(f"  faster-whisper '{model}'  : {'✅ Ready' if whisper_ok else '❌ Failed'}")
    print(f"  pyannote diarization     : {'✅ Ready' if diarize_ok else '❌ Failed'}")
    print("═" * 64)
    if whisper_ok and diarize_ok:
        print("\n🎉 All models cached! wishcribe now works fully offline.")
        print("   Run:  wishcribe --video meeting.mp4\n")
    else:
        print("\n⚠️  Some models failed to download. See errors above.\n")
