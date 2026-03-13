"""
wishcribe.download
------------------
Pre-download and cache all required model weights before transcribing.

Downloads:
  1. faster-whisper model  → ~/.cache/huggingface/hub/
                             models--Systran--faster-whisper-*      (most models)
                             models--mobiuslabsgmbh--faster-whisper-* (turbo)
  2. pyannote diarization  → ~/.cache/huggingface/hub/models--pyannote--...

After this runs, wishcribe works fully offline forever.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

from .transcribe import DEFAULT_WHISPER_MODEL, _FW_MODEL_MAP
from .diarize import _find_cached_model

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
    model      : Whisper model to download. Default: 'large-v2' (or 'turbo' on Apple Silicon)
    model_path : Path to a local pyannote model (skips HuggingFace download)
    force      : Delete existing cache and re-download fresh
    verbose    : Print progress messages

    Raises
    ------
    RuntimeError if any required download fails
    """
    from .deps import ensure_dependencies
    from .transcribe import _is_apple_silicon, DEFAULT_WHISPER_MODEL_APPLE
    ensure_dependencies(use_api=False)

    # On Apple Silicon, transcription defaults to turbo — mirror that here so
    # `wishcribe download` (no --model flag) pre-caches the model that will actually
    # be used, rather than large-v2 which is the non-Apple default.
    if model == DEFAULT_WHISPER_MODEL and _is_apple_silicon():
        model = DEFAULT_WHISPER_MODEL_APPLE

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
    # Most faster-whisper models live under Systran/, but turbo uses mobiuslabsgmbh/
    # Check the resolved model ID (not the raw alias) so this works correctly
    # regardless of whether the caller used 'turbo' or 'large-v3-turbo'.
    if fw_model == "large-v3-turbo":
        return os.path.join(_FW_CACHE_BASE, f"models--mobiuslabsgmbh--faster-whisper-{safe}")
    return os.path.join(_FW_CACHE_BASE, f"models--Systran--faster-whisper-{safe}")


def _openai_whisper_cache(model: str) -> str:
    return os.path.expanduser(f"~/.cache/whisper/{model}.pt")


def _whisper_is_cached(model: str) -> bool:
    """
    Return True only if the model is fully cached with model.bin present.

    On Apple Silicon with mlx-whisper installed, returns True immediately —
    MLX models are downloaded automatically from HuggingFace on first use
    and do not live in the faster-whisper cache directories.

    For faster-whisper: checks for model.bin inside the HuggingFace hub
    snapshots directory. A directory that exists but has no model.bin
    (partial/interrupted download) returns False so the user is prompted
    to re-download rather than hitting a cryptic runtime error.
    """
    # MLX path: models are fetched at runtime by HuggingFace — no pre-check needed
    from .transcribe import _is_apple_silicon
    if _is_apple_silicon():
        try:
            import mlx_whisper  # noqa: F401
            return True  # MLX will handle download at transcription time
        except ImportError:
            pass  # fall through to faster-whisper cache check
    fw_dir = _fw_cache_dir(model)

    # Check faster-whisper cache: look for model.bin in any snapshot
    if os.path.isdir(fw_dir):
        snapshots_dir = os.path.join(fw_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                model_bin = os.path.join(snapshots_dir, snapshot, "model.bin")
                if os.path.isfile(model_bin):
                    return True
        # No model.bin found in any snapshot — treat as not cached

    # Check openai-whisper flat cache (fallback backend).
    # openai-whisper stores all large variants (large-v1/v2/v3, turbo) as "large.pt".
    _OW_ALIAS = {"large-v1": "large", "large-v2": "large", "large-v3": "large", "turbo": "large"}
    ow_name = _OW_ALIAS.get(model, _FW_MODEL_MAP.get(model, model))
    ow_path = _openai_whisper_cache(ow_name)
    return os.path.isfile(ow_path)


def _download_whisper(model: str, verbose: bool = True) -> bool:
    from .transcribe import _is_apple_silicon, _mlx_model_id

    # ── Apple Silicon + MLX path ──────────────────────────────────────────────
    # On Apple Silicon with mlx-whisper installed, the MLX backend handles
    # transcription. _whisper_is_cached() bypasses the faster-whisper check in
    # this case (MLX models live in a different cache location). We explicitly
    # trigger the download here so `wishcribe download` pre-caches the model
    # as documented, rather than deferring it silently to first transcription.
    if _is_apple_silicon():
        try:
            import mlx_whisper  # noqa: F401
            mlx_model = _mlx_model_id(model)
            if verbose:
                print(f"\n📥 Downloading MLX-Whisper model for Apple Silicon...")
                print(f"   Repo: {mlx_model}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(mlx_model)
            except Exception as exc:
                # huggingface_hub missing or download failed — non-fatal:
                # model will auto-download on first transcription call instead.
                if verbose:
                    print(f"   ⚠️  Pre-download skipped ({exc}); model will download on first use")
            if verbose:
                print(f"✅ MLX-Whisper '{model}' ready ({mlx_model})")
            return True
        except ImportError:
            pass  # mlx_whisper not installed — fall through to faster-whisper

    # ── faster-whisper / openai-whisper path ──────────────────────────────────
    if _whisper_is_cached(model):
        if verbose:
            print(f"✅ Whisper '{model}' already cached")
        return True

    _SIZES = {
        "tiny": "75 MB", "base": "139 MB", "small": "461 MB",
        "medium": "1.4 GB", "large-v1": "2.9 GB", "large-v2": "2.9 GB",
        "large-v3": "3.1 GB", "turbo": "1.6 GB", "large": "2.9 GB",
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
            ow = {"large-v2": "large", "large-v3": "large", "large-v1": "large", "turbo": "large"}.get(model, model)
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
    # Only purge the specific Whisper model cache — never the pyannote cache.
    # Pyannote is ~1 GB and takes minutes to re-download; it is independent of
    # which Whisper model is being re-downloaded.
    #
    # openai-whisper stores large variants (large-v1/v2/v3, turbo) all as "large.pt",
    # so we resolve via the same alias map used in transcription.

    # ── Apple Silicon + MLX: purge the mlx-community HF cache ────────────────
    from .transcribe import _is_apple_silicon, _mlx_model_id
    if _is_apple_silicon():
        try:
            import mlx_whisper  # noqa: F401
            mlx_model = _mlx_model_id(model)
            # HF hub stores repos as models--org--repo-name under the cache base
            safe = mlx_model.replace("/", "--")
            mlx_cache = os.path.join(_FW_CACHE_BASE, f"models--{safe}")
            if os.path.isdir(mlx_cache):
                shutil.rmtree(mlx_cache, ignore_errors=True)
                if verbose:
                    print(f"🗑️  Deleted MLX cache: {mlx_cache}")
        except ImportError:
            pass  # mlx_whisper not installed — no MLX cache to purge

    # ── faster-whisper and openai-whisper caches ──────────────────────────────
    _OW_ALIAS = {"large-v1": "large", "large-v2": "large", "large-v3": "large", "turbo": "large"}
    ow_name = _OW_ALIAS.get(model, model)
    paths = [_fw_cache_dir(model), _openai_whisper_cache(ow_name)]
    for path in paths:
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
    from .transcribe import _is_apple_silicon
    apple = _is_apple_silicon()
    mlx_active = False
    if apple:
        try:
            import mlx_whisper  # noqa: F401
            mlx_active = True
        except ImportError:
            pass
    backend_label = "MLX-Whisper (Apple Silicon)" if mlx_active else "faster-whisper"

    print("\n" + "═" * 64)
    print("📦  WISHCRIBE — MODEL DOWNLOADER")
    print("═" * 64)
    print(f"  Whisper model : {model} ({backend_label})")
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
    from .transcribe import _is_apple_silicon
    apple = _is_apple_silicon()
    mlx_active = False
    if apple:
        try:
            import mlx_whisper  # noqa: F401
            mlx_active = True
        except ImportError:
            pass
    backend_label = "MLX-Whisper" if mlx_active else "faster-whisper"

    print("\n" + "═" * 64)
    print("📋  DOWNLOAD SUMMARY")
    print("═" * 64)
    print(f"  {backend_label} '{model}'  : {'✅ Ready' if whisper_ok else '❌ Failed'}")
    print(f"  pyannote diarization     : {'✅ Ready' if diarize_ok else '❌ Failed'}")
    print("═" * 64)
    if whisper_ok and diarize_ok:
        print("\n🎉 All models cached! wishcribe now works fully offline.")
        print("   Run:  wishcribe --video meeting.mp4\n")
    else:
        print("\n⚠️  Some models failed to download. See errors above.\n")
