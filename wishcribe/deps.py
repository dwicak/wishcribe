"""
wishcribe.deps
--------------
Auto-install missing runtime dependencies.
Prefers faster-whisper for maximum transcription speed.
"""
from __future__ import annotations

import importlib
import subprocess
import sys


def ensure_dependencies(use_api: bool = False) -> None:
    """Install any missing packages at runtime."""

    # Prefer faster-whisper; fall back to openai-whisper if neither installed
    has_faster = _is_installed("faster_whisper")
    has_openai = _is_installed("whisper")

    if not has_faster and not has_openai:
        print("📦 Installing faster-whisper (4-8x faster than openai-whisper)...")
        _install("faster-whisper")

    # Always-required packages
    # Note: check "pyannote.audio" (not bare "pyannote") — the core namespace
    # package can exist without the audio subpackage.
    for import_name, pkg_spec in [
        ("torch",         "torch>=2.0.0"),
        ("pyannote.audio", "pyannote.audio>=3.1.0"),
        ("moviepy",       "moviepy>=1.0.3"),
    ]:
        if not _is_installed(import_name):
            _install(pkg_spec)

    if use_api and not _is_installed("openai"):
        _install("openai>=1.0.0")


def _is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _install(pkg: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
