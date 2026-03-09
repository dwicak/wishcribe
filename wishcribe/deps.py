"""
Auto-detect and install missing Python dependencies at runtime.
ffmpeg must be installed separately (brew / apt).
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

_BASE_DEPS  = ["moviepy", "pyannote.audio", "torch"]
_LOCAL_DEPS = ["openai-whisper"]
_API_DEPS   = ["openai"]

_IMPORT_MAP = {
    "moviepy":        "moviepy",
    "pyannote.audio": "pyannote.audio",
    "torch":          "torch",
    "openai-whisper": "whisper",
    "openai":         "openai",
}


def _is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _pip_install(packages: list[str]) -> None:
    print(f"📦 Installing: {', '.join(packages)} ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    )


def ensure_dependencies(use_api: bool = False) -> None:
    """Check all required packages and auto-install any that are missing."""
    required = _BASE_DEPS + (_API_DEPS if use_api else _LOCAL_DEPS)
    missing  = [pkg for pkg in required if not _is_installed(_IMPORT_MAP[pkg])]

    if missing:
        print("\n⚙️  Some dependencies are missing. Installing automatically...")
        _pip_install(missing)
        print("✅ Dependencies installed.\n")

    if not shutil.which("ffmpeg"):
        print("⚠️  WARNING: ffmpeg not found on PATH.")
        print("   Install it:")
        print("     macOS  : brew install ffmpeg")
        print("     Ubuntu : sudo apt install ffmpeg")
        print("     Windows: https://ffmpeg.org/download.html\n")
