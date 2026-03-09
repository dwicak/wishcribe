"""Transcription: local Whisper model or OpenAI Whisper API."""
from __future__ import annotations

import os
from typing import Optional

# Default Whisper model — large gives the best accuracy
DEFAULT_WHISPER_MODEL = "large"


def transcribe_local(
    audio_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    import whisper

    if verbose:
        print(f"\n🗣️  Transcribing locally with Whisper '{model_size}'...")
        if model_size == "large":
            print("   (large model — best accuracy, ~2.9 GB, may take a moment to load)")

    model = whisper.load_model(model_size)
    options = {"word_timestamps": False, "verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    if verbose:
        print("✅ Transcription complete")
    return result["segments"]


def transcribe_api(
    audio_path: str,
    api_key: str,
    language: Optional[str] = None,
    verbose: bool = True,
) -> list:
    from openai import OpenAI

    if verbose:
        print("\n🗣️  Transcribing via OpenAI API...")
        size_mb = os.path.getsize(audio_path) / 1_048_576
        if size_mb > 25:
            print(f"⚠️  File is {size_mb:.1f} MB — OpenAI API limit is 25 MB")

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        params = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            params["language"] = language
        result = client.audio.transcriptions.create(file=f, **params)

    if verbose:
        print("✅ Transcription complete")
    return result.segments
