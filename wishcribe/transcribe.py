"""
wishcribe.transcribe
--------------------
Transcription via faster-whisper (CTranslate2 backend) — same accuracy as
openai-whisper large but 4-8x faster, with batched inference and VAD filtering.

Falls back to openai-whisper if faster-whisper is not installed.
"""
from __future__ import annotations

import gc
import os
from typing import Optional

DEFAULT_WHISPER_MODEL = "large-v2"

# User-facing aliases → faster-whisper model IDs
_FW_MODEL_MAP = {
    "tiny":            "tiny",
    "base":            "base",
    "small":           "small",
    "medium":          "medium",
    "large":           "large-v2",   # legacy alias
    "large-v1":        "large-v1",
    "large-v2":        "large-v2",
    "large-v3":        "large-v3",
    "turbo":           "large-v3-turbo",
    "distil-large-v3": "distil-whisper/distil-large-v3",
}

# Aliases back to openai-whisper model names (fallback path)
_OW_MODEL_MAP = {
    "large-v2": "large", "large-v3": "large", "large-v1": "large",
    "turbo": "large", "distil-large-v3": "large",
}

_MODEL_INFO = {
    "tiny":            "75 MB  — fastest, fair accuracy",
    "base":            "139 MB — fast, good accuracy",
    "small":           "461 MB — moderate, better accuracy",
    "medium":          "1.4 GB — good speed/accuracy balance",
    "large":           "2.9 GB — best accuracy (alias for large-v2)",
    "large-v2":        "2.9 GB — best accuracy ⭐ (default)",
    "large-v3":        "3.1 GB — newest large model",
    "turbo":           "1.6 GB — large-v3-turbo, fast + accurate",
    "distil-large-v3": "1.5 GB — distilled, near large-v2 accuracy",
}


def transcribe_local(
    audio_path: str,
    model: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    verbose: bool = True,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    device: Optional[str] = None,
) -> list[dict]:
    """
    Transcribe using faster-whisper (CTranslate2). Falls back to openai-whisper.
    Returns list of dicts: {start, end, text}
    """
    try:
        import faster_whisper  # noqa: F401
        return _transcribe_faster_whisper(
            audio_path, model, language, verbose, batch_size, compute_type, device
        )
    except ImportError:
        if verbose:
            print("⚠️  faster-whisper not found — using openai-whisper (slower)")
            print("   Install for 4-8x speedup:  pip install faster-whisper")
        return _transcribe_openai_whisper(audio_path, model, language, verbose)


def _transcribe_faster_whisper(
    audio_path: str,
    model: str,
    language: Optional[str],
    verbose: bool,
    batch_size: int,
    compute_type: Optional[str],
    device: Optional[str],
) -> list[dict]:
    """Use faster-whisper with batching and VAD for maximum speed."""
    import torch
    from faster_whisper import WhisperModel

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto compute type — float16 on modern GPU, int8 on CPU (best speed/quality)
    if compute_type is None:
        if device == "cuda":
            try:
                cap = torch.cuda.get_device_capability()
                compute_type = "float16" if cap[0] >= 7 else "float32"
            except Exception:
                compute_type = "float16"
        else:
            compute_type = "int8"

    fw_model = _FW_MODEL_MAP.get(model, model)

    if verbose:
        print(f"⚡ Transcribing with faster-whisper '{fw_model}'")
        print(f"   Device: {device.upper()}  |  Compute: {compute_type}  |  Batch: {batch_size}")

    whisper_model = WhisperModel(fw_model, device=device, compute_type=compute_type)

    vad_params = {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
        "threshold": 0.5,
    }

    # Try BatchedInferencePipeline (faster-whisper >= 1.0) — only catch ImportError here.
    # Other errors (OOM, CUDA) should propagate so the user knows something went wrong.
    batched_ok = False
    try:
        from faster_whisper import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model=whisper_model)
        segments_iter, info = pipeline.transcribe(
            audio_path,
            language=language,
            batch_size=batch_size,
            vad_filter=True,
            vad_parameters=vad_params,
        )
        batched_ok = True
    except ImportError:
        # BatchedInferencePipeline not available — older faster-whisper
        pass

    if not batched_ok:
        # Non-batched fallback (faster-whisper < 1.0)
        if verbose:
            print("   (batched pipeline unavailable — using standard transcription)")
        segments_iter, info = whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=vad_params,
            condition_on_previous_text=False,  # reduces hallucination
        )

    if verbose:
        lang = getattr(info, "language", language or "unknown")
        prob = getattr(info, "language_probability", 0.0)
        print(f"   Detected language: {lang} (confidence {prob:.0%})")
        print("   Transcribing", end="", flush=True)

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})
        if verbose:
            print(".", end="", flush=True)

    if verbose:
        print(f" {len(segments)} segments")

    # Free GPU memory before diarization loads
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return segments


def _transcribe_openai_whisper(
    audio_path: str,
    model: str,
    language: Optional[str],
    verbose: bool,
) -> list[dict]:
    """Fallback: original openai-whisper (no batching, slower)."""
    import whisper

    ow_model = _OW_MODEL_MAP.get(model, model)
    if verbose:
        print(f"🎙️  Transcribing with openai-whisper '{ow_model}'...")

    wm = whisper.load_model(ow_model)
    kwargs = {"verbose": False}
    if language:
        kwargs["language"] = language

    result = wm.transcribe(audio_path, **kwargs)
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
        if s["text"].strip()
    ]


def transcribe_api(
    audio_path: str,
    api_key: str,
    language: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """Transcribe using OpenAI Whisper API (cloud, no local GPU needed)."""
    from openai import OpenAI

    if verbose:
        print("☁️  Transcribing via OpenAI Whisper API...")
        size_mb = os.path.getsize(audio_path) / 1_048_576
        if size_mb > 24:
            print(f"⚠️  File is {size_mb:.1f} MB — OpenAI API limit is 25 MB")

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        params = {
            "model": "whisper-1",
            "file": f,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            params["language"] = language
        result = client.audio.transcriptions.create(**params)

    raw_segs = getattr(result, "segments", None) or []
    segments = []
    for s in raw_segs:
        text = (s.text if hasattr(s, "text") else s.get("text", "")).strip()
        if text:
            segments.append({
                "start": s.start if hasattr(s, "start") else s["start"],
                "end":   s.end   if hasattr(s, "end")   else s["end"],
                "text":  text,
            })

    if verbose:
        print(f"   ✅ {len(segments)} segments from API")
    return segments
