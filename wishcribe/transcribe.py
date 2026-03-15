"""
wishcribe.transcribe
--------------------
Transcription backends in priority order:

  1. MLX-Whisper  — Apple Silicon only (M1/M2/M3/M4). Uses the Neural Engine /
                    GPU via Apple's MLX framework. Fastest on Mac, no CUDA needed.
  2. faster-whisper — CTranslate2 backend. 4-8x faster than openai-whisper.
                    Works on CUDA GPU and CPU. Batched inference + VAD built in.
  3. openai-whisper — Original CPU/GPU backend. Fallback when faster-whisper
                    is not installed.

New in v1.2.0
-------------
  - MLX-Whisper backend auto-selected on Apple Silicon (items 9, 10, 11)
  - Default model changed to 'turbo' on Apple Silicon (item 11)
  - OMP_NUM_THREADS auto-tuned to physical performance cores on Apple Silicon (item 12)
  - --initial-prompt  : inject domain context to guide transcription (item 4)
  - --temperature     : sampling temperature (0.0 = greedy, item 5)
  - --beam-size       : beam search width for non-batched path (item 6)

New in v1.3.0
-------------
  - word_timestamps     : word-level timing in SRT/JSON output (item 7)
  - no_speech_threshold : discard silent segments, suppress hallucination (item 8)
  - vad_filter=False    : --no-vad escape hatch to bypass VAD (item 13)
  - vad_threshold / vad_min_silence_ms / vad_speech_pad_ms (item 14)
  - _apple_chip_name()  : reads sysctl for exact chip label in banner (item 16)
"""
from __future__ import annotations

import gc
import os
import platform
import subprocess
from typing import Optional

# ── Default model ─────────────────────────────────────────────────────────────

DEFAULT_WHISPER_MODEL = "large-v2"
DEFAULT_WHISPER_MODEL_APPLE = "turbo"  # faster on Apple Silicon Neural Engine

# ── Apple Silicon detection ───────────────────────────────────────────────────

def _is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon (M1/M2/M3/M4)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _apple_chip_name() -> str:
    """
    Return a human-readable chip label, e.g. 'Apple M3 Pro'.
    Uses sysctl machdep.cpu.brand_string on Apple Silicon.
    Falls back to 'Apple Silicon (M-series)' if unavailable.
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "Apple Silicon (M-series)"


def _apple_perf_cores() -> int:
    """
    Return the number of performance (P) cores on Apple Silicon.
    Falls back to os.cpu_count() if sysctl is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        cores = int(out)
        return cores if cores > 0 else (os.cpu_count() or 4)
    except Exception:
        return os.cpu_count() or 4


# ── Model alias maps ──────────────────────────────────────────────────────────

# User-facing aliases → faster-whisper model IDs
_FW_MODEL_MAP = {
    "tiny":     "tiny",
    "base":     "base",
    "small":    "small",
    "medium":   "medium",
    "large":    "large-v2",       # legacy alias
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "turbo":    "large-v3-turbo",
    # distil-large-v3 removed — broken on Apple Silicon (ct2 conversion failure)
}

# MLX model repos by (model, available RAM).
# Each entry is a list of (min_ram_gb, exact_hf_repo) pairs in descending RAM order.
# _mlx_model_id() picks the first entry where ram >= min_ram_gb.
#
# Repos verified against mlx-community on HuggingFace (March 2026).
# Only use quantized repos that are confirmed to exist — not suffix math.
# Models where no quantized variant is confirmed fall back to turbo on low RAM.
_MLX_REPOS: dict[str, list[tuple[int, str]]] = {
    "tiny": [
        (0,  "mlx-community/whisper-tiny-mlx"),             # tiny is tiny — no quant needed
    ],
    "base": [
        (8,  "mlx-community/whisper-base-mlx"),
        (0,  "mlx-community/whisper-base-mlx-4bit"),        # confirmed on HF
    ],
    "small": [
        (8,  "mlx-community/whisper-small-mlx"),
        (0,  "mlx-community/whisper-small-mlx-8bit"),       # confirmed on HF
    ],
    "medium": [
        (16, "mlx-community/whisper-medium-mlx"),
        (0,  "mlx-community/whisper-medium-mlx-4bit"),      # confirmed on HF
    ],
    "large": [
        (16, "mlx-community/whisper-large-v2-mlx"),
        (8,  "mlx-community/whisper-large-v3-turbo-4bit"),  # no confirmed large-v2-4bit; turbo-4bit is close
        (0,  "mlx-community/whisper-large-v3-turbo-8bit"),
    ],
    "large-v1": [
        (16, "mlx-community/whisper-large-v1-mlx"),
        (8,  "mlx-community/whisper-large-v3-turbo-4bit"),
        (0,  "mlx-community/whisper-large-v3-turbo-8bit"),
    ],
    "large-v2": [
        (16, "mlx-community/whisper-large-v2-mlx"),
        (8,  "mlx-community/whisper-large-v3-turbo-4bit"),  # no confirmed large-v2-4bit on HF
        (0,  "mlx-community/whisper-large-v3-turbo-8bit"),
    ],
    "large-v3": [
        (16, "mlx-community/whisper-large-v3-mlx"),
        (8,  "mlx-community/whisper-large-v3-mlx-4bit"),    # confirmed on HF
        (0,  "mlx-community/whisper-large-v3-mlx-8bit"),    # confirmed on HF
    ],
    "turbo": [
        (16, "mlx-community/whisper-large-v3-turbo"),       # confirmed on HF
        (8,  "mlx-community/whisper-large-v3-turbo-4bit"),  # confirmed on HF
        (0,  "mlx-community/whisper-large-v3-turbo-8bit"),  # confirmed on HF
    ],
}

# Aliases back to openai-whisper model names (fallback path)
_OW_MODEL_MAP = {
    "large-v2": "large", "large-v3": "large", "large-v1": "large",
    "turbo": "large",
}

_MODEL_INFO = {
    "tiny":     "75 MB  — fastest, fair accuracy",
    "base":     "139 MB — fast, good accuracy",
    "small":    "461 MB — moderate, better accuracy",
    "medium":   "1.4 GB — good speed/accuracy balance",
    "large":    "2.9 GB — best accuracy (alias for large-v2)",
    "large-v1": "2.9 GB — original large model",
    "large-v2": "2.9 GB — best accuracy ⭐ (default on non-Apple)",
    "large-v3": "3.1 GB — newest large model",
    "turbo":    "1.6 GB — large-v3-turbo, fast + accurate ⭐ (default on Apple Silicon)",
}


# ── Public entry point ────────────────────────────────────────────────────────

def transcribe_local(
    audio_path: str,
    model: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    verbose: bool = True,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    device: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0,
    beam_size: int = 5,
    # Item 7: word-level timestamps
    word_timestamps: bool = False,
    # Item 8: suppress hallucinated segments in silent regions
    no_speech_threshold: float = 0.6,
    # Items 13+14: VAD controls
    vad_filter: bool = True,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 500,
    vad_speech_pad_ms: int = 200,
) -> list[dict]:
    """
    Transcribe using the best available backend.

    Backend priority: MLX (Apple Silicon) → faster-whisper → openai-whisper.

    Parameters
    ----------
    audio_path          : Path to 16 kHz mono WAV file.
    model               : Whisper model alias. On Apple Silicon defaults to 'turbo'.
    language            : BCP-47 language code. None = auto-detect.
    verbose             : Print progress to stdout.
    batch_size          : faster-whisper batch size (ignored by MLX and openai-whisper).
    compute_type        : CTranslate2 compute type (faster-whisper only). Auto-detected.
    device              : 'cuda' or 'cpu' (faster-whisper/openai-whisper only).
    initial_prompt      : Domain context injected before transcription to guide the model.
                          Note: disables batched inference (falls back to beam search).
    temperature         : Sampling temperature. 0.0 = greedy (deterministic, recommended).
                          Note: non-zero temperature disables batched inference.
    beam_size           : Beam search width for the non-batched path.
    word_timestamps     : Return word-level timing in each segment's 'words' list.
                          Supported by faster-whisper and openai-whisper.
                          MLX-Whisper: silently ignored (word timestamps not supported).
    no_speech_threshold : Probability below which a segment is treated as non-speech
                          and discarded (default 0.6). Suppresses hallucinations in
                          silent regions. faster-whisper and openai-whisper only.
    vad_filter          : Apply Voice Activity Detection before transcription (default
                          True). Set False to disable if VAD trims real speech.
                          MLX-Whisper: silently ignored (no VAD support).
    vad_threshold       : VAD speech probability threshold (default 0.5).
    vad_min_silence_ms  : Minimum silence (ms) to split chunks (default 500).
    vad_speech_pad_ms   : Padding added around detected speech (ms, default 200).

    Returns
    -------
    List of dicts: [{start, end, text, words?}, ...]
    """
    # Item 11: on Apple Silicon, default to turbo (Neural Engine optimised)
    if model == DEFAULT_WHISPER_MODEL and _is_apple_silicon():
        model = DEFAULT_WHISPER_MODEL_APPLE

    # Item 9: try MLX-Whisper first on Apple Silicon
    if _is_apple_silicon():
        try:
            import mlx_whisper  # noqa: F401
            return _transcribe_mlx(
                audio_path, model, language, verbose, initial_prompt, temperature
            )
        except ImportError:
            if verbose:
                print("💡 mlx-whisper not installed — using faster-whisper")
                print("   For faster Apple Silicon transcription:  pip install mlx-whisper")

    # faster-whisper (CUDA or CPU)
    try:
        import faster_whisper  # noqa: F401
        return _transcribe_faster_whisper(
            audio_path, model, language, verbose, batch_size,
            compute_type, device, initial_prompt, temperature, beam_size,
            word_timestamps=word_timestamps,
            no_speech_threshold=no_speech_threshold,
            vad_filter=vad_filter,
            vad_threshold=vad_threshold,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
        )
    except ImportError:
        if verbose:
            print("⚠️  faster-whisper not found — using openai-whisper (slower)")
            print("   Install for 4-8x speedup:  pip install faster-whisper")
        return _transcribe_openai_whisper(
            audio_path, model, language, verbose, initial_prompt, temperature,
            word_timestamps=word_timestamps,
        )


# ── MLX-Whisper backend (Apple Silicon) ──────────────────────────────────────

def _mlx_ram_gb() -> int:
    """Return total unified memory in GB on Apple Silicon via sysctl."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out) // (1024 ** 3)
    except Exception:
        return 8  # safe fallback


def _mlx_model_id(model: str) -> str:
    """
    Choose an MLX model repo from mlx-community, auto-selected by available RAM.

    Uses an explicit, HuggingFace-verified map — not suffix math — so every
    returned repo is confirmed to exist on HuggingFace.

    >= 16 GB → full precision
    >= 8 GB  → 4-bit quantized (where available)
    <  8 GB  → 8-bit quantized (most conservative)
    """
    entries = _MLX_REPOS.get(model)
    if not entries:
        # Unknown model — fall back to full-precision naming convention
        return f"mlx-community/whisper-{model}-mlx"
    ram = _mlx_ram_gb()
    for min_ram, repo in entries:
        if ram >= min_ram:
            return repo
    return entries[-1][1]  # safest (lowest RAM requirement)


def _transcribe_mlx(
    audio_path: str,
    model: str,
    language: Optional[str],
    verbose: bool,
    initial_prompt: Optional[str],
    temperature: float,
) -> list[dict]:
    """Transcribe using mlx-whisper on Apple Silicon (Neural Engine / GPU)."""
    import mlx_whisper

    mlx_model = _mlx_model_id(model)
    ram_gb = _mlx_ram_gb()

    # Item 12: tune OMP_NUM_THREADS to P-cores to avoid thrashing E-cores
    perf_cores = _apple_perf_cores()
    os.environ.setdefault("OMP_NUM_THREADS", str(perf_cores))

    if verbose:
        print(f"🍎 Transcribing with MLX-Whisper (Apple Silicon)")
        print(f"   Model: {mlx_model}  |  RAM: {ram_gb} GB  |  P-cores: {perf_cores}")
        if initial_prompt:
            snippet = initial_prompt[:60] + ("…" if len(initial_prompt) > 60 else "")
            print(f'   Prompt: "{snippet}"')
        print("   Transcribing", end="", flush=True)

    kwargs: dict = {
        "path_or_hf_repo": mlx_model,
        "verbose": False,
    }
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    # Note: mlx_whisper.transcribe() does not accept a temperature kwarg —
    # it always uses greedy decoding internally. temperature is silently ignored here.

    # mlx_whisper.transcribe() runs to completion and returns a dict —
    # it is NOT a streaming generator, so we can't print dots progressively.
    # We print a single "done" line after it returns.
    # The try/finally ensures a newline is printed if an exception is raised
    # mid-run, so the error message never appears on the same line as the prompt.
    mlx_ok = False
    try:
        result = mlx_whisper.transcribe(audio_path, **kwargs)
        mlx_ok = True
    finally:
        if verbose and not mlx_ok:
            print()  # newline so the traceback starts on a fresh line

    raw = result.get("segments", []) if isinstance(result, dict) else []
    segments = []
    for s in raw:
        text = (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")).strip()
        if text:
            start = s.get("start", 0.0) if isinstance(s, dict) else getattr(s, "start", 0.0)
            end   = s.get("end",   0.0) if isinstance(s, dict) else getattr(s, "end",   0.0)
            segments.append({"start": start, "end": end, "text": text})

    if verbose:
        print(f" done — {len(segments)} segments")

    return segments


# ── faster-whisper backend ────────────────────────────────────────────────────

def _transcribe_faster_whisper(
    audio_path: str,
    model: str,
    language: Optional[str],
    verbose: bool,
    batch_size: int,
    compute_type: Optional[str],
    device: Optional[str],
    initial_prompt: Optional[str],
    temperature: float,
    beam_size: int,
    *,
    word_timestamps: bool = False,
    no_speech_threshold: float = 0.6,
    vad_filter: bool = True,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 500,
    vad_speech_pad_ms: int = 200,
) -> list[dict]:
    """Use faster-whisper with batching and VAD for maximum speed."""
    from faster_whisper import WhisperModel

    # Only import torch when we actually need it for device/compute detection.
    # Skipping the import when both are already specified saves ~100-300 ms startup.
    if device is None or compute_type is None:
        import torch

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Item 12: tune OMP_NUM_THREADS on Apple Silicon CPU runs
    if device == "cpu" and _is_apple_silicon():
        perf_cores = _apple_perf_cores()
        os.environ.setdefault("OMP_NUM_THREADS", str(perf_cores))

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

    # Determine whether batching will be attempted before printing the banner
    use_batched = not initial_prompt and temperature == 0.0

    if verbose:
        batch_str = f"  |  Batch: {batch_size}" if use_batched else "  |  Batch: disabled"
        vad_str   = "" if vad_filter else "  |  VAD: off"
        print(f"⚡ Transcribing with faster-whisper '{fw_model}'")
        print(f"   Device: {device.upper()}  |  Compute: {compute_type}{batch_str}{vad_str}")
        if initial_prompt:
            snippet = initial_prompt[:60] + ("…" if len(initial_prompt) > 60 else "")
            print(f'   Prompt: "{snippet}"')

    try:
        whisper_model = WhisperModel(fw_model, device=device, compute_type=compute_type)
    except Exception as exc:
        exc_str = str(exc)
        if "model.bin" in exc_str or "Unable to open file" in exc_str or "Failed to load model" in exc_str:
            raise RuntimeError(
                f"Whisper model '{model}' is not properly cached — model.bin missing or corrupt.\n"
                f"   Fix:  wishcribe download --model {model} --force\n"
                f"   This will delete the incomplete cache and re-download cleanly."
            ) from exc
        raise  # any other error propagates as-is

    # Items 13+14: build VAD params only when vad_filter is True.
    # CRITICAL: passing vad_parameters when vad_filter=False raises a TypeError
    # in faster-whisper — the kwarg is only consumed by the VAD path.
    vad_params = None
    if vad_filter:
        vad_params = {
            "threshold":              vad_threshold,
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms":           vad_speech_pad_ms,
        }

    # Try BatchedInferencePipeline (faster-whisper >= 1.0).
    # Batching is disabled when initial_prompt or non-zero temperature is set,
    # because BatchedInferencePipeline does not support those parameters.
    batched_ok = False
    pipeline_created = False  # tracked separately so the finally block can del it
    if use_batched:
        try:
            from faster_whisper import BatchedInferencePipeline
            pipeline = BatchedInferencePipeline(model=whisper_model)
            pipeline_created = True  # set immediately so finally block cleans it up
            transcribe_kwargs: dict = dict(
                language=language,
                batch_size=batch_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                no_speech_threshold=no_speech_threshold,
            )
            if vad_params is not None:
                transcribe_kwargs["vad_parameters"] = vad_params
            segments_iter, info = pipeline.transcribe(audio_path, **transcribe_kwargs)
            batched_ok = True
        except ImportError:
            # BatchedInferencePipeline not available — older faster-whisper
            pass

    if not batched_ok:
        if verbose:
            if not use_batched:
                print("   (using standard transcription — batching disabled for prompt/temperature)")
            else:
                print("   (batched pipeline unavailable — using standard transcription)")
        # Item 8: restore condition_on_previous_text=True (faster-whisper default).
        # v1.1.1 set this to False to reduce hallucination, but that breaks coherence
        # across Whisper's 30 s internal windows. The correct solution is to pair
        # it with no_speech_threshold which discards truly silent segments.
        standard_kwargs: dict = dict(
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            condition_on_previous_text=True,
            no_speech_threshold=no_speech_threshold,
            word_timestamps=word_timestamps,
        )
        if vad_params is not None:
            standard_kwargs["vad_parameters"] = vad_params
        segments_iter, info = whisper_model.transcribe(audio_path, **standard_kwargs)

    if verbose:
        lang = getattr(info, "language", language or "unknown")
        prob = getattr(info, "language_probability", 0.0)
        print(f"   Detected language: {lang} (confidence {prob:.0%})")
        print("   Transcribing", end="", flush=True)

    segments = []
    completed = False
    try:
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue  # skip empty segments silently — no dot for blanks

            # Item 7: extract word-level timestamps from faster-whisper.
            # faster-whisper returns Word namedtuples with .word (not .text).
            # Normalise to plain dicts so all backends share the same schema.
            words = None
            if word_timestamps and seg.words:
                words = [
                    {
                        "word":        w.word,
                        "start":       w.start,
                        "end":         w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]

            entry: dict = {"start": seg.start, "end": seg.end, "text": text}
            if words is not None:
                entry["words"] = words
            segments.append(entry)

            if verbose:
                print(".", end="", flush=True)
        completed = True
    finally:
        # If iteration was interrupted by an exception, print a newline so the
        # error message doesn't appear on the same line as the progress dots.
        if verbose and not completed:
            print()
        # Always free GPU memory — even if an exception is raised during iteration
        # (e.g. CUDA OOM mid-batch). Without this, whisper_model stays alive and
        # holds VRAM, preventing diarization from loading.
        # Use pipeline_created (not batched_ok) so we clean up even if
        # pipeline.transcribe() raised before batched_ok was set True.
        if pipeline_created:
            del pipeline
        del whisper_model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if verbose:
        print(f" {len(segments)} segments")

    return segments


# ── openai-whisper fallback ───────────────────────────────────────────────────

def _transcribe_openai_whisper(
    audio_path: str,
    model: str,
    language: Optional[str],
    verbose: bool,
    initial_prompt: Optional[str],
    temperature: float,
    *,
    word_timestamps: bool = False,
) -> list[dict]:
    """Fallback: original openai-whisper (no batching, slower)."""
    import whisper

    ow_model = _OW_MODEL_MAP.get(model, model)
    if verbose:
        print(f"🎙️  Transcribing with openai-whisper '{ow_model}'...")

    wm = whisper.load_model(ow_model)
    kwargs: dict = {"verbose": False}
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    # Always pass temperature — openai-whisper's default is a [0.0,0.2…1.0] fallback
    # schedule, NOT greedy. Passing temperature=0.0 explicitly disables that schedule
    # and forces true greedy decoding, matching faster-whisper behavior.
    kwargs["temperature"] = temperature
    # Item 7: word timestamps (openai-whisper key is word_timestamps)
    if word_timestamps:
        kwargs["word_timestamps"] = True

    result = wm.transcribe(audio_path, **kwargs)

    segments = []
    for s in result.get("segments", []):
        text = s["text"].strip()
        if not text:
            continue
        entry: dict = {"start": s["start"], "end": s["end"], "text": text}
        # Item 7: openai-whisper returns word timestamps as a list of dicts with
        # keys "word", "start", "end" already — normalise probability to 1.0
        # (openai-whisper does not return per-word probabilities).
        if word_timestamps and s.get("words"):
            entry["words"] = [
                {
                    "word":        w.get("word", ""),
                    "start":       w.get("start", s["start"]),
                    "end":         w.get("end",   s["end"]),
                    "probability": w.get("probability", 1.0),
                }
                for w in s["words"]
            ]
        segments.append(entry)

    # Free model memory (same pattern as faster-whisper path)
    del wm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return segments


# ── OpenAI cloud API ──────────────────────────────────────────────────────────

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
