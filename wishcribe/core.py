"""
wishcribe.core
--------------
Main pipeline: extract audio → transcribe → diarize → merge → write outputs.

Speed improvements over v1 (openai-whisper):
  - faster-whisper backend (4-8x faster via CTranslate2)
  - Batched inference (multiple audio chunks in parallel)
  - VAD pre-filtering (skips silence, reduces hallucination)
  - float16 on GPU / int8 on CPU (automatic)
  - GPU memory freed after transcription before diarization runs

Apple Silicon (v1.2.0):
  - MLX-Whisper auto-selected on M1/M2/M3/M4 (Neural Engine / GPU)
  - Default model is 'turbo' on Apple Silicon
  - OMP_NUM_THREADS tuned to performance cores

Accuracy controls (v1.2.0):
  - initial_prompt : domain context to guide transcription vocabulary
  - temperature    : sampling temperature (0.0 = greedy, default)
  - beam_size      : beam search width for non-batched path

New in v1.3.0:
  - word_timestamps     : word-level timing in SRT/JSON (item 7)
  - no_speech_threshold : suppress hallucinated segments (item 8)
  - --no-vad            : escape hatch to disable VAD (item 13)
  - vad_threshold / vad_min_silence_ms / vad_speech_pad_ms (item 14)
  - Apple Silicon chip name shown in banner (item 16)
  - Auto-disable diarization with early warning if no token/cache (item 17)

Robustness:
  - All errors raise exceptions (no sys.exit in library code)
  - Upfront validation of model cache and token before pipeline starts
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Optional

from .deps import ensure_dependencies
from .audio import extract_audio
from .diarize import run_diarization
from .transcribe import (
    transcribe_local, transcribe_api,
    DEFAULT_WHISPER_MODEL, DEFAULT_WHISPER_MODEL_APPLE,
    _MODEL_INFO, _is_apple_silicon, _apple_chip_name,
)
from .merge import merge_segments
from .output import write_txt, write_srt, write_json
from .models import Segment


def _resolve_token(hf_token: Optional[str]) -> Optional[str]:
    """Accept HF token from argument or environment variable."""
    return hf_token or os.environ.get("WISHCRIBE_HF_TOKEN") or os.environ.get("HF_TOKEN")


def transcribe(
    input_path: str,
    *,
    hf_token: Optional[str] = None,
    model_path: Optional[str] = None,
    model: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    output_dir: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    save_txt: bool = True,
    save_srt: bool = True,
    save_json: bool = False,
    verbose: bool = True,
    diarize: bool = True,
    # Speed controls (faster-whisper)
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    device: Optional[str] = None,
    # Accuracy controls (v1.2.0)
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0,
    beam_size: int = 5,
    # v1.3.0 — word timestamps (item 7)
    word_timestamps: bool = False,
    # v1.3.0 — no-speech suppression (item 8)
    no_speech_threshold: float = 0.6,
    # v1.3.0 — VAD controls (items 13+14)
    vad_filter: bool = True,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 500,
    vad_speech_pad_ms: int = 200,
) -> list[Segment]:
    """
    Transcribe an audio/video file with per-speaker labels.

    Parameters
    ----------
    input_path     : Path to video or audio file (mp4, mkv, mp3, wav, m4a …)
    hf_token       : HuggingFace token. Also reads WISHCRIBE_HF_TOKEN / HF_TOKEN env.
    model          : Whisper model. Default: 'large-v2' (or 'turbo' on Apple Silicon).
                     Options: tiny | base | small | medium | large-v2 | large-v3 | turbo
    language       : BCP-47 code e.g. 'id', 'en'. None = auto-detect.
    num_speakers   : Exact speaker count (improves diarization when known).
    diarize        : False = skip speaker diarization (faster, no token needed).
    batch_size     : faster-whisper batch size. Higher = faster on GPU. Default 16.
    compute_type   : 'float16' (GPU), 'int8' (CPU/low-mem). Auto-detected.
    device         : 'cuda' or 'cpu'. Auto-detected.
    initial_prompt      : Domain context injected before transcription to guide the model
                          (e.g. "Medical: hypertension, tachycardia."). Helps with
                          specialised vocabulary and consistent casing/punctuation.
                          Note: disables batched inference (uses beam search instead).
    temperature         : Sampling temperature. 0.0 = greedy (default). Higher values
                          (0.2-1.0) increase diversity. Non-zero disables batched inference.
    beam_size           : Beam search width for non-batched path (default: 5).
    word_timestamps     : Embed word-level timing in each segment (default False).
                          Reflected in SRT output and JSON 'words' field.
                          Not supported by MLX-Whisper (silently ignored).
    no_speech_threshold : Probability below which a segment is discarded as non-speech
                          (default 0.6). Suppresses hallucinations in silent regions
                          while preserving cross-window coherence.
    vad_filter          : Apply Voice Activity Detection before transcription (default
                          True). Set False (--no-vad) if VAD incorrectly trims speech.
                          MLX-Whisper: silently ignored (no VAD support).
    vad_threshold       : VAD speech probability threshold (default 0.5).
    vad_min_silence_ms  : Minimum silence gap (ms) to split chunks (default 500).
    vad_speech_pad_ms   : Padding added around speech regions (ms, default 200).
    output_dir          : Where to save files. Default: same folder as input.
    use_api             : Use OpenAI Whisper API instead of local model.
    api_key             : OpenAI API key (required when use_api=True).
    save_txt            : Write <stem>_transcript.txt
    save_srt            : Write <stem>_transcript.srt
    save_json           : Write <stem>_transcript.json
    verbose             : Print progress to stdout.

    Returns
    -------
    List of Segment(start, end, speaker, text)

    Raises
    ------
    FileNotFoundError  : if input_path does not exist
    ValueError         : if use_api=True but no api_key given
    RuntimeError       : on audio extraction, transcription, or diarization failure
    """
    ensure_dependencies(use_api=use_api)

    hf_token = _resolve_token(hf_token)

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if use_api and not api_key:
        raise ValueError("api_key is required when use_api=True")

    # On Apple Silicon, transcribe_local() silently switches the default model
    # (large-v2) to turbo. Resolve that here so validation checks the model
    # that will actually be used — not the raw user-supplied default.
    effective_model = model
    if not use_api and model == DEFAULT_WHISPER_MODEL and _is_apple_silicon():
        effective_model = DEFAULT_WHISPER_MODEL_APPLE

    # Validate model cache and token BEFORE pipeline starts — fail fast before
    # spending time on audio extraction or transcription.
    if not use_api:
        _validate_model_cached(effective_model)
    if diarize:
        try:
            _validate_diarize_ready(hf_token, model_path)
        except RuntimeError as exc:
            # No token and no cached model — auto-disable diarization instead of
            # failing deep in the pipeline.  Print a clear warning and continue
            # without speaker labels so the transcription still completes.
            if verbose:
                print(
                    "⚠️  Diarization skipped — no HuggingFace token and no cached model.\n"
                    "   To enable speaker labels, either:\n"
                    "     wishcribe download --hf-token hf_xxx   (download once, then offline)\n"
                    "     wishcribe --video file.mp4 --hf-token hf_xxx\n"
                    "     export WISHCRIBE_HF_TOKEN=hf_xxx\n"
                    "   Continuing without speaker labels...\n"
                )
            diarize = False

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    if verbose:
        _banner(
            input_path.name, model, language, use_api, num_speakers,
            hf_token, model_path, diarize, batch_size, compute_type, device,
            initial_prompt, temperature, beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

    with tempfile.TemporaryDirectory() as tmp:
        # Step 1: Extract audio
        audio_path = extract_audio(str(input_path), tmp, verbose=verbose)

        # Step 2: Transcribe
        # Run FIRST so GPU memory is freed before diarization loads.
        if use_api:
            whisper_segs = transcribe_api(audio_path, api_key, language, verbose=verbose)
        else:
            whisper_segs = transcribe_local(
                audio_path, effective_model, language, verbose=verbose,
                batch_size=batch_size, compute_type=compute_type, device=device,
                initial_prompt=initial_prompt, temperature=temperature, beam_size=beam_size,
                word_timestamps=word_timestamps,
                no_speech_threshold=no_speech_threshold,
                vad_filter=vad_filter,
                vad_threshold=vad_threshold,
                vad_min_silence_ms=vad_min_silence_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
            )

        # Step 3: Diarize (after transcription — GPU memory now freed)
        diarization = None
        if diarize:
            diarization = run_diarization(
                audio_path,
                hf_token=hf_token,
                num_speakers=num_speakers,
                model_path=model_path,
                verbose=verbose,
            )

    # Step 4: Merge (audio tempdir now cleaned up — merge uses in-memory data only)
    if verbose:
        print("\n🔗 Merging transcription with speaker labels...")

    segments = merge_segments(whisper_segs, diarization)

    if verbose:
        print(f"✅ Merged {len(segments)} segments")

    # Step 5: Write outputs
    if save_txt:
        write_txt(segments, str(out_dir / f"{stem}_transcript.txt"))
    if save_srt:
        write_srt(segments, str(out_dir / f"{stem}_transcript.srt"),
                  word_timestamps=word_timestamps)
    if save_json:
        write_json(segments, str(out_dir / f"{stem}_transcript.json"))

    if verbose:
        _print_summary(segments)

    return segments


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_model_cached(model: str) -> None:
    """
    Check that the Whisper model is cached before starting the pipeline.
    Raises RuntimeError with a clear actionable message if not found.
    """
    if not _whisper_is_cached_safe(model):
        raise RuntimeError(
            f"Whisper model '{model}' is not cached.\n"
            f"   Run:  wishcribe download --model {model}\n"
            f"         (or wishcribe download --model {model} --force  to re-download)"
        )


def _whisper_is_cached_safe(model: str) -> bool:
    """Safe wrapper around _whisper_is_cached — returns True on any import error."""
    try:
        from .download import _whisper_is_cached
        return _whisper_is_cached(model)
    except Exception:
        return True  # can't check — don't block the user


def _validate_diarize_ready(hf_token, model_path) -> None:
    """
    Check that diarization can proceed: either a local model_path is given,
    or the pyannote model is already cached, or a token is provided.
    Raises RuntimeError with setup instructions if nothing is available.
    """
    if model_path:
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"model_path not found: {model_path}\n"
                f"   Provide a valid local pyannote model directory."
            )
        return  # local path is valid — proceed

    from .diarize import _find_cached_model
    if _find_cached_model():
        return  # already cached — proceed

    if hf_token:
        return  # token provided — download will happen during diarization

    # Nothing available — fail early with instructions
    raise RuntimeError(
        "Speaker diarization requires a HuggingFace token (model not cached).\n"
        "   Option 1 — provide a token:\n"
        "              wishcribe --video file.mp4 --hf-token hf_xxx\n"
        "              export WISHCRIBE_HF_TOKEN=hf_xxx  (then no flag needed)\n"
        "   Option 2 — download the model first:\n"
        "              wishcribe download --hf-token hf_xxx\n"
        "   Option 3 — skip speaker labels entirely:\n"
        "              wishcribe --video file.mp4 --no-diarize"
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _banner(name, model, language, use_api, num_speakers, hf_token, model_path,
            diarize, batch_size, compute_type, device,
            initial_prompt=None, temperature=0.0, beam_size=5,
            word_timestamps=False, vad_filter=True):
    if not diarize:
        diarize_str = "disabled (--no-diarize)"
    elif model_path:
        diarize_str = f"Local path: {model_path}"
    elif hf_token:
        diarize_str = "HuggingFace (token provided)"
    else:
        diarize_str = "Local cache (offline)"

    apple = _is_apple_silicon()

    # Resolve the effective model (Apple Silicon defaults to turbo)
    effective_model = model
    if model == DEFAULT_WHISPER_MODEL and apple:
        effective_model = DEFAULT_WHISPER_MODEL_APPLE
    model_desc = _MODEL_INFO.get(effective_model, "")
    model_display = f"{effective_model}  ({model_desc})" if model_desc else effective_model

    if apple:
        chip = _apple_chip_name()
        # Determine which backend is actually active
        try:
            import mlx_whisper  # noqa: F401
            backend_str = f"MLX-Whisper  [{chip} — Neural Engine / GPU]"
        except ImportError:
            backend_str = f"faster-whisper  [{chip} — install mlx-whisper for Neural Engine]"
    else:
        backend_str = "faster-whisper + batched inference + VAD"

    _device  = device or ("Apple Silicon (unified memory)" if apple else "auto (cuda if available, else cpu)")
    _compute = compute_type or ("MLX native" if apple else "auto (float16/GPU, int8/CPU)")

    print("\n" + "═" * 64)
    print("⚡  WISHCRIBE — FAST MULTI-SPEAKER TRANSCRIBER")
    print("═" * 64)
    print(f"  File       : {name}")
    print(f"  Whisper    : {model_display}")
    print(f"  Backend    : {backend_str}")
    print(f"  Device     : {_device}")
    if not apple:
        print(f"  Compute    : {_compute}  |  Batch size: {batch_size}")
    print(f"  Language   : {language or 'auto-detect'}")
    print(f"  Transcribe : {'OpenAI API' if use_api else 'Local'}")
    print(f"  Diarize    : {diarize_str}")
    if num_speakers:
        print(f"  Speakers   : {num_speakers} (specified)")
    if initial_prompt:
        snippet = initial_prompt[:50] + ("…" if len(initial_prompt) > 50 else "")
        print(f"  Prompt     : \"{snippet}\"")
    # temperature and beam_size only apply to faster-whisper / openai-whisper.
    # MLX-Whisper uses greedy decoding internally and ignores both — suppress
    # them from the banner on Apple Silicon + mlx installed to avoid misleading users.
    mlx_active = apple and importlib.util.find_spec("mlx_whisper") is not None
    if not mlx_active:
        if temperature != 0.0:
            print(f"  Temperature: {temperature}")
        if beam_size != 5:
            print(f"  Beam size  : {beam_size}")
    if word_timestamps:
        print(f"  Words      : enabled (word-level timestamps in SRT/JSON)")
    if not vad_filter:
        print(f"  VAD        : disabled (--no-vad)")
    print("═" * 64 + "\n")


def _print_summary(segments: list[Segment]):
    from collections import Counter
    from .utils import fmt_time

    print("\n" + "═" * 64)
    print("📋  TRANSCRIPT PREVIEW")
    print("═" * 64)
    prev = None
    for seg in segments[:25]:
        if seg.speaker != prev:
            label = f"[{seg.speaker}] " if seg.speaker else ""
            print(f"\n\033[1m{label}\033[0m{fmt_time(seg.start)}")
            prev = seg.speaker
        print(f"  {seg.text}")
    if len(segments) > 25:
        print(f"\n  … ({len(segments) - 25} more segments in output files)")
    print("═" * 64)

    # Speaker stats (only for diarized output)
    labeled = [s for s in segments if s.speaker]
    if labeled:
        counts = Counter(s.speaker for s in labeled)
        print("\n📊  SPEAKER STATS")
        for speaker, count in sorted(counts.items()):
            total = sum(s.end - s.start for s in labeled if s.speaker == speaker)
            print(f"  {speaker}: {count} segments, ~{fmt_time(total)} of speech")
    print()
