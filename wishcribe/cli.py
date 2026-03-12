"""
wishcribe — command-line interface

Commands
--------
  wishcribe download   Pre-download all model weights (run once before first use)
  wishcribe run        Transcribe a video/audio file (default command)

Examples
--------
  # One-time setup
  wishcribe download --hf-token hf_xxx

  # Transcribe with speaker labels (GPU, fast)
  wishcribe --video meeting.mp4 --bahasa id

  # Transcribe without diarization (no token needed)
  wishcribe --video meeting.mp4 --no-diarize

  # Low GPU memory mode
  wishcribe --video meeting.mp4 --batch-size 4 --compute-type int8

  # CPU-only mode
  wishcribe --video meeting.mp4 --device cpu
"""

import argparse
import os
import sys

from .core import transcribe
from .download import download_all
from .transcribe import DEFAULT_WHISPER_MODEL

_MODEL_CHOICES = [
    "tiny", "base", "small", "medium",
    "large", "large-v2", "large-v3", "turbo", "distil-large-v3"
]


def _resolve_token(args_token):
    return args_token or os.environ.get("WISHCRIBE_HF_TOKEN") or os.environ.get("HF_TOKEN") or None


# ── Shared argument groups ────────────────────────────────────────────────────

def _add_model_args(parser):
    diar = parser.add_argument_group(
        "Diarization",
        "Token read from --hf-token or WISHCRIBE_HF_TOKEN env var.",
    )
    diar.add_argument("--hf-token",   default=None, metavar="TOKEN",
                      help="HuggingFace read token")
    diar.add_argument("--model-path", default=None, metavar="PATH",
                      help="Local pyannote model folder path")

    wh = parser.add_argument_group("Whisper model")
    wh.add_argument("--model", default=DEFAULT_WHISPER_MODEL, choices=_MODEL_CHOICES,
                    help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})")


def _add_speed_args(parser):
    sp = parser.add_argument_group("Speed / hardware")
    sp.add_argument("--batch-size",   type=int, default=16, metavar="N",
                    help="Inference batch size — higher=faster on GPU (default: 16, lower to 4-8 if OOM)")
    sp.add_argument("--compute-type", default=None, metavar="TYPE",
                    choices=["float16", "int8", "float32"],
                    help="Compute type: float16 (GPU, fast), int8 (CPU/low-mem), float32 (CPU, accurate)")
    sp.add_argument("--device",       default=None, choices=["cuda", "cpu"],
                    help="Device to use (default: auto-detect cuda if available)")


# ── download subcommand ───────────────────────────────────────────────────────

def _cmd_download(args):
    try:
        download_all(
            hf_token=_resolve_token(args.hf_token),
            model=args.model,
            model_path=args.model_path,
            force=args.force,
            verbose=True,
        )
    except RuntimeError as exc:
        print(f"\n❌ Download failed: {exc}")
        sys.exit(1)


def _build_download_parser(subparsers):
    p = subparsers.add_parser(
        "download",
        help="Pre-download and cache all model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_model_args(p)
    p.add_argument("--force", action="store_true",
                   help="Delete existing cache and re-download (use when switching models)")
    p.set_defaults(func=_cmd_download)
    return p


# ── run subcommand ────────────────────────────────────────────────────────────

def _cmd_run(args):
    if args.use_api and not args.api_key:
        print("❌ --api-key is required when using --use-api")
        sys.exit(1)

    try:
        transcribe(
            input_path=args.video,
            hf_token=_resolve_token(args.hf_token),
            model_path=args.model_path,
            model=args.model,
            language=args.bahasa,
            num_speakers=args.speakers,
            diarize=not args.no_diarize,
            output_dir=getattr(args, "output", None),
            use_api=args.use_api,
            api_key=args.api_key,
            save_txt=not args.no_txt,
            save_srt=not args.no_srt,
            save_json=args.json,
            verbose=not getattr(args, "quiet", False),
            batch_size=getattr(args, "batch_size", 16),
            compute_type=getattr(args, "compute_type", None),
            device=getattr(args, "device", None),
        )
    except FileNotFoundError as exc:
        print(f"\n❌ File not found: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n❌ Invalid argument: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")
        sys.exit(130)


def _build_run_parser(subparsers):
    p = subparsers.add_parser(
        "run",
        help="Transcribe a video or audio file with speaker labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to video or audio file")
    _add_model_args(p)
    _add_speed_args(p)
    p.add_argument("--bahasa",     default=None, metavar="LANG",
                   help="Language code e.g. 'id', 'en' (default: auto-detect)")
    p.add_argument("--speakers",   type=int, default=None,
                   help="Number of speakers — improves accuracy when known")
    p.add_argument("--no-diarize", action="store_true",
                   help="Skip speaker diarization — no HuggingFace token needed")
    p.add_argument("--use-api",    action="store_true",
                   help="Use OpenAI Whisper API instead of local model")
    p.add_argument("--api-key",    default=None,
                   help="OpenAI API key (required with --use-api)")
    p.add_argument("--quiet",      action="store_true", help="Suppress progress output")
    out = p.add_argument_group("Output")
    out.add_argument("--output",   default=None, help="Output folder (default: same as input)")
    out.add_argument("--json",     action="store_true", help="Also save .json")
    out.add_argument("--no-txt",   action="store_true", help="Skip .txt output")
    out.add_argument("--no-srt",   action="store_true", help="Skip .srt output")
    p.set_defaults(func=_cmd_run)
    return p


# ── Root parser ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="wishcribe",
        description=(
            "⚡ wishcribe — Fast multi-speaker transcription\n"
            "   faster-whisper + pyannote.audio (fully offline after first run)\n\n"
            f"   Default model: {DEFAULT_WHISPER_MODEL} (best accuracy)\n\n"
            "Commands:\n"
            "  download   Pre-download all models (run once)\n"
            "  run        Transcribe a file\n\n"
            "Quick start:\n"
            "  wishcribe download --hf-token hf_xxx   # one-time setup\n"
            "  wishcribe --video meeting.mp4           # transcribe"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")
    _build_download_parser(subparsers)
    _build_run_parser(subparsers)

    # Legacy shorthand: wishcribe --video file.mp4 (without 'run' subcommand)
    parser.add_argument("--video",        default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hf-token",     default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-path",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model",        default=DEFAULT_WHISPER_MODEL,
                        choices=_MODEL_CHOICES,            help=argparse.SUPPRESS)
    parser.add_argument("--bahasa",       default=None,  help=argparse.SUPPRESS)
    parser.add_argument("--speakers",     type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-diarize",   action="store_true",    help=argparse.SUPPRESS)
    parser.add_argument("--use-api",      action="store_true",    help=argparse.SUPPRESS)
    parser.add_argument("--api-key",      default=None,  help=argparse.SUPPRESS)
    parser.add_argument("--output",       default=None,  help=argparse.SUPPRESS)
    parser.add_argument("--json",         action="store_true",    help=argparse.SUPPRESS)
    parser.add_argument("--no-txt",       action="store_true",    help=argparse.SUPPRESS)
    parser.add_argument("--no-srt",       action="store_true",    help=argparse.SUPPRESS)
    parser.add_argument("--quiet",        action="store_true",    help=argparse.SUPPRESS)
    parser.add_argument("--batch-size",   type=int, default=16,  help=argparse.SUPPRESS)
    parser.add_argument("--compute-type", default=None,           help=argparse.SUPPRESS)
    parser.add_argument("--device",       default=None,           help=argparse.SUPPRESS)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
        return

    if args.video:
        _cmd_run(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
