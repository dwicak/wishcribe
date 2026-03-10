"""
wishcribe — command-line interface

Commands
--------
  wishcribe download   Pre-download all model weights (run once before first use)
  wishcribe run        Transcribe a video/audio file (default command)

HuggingFace token
-----------------
  Your token can be supplied in 3 ways (in priority order):
    1. --hf-token flag         : wishcribe --video file.mp4 --hf-token hf_xxx
    2. Environment variable    : export WISHCRIBE_HF_TOKEN=hf_xxx  (add to ~/.zshrc)
    3. Omit entirely           : only works if pyannote model is fully cached

  The environment variable is the recommended approach — set it once in
  ~/.zshrc and never type --hf-token again. It is never committed to Git.

Examples
--------
  # One-time setup — download all models
  wishcribe download --hf-token hf_xxx

  # Transcribe (token auto-read from WISHCRIBE_HF_TOKEN env var)
  wishcribe --video meeting.mp4

  # Transcribe with explicit token
  wishcribe --video meeting.mp4 --hf-token hf_xxx
"""

import argparse
import os
import sys

from .core import transcribe
from .download import download_all
from .transcribe import DEFAULT_WHISPER_MODEL


def _resolve_token(args_token: str) -> str:
    """
    Resolve HuggingFace token in priority order:
      1. --hf-token argument
      2. WISHCRIBE_HF_TOKEN environment variable
      3. None (may still work if model fully cached)
    """
    return args_token or os.environ.get("WISHCRIBE_HF_TOKEN") or None


# ── Shared arguments ──────────────────────────────────────────────────────────

def _add_model_args(parser):
    diar = parser.add_argument_group(
        "Diarization model",
        "Token is read from --hf-token or WISHCRIBE_HF_TOKEN environment variable.",
    )
    diar.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace token (or set WISHCRIBE_HF_TOKEN env var instead)",
    )
    diar.add_argument(
        "--model-path", default=None, metavar="PATH",
        help="Path to a locally saved pyannote model folder",
    )

    whisper = parser.add_argument_group("Whisper model")
    whisper.add_argument(
        "--model", default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=(
            f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})\n"
            "  tiny   —  75 MB  very fast  fair accuracy\n"
            "  base   — 139 MB  fast       good accuracy\n"
            "  small  — 461 MB  moderate   better accuracy\n"
            "  medium — 1.4 GB  slow       very good accuracy\n"
            "  large  — 2.9 GB  slowest    BEST accuracy ⭐"
        ),
    )


# ── `download` subcommand ─────────────────────────────────────────────────────

def _cmd_download(args):
    download_all(
        hf_token=_resolve_token(args.hf_token),
        model=args.model,
        model_path=args.model_path,
        force=args.force,
        verbose=True,
    )


def _build_download_parser(subparsers):
    p = subparsers.add_parser(
        "download",
        help="Pre-download and cache all model weights (run once before first use)",
        description=(
            "Download and cache both the Whisper transcription model and the\n"
            "pyannote speaker diarization model.\n\n"
            "Token can be passed via --hf-token or WISHCRIBE_HF_TOKEN environment variable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_model_args(p)
    p.add_argument(
        "--force", action="store_true",
        help="Re-download models even if already cached (use when switching model versions)",
    )
    p.set_defaults(func=_cmd_download)
    return p


# ── `run` subcommand ──────────────────────────────────────────────────────────

def _cmd_run(args):
    if args.use_api and not args.api_key:
        print("❌ --api-key is required when using --use-api")
        sys.exit(1)

    transcribe(
        input_path=args.video,
        hf_token=_resolve_token(args.hf_token),
        model_path=args.model_path,
        model=args.model,
        language=args.bahasa,
        num_speakers=args.speakers,
        diarize=not args.no_diarize,
        output_dir=args.output,
        use_api=args.use_api,
        api_key=args.api_key,
        save_txt=not args.no_txt,
        save_srt=not args.no_srt,
        save_json=args.json,
        verbose=True,
    )


def _build_run_parser(subparsers):
    p = subparsers.add_parser(
        "run",
        help="Transcribe a video or audio file with speaker labels (default command)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to video or audio file")
    _add_model_args(p)
    p.add_argument("--bahasa", default=None, metavar="LANG",
                   help="Language code e.g. 'id', 'en'  (default: auto-detect)")
    p.add_argument("--speakers", type=int, default=None,
                   help="Number of speakers — optional, improves accuracy when known")
    p.add_argument("--no-diarize", action="store_true",
                   help="Skip speaker diarization — transcribe only, no HuggingFace token needed")
    p.add_argument("--use-api", action="store_true",
                   help="Use OpenAI Whisper API instead of local model")
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key (required with --use-api)")
    out = p.add_argument_group("Output")
    out.add_argument("--output", default=None,
                     help="Output folder (default: same as input file)")
    out.add_argument("--json",   action="store_true", help="Also save .json")
    out.add_argument("--no-txt", action="store_true", help="Skip .txt output")
    out.add_argument("--no-srt", action="store_true", help="Skip .srt output")
    p.set_defaults(func=_cmd_run)
    return p


# ── Root parser + legacy fallback ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="wishcribe",
        description=(
            "✍️  wishcribe — Multi-speaker transcription (Whisper + pyannote, offline)\n"
            f"   Default Whisper model: {DEFAULT_WHISPER_MODEL} (best accuracy)\n\n"
            "Token: pass --hf-token OR set HF_TOKEN environment variable\n"
            "  export WISHCRIBE_HF_TOKEN=hf_xxx   (add to ~/.zshrc — never exposed to GitHub)\n\n"
            "Commands:\n"
            "  download   Pre-download all models (run once)\n"
            "  run        Transcribe a file (default)\n\n"
            "Quick start:\n"
            "  wishcribe download --hf-token hf_xxx   # one-time setup\n"
            "  wishcribe --video meeting.mp4           # transcribe"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")
    _build_download_parser(subparsers)
    _build_run_parser(subparsers)

    # Legacy shorthand: wishcribe --video file.mp4 (without 'run')
    parser.add_argument("--video",      default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hf-token",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model", default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--bahasa",   default=None,  help=argparse.SUPPRESS)
    parser.add_argument("--speakers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-diarize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-api",  action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--api-key",  default=None,  help=argparse.SUPPRESS)
    parser.add_argument("--output",   default=None,  help=argparse.SUPPRESS)
    parser.add_argument("--json",     action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-txt",   action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-srt",   action="store_true", help=argparse.SUPPRESS)

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
