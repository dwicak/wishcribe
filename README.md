# wishcribe ⚡

**Fast multi-speaker audio/video transcription — fully offline after first run.**

`faster-whisper` + `pyannote.audio` + native **Apple Silicon** support (MLX-Whisper).

---

## Features

- 🎙️ **Multi-speaker transcription** — speaker labels per segment (`[SPEAKER_00]`, `[SPEAKER_01]`, …)
- ⚡ **4–8× faster than openai-whisper** via CTranslate2 batched inference + VAD
- 🍎 **Apple Silicon native** — MLX-Whisper auto-selected on M1/M2/M3/M4 (Neural Engine / GPU)
- 📡 **Fully offline** after one-time model download
- 🎯 **Accuracy controls** — `--initial-prompt`, `--temperature`, `--beam-size`
- 🎬 **Video + audio** — mp4, mkv, mov, avi, webm, ts, wmv, flv, mp3, wav, m4a, flac, ogg, aac, opus, wma
- 📄 **Multiple output formats** — `.txt`, `.srt`, `.json`
- 🐍 **Python API + CLI**

---

## Installation

```bash
pip install wishcribe
```

### Apple Silicon (M1/M2/M3/M4) — fastest

```bash
pip install "wishcribe[apple]"
```

This installs `mlx-whisper` for Neural Engine / GPU acceleration. Automatically selected when running on Apple Silicon.

### Legacy fallback

```bash
pip install "wishcribe[legacy]"   # openai-whisper (slower, no CTranslate2 needed)
pip install "wishcribe[api]"      # OpenAI cloud API (no local GPU needed)
```

---

## Quick start

### Step 1 — one-time model download

```bash
wishcribe download --hf-token hf_xxxxxxxxxx
```

This downloads and caches all required models:
- **Whisper** model (default: `large-v2` on non-Apple, `turbo` on Apple Silicon)
- **pyannote** speaker diarization model (~1 GB)

After this, wishcribe works **fully offline forever**.

**HuggingFace setup checklist (first time only):**
1. Sign up → https://huggingface.co/join
2. Accept model license → https://huggingface.co/pyannote/speaker-diarization-community-1
3. Create a Read token → https://huggingface.co/settings/tokens

### Step 2 — transcribe

```bash
wishcribe --video meeting.mp4
```

Output files are saved next to the input: `meeting_transcript.txt` and `meeting_transcript.srt`.

---

## CLI reference

### `wishcribe download`

```bash
wishcribe download [OPTIONS]

Options:
  --hf-token TOKEN      HuggingFace read token (required on first run)
  --model MODEL         Whisper model to download (default: large-v2)
  --force               Delete existing cache and re-download
```

### `wishcribe` / `wishcribe run`

```
wishcribe --video FILE [OPTIONS]
wishcribe run --video FILE [OPTIONS]

Input:
  --video PATH          Path to video or audio file (required)
  --bahasa LANG         Language code: 'id', 'en', 'zh', etc. (default: auto-detect)

Whisper model:
  --model MODEL         tiny | base | small | medium | large | large-v1 |
                        large-v2 | large-v3 | turbo  (default: large-v2)

Accuracy:
  --initial-prompt TEXT Domain context to guide transcription vocabulary and style.
                        E.g. "Medical: hypertension, tachycardia."
                        Disables batched inference (uses beam search instead).
  --temperature FLOAT   Sampling temperature (default: 0.0 = greedy/deterministic).
                        Higher values (0.2-1.0) add diversity but may reduce accuracy.
                        Non-zero temperature disables batched inference.
  --beam-size N         Beam search width (default: 5). Larger = more accurate, slower.
                        Only used when batched inference is disabled.

Speed / hardware:
  --batch-size N        Batch size (default: 16). Lower to 4-8 on OOM.
  --compute-type TYPE   float16 (GPU, fast) | int8 (CPU/low-mem) | float32 (CPU, accurate)
  --device DEVICE       cuda | cpu  (default: auto-detect)

Speaker diarization:
  --hf-token TOKEN      HuggingFace token (or set WISHCRIBE_HF_TOKEN env var)
  --model-path PATH     Local pyannote model folder (skips HuggingFace)
  --speakers N          Number of speakers — improves accuracy when known
  --no-diarize          Skip speaker diarization (no token needed, faster)

Output:
  --output DIR          Output folder (default: same as input file)
  --json                Also save .json transcript
  --no-txt              Skip .txt output
  --no-srt              Skip .srt output
  --quiet               Suppress progress output

Cloud API:
  --use-api             Use OpenAI Whisper API instead of local model
  --api-key KEY         OpenAI API key (required with --use-api)
```

---

## Examples

```bash
# Indonesian meeting with 3 speakers
wishcribe --video rapat.mp4 --bahasa id --speakers 3

# English podcast, no speaker labels needed
wishcribe --video podcast.mp3 --no-diarize

# Medical dictation — guide vocabulary with a prompt
wishcribe --video dictation.mp4 --initial-prompt "Medical: hypertension, tachycardia, bradycardia."

# Higher accuracy — beam search width 10
wishcribe --video interview.mp4 --beam-size 10

# Low-memory CPU-only mode
wishcribe --video lecture.mp4 --device cpu --compute-type int8 --batch-size 4

# Save JSON output too
wishcribe --video meeting.mp4 --json

# Re-download turbo model (e.g. after switching models)
wishcribe download --model turbo --force

# Specify HF token via environment variable (no --hf-token flag needed)
export WISHCRIBE_HF_TOKEN=hf_xxxxxxxxxx
wishcribe --video meeting.mp4
```

---

## Python API

```python
from wishcribe import transcribe, download

# One-time setup
download(hf_token="hf_xxxxxxxxxx")

# Basic transcription (saves .txt and .srt next to the input file)
segments = transcribe("meeting.mp4")

# With options
segments = transcribe(
    "meeting.mp4",
    model="large-v2",           # or "turbo" — auto-selected on Apple Silicon
    language="id",              # BCP-47 language code; None = auto-detect
    num_speakers=2,             # known speaker count improves diarization
    output_dir="./output",      # where to save files
    save_json=True,             # also write .json
)

# Skip diarization (no token needed)
segments = transcribe("lecture.mp4", diarize=False)

# Accuracy controls
segments = transcribe(
    "dictation.mp4",
    initial_prompt="Medical: hypertension, tachycardia.",
    beam_size=10,
)

# Sampling temperature (disables batched inference)
segments = transcribe("podcast.mp3", temperature=0.2)

# Speed controls
segments = transcribe(
    "meeting.mp4",
    batch_size=8,
    compute_type="int8",
    device="cpu",
)

# Each segment has: .start  .end  .speaker  .text
for seg in segments:
    print(f"[{seg.speaker}] {seg.start:.1f}s -> {seg.end:.1f}s  {seg.text}")

# HF token via environment variable
import os
os.environ["WISHCRIBE_HF_TOKEN"] = "hf_xxxxxxxxxx"
segments = transcribe("meeting.mp4")   # token picked up automatically
```

---

## Model guide

| Model | Size | Speed | Accuracy | Notes |
|-------|------|-------|----------|-------|
| `tiny` | 75 MB | ⚡⚡⚡⚡ | ★★ | Fastest; fair accuracy |
| `base` | 139 MB | ⚡⚡⚡ | ★★★ | Good for clear speech |
| `small` | 461 MB | ⚡⚡⚡ | ★★★ | Better accuracy |
| `medium` | 1.4 GB | ⚡⚡ | ★★★★ | Good speed/accuracy balance |
| `turbo` | 1.6 GB | ⚡⚡⚡ | ★★★★ | **Default on Apple Silicon** ⭐ |
| `large-v2` | 2.9 GB | ⚡ | ★★★★★ | **Default on non-Apple** ⭐ |
| `large-v3` | 3.1 GB | ⚡ | ★★★★★ | Newest large model |
| `large-v1` | 2.9 GB | ⚡ | ★★★★ | Original large model |

On **Apple Silicon**, `turbo` is the default — it uses the Neural Engine and is significantly faster than `large-v2` with only a marginal accuracy difference.

---

## Backend selection

wishcribe automatically picks the best available backend:

```
Apple Silicon (M1/M2/M3/M4)
  └── mlx-whisper installed?  → MLX-Whisper  (Neural Engine / GPU) ⚡⚡⚡
  └── else                    → faster-whisper (CPU)

Other platforms
  └── faster-whisper installed? → faster-whisper + batched inference + VAD ⚡⚡
  └── else                      → openai-whisper (fallback, slower) ⚡
```

**MLX-Whisper** (Apple Silicon) automatically selects the right quantized model based on available RAM:

| RAM | Model loaded |
|-----|-------------|
| >= 16 GB | Full precision |
| >= 8 GB | 4-bit quantized |
| < 8 GB | 8-bit quantized |

---

## Accuracy tips

**`--initial-prompt`** injects a text hint before transcription. Use it to:
- Improve domain-specific vocabulary (medical, legal, tech)
- Set consistent casing and punctuation style
- Guide the model when it struggles with acronyms or names

```bash
wishcribe --video standup.mp4 --initial-prompt "Engineering: Kubernetes, CI/CD, pull request, deployment."
```

> **Note:** `--initial-prompt` and non-zero `--temperature` disable batched inference and fall back to standard beam search. This is slower but gives more control.

**`--beam-size`** (default: 5) controls beam search width in the non-batched path. Increasing to 8–10 improves accuracy slightly at the cost of speed.

**`--temperature`** (default: 0.0 = greedy) adds sampling randomness. Useful when the model gets stuck in repetitive output. Try 0.1–0.2.

---

## Output format

### `.txt`

```
[SPEAKER_00] 0:00:01
  Good morning everyone, let's get started.

[SPEAKER_01] 0:00:05
  Thanks for joining. Today we'll cover the Q3 results.
```

### `.srt`

```
1
00:00:01,000 --> 00:00:04,500
[SPEAKER_00] Good morning everyone, let's get started.

2
00:00:05,200 --> 00:00:09,800
[SPEAKER_01] Thanks for joining. Today we'll cover the Q3 results.
```

### `.json`

```json
[
  {"start": 1.0, "end": 4.5, "speaker": "SPEAKER_00", "text": "Good morning everyone, let's get started."},
  {"start": 5.2, "end": 9.8, "speaker": "SPEAKER_01", "text": "Thanks for joining. Today we'll cover the Q3 results."}
]
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `WISHCRIBE_HF_TOKEN` | HuggingFace token — avoids passing `--hf-token` every time |
| `HF_TOKEN` | Fallback if `WISHCRIBE_HF_TOKEN` is not set |
| `OMP_NUM_THREADS` | CPU thread count — auto-tuned to performance cores on Apple Silicon |

Add to `~/.zshrc` or `~/.bashrc`:

```bash
export WISHCRIBE_HF_TOKEN="hf_xxxxxxxxxx"
```

---

## Troubleshooting

**`Whisper model 'large-v2' is not cached`**
```bash
wishcribe download --model large-v2
# or force a clean re-download:
wishcribe download --model large-v2 --force
```

**`Diarization model not found in local cache`**
```bash
wishcribe download --hf-token hf_xxxxxxxxxx
```

Make sure you've accepted the license at https://huggingface.co/pyannote/speaker-diarization-community-1

**CUDA out of memory**
```bash
wishcribe --video meeting.mp4 --batch-size 4
# or CPU mode:
wishcribe --video meeting.mp4 --device cpu --compute-type int8
```

**`mlx-whisper not installed`** (Apple Silicon)
```bash
pip install mlx-whisper
# or reinstall with apple extra:
pip install "wishcribe[apple]"
```

**Inaccurate transcription**
```bash
# Add domain context
wishcribe --video meeting.mp4 --initial-prompt "Context: ..."

# Increase beam search
wishcribe --video meeting.mp4 --beam-size 10

# Specify language (skip auto-detect)
wishcribe --video meeting.mp4 --bahasa id
```

---

## Supported formats

**Video:** mp4, mkv, mov, avi, webm, ts, wmv, flv

**Audio:** mp3, wav, m4a, flac, ogg, aac, opus, wma

---

## Requirements

- Python 3.9+
- [moviepy](https://github.com/Zulko/moviepy) >= 2.0 + ffmpeg (audio extraction)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) >= 1.0 (transcription)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) >= 3.1 (diarization)
- [torch](https://pytorch.org/) >= 2.0

**Apple Silicon only:**
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) >= 0.4 (`pip install "wishcribe[apple]"`)

---

## Changelog

### v1.2.0
- 🍎 **MLX-Whisper backend** — Apple Silicon M1/M2/M3/M4, auto-selected when `mlx-whisper` is installed
- 🍎 **Default model `turbo`** on Apple Silicon (faster + accurate via Neural Engine)
- 🍎 **OMP_NUM_THREADS** auto-tuned to physical performance cores on Apple Silicon
- 🍎 **Auto-quantization** — picks 4-bit or 8-bit MLX model based on available unified memory
- 🎯 **`--initial-prompt`** — inject domain context for specialised vocabulary
- 🎯 **`--temperature`** — sampling temperature (0.0 = true greedy, always deterministic)
- 🎯 **`--beam-size`** — beam search width for non-batched path
- 🛠️ **Pipeline memory leak fix** — GPU memory freed even when `BatchedInferencePipeline.transcribe()` raises mid-batch
- 🛠️ **Diarization memory fix** — GPU/MPS memory freed even if segment extraction fails
- 🛠️ **`wishcribe download`** mirrors transcription default on Apple Silicon (turbo, not large-v2)
- 🛠️ **`large-v1`** added to CLI model choices
- 🛠️ **MLX cache purge** on `--force` re-download
- 🛠️ **`moviepy>=2.0.0`** pin (2.x API required)

### v1.1.1
- Fix: `_suppress_fd2()` degrades gracefully on unusual fd environments
- Fix: `sys.exit()` replaced with `RuntimeError` throughout (library-safe)
- Fix: GPU memory freed after transcription before diarization loads

### v1.1.0
- faster-whisper backend (4–8x speedup over openai-whisper)
- Batched inference + VAD (silence filtering)
- float16/int8 auto-selection by device
- `wishcribe download` pre-caches all models

### v1.0.0
- Initial release (openai-whisper + pyannote.audio)

---

## License

MIT — see [LICENSE](LICENSE)

## Links

- **PyPI:** https://pypi.org/project/wishcribe/
- **GitHub:** https://github.com/dwicak/wishcribe
- **Issues:** https://github.com/dwicak/wishcribe/issues
