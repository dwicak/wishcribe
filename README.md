# wishcribe

Fast multi-speaker audio/video transcription — **faster-whisper + pyannote.audio**, fully offline after first run.

**v1.1.0** upgrades the transcription backend to [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2), giving **4–8× faster** transcription at the same accuracy, with batched inference and VAD silence-filtering built in.

```
[SPEAKER_00] 00:00:01
  Selamat datang di rapat hari ini.

[SPEAKER_01] 00:00:05
  Terima kasih. Mari kita mulai.

[SPEAKER_00] 00:00:10
  Baik, topik pertama adalah anggaran kuartal ini.
```

Or without speaker labels (no HuggingFace token needed):

```
00:00:01
  Selamat datang di rapat hari ini.

00:00:05
  Terima kasih. Mari kita mulai.
```

---

## What's new in v1.1.0

- **4–8× faster transcription** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend)
- **Batched inference** — multiple audio chunks transcribed in parallel (`--batch-size`, default 16)
- **VAD filtering** — silence is skipped before transcription, reducing hallucination
- **Auto compute type** — `float16` on modern GPU, `int8` on CPU (auto-detected, overridable)
- **New models** — `large-v3`, `turbo` (large-v3-turbo), `distil-large-v3`
- **New flags** — `--batch-size`, `--compute-type`, `--device`
- **GPU memory freed** between transcription and diarization (avoids OOM on 8 GB VRAM)
- **openai-whisper fallback** — automatically used if faster-whisper is not installed

---

## Requirements

- Python 3.9 or higher
- ffmpeg
- ~4 GB free disk space (for model weights)
- Internet connection (first run only — fully offline after)

---

## Installing Python

### Windows

1. Go to **https://www.python.org/downloads/windows/**
2. Click **"Download Python 3.x.x"** (latest version)
3. Run the installer
4. ⚠️ **Important:** Check **"Add Python to PATH"** before clicking Install
5. Open **Command Prompt** and verify:
   ```
   python --version
   pip --version
   ```

> **Tip:** Use **Command Prompt** or **PowerShell** to run wishcribe commands.  
> To open: press `Win + R`, type `cmd`, press Enter.

### macOS

```bash
python3 --version          # check if already installed
brew install python        # install via Homebrew if not
```

> If you don't have Homebrew: https://brew.sh

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

## Installing ffmpeg

ffmpeg is required to extract audio from video files.

### Windows

1. Go to **https://ffmpeg.org/download.html** → Windows → BtbN builds
2. Download `ffmpeg-master-latest-win64-gpl.zip`
3. Extract to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your system PATH
5. Verify: `ffmpeg -version`

### macOS

```bash
brew install ffmpeg
```

### Ubuntu / Debian

```bash
sudo apt install ffmpeg
```

---

## Installation

```bash
pip install wishcribe
```

> **Windows:** If `pip` is not found, try `pip3` or `python -m pip install wishcribe`

---

## Two modes

| Mode | Command | HuggingFace token? | Output |
|---|---|---|---|
| **With speaker labels** | *(default)* | ✅ Required | `[SPEAKER_00]`, `[SPEAKER_01]` … |
| **Transcription only** | `--no-diarize` | ❌ Not needed | Timestamps only |

Use `--no-diarize` if you want a transcript without identifying who speaks when.

---

## HuggingFace setup (required for speaker labels)

> Skip this section if using `--no-diarize` only.

wishcribe uses **pyannote/speaker-diarization-community-1** for speaker detection. You need to accept the license once before downloading:

1. Sign up: **https://huggingface.co/join**
2. Accept license: **https://huggingface.co/pyannote/speaker-diarization-community-1**
3. Create a Read token: **https://huggingface.co/settings/tokens**

> ⚠️ The license must be accepted **before** running `wishcribe download`. Without it the download fails with a 401 error.

---

## Quick start

### With speaker labels (full mode)

```bash
# Step 1 — download all models once (~4 GB total, then fully offline)
wishcribe download --hf-token hf_xxx

# Step 2 — transcribe
wishcribe --video meeting.mp4 --bahasa id --speakers 2
```

### Without speaker labels (no token needed)

```bash
wishcribe --video meeting.mp4 --bahasa id --no-diarize
```

### Low GPU memory (8 GB VRAM or less)

```bash
wishcribe --video meeting.mp4 --batch-size 4 --compute-type int8
```

### CPU-only

```bash
wishcribe --video meeting.mp4 --device cpu
```

---

## Avoid typing --hf-token every time

Set your token as an environment variable once:

### macOS / Linux

```bash
# Add to ~/.zshrc or ~/.bash_profile
export WISHCRIBE_HF_TOKEN="hf_xxx"

source ~/.zshrc    # reload
```

### Windows

```bash
# Current session only
set WISHCRIBE_HF_TOKEN=hf_xxx

# Permanently: Win + S → "Environment Variables" → New → WISHCRIBE_HF_TOKEN
```

After setting it, `--hf-token` is no longer needed:

```bash
wishcribe --video meeting.mp4 --bahasa id --speakers 2
```

> 🔒 Environment variables live on your machine only — never committed to Git or uploaded anywhere.

---

## CLI reference

### `wishcribe download`

Pre-download and cache all model weights (run once, then fully offline).

```bash
wishcribe download --hf-token hf_xxx                   # default large-v2 + diarization
wishcribe download --hf-token hf_xxx --model turbo     # download turbo instead
wishcribe download --hf-token hf_xxx --force           # delete cache and re-download fresh
wishcribe download --model-path /path/to/local-model   # use a local pyannote folder
```

### `wishcribe --video` (transcribe)

```bash
wishcribe --video meeting.mp4 --bahasa id --speakers 2        # full mode
wishcribe --video meeting.mp4 --bahasa id --no-diarize        # no speaker labels
wishcribe --video meeting.mp4 --model turbo                   # faster model
wishcribe --video meeting.mp4 --batch-size 4                  # lower GPU memory
wishcribe --video meeting.mp4 --compute-type int8             # quantized, less VRAM
wishcribe --video meeting.mp4 --device cpu                    # CPU only
wishcribe --video meeting.mp4 --use-api --api-key sk-xxx      # OpenAI cloud API
wishcribe --video meeting.mp4 --output ./results --json       # custom folder + JSON
```

### All options

| Argument | Description | Default |
|---|---|---|
| `--video` | Path to video or audio file **(required)** | — |
| `--hf-token` | HuggingFace token (or set `WISHCRIBE_HF_TOKEN` env var) | — |
| `--no-diarize` | Skip speaker diarization — no token needed | `False` |
| `--model` | `tiny` / `base` / `small` / `medium` / `large` / `large-v2` / `large-v3` / `turbo` / `distil-large-v3` | `large-v2` |
| `--bahasa` | Language code e.g. `id`, `en` | auto-detect |
| `--speakers` | Number of speakers (improves accuracy when known) | auto |
| `--batch-size` | Transcription batch size — higher = faster on GPU | `16` |
| `--compute-type` | `float16` (GPU fast) / `int8` (low-mem) / `float32` (CPU) | auto |
| `--device` | `cuda` or `cpu` | auto |
| `--model-path` | Path to local pyannote model folder | — |
| `--output` | Output folder | same as input |
| `--use-api` | Use OpenAI Whisper API (no local GPU) | `False` |
| `--api-key` | OpenAI API key (required with `--use-api`) | — |
| `--json` | Also save `.json` output | `False` |
| `--no-txt` | Skip `.txt` output | `False` |
| `--no-srt` | Skip `.srt` output | `False` |
| `--quiet` | Suppress progress output | `False` |

---

## Python API

```python
from wishcribe import download, transcribe

# ── One-time setup ─────────────────────────────────────────────
download(hf_token="hf_xxx")
# download(hf_token="hf_xxx", model="turbo")    # different model
# download(hf_token="hf_xxx", force=True)       # re-download fresh

# ── With speaker labels (default) ─────────────────────────────
segments = transcribe(
    "meeting.mp4",
    language="id",
    num_speakers=2,       # optional but improves accuracy
    output_dir="./out",
)

# ── Without speaker labels ─────────────────────────────────────
segments = transcribe("meeting.mp4", diarize=False, language="id")

# ── Speed / hardware controls ──────────────────────────────────
segments = transcribe(
    "meeting.mp4",
    model="turbo",          # large-v3-turbo: fast + accurate
    batch_size=16,          # default — lower to 4 if OOM
    compute_type="float16", # auto-detected; "int8" for CPU/low-mem
    device="cuda",          # auto-detected
)

# ── OpenAI cloud API ───────────────────────────────────────────
segments = transcribe("meeting.mp4", use_api=True, api_key="sk-xxx")

# ── Output control ─────────────────────────────────────────────
segments = transcribe(
    "meeting.mp4",
    save_txt=True,   # _transcript.txt  (default on)
    save_srt=True,   # _transcript.srt  (default on)
    save_json=True,  # _transcript.json (default off)
)

# ── Iterate results ────────────────────────────────────────────
for seg in segments:
    print(f"[{seg.speaker}] {seg.start:.1f}s — {seg.text}")

# Each Segment: .start  .end  .speaker  .text
# seg.to_dict() → {start, end, duration, speaker, text}
```

---

## Whisper model guide

| Model | Size | Speed | Accuracy | Notes |
|---|---|---|---|---|
| `tiny` | 75 MB | ⚡⚡⚡⚡ | Fair | Fast testing / drafts |
| `base` | 139 MB | ⚡⚡⚡ | Good | |
| `small` | 461 MB | ⚡⚡ | Better | |
| `medium` | 1.4 GB | ⚡ | Very good | Recommended for CPU |
| **`large-v2`** | **2.9 GB** | — | **Best ⭐** | **Default — highest accuracy** |
| `large-v3` | 3.1 GB | — | Best | Newest large model |
| `turbo` | 1.6 GB | ⚡⚡ | Very good | Best speed/accuracy ratio |
| `distil-large-v3` | 1.5 GB | ⚡⚡ | Very good | Distilled, near large-v2 |

**Recommendation:** use `large-v2` (default) for best accuracy, or `turbo` for a fast/accurate balance.

---

## Output files

| File | Description |
|---|---|
| `<name>_transcript.txt` | Human-readable, grouped by speaker turn |
| `<name>_transcript.srt` | SRT subtitles — importable into video editors |
| `<name>_transcript.json` | JSON array with `start`, `end`, `duration`, `speaker`, `text` (opt-in with `--json`) |

---

## Supported formats

**Video:** `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.ts`, `.wmv`, `.flv`  
**Audio:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.opus`, `.wma`  
**Languages:** 90+ (Whisper auto-detects if `--bahasa` is not set)

---

## Using a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv wishcribe-env
source wishcribe-env/bin/activate
pip install wishcribe

# Windows
python -m venv wishcribe-env
wishcribe-env\Scripts\activate
pip install wishcribe
```

Activate at the start of each terminal session:

```bash
source wishcribe-env/bin/activate    # macOS / Linux
wishcribe-env\Scripts\activate        # Windows
```

---

## Troubleshooting

**`401 Client Error` / `Access to model is restricted`**  
Accept the license at https://huggingface.co/pyannote/speaker-diarization-community-1 and verify your token is a valid Read token.

**`wishcribe: command not found`**  
```bash
pip install --upgrade wishcribe
# Windows fallback:
python -m wishcribe --video meeting.mp4
```

**`ffmpeg not found`**  
Install ffmpeg for your OS (see above).

**Out of GPU memory (CUDA OOM)**  
Lower batch size or use int8 quantization:
```bash
wishcribe --video meeting.mp4 --batch-size 4 --compute-type int8
# or CPU only:
wishcribe --video meeting.mp4 --device cpu
```

**Dependency conflicts (e.g. with TensorFlow)**  
Use a virtual environment to isolate wishcribe cleanly.

**Want to skip HuggingFace entirely?**  
Use `--no-diarize` — no token required, works right after `pip install wishcribe`.

---

## License

MIT — free to use, modify, and distribute.
