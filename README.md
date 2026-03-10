# Wishcribe

Multi-speaker audio/video transcription — **Whisper large + pyannote.audio**, fully offline after first run.

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

## Requirements

- Python 3.9 or higher
- ffmpeg
- 4 GB free disk space (for model weights)
- Internet connection (first run only)

---

## Installing Python

### Windows

1. Go to **https://www.python.org/downloads/windows/**
2. Click **"Download Python 3.x.x"** (latest version)
3. Run the installer
4. ⚠️ **Important:** On the first screen, check **"Add Python to PATH"** before clicking Install
5. Click **"Install Now"**
6. Once done, open **Command Prompt** and verify:
   ```
   python --version
   pip --version
   ```
   Both should print a version number.

> **Tip for Windows:** Use **Command Prompt** or **PowerShell** to run wishcribe commands.  
> To open Command Prompt: press `Win + R`, type `cmd`, press Enter.

### macOS

```bash
# Check if Python is already installed
python3 --version

# If not installed, use Homebrew
brew install python
```

> If you don't have Homebrew: https://brew.sh

### Ubuntu / Debian Linux

```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

## Installing ffmpeg

ffmpeg is required to extract audio from video files.

### Windows

1. Go to **https://ffmpeg.org/download.html**
2. Click **"Windows"** → **"Windows builds by BtbN"**
3. Download `ffmpeg-master-latest-win64-gpl.zip`
4. Extract the zip file to `C:\ffmpeg`
5. Add ffmpeg to PATH:
   - Press `Win + S` → search **"Environment Variables"**
   - Click **"Edit the system environment variables"**
   - Click **"Environment Variables"**
   - Under **"System variables"**, find **Path** → click **Edit**
   - Click **New** → type `C:\ffmpeg\bin`
   - Click OK on all windows
6. Open a new Command Prompt and verify:
   ```
   ffmpeg -version
   ```

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

Once Python and ffmpeg are installed:

```bash
pip install wishcribe
```

> **Windows users:** If `pip` is not found, try `pip3` or `python -m pip install wishcribe`

---

## Two modes of transcription

Wishcribe supports two modes:

| Mode | Command | HuggingFace token? | Output |
|---|---|---|---|
| **With speaker labels** | *(default)* | ✅ Required | `[SPEAKER_00]`, `[SPEAKER_01]` … |
| **Transcription only** | `--no-diarize` | ❌ Not needed | Timestamps only, no speaker labels |

Use `--no-diarize` if you just want a fast transcript without identifying who speaks when.

---

## HuggingFace setup (required for speaker labels)

> Skip this section if you only want to use `--no-diarize` mode.

Wishcribe uses pyannote.audio for speaker detection. You need to accept the model license on HuggingFace before downloading.

1. Sign up at **https://huggingface.co/join**
2. Accept license: **https://huggingface.co/pyannote/speaker-diarization-community-1**
3. Create a Read token: **https://huggingface.co/settings/tokens**

> ⚠️ **The license must be accepted before running `wishcribe download`.** Without it, the download will fail with a 401 error.

---

## Quick start

### With speaker labels (full mode)

```bash
# Step 1 — Download all models once
wishcribe download --hf-token hf_xxx

# Step 2 — Transcribe
wishcribe --video meeting.mp4 --bahasa id --speakers 2
```

### Without speaker labels (no token needed)

```bash
# No download step needed — just transcribe
wishcribe --video meeting.mp4 --bahasa id --no-diarize
```

---

## Avoid typing --hf-token every time

Set your token as an environment variable once and wishcribe will read it automatically:

### macOS / Linux

```bash
# Add this to your ~/.zshrc or ~/.bash_profile
export WISHCRIBE_HF_TOKEN="hf_xxx"

# Reload
source ~/.zshrc
```

### Windows

```bash
# In Command Prompt (current session only)
set WISHCRIBE_HF_TOKEN=hf_xxx

# Or permanently via System Environment Variables:
# Win + S → "Environment Variables" → New → Name: WISHCRIBE_HF_TOKEN, Value: hf_xxx
```

After setting it, run without `--hf-token`:

```bash
wishcribe --video meeting.mp4 --bahasa id --speakers 2
```

> 🔒 **Your token is safe** — environment variables live on your machine only and are never committed to Git or uploaded to GitHub.

---

## Usage — CLI

### Download command

```bash
# Download default Whisper large model + pyannote diarization model
wishcribe download --hf-token hf_xxx

# Download a smaller/faster Whisper model instead
wishcribe download --hf-token hf_xxx --model medium

# Use a manually downloaded pyannote model folder
wishcribe download --model-path /path/to/pyannote-model
```

### Transcribe command

```bash
# With speaker labels (default)
wishcribe --video meeting.mp4 --bahasa id --speakers 2

# Without speaker labels — no HuggingFace token needed, faster
wishcribe --video meeting.mp4 --bahasa id --no-diarize

# Override Whisper model
wishcribe --video meeting.mp4 --model medium

# Use OpenAI API for transcription
wishcribe --video meeting.mp4 --use-api --api-key sk-xxx

# Save to a custom folder + include JSON
wishcribe --video meeting.mp4 --output ./results --json
```

### All options

| Argument | Description | Default |
|---|---|---|
| `--video` | Path to video or audio file **(required)** | — |
| `--hf-token` | HuggingFace token (or set `WISHCRIBE_HF_TOKEN` env var) | — |
| `--no-diarize` | Skip speaker diarization — no token needed | `False` |
| `--model-path` | Path to local pyannote model folder | — |
| `--model` | `tiny`/`base`/`small`/`medium`/`large` | **`large`** |
| `--bahasa` | Language code e.g. `id`, `en` | auto-detect |
| `--speakers` | Number of speakers (optional, ignored with `--no-diarize`) | auto |
| `--output` | Output folder | same as input |
| `--use-api` | Use OpenAI Whisper API | `False` |
| `--api-key` | OpenAI API key (with `--use-api`) | — |
| `--json` | Also save `.json` | `False` |
| `--no-txt` | Skip `.txt` output | `False` |
| `--no-srt` | Skip `.srt` output | `False` |

---

## Usage — Python

```python
from wishcribe import download, transcribe

# ── With speaker labels ────────────────────────────────────────
# Step 1 — download models once
download(hf_token="hf_xxx")

# Step 2 — transcribe with speaker labels
segments = transcribe(
    "meeting.mp4",
    hf_token="hf_xxx",     # or set WISHCRIBE_HF_TOKEN env var
    model="large",          # default — best accuracy
    language="id",
    num_speakers=2,
    output_dir="./out",
)

# ── Without speaker labels ─────────────────────────────────────
# No download step needed, no token needed
segments = transcribe(
    "meeting.mp4",
    diarize=False,
    language="id",
)

for seg in segments:
    if seg.speaker:
        print(f"[{seg.speaker}] {seg.start:.1f}s  {seg.text}")
    else:
        print(f"{seg.start:.1f}s  {seg.text}")
```

---

## Using a virtual environment (recommended)

To avoid conflicts with other Python packages on your system:

### Windows
```bash
python -m venv wishcribe-env
wishcribe-env\Scripts\activate
pip install wishcribe
```

### macOS / Linux
```bash
python3 -m venv wishcribe-env
source wishcribe-env/bin/activate
pip install wishcribe
```

Every time you open a new terminal, activate the environment first:
```bash
# Windows
wishcribe-env\Scripts\activate

# macOS / Linux
source wishcribe-env/bin/activate
```

---

## Whisper model guide

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `tiny` | 75 MB | Very fast | Fair |
| `base` | 139 MB | Fast | Good |
| `small` | 461 MB | Moderate | Better |
| `medium` | 1.4 GB | Slow | Very good |
| **`large`** | **2.9 GB** | **Slowest** | **Best ⭐ (default)** |

---

## Output files

| File | Description |
|---|---|
| `<n>_transcript.txt` | Plain text grouped by speaker (or by time if `--no-diarize`) |
| `<n>_transcript.srt` | SRT subtitles with speaker labels (or without if `--no-diarize`) |
| `<n>_transcript.json` | Raw JSON array (opt-in with `--json`) |

---

## Supported formats

**Video:** mp4, mkv, avi, mov, webm, and more  
**Audio:** mp3, wav, m4a, flac, ogg, aac, opus, and more  
**Languages:** 90+ (Whisper auto-detects if `--bahasa` not set)

---

## Troubleshooting

**`401 Client Error` / `Access to model is restricted`**  
Make sure the license is accepted and your token is valid:
```bash
wishcribe --video meeting.mp4 --bahasa id --hf-token hf_xxx
# or set once: export WISHCRIBE_HF_TOKEN=hf_xxx
```
Accept the license here: https://huggingface.co/pyannote/speaker-diarization-community-1

**Want to skip the HuggingFace setup entirely?**  
Use `--no-diarize` — no token needed, works immediately after `pip install wishcribe`.

**`wishcribe: command not found`**
```bash
pip install wishcribe --upgrade
# or on Windows:
python -m wishcribe --video meeting.mp4
```

**`ffmpeg not found`**  
Follow the ffmpeg installation steps above for your OS.

**Dependency conflicts (e.g. with tensorflow)**  
Use a virtual environment (see section above) to isolate wishcribe cleanly.

**Out of memory with `large` model**  
Switch to a smaller model:
```bash
wishcribe --video meeting.mp4 --model medium
```

---

## License

MIT — free to use, modify, and distribute.
