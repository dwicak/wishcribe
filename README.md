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

   ![Add Python to PATH](https://www.python.org/static/img/python-logo.png)

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

## Quick start

### Step 1 — Download all models once

```bash
wishcribe download --hf-token hf_xxx
```

This downloads and caches:
- **Whisper `large`** (~2.9 GB) → saved locally
- **pyannote diarization** (~1 GB) → saved locally

> Get your free HuggingFace token at: https://huggingface.co/settings/tokens  
> ⚠️ **Accept both licenses before running:**  
> • https://huggingface.co/pyannote/speaker-diarization-3.1  
> • https://huggingface.co/pyannote/segmentation-3.0

### Step 2 — Transcribe (fully offline forever after)

```bash
wishcribe --video meeting.mp4
```

**That's it.** No token, no internet, no extra flags needed.

---

## Usage — CLI

### Download command

```bash
# Download default Whisper large model
wishcribe download --hf-token hf_xxx

# Download a smaller/faster model instead
wishcribe download --hf-token hf_xxx --model medium

# Use a manually downloaded pyannote model folder
wishcribe download --model-path /path/to/pyannote-model
```

### Transcribe command

```bash
# Basic — Whisper large by default
wishcribe --video meeting.mp4

# With language + speaker count
wishcribe --video meeting.mp4 --bahasa id --speakers 3

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
| `--hf-token` | HuggingFace token — first-time only | — |
| `--model-path` | Path to local pyannote model folder | — |
| `--model` | `tiny`/`base`/`small`/`medium`/`large` | **`large`** |
| `--bahasa` | Language code e.g. `id`, `en` | auto-detect |
| `--speakers` | Number of speakers (optional) | auto |
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

# Step 1 — download models once
download(hf_token="hf_xxx")

# Step 2 — transcribe offline
segments = transcribe("meeting.mp4")

# With options
segments = transcribe(
    "meeting.mp4",
    model="large",     # default — best accuracy
    language="id",
    num_speakers=3,
    output_dir="./out",
)

for seg in segments:
    print(f"[{seg.speaker}] {seg.start:.1f}s  {seg.text}")
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
| `<n>_transcript.txt` | Plain text grouped by speaker |
| `<n>_transcript.srt` | SRT subtitles with speaker labels |
| `<n>_transcript.json` | Raw JSON array (opt-in) |

---

## Supported formats

**Video:** mp4, mkv, avi, mov, webm, and more  
**Audio:** mp3, wav, m4a, flac, ogg, aac, opus, and more  
**Languages:** 90+ (Whisper auto-detects if `--bahasa` not set)

---

## Troubleshooting

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
