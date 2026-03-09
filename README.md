# wishcribe ✍️

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

### Step 1 — download all models (run once)

```bash
wishcribe download --hf-token hf_xxx
```

This downloads and caches:
- **Whisper `large`** (~2.9 GB) → `~/.cache/whisper/large.pt`
- **pyannote diarization** (~1 GB) → `~/.cache/huggingface/hub/...`

Output:
```
📦  WISHCRIBE — MODEL DOWNLOADER
══════════════════════════════════════════
  Whisper model : large
  Diarization   : HuggingFace download (token provided)
══════════════════════════════════════════

📥 Downloading Whisper 'large' model (2.9 GB)...
✅ Whisper 'large' downloaded and cached  (2.9 GB)

📥 Downloading pyannote diarization model (~1 GB)...
✅ Diarization model downloaded and cached

🎉 All models cached! wishcribe now works fully offline.
   Run transcription with:
   wishcribe --video meeting.mp4
```

### Step 2 — transcribe (fully offline, forever)

```bash
wishcribe --video meeting.mp4
```

**That's it.** No token, no internet, no extra flags.

---

## Usage — CLI

### Download command

```bash
# Download default model (large)
wishcribe download --hf-token hf_xxx

# Download a specific model size
wishcribe download --hf-token hf_xxx --model medium

# Use a local pyannote model folder (no HuggingFace needed)
wishcribe download --model-path /path/to/pyannote-model
```

### Run / transcribe command

```bash
# Basic (Whisper large by default)
wishcribe --video meeting.mp4
wishcribe run --video meeting.mp4    # same thing

# With language + speaker count
wishcribe --video meeting.mp4 --bahasa id --speakers 3

# Override Whisper model
wishcribe --video meeting.mp4 --model medium
wishcribe --video meeting.mp4 --model small

# Use OpenAI API for transcription (diarization still offline)
wishcribe --video meeting.mp4 --use-api --api-key sk-xxx

# Custom output folder + save JSON
wishcribe --video meeting.mp4 --output ./results --json
```

### All run options

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

## How offline mode works

| Cache location | What's stored |
|---|---|
| `~/.cache/whisper/large.pt` | Whisper large model weights (2.9 GB) |
| `~/.cache/huggingface/hub/models--pyannote--...` | Diarization model (~1 GB) |

Once cached, both load instantly from disk — no internet ever needed.

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

## HuggingFace setup (for download command)

1. Sign up at https://huggingface.co
2. Accept the license: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create a Read token: https://huggingface.co/settings/tokens

Only needed once for `wishcribe download`.

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

## Publishing

```bash
make build      # build dist/
make publish    # upload to PyPI → pip install wishcribe
```

---

## License

MIT
