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

## Installation

```bash
pip install wishcribe
```

> **ffmpeg is also required** (one-time system install):
> ```bash
> brew install ffmpeg        # macOS
> sudo apt install ffmpeg    # Ubuntu/Debian
> ```

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

## Publishing

```bash
make build      # build dist/
make publish    # upload to PyPI → pip install wishcribe
```

---

## License

MIT
