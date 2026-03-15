# Wishcribe

Fast multi-speaker audio/video transcription — fully offline after first run.

**faster-whisper** + **pyannote.audio** + native **Apple Silicon** support (MLX-Whisper).

---

## Features

- **Multi-speaker transcription** — speaker labels per segment (`[SPEAKER_00]`, `[SPEAKER_01]`, …)
- **4–8× faster** than openai-whisper via CTranslate2 batched inference + VAD
- **Apple Silicon native** — MLX-Whisper auto-selected on M1/M2/M3/M4 (Neural Engine / GPU), chip name shown in banner
- **Fully offline** after one-time model download
- **Word-level timestamps** — `--word-timestamps` embeds per-word timing in SRT/JSON
- **VAD controls** — `--no-vad`, `--vad-threshold`, `--vad-min-silence-ms`, `--vad-speech-pad-ms`
- **Silence suppression** — `--no-speech-threshold` discards hallucinated segments in quiet regions
- **Fast mode** — `--fast-mode` enables greedy decoding + VAD chunk packing + concurrent model warmup on M1 (~40-50% faster)
- **Auto-diarize fallback** — warns and continues without speaker labels if no token and no cached model
- **Accuracy controls** — `--initial-prompt`, `--temperature`, `--beam-size`
- **Video + audio** — mp4, mkv, mov, avi, webm, ts, wmv, flv, mp3, wav, m4a, flac, ogg, aac, opus, wma
- **Multiple output formats** — `.txt`, `.srt`, `.json`
- **Python API + CLI**

---

## Installation

```bash
pip install wishcribe
```

### Apple Silicon (M1/M2/M3/M4) — fastest

```bash
pip install "wishcribe[apple]"
```

Installs `mlx-whisper` for Neural Engine / GPU acceleration. Automatically selected when running on Apple Silicon.

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
- Whisper model (default: `large-v2` on non-Apple, `turbo` on Apple Silicon)
- pyannote speaker diarization model (~1 GB)

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

```
wishcribe download [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--hf-token TOKEN` | HuggingFace read token (required on first run) |
| `--model MODEL` | Whisper model to download (default: `large-v2`) |
| `--force` | Delete existing cache and re-download |

### `wishcribe` / `wishcribe run`

```
wishcribe --video FILE [OPTIONS]
wishcribe run --video FILE [OPTIONS]
```

**Input:**

| Option | Description |
|--------|-------------|
| `--video PATH` | Path to video or audio file (required) |
| `--bahasa LANG` | Language code: `id`, `en`, `zh`, etc. (default: auto-detect) |

**Whisper model:**

| Option | Description |
|--------|-------------|
| `--model MODEL` | `tiny` \| `base` \| `small` \| `medium` \| `large` \| `large-v1` \| `large-v2` \| `large-v3` \| `turbo` (default: `large-v2`) |

**Accuracy:**

| Option | Description |
|--------|-------------|
| `--initial-prompt TEXT` | Domain context to guide transcription vocabulary and style. Disables batched inference. |
| `--temperature FLOAT` | Sampling temperature (default: `0.0` = greedy). Non-zero disables batched inference. |
| `--beam-size N` | Beam search width (default: `5`). Only used when batched inference is disabled. |
| `--word-timestamps` | Embed word-level timing in SRT (one block per word) and JSON (`words` array). Not supported by MLX-Whisper. |
| `--no-speech-threshold FLOAT` | Discard segments below this non-speech probability (default: `0.6`). Suppresses hallucinations in silent regions. |
| `--fast-mode` | Enable M1-optimised fast transcription: greedy decoding (beam_size=1), VAD chunk packing, concurrent model warmup. ~40-50% faster on Apple Silicon. MLX-Whisper only. |

**VAD (Voice Activity Detection):**

| Option | Description |
|--------|-------------|
| `--no-vad` | Disable VAD. Use when VAD incorrectly trims real speech. |
| `--vad-threshold FLOAT` | VAD speech probability threshold (default: `0.5`). |
| `--vad-min-silence-ms MS` | Minimum silence gap in ms to split chunks (default: `500`). |
| `--vad-speech-pad-ms MS` | Padding added around detected speech in ms (default: `200`). |

**Speed / hardware:**

| Option | Description |
|--------|-------------|
| `--batch-size N` | Batch size (default: `16`). Lower to 4–8 on OOM. |
| `--compute-type TYPE` | `float16` (GPU, fast) \| `int8` (CPU/low-mem) \| `float32` (CPU, accurate) |
| `--device DEVICE` | `cuda` \| `cpu` (default: auto-detect) |

**Speaker diarization:**

| Option | Description |
|--------|-------------|
| `--hf-token TOKEN` | HuggingFace token (or set `WISHCRIBE_HF_TOKEN` env var) |
| `--model-path PATH` | Local pyannote model folder (skips HuggingFace) |
| `--speakers N` | Number of speakers — improves accuracy when known |
| `--no-diarize` | Skip speaker diarization (no token needed, faster) |

**Output:**

| Option | Description |
|--------|-------------|
| `--output DIR` | Output folder (default: same as input file) |
| `--json` | Also save `.json` transcript |
| `--no-txt` | Skip `.txt` output |
| `--no-srt` | Skip `.srt` output |
| `--quiet` | Suppress progress output |

**Cloud API:**

| Option | Description |
|--------|-------------|
| `--use-api` | Use OpenAI Whisper API instead of local model |
| `--api-key KEY` | OpenAI API key (required with `--use-api`) |

---

## Examples

```bash
# Indonesian meeting with 3 speakers
wishcribe --video rapat.mp4 --bahasa id --speakers 3

# English podcast, no speaker labels needed
wishcribe --video podcast.mp3 --no-diarize

# Medical dictation — guide vocabulary with a prompt
wishcribe --video dictation.mp4 --initial-prompt "Medical: hypertension, tachycardia, bradycardia."

# Word-level timestamps (karaoke-style SRT)
wishcribe --video interview.mp4 --word-timestamps

# Apple Silicon fast mode — ~40-50% faster, greedy decoding (v1.4.0)
wishcribe --video meeting.mp4 --fast-mode

# Suppress silence hallucinations more aggressively
wishcribe --video lecture.mp4 --no-speech-threshold 0.8

# Disable VAD if it trims real speech
wishcribe --video quiet_speaker.mp4 --no-vad

# Tune VAD sensitivity
wishcribe --video meeting.mp4 --vad-threshold 0.3 --vad-min-silence-ms 300

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
    model="large-v2",       # or "turbo" — auto-selected on Apple Silicon
    language="id",           # BCP-47 language code; None = auto-detect
    num_speakers=2,          # known speaker count improves diarization
    output_dir="./output",   # where to save files
    save_json=True,          # also write .json
)

# Skip diarization (no token needed)
segments = transcribe("lecture.mp4", diarize=False)

# Word-level timestamps in SRT and JSON
segments = transcribe("interview.mp4", word_timestamps=True)

# Suppress silence hallucinations
segments = transcribe("lecture.mp4", no_speech_threshold=0.8)

# Fast mode — ~40-50% faster on Apple Silicon M1 (v1.4.0)
# Greedy decoding + VAD chunk packing + concurrent model warmup
segments = transcribe("meeting.mp4", fast_mode=True)

# Disable VAD if it trims real speech
segments = transcribe("quiet.mp4", vad_filter=False)

# Tune VAD
segments = transcribe(
    "meeting.mp4",
    vad_threshold=0.3,
    vad_min_silence_ms=300,
    vad_speech_pad_ms=150,
)

# Accuracy controls
segments = transcribe(
    "dictation.mp4",
    initial_prompt="Medical: hypertension, tachycardia.",
    beam_size=10,
)

# Speed controls
segments = transcribe(
    "meeting.mp4",
    batch_size=8,
    compute_type="int8",
    device="cpu",
)

# Each segment has: .start  .end  .speaker  .text  .words (when word_timestamps=True)
for seg in segments:
    print(f"[{seg.speaker}] {seg.start:.1f}s -> {seg.end:.1f}s  {seg.text}")

# HF token via environment variable
import os
os.environ["WISHCRIBE_HF_TOKEN"] = "hf_xxxxxxxxxx"
segments = transcribe("meeting.mp4")  # token picked up automatically
```

---

## Model guide

| Model | Size | Speed | Accuracy | Notes |
|-------|------|-------|----------|-------|
| tiny | 75 MB | ⚡⚡⚡⚡ | ★★ | Fastest; fair accuracy |
| base | 139 MB | ⚡⚡⚡ | ★★★ | Good for clear speech |
| small | 461 MB | ⚡⚡⚡ | ★★★ | Better accuracy |
| medium | 1.4 GB | ⚡⚡ | ★★★★ | Good speed/accuracy balance |
| turbo | 1.6 GB | ⚡⚡⚡ | ★★★★ | **Default on Apple Silicon** ⭐ |
| large-v1 | 2.9 GB | ⚡ | ★★★★ | Original large model |
| large-v2 | 2.9 GB | ⚡ | ★★★★★ | **Default on non-Apple** ⭐ |
| large-v3 | 3.1 GB | ⚡ | ★★★★★ | Newest large model |

On Apple Silicon, `turbo` is the default — it uses the Neural Engine and is significantly faster than `large-v2` with only a marginal accuracy difference.

---

## Backend selection

wishcribe automatically picks the best available backend:

```
Apple Silicon (M1/M2/M3/M4)
└── mlx-whisper installed?  → MLX-Whisper (Neural Engine / GPU)
└── else                    → faster-whisper (CPU)

Other platforms
└── faster-whisper installed?  → faster-whisper + batched inference + VAD
└── else                       → openai-whisper (fallback, slower)
```

### MLX-Whisper (Apple Silicon)

Automatically selects the right quantized model based on available RAM:

| RAM | Model loaded |
|-----|-------------|
| ≥ 16 GB | Full precision |
| ≥ 8 GB | 4-bit quantized |
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

> Note: `--initial-prompt` and non-zero `--temperature` disable batched inference and fall back to standard beam search. This is slower but gives more control.

**`--beam-size`** (default: `5`) controls beam search width in the non-batched path. Increasing to 8–10 improves accuracy slightly at the cost of speed.

**`--temperature`** (default: `0.0` = greedy) adds sampling randomness. Useful when the model gets stuck in repetitive output. Try `0.1`–`0.2`.

**`--no-speech-threshold`** (default: `0.6`) discards segments where the model is uncertain whether the audio contains speech. Increase toward `1.0` to suppress more aggressively in recordings with long silences.

**`--no-vad`** disables Voice Activity Detection. Use this if VAD incorrectly cuts off quiet speakers, music beds, or recordings with ambient noise.

**`--fast-mode`** (v1.4.0, Apple Silicon M1 only) activates three stacked optimisations:

- **Greedy decoding** (`beam_size=1, best_of=1`) — eliminates beam search, ~40-50% faster on M1. M1's GPU handles parallel beam candidates less efficiently than CUDA, so the gain is larger than on NVIDIA hardware.
- **VAD chunk packing** — speech segments are greedily packed into 30 s windows before sending to the model. Reduces the number of MLX transcribe() calls and maximises GPU occupancy per call.
- **Concurrent model warmup** — the MLX model starts loading in a background thread immediately, overlapping with the VAD preprocessing step. On M1 unified memory there is no PCIe copy, so this parallelism is nearly free.

All three are best-effort — any failure falls back to standard full-file transcription silently.

---

## Output format

### `.txt`

```
[SPEAKER_00] 0:00:01
  Good morning everyone, let's get started.

[SPEAKER_01] 0:00:05
  Thanks for joining. Today we'll cover the Q3 results.
```

### `.srt` (segment-level, default)

```
1
00:00:01,000 --> 00:00:04,500
[SPEAKER_00] Good morning everyone, let's get started.

2
00:00:05,200 --> 00:00:09,800
[SPEAKER_01] Thanks for joining. Today we'll cover the Q3 results.
```

### `.srt` (word-level, with `--word-timestamps`)

```
1
00:00:01,000 --> 00:00:01,380
[SPEAKER_00] Good

2
00:00:01,420 --> 00:00:01,710
morning

3
00:00:01,740 --> 00:00:02,100
everyone,
```

### `.json`

```json
[
  {
    "start": 1.0,
    "end": 4.5,
    "duration": 3.5,
    "speaker": "SPEAKER_00",
    "text": "Good morning everyone, let's get started.",
    "words": [
      {"word": "Good", "start": 1.0, "end": 1.38, "probability": 0.99},
      {"word": "morning", "start": 1.42, "end": 1.71, "probability": 0.98}
    ]
  }
]
```

> `words` is only present when `--word-timestamps` is used. `duration` is always included.

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

**Whisper model `large-v2` is not cached**
```bash
wishcribe download --model large-v2
# or force a clean re-download:
wishcribe download --model large-v2 --force
```

**Diarization model not found in local cache**
```bash
wishcribe download --hf-token hf_xxxxxxxxxx
```
Make sure you've accepted the license at https://huggingface.co/pyannote/speaker-diarization-community-1

**Diarization auto-skipped with warning**

If no token and no cached model, wishcribe now continues with transcription only (no speaker labels) instead of crashing. To enable speaker labels:
```bash
export WISHCRIBE_HF_TOKEN=hf_xxxxxxxxxx
wishcribe --video meeting.mp4
```

**CUDA out of memory**
```bash
wishcribe --video meeting.mp4 --batch-size 4
# or CPU mode:
wishcribe --video meeting.mp4 --device cpu --compute-type int8
```

**VAD cutting off speech**
```bash
# Disable VAD entirely
wishcribe --video meeting.mp4 --no-vad
# Or lower the threshold (more sensitive)
wishcribe --video meeting.mp4 --vad-threshold 0.3
```

**Hallucinated text in silent regions**
```bash
# Increase no-speech threshold
wishcribe --video meeting.mp4 --no-speech-threshold 0.8
```

**mlx-whisper not installed (Apple Silicon)**
```bash
pip install mlx-whisper
# or reinstall with apple extra:
pip install "wishcribe[apple]"
```

**Fast mode not speeding things up**
```bash
# Verify mlx-whisper is installed (fast_mode only affects the MLX path)
pip install "wishcribe[apple]"
# Try on a longer file — warmup overhead is proportionally smaller
wishcribe --video long_meeting.mp4 --fast-mode
```

**Inaccurate transcription with --fast-mode**
```bash
# fast_mode uses greedy decoding — drop it for beam search accuracy
wishcribe --video meeting.mp4   # standard mode, beam_size=5
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
- `moviepy >= 2.0` + ffmpeg (audio extraction)
- `faster-whisper >= 1.0` (transcription)
- `pyannote.audio >= 3.1` (diarization)
- `torch >= 2.0`
- Apple Silicon only: `mlx-whisper >= 0.4`, `soundfile >= 0.12` (`pip install "wishcribe[apple]"`)

---

## Changelog

### v1.4.0
- **`--fast-mode`** — three stacked M1 speed optimisations on MLX-Whisper (Apple Silicon only):
  - Greedy decoding (`beam_size=1, best_of=1`): ~40-50% faster on M1 vs default beam search
  - VAD chunk packing: greedy 30 s window packing maximises GPU occupancy per MLX call
  - Concurrent model warmup: MLX model loads in background thread while Silero VAD runs on CPU
  - All three are best-effort — any failure falls through to standard full-file path silently
- `soundfile>=0.12.0` added to `[apple]` optional extra (required for VAD chunk packing)

### v1.3.2
- Reverted Lightning-Whisper-MLX backend (incompatible with current Apple Silicon environment)
- MLX-Whisper remains the sole Apple Silicon backend (priority 1 on M-series)
- All v1.3.0 features intact: word timestamps, VAD controls, no-speech threshold, auto-diarize fallback

### v1.3.0
- **Word-level timestamps** — `--word-timestamps` embeds per-word timing in SRT (karaoke-style, one block per word) and JSON (`words` array with `start`, `end`, `probability`)
- **No-speech threshold** — `--no-speech-threshold` (default `0.6`) discards hallucinated segments in silent regions; applied to both batched and standard transcription paths
- **VAD escape hatch** — `--no-vad` disables Voice Activity Detection for recordings where VAD incorrectly trims real speech
- **VAD sensitivity controls** — `--vad-threshold`, `--vad-min-silence-ms`, `--vad-speech-pad-ms`
- **Cross-window coherence fix** — `condition_on_previous_text` restored to `True` (was incorrectly set to `False` in v1.1.1); hallucination now suppressed via `no_speech_threshold` instead
- **Apple Silicon chip name in banner** — shows exact chip label (e.g. `Apple M3 Pro`) via `sysctl`
- **Auto-diarize fallback** — warns and continues without speaker labels instead of crashing when no token and no cached model
- **`AVFFrameReceiver` / `objc[]` warning suppression** — extended to pyannote diarization (onnxruntime fires these on every `runSession` call)
- **Actionable model.bin error** — clear message with exact `--force` command when Whisper cache is incomplete

### v1.2.1
- Fix: `_apple_perf_cores` function definition restored (was accidentally removed, caused `NameError` on Apple Silicon)
- Fix: `no_speech_threshold` now applied in batched inference path (was only in standard path)
- Fix: `write_txt` and `write_srt` write empty file (not bare newline) for empty segment lists
- Fix: Progress dot no longer printed for blank/empty segments
- Fix: PEP 8 blank line between `_suppress_fd2` and module-level constants in `diarize.py`

### v1.2.0
- MLX-Whisper backend — Apple Silicon M1/M2/M3/M4, auto-selected when `mlx-whisper` is installed
- Default model `turbo` on Apple Silicon (faster + accurate via Neural Engine)
- `OMP_NUM_THREADS` auto-tuned to physical performance cores on Apple Silicon
- Auto-quantization — picks 4-bit or 8-bit MLX model based on available unified memory
- `--initial-prompt` — inject domain context for specialised vocabulary
- `--temperature` — sampling temperature (0.0 = true greedy, always deterministic)
- `--beam-size` — beam search width for non-batched path
- Pipeline memory leak fix — GPU memory freed even when `BatchedInferencePipeline.transcribe()` raises mid-batch
- Diarization memory fix — GPU/MPS memory freed even if segment extraction fails
- `wishcribe download` mirrors transcription default on Apple Silicon (`turbo`, not `large-v2`)
- `large-v1` added to CLI model choices
- MLX cache purge on `--force` re-download
- `moviepy>=2.0.0` pin (2.x API required)

### v1.1.1 *(bug fixes, included in v1.2.0)*
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

---

## Links

- PyPI: https://pypi.org/project/wishcribe/
- GitHub: https://github.com/dwicak/wishcribe
- Issues: https://github.com/dwicak/wishcribe/issues
