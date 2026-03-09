"""
Unit tests for wishcribe.
Run with: pytest tests/ -v
"""
import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch, call

from wishcribe.merge import merge_segments
from wishcribe.models import Segment
from wishcribe.output import write_json, write_srt, write_txt
from wishcribe.utils import fmt_time, fmt_time_srt
from wishcribe.diarize import _find_cached_model
from wishcribe.transcribe import DEFAULT_WHISPER_MODEL
from wishcribe.download import (
    _whisper_is_cached, _whisper_cache_path,
    _download_whisper, _download_diarization,
)


# ── default model ─────────────────────────────────────────────────────────────

def test_default_whisper_model_is_large():
    assert DEFAULT_WHISPER_MODEL == "large"

def test_core_uses_large_as_default():
    import inspect
    from wishcribe.core import transcribe
    sig = inspect.signature(transcribe)
    assert sig.parameters["model"].default == "large"

def test_download_uses_large_as_default():
    import inspect
    from wishcribe.download import download_all
    sig = inspect.signature(download_all)
    assert sig.parameters["model"].default == "large"


# ── utils ─────────────────────────────────────────────────────────────────────

def test_fmt_time_zero():       assert fmt_time(0)    == "00:00:00"
def test_fmt_time_minutes():    assert fmt_time(61)   == "00:01:01"
def test_fmt_time_hours():      assert fmt_time(3661) == "01:01:01"
def test_fmt_time_srt_zero():   assert fmt_time_srt(0)       == "00:00:00,000"
def test_fmt_time_srt_millis(): assert fmt_time_srt(1.5)     == "00:00:01,500"
def test_fmt_time_srt_hours():  assert fmt_time_srt(3661.25) == "01:01:01,250"


# ── cache detection ───────────────────────────────────────────────────────────

def test_find_cached_model_missing():
    with patch("wishcribe.diarize._HF_CACHE_PATH", "/nonexistent/path"):
        assert _find_cached_model() is None

def test_find_cached_model_found():
    with tempfile.TemporaryDirectory() as d:
        snap = os.path.join(d, "snapshots", "abc123")
        os.makedirs(snap)
        with patch("wishcribe.diarize._HF_CACHE_PATH", d):
            assert _find_cached_model() == snap

def test_find_cached_model_picks_latest():
    with tempfile.TemporaryDirectory() as d:
        snaps = os.path.join(d, "snapshots")
        os.makedirs(snaps)
        old = os.path.join(snaps, "old_snap")
        new = os.path.join(snaps, "new_snap")
        os.makedirs(old)
        time.sleep(0.05)
        os.makedirs(new)
        with patch("wishcribe.diarize._HF_CACHE_PATH", d):
            assert _find_cached_model() == new


# ── download — whisper ────────────────────────────────────────────────────────

def test_whisper_cache_path():
    path = _whisper_cache_path("large")
    assert path.endswith("large.pt")
    assert ".cache/whisper" in path

def test_whisper_is_cached_false():
    with patch("wishcribe.download._whisper_cache_path", return_value="/nonexistent/large.pt"):
        assert _whisper_is_cached("large") is False

def test_whisper_is_cached_true():
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        with patch("wishcribe.download._whisper_cache_path", return_value=f.name):
            assert _whisper_is_cached("large") is True

def test_download_whisper_already_cached():
    """If model already cached, no whisper.load_model call should happen."""
    with patch("wishcribe.download._whisper_is_cached", return_value=True), \
         patch("wishcribe.download.os.path.getsize", return_value=3_000_000_000):
        result = _download_whisper("large", verbose=False)
        assert result is True

def test_download_whisper_not_cached_downloads():
    """If not cached, calls whisper.load_model to trigger download."""
    with patch("wishcribe.download._whisper_is_cached", return_value=False), \
         patch("wishcribe.download.os.path.getsize", return_value=3_000_000_000):
        mock_whisper = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            result = _download_whisper("large", verbose=False)
            mock_whisper.load_model.assert_called_once_with("large")
            assert result is True

def test_download_whisper_failure():
    """Returns False cleanly if download fails."""
    with patch("wishcribe.download._whisper_is_cached", return_value=False):
        mock_whisper = MagicMock()
        mock_whisper.load_model.side_effect = RuntimeError("network error")
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            result = _download_whisper("large", verbose=False)
            assert result is False


# ── download — diarization ────────────────────────────────────────────────────

def test_download_diarization_already_cached():
    """If model cached, returns True without downloading."""
    with patch("wishcribe.download._find_cached_model", return_value="/some/cache/path"):
        result = _download_diarization(None, None, verbose=False)
        assert result is True

def test_download_diarization_no_token_no_cache():
    """Returns False and prints helpful message when no token and no cache."""
    with patch("wishcribe.download._find_cached_model", return_value=None):
        result = _download_diarization(None, None, verbose=False)
        assert result is False

def test_download_diarization_custom_model_path_valid():
    """Returns True when a valid local model_path is given."""
    with tempfile.TemporaryDirectory() as d:
        result = _download_diarization(None, d, verbose=False)
        assert result is True

def test_download_diarization_custom_model_path_invalid():
    """Returns False when model_path does not exist."""
    result = _download_diarization(None, "/nonexistent/path", verbose=False)
    assert result is False

def test_download_diarization_with_token():
    """Calls Pipeline.from_pretrained when token provided and not cached."""
    mock_pipeline = MagicMock()
    mock_pyannote = MagicMock()
    mock_pyannote.audio.Pipeline = mock_pipeline

    with patch("wishcribe.download._find_cached_model", side_effect=[None, "/cached/path"]), \
         patch.dict("sys.modules", {"pyannote": mock_pyannote, "pyannote.audio": mock_pyannote.audio}):
        from pyannote.audio import Pipeline
        with patch("wishcribe.download.Pipeline", mock_pipeline):
            result = _download_diarization("hf_token_123", None, verbose=False)
            mock_pipeline.from_pretrained.assert_called_once()


# ── merge ─────────────────────────────────────────────────────────────────────

def test_merge_basic_overlap():
    whisper = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 3.0, "end": 5.0, "text": "World"},
    ]
    diarization = [(0.0, 2.5, "SPEAKER_00"), (2.5, 6.0, "SPEAKER_01")]
    result = merge_segments(whisper, diarization)
    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"

def test_merge_skips_empty_text():
    whisper = [
        {"start": 0.0, "end": 1.0, "text": ""},
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "Hi"},
    ]
    result = merge_segments(whisper, [(0.0, 5.0, "SPEAKER_00")])
    assert len(result) == 1
    assert result[0].text == "Hi"

def test_merge_fallback_when_no_diarization():
    result = merge_segments([{"start": 0.0, "end": 1.0, "text": "Test"}], [])
    assert result[0].speaker == "SPEAKER_00"

def test_merge_multi_speaker():
    whisper = [
        {"start": 0.0, "end": 2.0, "text": "First"},
        {"start": 3.0, "end": 5.0, "text": "Second"},
        {"start": 6.0, "end": 8.0, "text": "Third"},
    ]
    diarization = [
        (0.0, 2.5, "SPEAKER_00"),
        (2.5, 5.5, "SPEAKER_01"),
        (5.5, 9.0, "SPEAKER_00"),
    ]
    result = merge_segments(whisper, diarization)
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"
    assert result[2].speaker == "SPEAKER_00"


# ── output ────────────────────────────────────────────────────────────────────

SAMPLE = [
    Segment(0.0,  2.0,  "SPEAKER_00", "Hello there"),
    Segment(3.0,  5.0,  "SPEAKER_01", "Hi, how are you?"),
    Segment(6.0,  8.0,  "SPEAKER_00", "I am doing well."),
    Segment(9.0,  11.0, "SPEAKER_01", "Great to hear that."),
]

def test_write_txt_contains_speakers():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.txt")
        write_txt(SAMPLE, path)
        content = open(path, encoding="utf-8").read()
        assert "[SPEAKER_00]" in content
        assert "[SPEAKER_01]" in content

def test_write_txt_groups_same_speaker():
    segs = [Segment(0.0, 1.0, "SPEAKER_00", "Line one"),
            Segment(1.0, 2.0, "SPEAKER_00", "Line two")]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.txt")
        write_txt(segs, path)
        assert open(path).read().count("[SPEAKER_00]") == 1

def test_write_srt_format():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.srt")
        write_srt(SAMPLE, path)
        content = open(path, encoding="utf-8").read()
        assert "-->" in content
        assert "[SPEAKER_00]" in content

def test_write_json_structure():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        write_json(SAMPLE, path)
        data = json.loads(open(path, encoding="utf-8").read())
        assert len(data) == 4
        assert data[0]["speaker"] == "SPEAKER_00"
        assert data[1]["speaker"] == "SPEAKER_01"


# ── deps ──────────────────────────────────────────────────────────────────────

def test_deps_no_install_when_all_present():
    from wishcribe.deps import ensure_dependencies
    with patch("wishcribe.deps._is_installed", return_value=True), \
         patch("wishcribe.deps._pip_install") as mock_pip:
        ensure_dependencies(use_api=False)
        mock_pip.assert_not_called()

def test_deps_installs_missing():
    from wishcribe.deps import ensure_dependencies
    def fake(name): return name != "whisper"
    with patch("wishcribe.deps._is_installed", side_effect=fake), \
         patch("wishcribe.deps._pip_install") as mock_pip:
        ensure_dependencies(use_api=False)
        mock_pip.assert_called_once()


# ── Segment model ─────────────────────────────────────────────────────────────

def test_segment_to_dict():
    seg = Segment(1.0, 2.0, "SPEAKER_00", "Hello")
    assert seg.to_dict() == {
        "start": 1.0, "end": 2.0,
        "speaker": "SPEAKER_00", "text": "Hello",
    }
