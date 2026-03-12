"""Shared time formatting utilities."""


def fmt_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.
    Clamps negative values to 0 (VAD can produce tiny negative timestamps).
    """
    total = int(max(0.0, seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_time_srt(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS,mmm for SRT subtitles.
    Clamps negative values to 0 (VAD can produce tiny negative timestamps).
    """
    ms_total = int(max(0.0, seconds) * 1000)
    h  = ms_total // 3_600_000
    m  = (ms_total % 3_600_000) // 60_000
    s  = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
