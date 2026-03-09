"""Shared time formatting utilities."""


def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_time_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    ms_total = int(seconds * 1000)
    h  = ms_total // 3_600_000
    m  = (ms_total % 3_600_000) // 60_000
    s  = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
