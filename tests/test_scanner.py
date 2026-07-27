import numpy as np
import pandas as pd

from scanner import Config, find_sma150_breakout, parse_money


def make_frame(volume_offset: int = 0):
    n = 230
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.full(n, 0.50)
    close[:190] = 0.55
    close[190:209] = 0.48
    close[209:] = 0.58
    volume = np.full(n, 100_000.0)
    volume[209 + volume_offset] = 400_000.0
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": volume,
    })


def test_parse_money():
    assert parse_money("$0.83") == 0.83
    assert parse_money("12.5M") == 12_500_000
    assert parse_money("N/A") is None


def test_detects_same_day_high_volume_sma150_reclaim():
    result, reason = find_sma150_breakout(make_frame(0), Config())
    assert reason == "ok"
    assert result is not None
    assert result["volume_multiple"] >= 3.0
    assert 10 <= result["age"] <= 40


def test_accepts_volume_confirmation_within_three_sessions():
    result, reason = find_sma150_breakout(make_frame(2), Config())
    assert reason == "ok"
    assert result is not None
    assert result["volume_offset"] == 2


def test_rejects_when_volume_confirmation_is_outside_window():
    result, reason = find_sma150_breakout(make_frame(5), Config())
    assert result is None
    assert reason == "breakout_volume"
