import numpy as np
import pandas as pd

from scanner import Config, find_sma150_breakout, parse_money


def test_parse_money():
    assert parse_money("$0.83") == 0.83
    assert parse_money("12.5M") == 12_500_000
    assert parse_money("N/A") is None


def test_detects_recent_high_volume_sma150_reclaim():
    n = 230
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.full(n, 0.50)
    # Keep price below its SMA, then reclaim 20 sessions ago.
    close[:190] = 0.55
    close[190:209] = 0.48
    close[209:] = 0.58
    volume = np.full(n, 100_000.0)
    volume[209] = 400_000.0
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
        }
    )
    cfg = Config()
    result = find_sma150_breakout(df, cfg)
    assert result is not None
    assert result["volume_multiple"] >= 3.0
    assert 10 <= result["age"] <= 40
