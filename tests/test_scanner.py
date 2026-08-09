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


def test_common_stock_filter_excludes_warrants_units_rights():
    from scanner import is_probably_common_stock
    assert is_probably_common_stock("ABCD", "Acme Inc. Common Stock")
    assert not is_probably_common_stock("ABCDW", "Acme Inc. Warrant")
    assert not is_probably_common_stock("ABCDU", "Acme Inc. Unit")
    assert not is_probably_common_stock("ABCDR", "Acme Inc. Right")
    assert not is_probably_common_stock("ABCD", "Acme Inc. Warrants expiring 2027")
    assert not is_probably_common_stock("ABCD", "Acme Inc. 8% Series A Preferred")


def test_wilson_ci_is_conservative_on_tiny_samples():
    from scanner import wilson_ci
    lo, hi = wilson_ci(3, 3)
    assert lo < 0.9 and hi <= 1.0          # never claims certainty from n=3
    lo2, hi2 = wilson_ci(300, 300)
    assert lo2 > lo                        # interval tightens with sample size
    lo3, hi3 = wilson_ci(0, 0)
    assert (lo3, hi3) == (0.0, 1.0)


def test_journal_uses_adjusted_entry_and_flags_dead_streams(tmp_path, monkeypatch):
    import json
    from datetime import date, timedelta
    import scanner as sc

    cfg = sc.Config()
    journal_path = tmp_path / "signal_journal.json"
    old = (date.today() - timedelta(days=400)).isoformat()
    journal_path.write_text(json.dumps([
        {"ticker": "DEAD", "signal_date": old, "signal_price": 0.50,
         "published": True, "outcomes": {}}
    ]), encoding="utf-8")
    object.__setattr__(cfg, "signal_journal_json", journal_path)
    monkeypatch.setattr(sc, "load_ohlcv", lambda ticker, cfg: None)

    sc.update_signal_journal([], cfg, [])
    entry = json.loads(journal_path.read_text(encoding="utf-8"))[0]
    # A stale signal with no data is recorded as missing, never silently dropped.
    assert entry["outcomes"]["20"]["data_missing"] is True


def test_calibration_refuses_to_report_on_small_samples(tmp_path):
    import json
    import scanner as sc

    cfg = sc.Config()
    path = tmp_path / "j.json"
    path.write_text(json.dumps([
        {"ticker": "AAA", "signal_date": "2026-01-05", "published": True,
         "outcomes": {"20": {"return_pct": 12.0}}}
    ]), encoding="utf-8")
    object.__setattr__(cfg, "signal_journal_json", path)
    text = sc.calibration_summary(cfg, horizon=20, min_n=20)
    assert "amostra insuficiente" in text


def test_sma50_curling_up_default_requires_positive_slope_only():
    """Default mode (REQUIRE_SMA50_CURL_ACCELERATING=0): 'a curvar para cima'
    means sustained positive slope, not the exact inflection instant. Requiring
    both the SMA150 volume-confirmed breakout AND the precise SMA50 inflection
    on the same session emptied the funnel almost every day — two rare events
    that rarely land on the same date."""
    import dataclasses
    import numpy as np, pandas as pd
    from scanner import Config, sma50_curling_up

    cfg = Config()
    assert cfg.require_sma50_curl_accelerating is False  # locks the new default
    n = 260
    dates = pd.date_range("2024-01-02", periods=n, freq="B")

    # Established, steady uptrend: constant slope, no acceleration. Under the
    # new default this must PASS — it is genuinely "curving up" (rising),
    # even though it is not an inflection point.
    close_steady = np.linspace(0.30, 0.70, n)  # constant absolute slope throughout
    df_steady = pd.DataFrame({"date": dates, "open": close_steady, "high": close_steady * 1.01,
                               "low": close_steady * 0.99, "close": close_steady,
                               "volume": np.full(n, 200_000.0)})
    res_steady = sma50_curling_up(df_steady, cfg)
    assert res_steady is not None
    assert res_steady["slope_pct"] > 0
    assert res_steady["curling_up"] is True

    # Flat/declining: must still be rejected regardless of mode.
    close_flat = np.full(n, 0.40) + np.random.default_rng(1).normal(0, 0.001, n)
    df_flat = pd.DataFrame({"date": dates, "open": close_flat, "high": close_flat * 1.01,
                             "low": close_flat * 0.99, "close": close_flat,
                             "volume": np.full(n, 200_000.0)})
    res_flat = sma50_curling_up(df_flat, cfg)
    assert res_flat is not None
    assert res_flat["curling_up"] is False

    # Insufficient history: must return None (fail-safe, never a silent pass).
    assert sma50_curling_up(df_steady.head(60), cfg) is None

    # Strict opt-in mode: same steady uptrend must now be REJECTED, since it
    # is not accelerating — only the genuine inflection case (built below)
    # should pass when this mode is explicitly requested.
    strict_cfg = dataclasses.replace(cfg, require_sma50_curl_accelerating=True)
    res_steady_strict = sma50_curling_up(df_steady, strict_cfg)
    assert res_steady_strict["curling_up"] is False

    flat_len, ramp_len = 235, n - 235
    close_up = np.concatenate([
        np.full(flat_len, 0.50),
        np.linspace(0.50, 0.66, ramp_len),
    ])
    df_up = pd.DataFrame({"date": dates, "open": close_up, "high": close_up * 1.02,
                           "low": close_up * 0.98, "close": close_up,
                           "volume": np.full(n, 200_000.0)})
    res_up_strict = sma50_curling_up(df_up, strict_cfg)
    assert res_up_strict["curling_up"] is True
    assert res_up_strict["accelerating"] is True
    # And under the default (non-strict) mode, the same inflection data also
    # passes, since positive-and-accelerating implies positive.
    res_up_default = sma50_curling_up(df_up, cfg)
    assert res_up_default["curling_up"] is True


def test_sma50_gate_is_additional_not_a_replacement():
    """The SMA50 curl check must not run instead of the SMA150 reclaim gate —
    both are required. This locks the ordering asserted by the user: 'Adicional
    (ambos têm de se verificar)'."""
    import inspect
    import scanner

    src = inspect.getsource(scanner.analyse_candidate)
    breakout_pos = src.index("find_sma150_breakout")
    curl_pos = src.index("sma50_curling_up")
    base_pos = src.index("detect_base(df")
    assert breakout_pos < curl_pos < base_pos, (
        "esperado: SMA150 -> SMA50 curl -> base, ambos os gates antes da deteção de base"
    )
