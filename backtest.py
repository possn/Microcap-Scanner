"""Walk-forward backtest for the Heartbeat Stage 2 setup.

Purpose
-------
The live scanner produces a 0-100 score. A score is a RANKING. It only becomes
a PROBABILITY after you measure, out of sample, how often each score bucket
actually produced a positive forward return. This script does that measurement.

Method
------
For every cached ticker, walk forward session by session. At each session `t`
the detection stack is re-run on `df.iloc[:t+1]` ONLY — the exact information a
live run would have had on that date. Benchmarks are sliced to the same date,
so relative strength and market regime are also point-in-time. Forward returns
are then measured from `t` over 5/10/20/40/60 sessions.

Known biases this script CANNOT remove (state them whenever quoting results)
---------------------------------------------------------------------------
1. SURVIVORSHIP: the cache is built from the CURRENT Nasdaq sub-$1 list.
   Companies that were delisted, acquired, or reverse-split out of the sub-$1
   range are absent. This inflates every statistic, and below $1 the attrition
   rate is high. Treat measured hit rates as an OPTIMISTIC upper bound.
2. NO SLIPPAGE/SPREAD: sub-$1 microcaps routinely show 2-5% spreads. Real
   entries are worse than the modelled close.
3. NO BORROW/HALT MODELLING: trading halts are common in this segment.
4. OVERLAPPING SAMPLES: consecutive signals on the same ticker are serially
   correlated, so the effective sample size is far below the nominal N. The
   per-ticker cap below mitigates but does not eliminate this.

Usage
-----
    python backtest.py --min-score 68 --step 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

import scanner
from scanner import (
    Config,
    detect_base,
    enhanced_metrics,
    find_sma150_breakout,
    technical_score,
    wilson_ci,
)

HORIZONS = (5, 10, 20, 40, 60)


def slice_benchmarks(full: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out = {}
    for key, frame in full.items():
        if frame is None:
            continue
        out[key] = frame[frame["date"] <= cutoff].reset_index(drop=True)
    return out


def evaluate_point(df: pd.DataFrame, cfg: Config) -> Optional[dict[str, Any]]:
    """Run the full live gate stack on a truncated frame. Returns None if the
    setup would not have been published on that date."""
    breakout, _ = find_sma150_breakout(df, cfg)
    if breakout is None:
        return None
    base = detect_base(df, breakout["idx"], cfg)
    if base is None:
        return None
    extra = enhanced_metrics(base, breakout, df)
    if extra["clv"] < cfg.min_close_location:
        return None
    if extra["rr"] < cfg.min_reward_risk:
        return None
    if extra["fails"] > cfg.max_failed_breakouts:
        return None
    score = technical_score(base, breakout, df, cfg, extra)
    return {"score": score, "state": extra["state"], "rr": extra["rr"]}


def forward_returns(df: pd.DataFrame, i: int) -> dict[int, float]:
    entry = float(df.iloc[i]["close"])
    out = {}
    for h in HORIZONS:
        if i + h < len(df):
            out[h] = 100 * (float(df.iloc[i + h]["close"]) / entry - 1)
    return out


def run(cfg: Config, step: int, min_score: float, max_per_ticker: int, min_i: int) -> dict[str, Any]:
    full_benchmarks = {}
    for symbol in ("QQQ", "IWM"):
        path = cfg.ohlcv_dir / f"{symbol}.csv"
        if path.exists():
            full_benchmarks[symbol] = scanner._normalise_ohlcv(pd.read_csv(path))

    files = sorted(p for p in cfg.ohlcv_dir.glob("*.csv") if p.stem not in {"QQQ", "IWM", "SPY"})
    samples: list[dict[str, Any]] = []
    for path in files:
        try:
            df = scanner._normalise_ohlcv(pd.read_csv(path))
        except Exception:
            continue
        if df is None or len(df) < cfg.min_history_sessions + max(HORIZONS):
            continue
        taken = 0
        last_signal = -10**9
        for i in range(max(min_i, cfg.min_history_sessions), len(df) - max(HORIZONS), step):
            if taken >= max_per_ticker:
                break
            if i - last_signal < 20:  # de-overlap: one signal per ticker per month
                continue
            window = df.iloc[: i + 1].reset_index(drop=True)
            cutoff = window["date"].iloc[-1]
            scanner.BENCHMARKS = slice_benchmarks(full_benchmarks, cutoff)
            scanner.MARKET_REGIME = scanner.compute_market_regime()
            try:
                hit = evaluate_point(window, cfg)
            except Exception:
                continue
            if hit is None or hit["score"] < min_score:
                continue
            fwd = forward_returns(df, i)
            if len(fwd) < len(HORIZONS):
                continue
            samples.append({
                "ticker": path.stem,
                "date": str(cutoff.date()),
                "score": hit["score"],
                "regime": scanner.MARKET_REGIME.get("label"),
                "returns": fwd,
            })
            taken += 1
            last_signal = i

    return summarise(samples)


def summarise(samples: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"n_signals": len(samples), "buckets": {}, "by_regime": {}}
    if not samples:
        return report

    def stats(values: list[float]) -> dict[str, Any]:
        n = len(values)
        wins = sum(1 for v in values if v > 0)
        lo, hi = wilson_ci(wins, n)
        return {
            "n": n,
            "hit_rate_pct": round(100 * wins / n, 1),
            "wilson_ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
            "median_pct": round(float(np.median(values)), 2),
            "mean_pct": round(float(np.mean(values)), 2),
            "p25_pct": round(float(np.percentile(values, 25)), 2),
            "p75_pct": round(float(np.percentile(values, 75)), 2),
        }

    for h in HORIZONS:
        buckets: dict[str, list[float]] = defaultdict(list)
        for s in samples:
            if h not in s["returns"]:
                continue
            score = s["score"]
            label = "<70" if score < 70 else "70-79" if score < 80 else "80-89" if score < 90 else "90+"
            buckets[label].append(s["returns"][h])
            buckets["TODOS"].append(s["returns"][h])
        report["buckets"][str(h)] = {k: stats(v) for k, v in buckets.items() if len(v) >= 5}

    regimes: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        if 20 in s["returns"]:
            regimes[s.get("regime") or "n/d"].append(s["returns"][20])
    report["by_regime"] = {k: stats(v) for k, v in regimes.items() if len(v) >= 5}
    report["caveats"] = [
        "Survivorship bias: universo construído a partir da lista atual — resultados são um limite SUPERIOR otimista.",
        "Sem spread, slippage nem halts. Abaixo de $1 o spread real corrói vários pontos percentuais.",
        "Amostras sobrepostas por ticker: N efetivo << N nominal; os IC95 são otimistas.",
        "Monotonia entre buckets de score é o teste que importa. Se não for monótona, o score não discrimina.",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=5, help="sessões entre avaliações")
    parser.add_argument("--min-score", type=float, default=0.0, help="score mínimo para registar sinal")
    parser.add_argument("--max-per-ticker", type=int, default=6)
    parser.add_argument("--min-index", type=int, default=200)
    parser.add_argument("--out", default="cache/backtest_report.json")
    args = parser.parse_args()

    cfg = Config()
    report = run(cfg, args.step, args.min_score, args.max_per_ticker, args.min_index)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    scanner.Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
