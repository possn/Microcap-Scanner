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
1. SURVIVORSHIP: the cache is built from the CURRENT Nasdaq universe eligible
   under today's price/market-cap config. Companies that were delisted,
   acquired, or moved outside that range are absent. This inflates every
   statistic; the effect is largest at the nano/micro end (high attrition)
   and smaller for established mid caps. Treat measured hit rates as an
   OPTIMISTIC upper bound, more so toward the small end of the range.
2. NO SLIPPAGE/SPREAD: real entries are worse than the modelled close.
   This is a mixed-regime universe now — sub-$1 nano/micro names routinely
   show 2-5% spreads, while liquid mid caps typically show a fraction of
   that. A single aggregate statistic blends two very different liquidity
   regimes; check `by_regime`/score buckets rather than the headline number.
3. NO BORROW/HALT MODELLING: trading halts are common at the nano/micro end
   of this universe, rarer for mid caps.
4. OVERLAPPING SAMPLES: consecutive signals on the same ticker are serially
   correlated, so the effective sample size is far below the nominal N. The
   per-ticker cap below mitigates but does not eliminate this.
5. SECTOR MAP IS TODAY'S SNAPSHOT: the sector-ETF gate needs a ticker->sector
   mapping, persisted by scanner.py at `cache/universe_sectors.json` from the
   MOST RECENT live run. A ticker's sector rarely changes, so this is a mild
   approximation, but it means the gate can only be replicated for tickers
   that were part of a recent live scan. If the file is missing, the gate is
   skipped entirely (documented, not silent) and the backtest reduces to the
   pre-1.5.1 behaviour.

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
    classify_sector,
    detect_base,
    enhanced_metrics,
    find_sma150_breakout,
    sma50_curling_up,
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
    curl = sma50_curling_up(df, cfg)
    if curl is None or not curl["curling_up"]:
        return None
    base = detect_base(df, breakout["idx"], cfg)
    if base is None:
        return None
    # market_cap não é armazenado na cache de OHLCV, por isso aqui a seleção
    # de benchmark (choose_rs_benchmark) recai sempre em IWM. Para candidatas
    # mid cap reais isto sub-estima a força relativa face ao MDY correto —
    # o resultado do backtest é, nessa faixa, uma aproximação conservadora.
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


def sector_curling_at(etf_full: Optional[pd.DataFrame], cutoff, cfg: Config) -> Optional[dict[str, Any]]:
    """Estado da SMA50 do ETF de setor, recalculado apenas com dados até
    `cutoff` — a mesma disciplina ponto-no-tempo usada para os benchmarks."""
    if etf_full is None:
        return None
    sliced = etf_full[etf_full["date"] <= cutoff]
    if len(sliced) < 50:
        return None
    return sma50_curling_up(sliced.reset_index(drop=True), cfg)


def run(cfg: Config, step: int, min_score: float, max_per_ticker: int, min_i: int) -> dict[str, Any]:
    full_benchmarks = {}
    for symbol in ("QQQ", "IWM", "MDY"):
        path = cfg.ohlcv_dir / f"{symbol}.csv"
        if path.exists():
            full_benchmarks[symbol] = scanner._normalise_ohlcv(pd.read_csv(path))

    sector_map: dict[str, dict[str, Any]] = {}
    if cfg.sector_map_json.exists():
        try:
            sector_map = json.loads(cfg.sector_map_json.read_text(encoding="utf-8"))
        except Exception:
            sector_map = {}
    elif cfg.require_sector_etf_curl:
        print("AVISO: cache/universe_sectors.json não encontrado — gate de setor "
              "desligado neste backtest (corre scanner.py pelo menos uma vez primeiro).")

    full_sector_etfs: dict[str, pd.DataFrame] = {}
    if cfg.require_sector_etf_curl and sector_map:
        for _, etf, _ in scanner.SECTOR_ETF_MAP:
            if etf in full_sector_etfs:
                continue
            path = cfg.ohlcv_dir / f"{etf}.csv"
            if path.exists():
                try:
                    full_sector_etfs[etf] = scanner._normalise_ohlcv(pd.read_csv(path))
                except Exception:
                    pass

    files = sorted(p for p in cfg.ohlcv_dir.glob("*.csv") if p.stem not in {"QQQ", "IWM", "SPY"})
    samples: list[dict[str, Any]] = []
    for path in files:
        try:
            df = scanner._normalise_ohlcv(pd.read_csv(path))
        except Exception:
            continue
        if df is None or len(df) < cfg.min_history_sessions + max(HORIZONS):
            continue
        ticker_sector = None
        if cfg.require_sector_etf_curl and sector_map:
            info = sector_map.get(path.stem)
            if info:
                ticker_sector = classify_sector(info.get("sector"), info.get("industry"))
        taken = 0
        last_signal = -10**9
        for i in range(max(min_i, cfg.min_history_sessions), len(df) - max(HORIZONS), step):
            if taken >= max_per_ticker:
                break
            if i - last_signal < 20:  # de-overlap: one signal per ticker per month
                continue
            window = df.iloc[: i + 1].reset_index(drop=True)
            cutoff = window["date"].iloc[-1]
            if cfg.require_sector_etf_curl and sector_map:
                # Setor desconhecido/não classificado ou dados do ETF insuficientes
                # nesta data => sem forma de verificar o gate => rejeita, tal como
                # em produção (analyse_candidate faz o mesmo por omissão).
                if ticker_sector is None:
                    continue
                _, etf = ticker_sector
                curl_status = sector_curling_at(full_sector_etfs.get(etf), cutoff, cfg)
                if curl_status is None or not curl_status["curling_up"]:
                    continue
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
        "Survivorship bias: universo construído a partir da lista atual — resultados são um limite SUPERIOR otimista, mais acentuado no extremo nano/micro cap do que em mid cap.",
        "Sem spread, slippage nem halts. Universo agora misto: sub-$1/nano cap tem spreads de vários pontos percentuais, mid cap líquida tem muito menos — ver score buckets/regime em vez do número agregado.",
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
