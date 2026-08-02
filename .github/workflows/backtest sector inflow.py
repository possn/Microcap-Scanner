"""
backtest_sector_inflow.py

Backtest do sector_inflow.py: mede se o sinal de "inflow sustentado"
(vol_ratio + up_share) precede retorno futuro positivo do ETF sectorial,
comparado com o retorno médio desse mesmo ETF fora dos dias de sinal.

Sem isto, o sinal é apenas descritivo (diz-te que houve volume direcional
no passado) — o backtest é o que testa se tem valor preditivo.

Uso:
    python backtest_sector_inflow.py --start 2022-01-01 --end 2026-08-01 --horizon 10

Requer: yfinance, pandas, numpy (mesmas dependências do Heartbeat)
Corre no GitHub Actions ou localmente com rede — não corre em sandbox isolado.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from sector_inflow import SECTOR_ETFS, compute_inflow_signal


def backtest_ticker(
    df: pd.DataFrame,
    horizon: int = 10,
    vol_window: int = 5,
    vol_ref_window: int = 50,
    vol_multiplier: float = 2.0,
    up_share_threshold: float = 0.6,
) -> dict:
    """
    Para um ETF: calcula o retorno forward de `horizon` dias em cada data,
    e compara a média desse retorno nos dias com inflow_signal=True vs.
    nos restantes dias (baseline).
    """
    sig = compute_inflow_signal(
        df, vol_window, vol_ref_window, vol_multiplier, up_share_threshold
    )
    sig["fwd_return"] = sig["Close"].shift(-horizon) / sig["Close"] - 1

    # remove linhas sem retorno forward válido (fim da série) ou sem sinal calculável (início)
    valid = sig.dropna(subset=["fwd_return", "inflow_signal"])

    signal_days = valid[valid["inflow_signal"]]
    baseline_days = valid[~valid["inflow_signal"]]

    return {
        "n_signal_days": len(signal_days),
        "n_baseline_days": len(baseline_days),
        "avg_fwd_return_signal": signal_days["fwd_return"].mean(),
        "avg_fwd_return_baseline": baseline_days["fwd_return"].mean(),
        "hit_rate_signal": (signal_days["fwd_return"] > 0).mean() if len(signal_days) else np.nan,
        "hit_rate_baseline": (baseline_days["fwd_return"] > 0).mean() if len(baseline_days) else np.nan,
    }


def run_backtest(start: str, end: str, horizon: int = 10) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for ticker, sector in SECTOR_ETFS.items():
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist.empty or len(hist) < 100:
            print(f"[aviso] dados insuficientes para {ticker}, a saltar")
            continue
        result = backtest_ticker(hist[["Close", "Volume"]], horizon=horizon)
        result["ticker"] = ticker
        result["sector"] = sector
        rows.append(result)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["edge"] = df["avg_fwd_return_signal"] - df["avg_fwd_return_baseline"]
    cols = [
        "ticker", "sector", "n_signal_days", "n_baseline_days",
        "avg_fwd_return_signal", "avg_fwd_return_baseline", "edge",
        "hit_rate_signal", "hit_rate_baseline",
    ]
    return df[cols].sort_values("edge", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--horizon", type=int, default=10, help="dias úteis à frente para medir retorno")
    args = parser.parse_args()

    results = run_backtest(args.start, args.end, args.horizon)

    if results.empty:
        print("Sem resultados — verifica ligação de rede / tickers.")
    else:
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(results.to_string(index=False))
        print(
            "\nLeitura: 'edge' = retorno médio forward nos dias de sinal MENOS "
            "retorno médio forward nos restantes dias. Se 'edge' for consistentemente "
            "próximo de zero ou negativo, o sinal não tem valor preditivo além do "
            "comportamento normal do ETF — é ruído, não informação.\n"
            "'n_signal_days' baixo (<20-30) invalida qualquer conclusão estatística, "
            "independentemente do 'edge' observado."
        )
