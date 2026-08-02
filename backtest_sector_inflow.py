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


def _fetch_all(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Descarrega os 11 ETFs uma vez só, para reaproveitar entre várias combinações de thresholds."""
    import yfinance as yf

    data = {}
    for ticker in SECTOR_ETFS:
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist.empty or len(hist) < 100:
            print(f"[aviso] dados insuficientes para {ticker}, a saltar")
            continue
        data[ticker] = hist[["Close", "Volume"]]
    return data


def run_backtest(
    start: str,
    end: str,
    horizon: int = 10,
    vol_window: int = 5,
    vol_ref_window: int = 50,
    vol_multiplier: float = 2.0,
    up_share_threshold: float = 0.6,
    price_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if price_data is None:
        price_data = _fetch_all(start, end)

    rows = []
    for ticker, df in price_data.items():
        result = backtest_ticker(
            df, horizon, vol_window, vol_ref_window, vol_multiplier, up_share_threshold
        )
        result["ticker"] = ticker
        result["sector"] = SECTOR_ETFS.get(ticker, ticker)
        rows.append(result)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    df_out["edge"] = df_out["avg_fwd_return_signal"] - df_out["avg_fwd_return_baseline"]
    cols = [
        "ticker", "sector", "n_signal_days", "n_baseline_days",
        "avg_fwd_return_signal", "avg_fwd_return_baseline", "edge",
        "hit_rate_signal", "hit_rate_baseline",
    ]
    return df_out[cols].sort_values("edge", ascending=False)


def grid_scan(start: str, end: str, horizon: int = 10) -> pd.DataFrame:
    """
    Testa combinações de vol_multiplier x up_share_threshold e reporta
    o total de sinais gerados (soma nos 11 ETFs) para cada combinação.
    Objetivo: encontrar uma zona de parâmetros com amostra estatística
    suficiente antes de avaliar 'edge' a sério.
    """
    price_data = _fetch_all(start, end)

    vol_multipliers = [1.2, 1.3, 1.5, 1.7, 2.0]
    up_share_thresholds = [0.5, 0.55, 0.6, 0.65]

    rows = []
    for vm in vol_multipliers:
        for ust in up_share_thresholds:
            res = run_backtest(
                start, end, horizon,
                vol_multiplier=vm, up_share_threshold=ust,
                price_data=price_data,
            )
            total_signals = res["n_signal_days"].sum() if not res.empty else 0
            n_sectors_with_signal = (res["n_signal_days"] > 0).sum() if not res.empty else 0
            rows.append({
                "vol_multiplier": vm,
                "up_share_threshold": ust,
                "total_signal_days": total_signals,
                "sectores_com_sinal": n_sectors_with_signal,
            })
    return pd.DataFrame(rows).sort_values("total_signal_days", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--horizon", type=int, default=10, help="dias úteis à frente para medir retorno")
    parser.add_argument("--vol_multiplier", type=float, default=2.0)
    parser.add_argument("--up_share_threshold", type=float, default=0.6)
    parser.add_argument("--vol_window", type=int, default=5)
    parser.add_argument("--vol_ref_window", type=int, default=50)
    parser.add_argument("--grid", action="store_true", help="corre grid-scan de thresholds em vez de um único backtest")
    args = parser.parse_args()

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    if args.grid:
        results = grid_scan(args.start, args.end, args.horizon)
        print(results.to_string(index=False))
        print(
            "\nLeitura: 'total_signal_days' = soma de sinais nos 11 ETFs no período. "
            "Escolhe a combinação com pelo menos ~150-300 sinais totais (ou ~15-30 por "
            "sector, idealmente) antes de correr o backtest normal com esses parâmetros. "
            "Combinações mais soltas (vol_multiplier baixo, up_share_threshold baixo) "
            "geram mais sinais mas cada um é mais fraco/menos seletivo — é um trade-off, "
            "não escolhas automaticamente a linha do topo sem pensar no que significa."
        )
    else:
        results = run_backtest(
            args.start, args.end, args.horizon,
            args.vol_window, args.vol_ref_window,
            args.vol_multiplier, args.up_share_threshold,
        )
        if results.empty:
            print("Sem resultados — verifica ligação de rede / tickers.")
        else:
            print(results.to_string(index=False))
            print(
                "\nLeitura: 'edge' = retorno médio forward nos dias de sinal MENOS "
                "retorno médio forward nos restantes dias. Se 'edge' for consistentemente "
                "próximo de zero ou negativo, o sinal não tem valor preditivo além do "
                "comportamento normal do ETF — é ruído, não informação.\n"
                "'n_signal_days' baixo (<20-30) invalida qualquer conclusão estatística, "
                "independentemente do 'edge' observado."
            )
