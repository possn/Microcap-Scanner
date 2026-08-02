"""
backtest_sector_momentum.py

Testa a hipótese de momentum de força relativa sectorial:
sectores com melhor retorno EXCESSO vs. SPY nos últimos N meses
(lookback) tendem a continuar a superar nos M meses seguintes (horizon)?

Base empírica: Jegadeesh & Titman (1993), Journal of Finance — efeito
momentum em horizontes de meses. Faber (2010) aplica especificamente a
rotação sectorial.

Diferença crítica face ao backtest_sector_inflow.py: aqui as observações
são mensais, não diárias — evita o problema de sobreposição de amostra
(dias de sinal consecutivos não são eventos independentes) identificado
no backtest anterior.

Uso:
    python backtest_sector_momentum.py --start 2010-01-01 --lookback 6 --horizon 1
    python backtest_sector_momentum.py --start 2010-01-01 --grid   # varre combinações

Requer: yfinance, pandas, numpy
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from sector_inflow import SECTOR_ETFS


def _fetch_monthly_closes(start: str, end: str) -> pd.DataFrame:
    """
    Descarrega preços diários dos 11 ETFs + SPY e devolve close mensal
    (último dia útil de cada mês), uma coluna por ticker.
    """
    import yfinance as yf

    tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
    closes = {}
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist.empty or len(hist) < 100:
            print(f"[aviso] dados insuficientes para {ticker}, a saltar")
            continue
        # normalizar índice para tz-naive antes de resample mensal
        s = hist["Close"]
        s.index = s.index.tz_localize(None)
        closes[ticker] = s

    df = pd.DataFrame(closes)
    monthly = df.resample("ME").last()
    return monthly


def backtest_momentum(
    monthly_closes: pd.DataFrame,
    lookback: int = 6,
    horizon: int = 1,
    top_n: int = 4,
    bottom_n: int = 4,
) -> dict:
    """
    lookback: meses usados para calcular o retorno trailing (força relativa)
    horizon: meses à frente medidos como retorno forward
    top_n / bottom_n: quantos sectores (de 11) entram no grupo "líder" / "lagging"
    """
    if "SPY" not in monthly_closes.columns:
        raise ValueError("SPY em falta nos dados — necessário para calcular excesso de retorno")

    sector_cols = [c for c in monthly_closes.columns if c != "SPY"]

    trailing_ret = monthly_closes.pct_change(lookback)
    spy_trailing = trailing_ret["SPY"]
    excess_ret = trailing_ret[sector_cols].sub(spy_trailing, axis=0)

    fwd_ret = monthly_closes[sector_cols].pct_change(horizon).shift(-horizon)

    top_returns = []
    bottom_returns = []
    n_dates = 0

    # itera por cada mês de rebalanceamento (exceto os que não têm lookback/horizon completos)
    for date in excess_ret.index:
        row = excess_ret.loc[date].dropna()
        if len(row) < (top_n + bottom_n):
            continue
        fwd_row = fwd_ret.loc[date]
        if fwd_row.isna().all():
            continue

        ranked = row.sort_values(ascending=False)
        top_tickers = ranked.index[:top_n]
        bottom_tickers = ranked.index[-bottom_n:]

        top_fwd = fwd_row[top_tickers].dropna()
        bottom_fwd = fwd_row[bottom_tickers].dropna()

        if len(top_fwd) == 0 or len(bottom_fwd) == 0:
            continue

        top_returns.extend(top_fwd.tolist())
        bottom_returns.extend(bottom_fwd.tolist())
        n_dates += 1

    top_arr = np.array(top_returns)
    bottom_arr = np.array(bottom_returns)

    return {
        "lookback_meses": lookback,
        "horizon_meses": horizon,
        "n_datas_rebalanceamento": n_dates,
        "n_obs_top": len(top_arr),
        "n_obs_bottom": len(bottom_arr),
        "avg_fwd_return_top": top_arr.mean() if len(top_arr) else np.nan,
        "avg_fwd_return_bottom": bottom_arr.mean() if len(bottom_arr) else np.nan,
        "edge": (top_arr.mean() - bottom_arr.mean()) if len(top_arr) and len(bottom_arr) else np.nan,
        "hit_rate_top": (top_arr > 0).mean() if len(top_arr) else np.nan,
        "hit_rate_bottom": (bottom_arr > 0).mean() if len(bottom_arr) else np.nan,
    }


def grid_scan(monthly_closes: pd.DataFrame) -> pd.DataFrame:
    lookbacks = [3, 6, 12]
    horizons = [1, 3]
    rows = []
    for lb in lookbacks:
        for h in horizons:
            res = backtest_momentum(monthly_closes, lookback=lb, horizon=h)
            rows.append(res)
    return pd.DataFrame(rows).sort_values("edge", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--lookback", type=int, default=6, help="meses de retorno trailing")
    parser.add_argument("--horizon", type=int, default=1, help="meses forward a medir")
    parser.add_argument("--top_n", type=int, default=4)
    parser.add_argument("--bottom_n", type=int, default=4)
    parser.add_argument("--grid", action="store_true", help="varre combinações de lookback x horizon")
    args = parser.parse_args()

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    closes = _fetch_monthly_closes(args.start, args.end)

    if closes.empty or "SPY" not in closes.columns:
        print("Sem dados suficientes — verifica ligação de rede.")
    elif args.grid:
        results = grid_scan(closes)
        print(results.to_string(index=False))
        print(
            "\nLeitura: compara 'edge' entre combinações de lookback/horizon. "
            "'n_datas_rebalanceamento' baixo (<30-40, ~3-4 anos de dados mensais) "
            "invalida qualquer conclusão. Nota: horizons >1 mês com rebalanceamento "
            "mensal têm janelas forward sobrepostas — trata o 'n' como um limite "
            "superior otimista da amostra efetiva, não como contagem de eventos "
            "independentes."
        )
    else:
        result = backtest_momentum(
            closes, args.lookback, args.horizon, args.top_n, args.bottom_n
        )
        for k, v in result.items():
            print(f"{k}: {v}")
        print(
            "\nLeitura: 'edge' = retorno médio forward do grupo TOP (maior força "
            "relativa) MENOS grupo BOTTOM. Positivo e consistente = evidência a "
            "favor de momentum sectorial. Perto de zero = sem edge detectável."
        )
