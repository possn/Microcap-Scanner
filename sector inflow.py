"""
sector_inflow.py

Deteta inflow sustentado nos 11 SPDR Select Sector ETFs, usando volume
condicionado por direção (up-day vs down-day), não volume bruto.

Lógica:
  - vol_ratio  = média de volume dos últimos N dias / média de referência (ex: 50 dias)
  - up_share   = fração do volume (nos últimos N dias) que ocorreu em dias de alta
  - sinal      = vol_ratio >= threshold  E  up_share >= threshold

Isto distingue acumulação (volume alto concentrado em dias de subida) de
distribuição (volume alto concentrado em dias de descida) e de ruído de
rebalanceamento de índice (volume alto sem viés direcional consistente).

Integração com o Heartbeat: reaproveita a mesma estrutura de scanner
(cache/, GitHub Actions, Telegram) — este módulo produz apenas o ranking
sectorial; o screening final dos 10 tickers por sector continua a usar
o setup nuclear (base + SMA150 + volume) já existente, restrito ao(s)
sector(es) com sinal de inflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Os 11 SPDR Select Sector ETFs (cobertura GICS do S&P500)
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


def compute_inflow_signal(
    df: pd.DataFrame,
    vol_window: int = 5,
    vol_ref_window: int = 50,
    vol_multiplier: float = 2.0,
    up_share_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    df: DataFrame com colunas ['Close', 'Volume'], índice de datas ascendente.
    Retorna o df com colunas adicionais, incluindo 'inflow_signal' (bool)
    e 'inflow_score' (contínuo, para ranking entre sectores).
    """
    out = df.copy()
    out["change"] = out["Close"].diff()
    out["up_day"] = out["change"] > 0

    out["up_volume"] = np.where(out["up_day"], out["Volume"], 0.0)
    out["down_volume"] = np.where(~out["up_day"], out["Volume"], 0.0)

    out["vol_ref"] = out["Volume"].rolling(vol_ref_window, min_periods=vol_ref_window).mean()
    out["vol_recent"] = out["Volume"].rolling(vol_window, min_periods=vol_window).mean()
    out["vol_ratio"] = out["vol_recent"] / out["vol_ref"]

    up_roll = out["up_volume"].rolling(vol_window, min_periods=vol_window).sum()
    down_roll = out["down_volume"].rolling(vol_window, min_periods=vol_window).sum()
    out["up_share"] = up_roll / (up_roll + down_roll)

    out["inflow_signal"] = (out["vol_ratio"] >= vol_multiplier) & (
        out["up_share"] >= up_share_threshold
    )

    # score contínuo para ranking (não binário) — útil para ordenar sectores
    # mesmo quando nenhum ou vários passam o threshold binário
    out["inflow_score"] = out["vol_ratio"].fillna(0) * out["up_share"].fillna(0)

    return out


def rank_sectors(
    price_data: dict[str, pd.DataFrame],
    vol_window: int = 5,
    vol_ref_window: int = 50,
    vol_multiplier: float = 2.0,
    up_share_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    price_data: dict {ticker: DataFrame com ['Close','Volume']}, uma entrada
    por ETF sectorial (tipicamente vindo de yfinance).
    Retorna um ranking do dia mais recente, ordenado por inflow_score desc.
    """
    rows = []
    for ticker, df in price_data.items():
        sig = compute_inflow_signal(
            df, vol_window, vol_ref_window, vol_multiplier, up_share_threshold
        )
        latest = sig.iloc[-1]
        rows.append(
            {
                "ticker": ticker,
                "sector": SECTOR_ETFS.get(ticker, ticker),
                "vol_ratio": latest["vol_ratio"],
                "up_share": latest["up_share"],
                "inflow_signal": bool(latest["inflow_signal"]),
                "inflow_score": latest["inflow_score"],
            }
        )
    ranking = pd.DataFrame(rows).sort_values("inflow_score", ascending=False)
    return ranking.reset_index(drop=True)


def fetch_sector_data(period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Descarrega OHLCV para os 11 ETFs sectoriais via yfinance.
    NOTA: requer acesso de rede a query1/query2.finance.yahoo.com — corre
    no ambiente do GitHub Actions do Heartbeat, não neste sandbox.
    """
    import yfinance as yf

    data = {}
    for ticker in SECTOR_ETFS:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if not hist.empty:
            data[ticker] = hist[["Close", "Volume"]]
    return data


if __name__ == "__main__":
    # Exemplo de uso real (requer rede):
    # data = fetch_sector_data()
    # ranking = rank_sectors(data)
    # print(ranking.to_string(index=False))
    pass
