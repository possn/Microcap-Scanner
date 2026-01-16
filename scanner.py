import os
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# -----------------------------
# Indicators
# -----------------------------
def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs())
    )
    return tr.rolling(n).mean()


def compute_bb_width(df: pd.DataFrame, n: int = 20) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / ma


def zscore(series: pd.Series, window: int = 100) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


# -----------------------------
# Telegram
# -----------------------------
TG_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]


def tg_send(text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


# -----------------------------
# Data fetchers
# -----------------------------
def fetch_csv(url: str, holdings: bool = False) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text

    if not holdings:
        return pd.read_csv(io.StringIO(text))

    # iShares holdings: skip metadata until header starts with "Ticker,"
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines[:250]):
        if ln.strip().lower().startswith("ticker,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Não encontrei o header 'Ticker,' no holdings CSV.")

    cleaned = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(cleaned), engine="python", on_bad_lines="skip")


def get_universe(holdings_url: str) -> list[str]:
    df = fetch_csv(holdings_url, holdings=True)

    # Be tolerant to minor column name variations
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols and "symbol" not in cols:
        raise RuntimeError(f"Holdings CSV sem coluna Ticker/Symbol: {df.columns.tolist()}")

    col = cols.get("ticker", cols.get("symbol"))
    tickers = df[col].astype(str).str.strip()
    tickers = tickers[tickers.str.len() > 0].unique().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def fetch_ohlcv(ticker: str, fmt: str) -> pd.DataFrame:
    os.makedirs("cache/ohlcv", exist_ok=True)
    path = f"cache/ohlcv/{ticker}.csv"

    cached = None
    if os.path.exists(path):
        try:
            cached = pd.read_csv(path)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"]).sort_values("date")
        except Exception:
            cached = None

    url = fmt.format(symbol=ticker.lower())
    df = fetch_csv(url)

    cols = {c.lower(): c for c in df.columns}
    need = ["date", "open", "high", "low", "close", "volume"]
    if not all(k in cols for k in need):
        raise RuntimeError(f"{ticker}: formato OHLCV inválido, colunas={df.columns.tolist()}")

    df = df[[cols["date"], cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Merge cache + new, de-duplicate by date
    if cached is not None and len(cached) > 0:
        df = pd.concat([cached, df], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.reset_index(drop=True)

    # Save back to cache
    df.to_csv(path, index=False)

    return df


# -----------------------------
# Main (Mode A - Conservative staging)
# -----------------------------
def main() -> None:
    holdings_url = os.environ["IWC_HOLDINGS_CSV_URL"]
    ohlcv_fmt = os.environ["OHLCV_URL_FMT"]
    max_n = int(os.environ.get("MAX_TICKERS", "300"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tickers = get_universe(holdings_url)[:max_n]
    results = []

    for i, t in enumerate(tickers):
        try:
            df = fetch_ohlcv(t, ohlcv_fmt)

            # Need enough history for percentiles/zscore
            if len(df) < 260:
                continue

            px = float(df.iloc[-1]["close"])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())

            # Liquidity + price filters
            if not (px >= 1.0 and dv20 >= 3_000_000):
                continue

            # --- Compression metrics ---
            bb_width = compute_bb_width(df, n=20)
            bb_z = zscore(bb_width, window=120)

            atr = compute_atr(df, n=14)
            atr_pct = atr / df["close"]

            atr_last = float(atr_pct.iloc[-1])
            atr_pctl = float((atr_pct.iloc[-252:] <= atr_last).mean())
            bbz_last = float(bb_z.iloc[-1])

            # Conservative staging gate
                        # Conservative staging gate
            is_staging = (bbz_last < -1.8) and (atr_pctl < 0.10)

            if is_staging and np.isfinite(bbz_last) and np.isfinite(atr_pctl):

                # -------- BASE QUALITY FILTER (4–10 weeks) --------
                base_min = 20     # ~4 semanas
                base_max = 50     # ~10 semanas

                base_pass = False

                for win in range(base_min, base_max + 1, 5):
                    base = df.iloc[-win:]

                    high_base = float(base["high"].max())
                    low_base = float(base["low"].min())

                    # drawdown interno da base
                    dd = (high_base - low_base) / high_base if high_base > 0 else 1.0

                    # comparar range da base vs range anterior (até 6 meses)
                    prev = df.iloc[-(win + 120):-win]
                    if len(prev) < 60:
                        continue

                    prev_range = float(prev["high"].max() - prev["low"].min())
                    base_range = float(high_base - low_base)

                    contraction_ratio = (base_range / prev_range) if prev_range > 0 else 1.0

                    if dd <= 0.35 and contraction_ratio <= 0.50:
                        base_pass = True
                        break

                if base_pass:
                    results.append((t, px, dv20, bbz_last, atr_pctl))
   
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            time.sleep(1)

    # Sort: most compressed first (more negative bbz), then higher liquidity
    results.sort(key=lambda x: (x[3], -x[2]))
    top = results[:15]

    msg = [f"[{now}] Microcap scanner (Modo A - staging)"]
    msg.append(f"Universo avaliado: {len(tickers)}")
    msg.append("")

    if not top:
        msg.append("FLAT: nenhum candidato passou o filtro conservador de compressão.")
    else:
        msg.append("AGUARDAR (compressão extrema):")
        for t, px, dv, bbz, atrp in top:
            msg.append(f"- {t} | close={px:.2f} | dv20=${dv/1e6:.1f}M | BBz={bbz:.2f} | ATRpctl={atrp:.2f}")

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
