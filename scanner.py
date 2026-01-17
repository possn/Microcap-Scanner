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
# CSV fetchers
# -----------------------------
def fetch_csv(url: str, holdings: bool = False) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text

    if not holdings:
        return pd.read_csv(io.StringIO(text))

    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines[:400]):
        if ln.strip().lower().startswith("ticker,"):
            header_idx = i
            break
    if header_idx is None:
        try:
            return pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")
        except Exception as e:
            raise RuntimeError("Não encontrei header holdings.") from e

    cleaned = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(cleaned), engine="python", on_bad_lines="skip")


def get_universe(holdings_url: str) -> list[str]:
    df = fetch_csv(holdings_url, holdings=True)
    cols = {c.lower(): c for c in df.columns}
    if "ticker" in cols:
        col = cols["ticker"]
    elif "symbol" in cols:
        col = cols["symbol"]
    else:
        raise RuntimeError(f"Holdings sem Ticker/Symbol: {df.columns.tolist()}")

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
    df = fetch_csv(url, holdings=False)

    cols = {c.lower(): c for c in df.columns}
    need = ["date", "open", "high", "low", "close", "volume"]
    if not all(k in cols for k in need):
        raise RuntimeError(f"{ticker}: formato OHLCV inválido")

    df = df[[cols["date"], cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if cached is not None and len(cached) > 0:
        df = pd.concat([cached, df], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    iwc_url = os.environ["IWC_HOLDINGS_CSV_URL"]
    iwm_url = os.environ["IWM_HOLDINGS_CSV_URL"]
    ijr_url = os.environ["IJR_HOLDINGS_CSV_URL"]

    ohlcv_fmt = os.environ["OHLCV_URL_FMT"]
    max_n = int(os.environ.get("MAX_TICKERS", "600"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    u1 = get_universe(iwc_url)
    u2 = get_universe(iwm_url)
    u3 = get_universe(ijr_url)

    tickers = list(set(u1 + u2 + u3))[:max_n]

    results = []  # (ticker, decision, px, dv20, bbz, atr_pctl, win, high_base)
    pipeline_pass = 0
    # Funnel counters
    c_total = 0
    c_hist = 0
    c_liq = 0
    c_comp = 0
    c_base = 0
    c_dry = 0
    near = []     # (ticker, px, dv20, bbz, atr_pctl, reason)

    for i, t in enumerate(tickers):
        try:
            df = fetch_ohlcv(t, ohlcv_fmt)
            if len(df) < 260:
                continue

            px = float(df["close"].iloc[-1])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())

            if not (px >= 1.0 and dv20 >= 2_000_000):
                continue

            bb_width = compute_bb_width(df, n=20)
            bb_z = zscore(bb_width, window=120)

            atr = compute_atr(df, n=14)
            atr_pct = atr / df["close"]

            bbz_last = float(bb_z.iloc[-1])
            atr_last = float(atr_pct.iloc[-1])
            atr_pctl = float((atr_pct.iloc[-252:] <= atr_last).mean())

            if not (np.isfinite(bbz_last) and np.isfinite(atr_pctl)):
                continue

            # Relaxed but still conservative
            if not (bbz_last < -0.8 and atr_pctl < 0.35):
                continue

            # Base 4–10 weeks
            base_min = 20
            base_max = 50
            base_pass = False
            best_win = None
            best_high = None

            for win in range(base_min, base_max + 1, 5):
                base = df.iloc[-win:]
                high_base = float(base["high"].max())
                low_base = float(base["low"].min())

                dd = (high_base - low_base) / high_base if high_base > 0 else 1.0

                prev = df.iloc[-(win + 120):-win]
                if len(prev) < 60:
                    continue

                prev_range = float(prev["high"].max() - prev["low"].min())
                base_range = float(high_base - low_base)
                contraction_ratio = (base_range / prev_range) if prev_range > 0 else 1.0

            if dd <= 0.45 and contraction_ratio <= 0.65:
                    base_pass = True
                    best_win = win
                    best_high = high_base
                    break

            if not base_pass or best_high is None or best_win is None:
                near.append((t, px, dv20, bbz_last, atr_pctl, "FAIL_BASE"))
                continue

            # Dry-up (relaxed)
            vol10 = float(df["volume"].iloc[-10:].mean())
            vol60 = float(df["volume"].iloc[-60:].mean())
            if not (vol60 > 0 and (vol10 / vol60) < 0.90):
                near.append((t, px, dv20, bbz_last, atr_pctl, "FAIL_DRYUP"))
                continue

            # EXECUTAR trigger
            close_today = float(df["close"].iloc[-1])
            open_today = float(df["open"].iloc[-1])
            close_prev = float(df["close"].iloc[-2])
            vol_today = float(df["volume"].iloc[-1])
            vol20 = float(df["volume"].iloc[-20:].mean())

            breakout = close_today > best_high
            vol_confirm = (vol20 > 0) and (vol_today >= 1.5 * vol20)
            no_exhaust_gap = open_today <= close_prev * 1.10

            decision = "EXECUTAR" if (breakout and vol_confirm and no_exhaust_gap) else "AGUARDAR"
            results.append((t, decision, px, dv20, bbz_last, atr_pctl, best_win, best_high))

        except Exception:
            pass

        if (i + 1) % 50 == 0:
            time.sleep(1)

    exec_list = [r for r in results if r[1] == "EXECUTAR"]
    watch_list = [r for r in results if r[1] == "AGUARDAR"]

    exec_list.sort(key=lambda x: -x[3])
    watch_list.sort(key=lambda x: (x[4], -x[3]))

    exec_top = exec_list[:5]
    watch_top = watch_list[:10]

    msg = [f"[{now}] Microcap scanner (Modo A) [BUILD=IJR_UNIVERSE_V1]"]
    msg.append(f"Universo combinado bruto: {len(set(u1 + u2 + u3))}")
    msg.append(f"Universo avaliado (cap {max_n}): {len(tickers)}")
    msg.append("")
    msg.append(f"FUNIL: total={c_total} | hist={c_hist} | liq={c_liq} | comp={c_comp} | base={c_base} | dry={c_dry}")
    msg.append("")
    msg.append(f"PIPELINE_OK: {len(results)}")
    msg.append("")

    if not exec_top:
        msg.append("EXECUTAR: (vazio)")
    else:
        msg.append("EXECUTAR (gatilho EOD confirmado):")
        for t, decision, px, dv, bbz, atrp, win, highb in exec_top:
            msg.append(f"- {t} | close={px:.2f} | dv20=${dv/1e6:.1f}M | base={win}d | trigger>={highb:.2f}")

    msg.append("")
    if not watch_top:
        msg.append("AGUARDAR: (vazio)")
    else:
        msg.append("AGUARDAR (staging):")
        for t, decision, px, dv, bbz, atrp, win, highb in watch_top:
            msg.append(f"- {t} | close={px:.2f} | dv20=${dv/1e6:.1f}M | BBz={bbz:.2f} | ATRpctl={atrp:.2f} | base={win}d | trigger>={highb:.2f}")

    if not exec_top and not watch_top:
        msg.append("")
        msg.append("FLAT: nenhum candidato passou os filtros finais (base+dry-up).")

        if near:
            near.sort(key=lambda x: (x[3], -x[2]))
            near_top = near[:10]
            msg.append("")
            msg.append("QUASE (passou compressão, falhou base/dry-up):")
            for t, px, dv, bbz, atrp, reason in near_top:
                msg.append(f"- {t} | {reason} | close={px:.2f} | dv20=${dv/1e6:.1f}M | BBz={bbz:.2f} | ATRpctl={atrp:.2f}")

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
