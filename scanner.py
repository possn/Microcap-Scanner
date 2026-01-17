# scanner.py  (V3)
# Daily US small+microcap pre-breakout scanner (free/public data, no paid APIs)
# Universe: iShares holdings CSV (IWC + IWM + IJR) via secrets
# Prices/Volume: Stooq EOD CSV via OHLCV_URL_FMT (handles .us suffix correctly)
#
# V3 improvements:
# - Rate-limit resilience: detects "daily hits limit" and stops further OHLCV calls early
# - Optional caps to approximate small+micro without market-cap (verifiable proxies):
#     MAX_PX, MAX_DV20 (set as GitHub Actions variables)
# - Prints base LOW + suggested stop (invalidation)
# - Always shows QUASE + DIAG
#
# Copy/paste this whole file and commit.

import os
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# -----------------------------
# Config (env)
# -----------------------------
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")

IWC_URL = os.environ.get("IWC_HOLDINGS_CSV_URL", "")
IWM_URL = os.environ.get("IWM_HOLDINGS_CSV_URL", "")
IJR_URL = os.environ.get("IJR_HOLDINGS_CSV_URL", "")

# Examples:
#   https://stooq.com/q/d/l/?s={symbol}.us&i=d
#   https://stooq.com/q/d/l/?s={symbol}&i=d
OHLCV_FMT = os.environ.get("OHLCV_URL_FMT", "https://stooq.com/q/d/l/?s={symbol}.us&i=d")

MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "350"))  # reduce Stooq limits by default

# Liquidity filters
MIN_PX = float(os.environ.get("MIN_PX", "1.0"))
MIN_DV20 = float(os.environ.get("MIN_DV20", "2000000"))   # $2M/day avg dollar vol
MIN_SV20 = float(os.environ.get("MIN_SV20", "500000"))    # 0.5M shares/day avg

# Optional caps (proxy to exclude large caps; still fully verifiable)
MAX_PX = float(os.environ.get("MAX_PX", "80"))
MAX_DV20 = float(os.environ.get("MAX_DV20", "300000000"))  # $300M/day

# Compression thresholds
BBZ_GATE = float(os.environ.get("BBZ_GATE", "-0.7"))
ATRPCTL_GATE = float(os.environ.get("ATRPCTL_GATE", "0.45"))

# Base + dry-up thresholds
BASE_DD_MAX = float(os.environ.get("BASE_DD_MAX", "0.55"))
CONTRACTION_MAX = float(os.environ.get("CONTRACTION_MAX", "0.85"))
DRYUP_MAX = float(os.environ.get("DRYUP_MAX", "0.95"))

# EXECUTAR trigger strictness
VOL_CONFIRM_MULT = float(os.environ.get("VOL_CONFIRM_MULT", "1.5"))  # set 1.2 for more signals
MAX_GAP_UP = float(os.environ.get("MAX_GAP_UP", "1.12"))

# Pacing/backoff
SLEEP_EVERY = int(os.environ.get("SLEEP_EVERY", "60"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "1.0"))


# -----------------------------
# Telegram
# -----------------------------
def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
    except Exception:
        pass


# -----------------------------
# Indicators
# -----------------------------
def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(n).mean()


def compute_bb_width(df: pd.DataFrame, n: int = 20) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / ma


def zscore(series: pd.Series, window: int = 120) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


# -----------------------------
# HTTP + Holdings parsing
# -----------------------------
def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def fetch_holdings_csv(url: str) -> pd.DataFrame:
    text = fetch_text(url)
    lines = text.splitlines()

    header_idx = None
    for i, ln in enumerate(lines[:800]):
        s = ln.strip().lower()
        if s.startswith("ticker,") or s.startswith("symbol,"):
            header_idx = i
            break

    if header_idx is None:
        if "<!doctype html" in text.lower():
            raise RuntimeError("Holdings URL devolveu HTML (bloqueio/redirect).")
        return pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")

    cleaned = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(cleaned), engine="python", on_bad_lines="skip")


def is_valid_ticker(t: str) -> bool:
    if not t:
        return False
    t = t.strip().upper()
    if t in {"-", "N/A", "NA", "CASH", "USD"}:
        return False
    if len(t) < 2:
        return False
    if not any(ch.isalpha() for ch in t):
        return False
    if any(ch in t for ch in [" ", "/", "\\"]):
        return False
    return True


def get_universe_from_holdings(url: str) -> list[str]:
    if not url:
        return []
    df = fetch_holdings_csv(url)
    cols = {c.lower(): c for c in df.columns}
    if "ticker" in cols:
        col = cols["ticker"]
    elif "symbol" in cols:
        col = cols["symbol"]
    else:
        raise RuntimeError(f"Holdings sem coluna Ticker/Symbol: {df.columns.tolist()[:20]}")

    raw = df[col].astype(str).str.strip().tolist()
    out = []
    for t in raw:
        t = t.replace(".", "-").upper()
        if is_valid_ticker(t):
            out.append(t)
    return sorted(set(out))


# -----------------------------
# OHLCV (Stooq) with cache
# -----------------------------
def ensure_dirs() -> None:
    os.makedirs("cache/ohlcv", exist_ok=True)


def ohlcv_cache_path(ticker: str) -> str:
    return f"cache/ohlcv/{ticker}.csv"


def build_ohlcv_url(ticker: str) -> str:
    # Prevent .us.us errors
    sym = ticker.lower().replace("-", ".")
    fmt_lower = OHLCV_FMT.lower()
    if ".us" in fmt_lower:
        return OHLCV_FMT.format(symbol=sym)
    return OHLCV_FMT.format(symbol=f"{sym}.us")


def fetch_ohlcv_stooq(ticker: str) -> pd.DataFrame:
    ensure_dirs()
    path = ohlcv_cache_path(ticker)

    cached = None
    if os.path.exists(path):
        try:
            cached = pd.read_csv(path)
        except Exception:
            cached = None

    url = build_ohlcv_url(ticker)
    text = fetch_text(url).strip()
    low = text.lower()

    if low.startswith("no data") or low == "no data":
        raise RuntimeError("NO_DATA")
    if "exceeded the daily hits limit" in low:
        raise RuntimeError("HITS_LIMIT")

    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]

    if len(df.columns) == 1:
        h = df.columns[0].lower()
        if "exceeded the daily hits limit" in h:
            raise RuntimeError("HITS_LIMIT")
        if "no data" in h:
            raise RuntimeError("NO_DATA")

    need = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        raise RuntimeError(f"OHLCV formato inesperado: {df.columns.tolist()}")

    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if cached is not None and len(cached) > 0:
        cached.columns = [c.strip().lower() for c in cached.columns]
        if "date" in cached.columns:
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"]).sort_values("date")
            df = pd.concat([cached, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    df.to_csv(path, index=False)
    return df.reset_index(drop=True)


# -----------------------------
# Base + dry-up + false breakout
# -----------------------------
def base_scan(df: pd.DataFrame) -> tuple[bool, int, float, float, float, float]:
    if len(df) < 260:
        return (False, 0, np.nan, np.nan, np.nan, np.nan)

    best = None
    for win in [20, 25, 30, 35, 40, 50, 60, 70, 80]:
        base = df.iloc[-win:]
        highb = float(base["high"].max())
        lowb = float(base["low"].min())
        if highb <= 0:
            continue
        dd = (highb - lowb) / highb

        prev = df.iloc[-(win + 120):-win]
        if len(prev) < 60:
            continue
        prev_range = float(prev["high"].max() - prev["low"].min())
        base_range = float(highb - lowb)
        contr = (base_range / prev_range) if prev_range > 0 else 1.0

        if dd <= BASE_DD_MAX and contr <= CONTRACTION_MAX:
            score = (BASE_DD_MAX - dd) + (CONTRACTION_MAX - contr)
            cand = (score, win, highb, lowb, dd, contr)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return (False, 0, np.nan, np.nan, np.nan, np.nan)

    _, win, highb, lowb, dd, contr = best
    return (True, int(win), float(highb), float(lowb), float(dd), float(contr))


def dryup_ok(df: pd.DataFrame) -> tuple[bool, float]:
    v10 = float(df["volume"].iloc[-10:].mean())
    v60 = float(df["volume"].iloc[-60:].mean())
    if v60 <= 0:
        return (False, np.nan)
    ratio = v10 / v60
    return (ratio < DRYUP_MAX, float(ratio))


def failed_breakout_recent(df: pd.DataFrame, level: float, lookback: int = 30) -> bool:
    if len(df) < lookback + 5:
        return False
    closes = df["close"].iloc[-(lookback + 5):].values
    for i in range(len(closes) - 4):
        if closes[i] > level:
            if closes[i + 1] < level or closes[i + 2] < level or closes[i + 3] < level:
                return True
    return False


def suggested_stop(low_base: float) -> float:
    return float(low_base * 0.99)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ensure_dirs()

    errors: list[str] = []
    hits_limited = False

    # Universe
    u1 = []
    u2 = []
    u3 = []
    try:
        u1 = get_universe_from_holdings(IWC_URL)
    except Exception as e:
        errors.append(f"IWC holdings: {str(e)[:140]}")
    try:
        u2 = get_universe_from_holdings(IWM_URL)
    except Exception as e:
        errors.append(f"IWM holdings: {str(e)[:140]}")
    try:
        u3 = get_universe_from_holdings(IJR_URL)
    except Exception as e:
        errors.append(f"IJR holdings: {str(e)[:140]}")

    universe_all = sorted(set(u1 + u2 + u3))
    tickers = universe_all[:MAX_TICKERS]

    # Diagnostics
    n_hist_ok = 0
    n_liq_ok = 0
    n_comp_ok = 0
    n_base_ok = 0
    n_dry_ok = 0
    n_no_data = 0
    n_hits_limit = 0

    results_exec = []
    results_watch = []
    results_near = []

    for i, t in enumerate(tickers):
        if hits_limited:
            break
        try:
            df = fetch_ohlcv_stooq(t)

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

            if len(df) < 260:
                continue
            n_hist_ok += 1

            px = float(df["close"].iloc[-1])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
            sv20 = float(df["volume"].iloc[-20:].mean())

            if px > MAX_PX or dv20 > MAX_DV20:
                continue
            if px < MIN_PX or dv20 < MIN_DV20 or sv20 < MIN_SV20:
                continue
            n_liq_ok += 1

            bbw = compute_bb_width(df, n=20)
            bbz = zscore(bbw, window=120)
            atrp = compute_atr(df, n=14) / df["close"]

            bbz_last = float(bbz.iloc[-1]) if len(bbz) else np.nan
            atr_last = float(atrp.iloc[-1]) if len(atrp) else np.nan

            atr_win = atrp.iloc[-252:].dropna()
            if len(atr_win) < 80 or not np.isfinite(atr_last):
                continue
            atr_pctl = float((atr_win <= atr_last).mean())

            if (not np.isfinite(bbz_last)) or (not np.isfinite(atr_pctl)):
                continue
            if not (bbz_last < BBZ_GATE and atr_pctl < ATRPCTL_GATE):
                continue
            n_comp_ok += 1

            base_ok, win, highb, lowb, dd, contr = base_scan(df)
            if not base_ok:
                results_near.append((t, px, dv20, bbz_last, atr_pctl, "FAIL_BASE"))
                continue
            n_base_ok += 1

            dry_ok, vratio = dryup_ok(df)
            if not dry_ok:
                results_near.append((t, px, dv20, bbz_last, atr_pctl, "FAIL_DRYUP"))
                continue
            n_dry_ok += 1

            close_today = float(df["close"].iloc[-1])
            open_today = float(df["open"].iloc[-1])
            close_prev = float(df["close"].iloc[-2])
            vol_today = float(df["volume"].iloc[-1])
            vol20 = float(df["volume"].iloc[-20:].mean())

            breakout = close_today > highb
            vol_confirm = (vol20 > 0) and (vol_today >= VOL_CONFIRM_MULT * vol20)
            no_big_gap = (close_prev > 0) and (open_today <= close_prev * MAX_GAP_UP)
            fb = failed_breakout_recent(df, level=highb, lookback=30)

            decision = "AGUARDAR"
            if breakout and vol_confirm and no_big_gap and (not fb):
                decision = "EXECUTAR"

            stp = suggested_stop(lowb)
            item = (t, decision, px, dv20, bbz_last, atr_pctl, win, highb, lowb, stp, dd, contr, vratio)
            if decision == "EXECUTAR":
                results_exec.append(item)
            else:
                results_watch.append(item)

        except Exception as e:
            s = str(e)
            if "NO_DATA" in s:
                n_no_data += 1
            elif "HITS_LIMIT" in s:
                n_hits_limit += 1
                hits_limited = True
            else:
                if len(errors) < 8:
                    errors.append(f"{t}: {s[:140]}")
            pass

        if (i + 1) % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    results_exec.sort(key=lambda x: (-x[3], x[4]))
    results_watch.sort(key=lambda x: (x[4], -x[3]))
    results_near.sort(key=lambda x: (x[3], -x[2]))

    exec_top = results_exec[:6]
    watch_top = results_watch[:12]
    near_top = results_near[:12]

    msg = [f"[{now}] Microcap scanner (Modo A) [BUILD=V3_RATE_LIMIT_SMALLCAP_PROXY]"]
    msg.append(f"Universo combinado bruto: {len(universe_all)}")
    msg.append(f"Universo avaliado (cap {MAX_TICKERS}): {len(tickers)}")
    msg.append("")
    msg.append(
        "DIAG: "
        f"hist_ok={n_hist_ok} | liq_ok={n_liq_ok} | comp_ok={n_comp_ok} | base_ok={n_base_ok} | dry_ok={n_dry_ok} | "
        f"no_data={n_no_data} | hits_limit={n_hits_limit} | exec={len(results_exec)} | watch={len(results_watch)}"
    )
    msg.append(f"CAPS: MAX_PX={MAX_PX:.0f} | MAX_DV20=${MAX_DV20/1e6:.0f}M | MIN_DV20=${MIN_DV20/1e6:.1f}M")
    msg.append("")

    if hits_limited:
        msg.append("NOTA: Stooq atingiu limite diário; run terminou cedo.")
        msg.append("")

    if errors:
        msg.append("ERROS (top):")
        for e in errors[:5]:
            msg.append(f"- {e}")
        msg.append("")

    if not exec_top:
        msg.append("EXECUTAR: (vazio)")
    else:
        msg.append("EXECUTAR (gatilho EOD confirmado):")
        for t, _, px, dv, bbz, atrp, win, highb, lowb, stp, dd, contr, vratio in exec_top:
            msg.append(f"- {t} | close={px:.2f} | dv20=${dv/1e6:.1f}M | trigger>={highb:.2f} | stop~{stp:.2f}")

    msg.append("")
    if not watch_top:
        msg.append("AGUARDAR: (vazio)")
    else:
        msg.append("AGUARDAR (pré-breakout):")
        for t, _, px, dv, bbz, atrp, win, highb, lowb, stp, dd, contr, vratio in watch_top:
            msg.append(
                f"- {t} | close={px:.2f} | dv20=${dv/1e6:.1f}M | BBz={bbz:.2f} ATRpctl={atrp:.2f} | "
                f"trigger>={highb:.2f} | stop~{stp:.2f}"
            )

    msg.append("")
    if near_top:
        msg.append("QUASE (passou compressão+liquidez, falhou base/dry-up):")
        for t, px, dv, bbz, atrp, reason in near_top:
            msg.append(f"- {t} | {reason} | close={px:.2f} | dv20=${dv/1e6:.1f}M | BBz={bbz:.2f} ATRpctl={atrp:.2f}")
    else:
        msg.append("QUASE: (vazio)")

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
