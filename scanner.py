# scanner.py  (V7_2_PREOPEN_RELAX_DAILY) — RELAXED (MORE IDEAS DAILY + ASOF + DETERMINISM)
# EOD-only, public/free, cache-first, ranked, pre-open actionable.
#
# OBJECTIVE:
# - Output proposals DAILY (even when strict filter yields few).
# - Keep determinism and ASOF schema for weekly eval.
# - Maintain EXEC logic but slightly more permissive to avoid "empty" runs.
#
# KEY CHANGES vs your V7_1:
#  1) Liquidity + compression gates are relaxed (still bounded).
#  2) WATCH rule is widened (atr_pctl, dist, overhead tolerance).
#  3) If total proposals are too low, produce WATCH_RELAX list from "near" candidates.
#  4) Keeps ASOF+schema migration + de-dup by (ASOF,ticker,signal).

import os, io, time, csv
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# =========================
# ENV / CONFIG
# =========================
CACHE_ONLY = os.environ.get("CACHE_ONLY", "0") == "1"

TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")

IWC_URL = os.environ.get("IWC_HOLDINGS_CSV_URL", "")
IWM_URL = os.environ.get("IWM_HOLDINGS_CSV_URL", "")
IJR_URL = os.environ.get("IJR_HOLDINGS_CSV_URL", "")

OHLCV_FMT = os.environ.get("OHLCV_URL_FMT", "https://stooq.com/q/d/l/?s={symbol}.us&i=d")

MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "450"))
CANDIDATE_POOL = int(os.environ.get("CANDIDATE_POOL", "2000"))

MIN_PX = float(os.environ.get("MIN_PX", "1.0"))
MAX_PX = float(os.environ.get("MAX_PX", "30"))

MIN_DV20 = float(os.environ.get("MIN_DV20", "1500000"))
MAX_DV20 = float(os.environ.get("MAX_DV20", "120000000"))
MIN_SV20 = float(os.environ.get("MIN_SV20", "250000"))

BBZ_GATE = float(os.environ.get("BBZ_GATE", "-0.60"))
ATRPCTL_GATE = float(os.environ.get("ATRPCTL_GATE", "0.50"))

BASE_DD_MAX = float(os.environ.get("BASE_DD_MAX", "0.62"))
CONTRACTION_MAX = float(os.environ.get("CONTRACTION_MAX", "0.92"))
DRYUP_MAX = float(os.environ.get("DRYUP_MAX", "1.10"))

VOL_CONFIRM_MULT_BASE = float(os.environ.get("VOL_CONFIRM_MULT", "1.12"))
MAX_GAP_UP = float(os.environ.get("MAX_GAP_UP", "1.14"))
DIST_MAX_PCT = float(os.environ.get("DIST_MAX_PCT", "18.0"))
EXEC_MAX_OVERSHOOT_PCT = float(os.environ.get("EXEC_MAX_OVERSHOOT_PCT", "8.0"))

EXEC_BBZ_MAX = float(os.environ.get("EXEC_BBZ_MAX", "-0.90"))
EXEC_ATRPCTL_MAX = float(os.environ.get("EXEC_ATRPCTL_MAX", "0.38"))

MIN_R_PCT = float(os.environ.get("MIN_R_PCT", "6.0"))

OVERHEAD_WINDOW = int(os.environ.get("OVERHEAD_WINDOW", "200"))
OVERHEAD_BAND_PCT = float(os.environ.get("OVERHEAD_BAND_PCT", "8.0"))
OVERHEAD_MAX_TOUCHES = int(os.environ.get("OVERHEAD_MAX_TOUCHES", "12"))

WATCH_BOOST_MIN_DAYS = int(os.environ.get("WATCH_BOOST_MIN_DAYS", "3"))
WATCH_STALE_DAYS = int(os.environ.get("WATCH_STALE_DAYS", "10"))

# "Guarantee ideas"
MIN_DAILY_PROPOSALS = int(os.environ.get("MIN_DAILY_PROPOSALS", "6"))  # total WATCH+EXEC desired
WATCH_RELAX_TOP = int(os.environ.get("WATCH_RELAX_TOP", "8"))

HORIZON = int(os.environ.get("HORIZON_SESS", "40"))
RET_WINDOWS = [5, 10, 20, 40]

REG_QQQ = os.environ.get("REG_QQQ", "QQQ")
REG_VIX = os.environ.get("REG_VIX", "VIX")

SLEEP_EVERY = int(os.environ.get("SLEEP_EVERY", "25"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "2.0"))

# paths
CACHE_DIR = Path("cache")
OHLCV_DIR = CACHE_DIR / "ohlcv"
SIGNALS_CSV = CACHE_DIR / "signals.csv"

# =========================
# Telegram
# =========================
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

# =========================
# Utils / Indicators
# =========================
def ensure_dirs() -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

# =========================
# Holdings parsing
# =========================
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
            raise RuntimeError("Holdings HTML (blocked/redirect).")
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
        raise RuntimeError("Holdings sem Ticker/Symbol.")
    raw = df[col].astype(str).str.strip().tolist()
    out = []
    for t in raw:
        t = t.replace(".", "-").upper()
        if is_valid_ticker(t):
            out.append(t)
    return out

# =========================
# OHLCV / cache
# =========================
def cache_path(t: str) -> Path:
    return OHLCV_DIR / f"{t}.csv"

def build_ohlcv_url_equity(ticker: str) -> str:
    sym = ticker.lower().replace("-", ".")
    if ".us" in OHLCV_FMT.lower():
        return OHLCV_FMT.format(symbol=sym)
    return OHLCV_FMT.format(symbol=f"{sym}.us")

def build_ohlcv_url_symbol(symbol: str) -> tuple[str, str]:
    sym = symbol.lower().replace("-", ".")
    url1 = OHLCV_FMT.format(symbol=sym)
    url2 = OHLCV_FMT.format(symbol=f"{sym}.us")
    return url1, url2

def load_cached_ohlcv_local(ticker: str, min_rows: int = 120) -> pd.DataFrame | None:
    p = cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        need = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            return None
        df = df[need].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)
        if len(df) < min_rows:
            return None
        return df
    except Exception:
        return None

def read_cached_dv20(ticker: str) -> float | None:
    df = load_cached_ohlcv_local(ticker, min_rows=30)
    if df is None:
        return None
    try:
        dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
        if not np.isfinite(dv20) or dv20 <= 0:
            return None
        return dv20
    except Exception:
        return None

def fetch_ohlcv_equity(ticker: str) -> pd.DataFrame:
    """
    Fetch EOD OHLCV for ticker, with disk cache.
    - Tenta Stooq e atualiza CSV local.
    - Se falhar / rate-limit, usa cache existente (se houver).
    """
    ensure_dirs()
    path = cache_path(ticker)

    cached = None
    if path.exists():
        try:
            cached = pd.read_csv(path)
            cached.columns = [c.strip().lower() for c in cached.columns]
            if "date" in cached.columns:
                cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                cached = cached.dropna(subset=["date"]).sort_values("date")
        except Exception:
            cached = None

    try:
        url = build_ohlcv_url_equity(ticker)
        text = fetch_text(url).strip()
        low = text.lower()

        if low.startswith("no data") or low == "no data":
            raise RuntimeError("NO_DATA")
        if "exceeded the daily hits limit" in low:
            raise RuntimeError("HITS_LIMIT")

        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
        need = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            raise RuntimeError("BAD_FORMAT")

        df = df[need].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

        if cached is not None and len(cached) > 0 and all(c in cached.columns for c in need):
            merged = pd.concat([cached[need], df[need]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            df = merged

        df.to_csv(path, index=False)
        return df.reset_index(drop=True)

    except Exception:
        if cached is not None and len(cached) >= 120:
            need = ["date", "open", "high", "low", "close", "volume"]
            if all(c in cached.columns for c in need):
                cached = cached[need].copy()
                cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                cached = cached.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
                for c in ["open","high","low","close","volume"]:
                    cached[c] = pd.to_numeric(cached[c], errors="coerce")
                cached = cached.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)
                if len(cached) >= 120:
                    return cached
        raise

def fetch_ohlcv_symbol_best_effort(symbol: str) -> pd.DataFrame | None:
    url1, url2 = build_ohlcv_url_symbol(symbol)
    for url in [url1, url2]:
        try:
            text = fetch_text(url).strip()
            low = text.lower()
            if low.startswith("no data") or low == "no data":
                continue
            if "exceeded the daily hits limit" in low:
                continue
            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip().lower() for c in df.columns]
            need = ["date", "open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in need):
                continue
            df = df[need].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)
            if len(df) >= 120:
                return df
        except Exception:
            continue
    return None

# =========================
# Model components
# =========================
def base_scan(df: pd.DataFrame) -> tuple[bool, float, float, float, float, int]:
    if len(df) < 260:
        return (False, np.nan, np.nan, np.nan, np.nan, 0)

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
            cand = (score, highb, lowb, dd, contr, win)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return (False, np.nan, np.nan, np.nan, np.nan, 0)

    _, highb, lowb, dd, contr, win = best
    return (True, float(highb), float(lowb), float(dd), float(contr), int(win))

def dryup_ratio(df: pd.DataFrame) -> float:
    v10 = float(df["volume"].iloc[-10:].mean())
    v60 = float(df["volume"].iloc[-60:].mean())
    if v60 <= 0:
        return np.nan
    return v10 / v60

def overhead_supply_touches(df: pd.DataFrame, trig: float) -> int:
    if trig <= 0:
        return 0
    w = df["close"].iloc[-OVERHEAD_WINDOW:]
    upper = trig * (1.0 + OVERHEAD_BAND_PCT / 100.0)
    return int(((w >= trig) & (w <= upper)).sum())

def liquidity_sweet_spot_bonus(dv20: float) -> float:
    if dv20 <= 0:
        return -0.5
    if 4_000_000 <= dv20 <= 45_000_000:
        return 0.6
    if dv20 < 4_000_000:
        return -0.15
    return -0.25

def score_candidate(
    bbz: float, atrpctl: float, dd: float, contr: float, dry: float, dv20: float,
    dist_to_trig_pct: float, overhead_touches: int,
    watch_boost: float, stale_penalty: float
) -> float:
    s = 0.0
    s += (-bbz) * 2.1
    s += (0.60 - atrpctl) * 1.4
    s += (BASE_DD_MAX - dd) * 1.0
    s += (CONTRACTION_MAX - contr) * 0.9
    s += (DRYUP_MAX - dry) * 0.8
    s += liquidity_sweet_spot_bonus(dv20)

    if np.isfinite(dist_to_trig_pct):
        s += max(0.0, (DIST_MAX_PCT - dist_to_trig_pct)) * 0.08

    if overhead_touches >= 60:
        s -= 1.3
    elif overhead_touches >= 30:
        s -= 0.7
    elif overhead_touches > OVERHEAD_MAX_TOUCHES:
        s -= 0.35

    s += watch_boost
    s -= stale_penalty
    return float(s)

# =========================
# Regime logic (EOD)
# =========================
def regime_snapshot() -> dict:
    out = {"mode": "TRANSITION", "vol_mult_adj": 0.0, "dist_adj": 0.0, "qqq_trend": None, "vix_trend": None}
    qqq = fetch_ohlcv_symbol_best_effort(REG_QQQ)
    if qqq is None or len(qqq) < 200:
        return out

    qqq["ma50"] = qqq["close"].rolling(50).mean()
    qqq["ma200"] = qqq["close"].rolling(200).mean()
    if not (np.isfinite(qqq["ma50"].iloc[-1]) and np.isfinite(qqq["ma200"].iloc[-1])):
        return out

    q_close = float(qqq["close"].iloc[-1])
    q_ma50 = float(qqq["ma50"].iloc[-1])
    q_ma50_prev = float(qqq["ma50"].iloc[-6]) if len(qqq) >= 60 else float(qqq["ma50"].iloc[-2])
    ma50_slope = q_ma50 - q_ma50_prev

    qqq_trend = ("UP" if (q_close > q_ma50 and ma50_slope > 0) else
                 "DOWN" if (q_close < q_ma50 and ma50_slope < 0) else "MIXED")
    out["qqq_trend"] = qqq_trend

    vix = fetch_ohlcv_symbol_best_effort(REG_VIX)
    if vix is not None and len(vix) >= 30:
        vix["ma20"] = vix["close"].rolling(20).mean()
        if np.isfinite(vix["ma20"].iloc[-1]):
            v_close = float(vix["close"].iloc[-1])
            v_ma20 = float(vix["ma20"].iloc[-1])
            v_ma20_prev = float(vix["ma20"].iloc[-6]) if len(vix) >= 30 else float(vix["ma20"].iloc[-2])
            vix_trend = ("UP" if (v_close > v_ma20 and v_ma20 > v_ma20_prev) else
                         "DOWN" if (v_close < v_ma20 and v_ma20 < v_ma20_prev) else "MIXED")
            out["vix_trend"] = vix_trend

    if qqq_trend == "UP" and (out["vix_trend"] in [None, "DOWN", "MIXED"]):
        out["mode"] = "RISK_ON"
        out["vol_mult_adj"] = -0.02
        out["dist_adj"] = +2.0
    elif qqq_trend == "DOWN" and (out["vix_trend"] == "UP"):
        out["mode"] = "RISK_OFF"
        out["vol_mult_adj"] = +0.08
        out["dist_adj"] = -3.0
    else:
        out["mode"] = "TRANSITION"
        out["vol_mult_adj"] = +0.02
        out["dist_adj"] = -1.0

    return out

# =========================
# signals.csv helpers (ASOF + NO DUPES + SCHEMA MIGRATION)
# =========================
SIGNALS_COLS = [
    "date",
    "asof",
    "ticker","signal","score","close","trig","stop","dist_pct","dv20",
    "bbz","atrpctl","dd","contr","dry","overhead_touches","vol_mult","overshoot_pct",
    "R_pct","regime",
    "ret_5","ret_10","ret_20","ret_40","hit_30","hit_50","resolved"
]

def _read_signals_header() -> list[str] | None:
    if not SIGNALS_CSV.exists():
        return None
    try:
        with SIGNALALS_CSV.open("r", encoding="utf-8") as f:  # noqa: F821
            first = f.readline().strip()
        if not first:
            return None
        return [c.strip() for c in first.split(",")]
    except Exception:
        # fallback safe
        try:
            with SIGNALS_CSV.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            if not first:
                return None
            return [c.strip() for c in first.split(",")]
        except Exception:
            return None

def signals_csv_ensure_schema() -> None:
    ensure_dirs()

    if not SIGNALS_CSV.exists():
        with SIGNALS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SIGNALS_COLS)
        return

    hdr = _read_signals_header()
    if not hdr:
        with SIGNALS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SIGNALS_COLS)
        return

    if "asof" in [h.lower() for h in hdr]:
        return

    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        with SIGNALS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SIGNALS_COLS)
        return

    if "asof" not in df.columns:
        df.insert(1, "asof", "")

    for c in SIGNALS_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[SIGNALS_COLS].copy()
    df.to_csv(SIGNALS_CSV, index=False)

_KEYS_BY_ASOF: dict[str, set[tuple[str,str,str]]] = {}

def load_existing_keys_for_asof(asof_str: str) -> set[tuple[str,str,str]]:
    if asof_str in _KEYS_BY_ASOF:
        return _KEYS_BY_ASOF[asof_str]

    keys: set[tuple[str,str,str]] = set()
    if not SIGNALS_CSV.exists():
        _KEYS_BY_ASOF[asof_str] = keys
        return keys

    try:
        cols = pd.read_csv(SIGNALS_CSV, nrows=1).columns
        usecols = ["asof","ticker","signal"] if "asof" in cols else ["date","ticker","signal"]
        df = pd.read_csv(SIGNALS_CSV, usecols=usecols).dropna()
        if "asof" in df.columns:
            df["asof"] = df["asof"].astype(str)
            df = df[df["asof"] == asof_str]
            dcol = "asof"
        else:
            df["date"] = df["date"].astype(str)
            df = df[df["date"] == asof_str]
            dcol = "date"

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["signal"] = df["signal"].astype(str).str.upper().str.strip()

        for _, r in df.iterrows():
            keys.add((str(r[dcol]), r["ticker"], r["signal"]))
    except Exception:
        pass

    _KEYS_BY_ASOF[asof_str] = keys
    return keys

def append_signal_row(row: dict) -> None:
    signals_csv_ensure_schema()

    asof = str(row.get("asof", "")).strip()
    t = str(row.get("ticker", "")).strip().upper()
    s = str(row.get("signal", "")).strip().upper()
    if not asof or not t or not s:
        return

    keyset = load_existing_keys_for_asof(asof)
    k = (asof, t, s)
    if k in keyset:
        return

    with SIGNALS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIGNALS_COLS)
        w.writerow({c: row.get(c, "") for c in SIGNALS_COLS})

    keyset.add(k)

def recent_watch_stats(ticker: str, lookback_days: int = 15) -> dict:
    if not SIGNALS_CSV.exists():
        return {"days": 0, "dists": []}
    try:
        df = pd.read_csv(SIGNALS_CSV, usecols=["ticker","dist_pct","asof","date"])
    except Exception:
        return {"days": 0, "dists": []}
    if df.empty:
        return {"days": 0, "dists": []}

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        return {"days": 0, "dists": []}

    if "asof" in df.columns:
        df["asof"] = pd.to_datetime(df["asof"], errors="coerce")
        daycol = "asof"
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        daycol = "date"

    df = df.dropna(subset=[daycol]).sort_values(daycol)
    cutoff = df[daycol].max() - pd.Timedelta(days=lookback_days)
    df = df[df[daycol] >= cutoff]

    dd = pd.to_numeric(df.get("dist_pct", np.nan), errors="coerce").dropna()
    dists = dd.tolist()[-WATCH_STALE_DAYS:]
    return {"days": int(len(df)), "dists": dists}

def watch_boost_and_stale_penalty(dist_series: list[float]) -> tuple[float, float]:
    if not dist_series:
        return (0.0, 0.0)

    days = len(dist_series)
    boost = 0.0
    penalty = 0.0

    if days >= WATCH_BOOST_MIN_DAYS:
        tail = dist_series[-WATCH_BOOST_MIN_DAYS:]
        if all(np.isfinite(x) for x in tail) and tail[-1] < tail[0]:
            boost = 0.30

    if days >= WATCH_STALE_DAYS:
        tail = dist_series[-WATCH_STALE_DAYS:]
        if all(np.isfinite(x) for x in tail):
            if (tail[0] - tail[-1]) < 1.0:
                penalty = 0.35

    return (boost, penalty)

# =========================
# outcomes update (EOD-only, from cache) — USE ASOF
# =========================
def update_outcomes_using_cache() -> dict:
    signals_csv_ensure_schema()
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return {"updated": 0, "resolved_total": 0}
    if df.empty or "resolved" not in df.columns:
        return {"updated": 0, "resolved_total": 0}

    df["resolved"] = df["resolved"].astype(str)
    unresolved_idx = df.index[df["resolved"] != "1"].tolist()
    if not unresolved_idx:
        return {"updated": 0, "resolved_total": int((df["resolved"] == "1").sum())}

    updated = 0
    for idx in unresolved_idx[:800]:
        try:
            ticker = str(df.at[idx, "ticker"]).strip().upper()

            sig_day = None
            if "asof" in df.columns:
                sig_day = pd.to_datetime(df.at[idx, "asof"], errors="coerce")
            if sig_day is None or pd.isna(sig_day):
                sig_day = pd.to_datetime(df.at[idx, "date"], errors="coerce")
            if pd.isna(sig_day):
                continue

            o = load_cached_ohlcv_local(ticker, min_rows=260)
            if o is None:
                continue

            o_dates = o["date"].dt.date.values
            target = sig_day.date()

            pos_candidates = np.where(o_dates == target)[0]
            if len(pos_candidates) == 0:
                pos_candidates = np.where(o_dates <= target)[0]
                if len(pos_candidates) == 0:
                    continue
                pos = int(pos_candidates[-1])
            else:
                pos = int(pos_candidates[0])

            if pos + HORIZON >= len(o):
                continue

            entry_close = float(pd.to_numeric(df.at[idx, "close"], errors="coerce"))
            if not np.isfinite(entry_close) or entry_close <= 0:
                entry_close = float(o["close"].iloc[pos])

            rets = {}
            for w in RET_WINDOWS:
                c_fwd = float(o["close"].iloc[pos + w])
                rets[w] = (c_fwd / entry_close) - 1.0

            horizon_slice = pd.to_numeric(o["close"].iloc[pos:pos + HORIZON + 1], errors="coerce").dropna()
            max_ret = (float(horizon_slice.max()) / entry_close) - 1.0 if len(horizon_slice) else np.nan

            hit30 = 1 if (np.isfinite(max_ret) and max_ret >= 0.30) else 0
            hit50 = 1 if (np.isfinite(max_ret) and max_ret >= 0.50) else 0

            df.at[idx, "ret_5"] = round(rets[5], 6)
            df.at[idx, "ret_10"] = round(rets[10], 6)
            df.at[idx, "ret_20"] = round(rets[20], 6)
            df.at[idx, "ret_40"] = round(rets[40], 6)
            df.at[idx, "hit_30"] = hit30
            df.at[idx, "hit_50"] = hit50
            df.at[idx, "resolved"] = "1"
            updated += 1

        except Exception:
            continue

    if updated > 0:
        df.to_csv(SIGNALS_CSV, index=False)

    return {"updated": updated, "resolved_total": int((df["resolved"] == "1").sum())}

def empirical_prob_by_score_bin(regime: str) -> str:
    if not SIGNALS_CSV.exists():
        return "EMP: sem histórico"
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return "EMP: sem histórico"
    if df.empty or "resolved" not in df.columns:
        return "EMP: sem histórico"

    df = df[df["resolved"].astype(str) == "1"].copy()
    if df.empty or "score" not in df.columns:
        return "EMP: sem histórico"

    df = df[df["signal"].astype(str).str.contains("EXEC", na=False)]
    if df.empty:
        return "EMP: sem EXEC resolvido"

    if "regime" in df.columns:
        df2 = df[df["regime"].astype(str) == regime]
        if len(df2) >= 20:
            df = df2

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    if len(df) < 30:
        return f"EMP: n={len(df)} (insuficiente)"

    df["bin"] = pd.qcut(df["score"], 5, duplicates="drop")
    out = []
    for b, g in df.groupby("bin"):
        n = len(g)
        p30 = float(pd.to_numeric(g.get("hit_30", 0), errors="coerce").fillna(0).mean())
        p50 = float(pd.to_numeric(g.get("hit_50", 0), errors="coerce").fillna(0).mean())
        out.append((str(b), n, p30, p50))

    parts = [f"{i+1}:{n}|{p30*100:.0f}/{p50*100:.0f}" for i, (_, n, p30, p50) in enumerate(out)]
    return "EMP bins(n|P30/P50%): " + " ".join(parts)

# =========================
# MAIN
# =========================
def main() -> None:
    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    run_day = now_dt.strftime("%Y-%m-%d")

    ensure_dirs()
    signals_csv_ensure_schema()

    upd = update_outcomes_using_cache()

    reg = regime_snapshot()
    mode = reg["mode"]
    VOL_CONFIRM_MULT = max(1.05, VOL_CONFIRM_MULT_BASE + reg["vol_mult_adj"])
    DIST_LIMIT = max(8.0, DIST_MAX_PCT + reg["dist_adj"])

    # universe (DETERMINISTIC)
    all_tickers = (
        get_universe_from_holdings(IWC_URL)
        + get_universe_from_holdings(IWM_URL)
        + get_universe_from_holdings(IJR_URL)
    )
    universe = sorted(set([t.strip().upper() for t in all_tickers if isinstance(t, str) and t.strip()]))

    # deterministic scan order: dv20 desc then ticker asc
    scored: list[tuple[str, float]] = []
    unscored: list[str] = []
    pool = universe[: min(CANDIDATE_POOL, len(universe))]
    for t in pool:
        dv = read_cached_dv20(t)
        if dv is None:
            unscored.append(t)
        else:
            scored.append((t, float(dv)))

    scored.sort(key=lambda x: (-x[1], x[0]))
    unscored.sort()
    ordered = [t for t, _ in scored] + unscored
    tickers = ordered[:MAX_TICKERS]

    execA: list[tuple] = []
    execB: list[tuple] = []
    watch_clean: list[tuple] = []
    watch_over: list[tuple] = []
    near: list[tuple] = []        # strict near (debug)
    near_relax: list[tuple] = []  # candidates for WATCH_RELAX

    hits_limited = False
    no_data = 0
    hist_ok = liq_ok = comp_ok = base_ok = dry_ok = 0
    cache_used = 0
    cache_miss = 0

    asof_date_global = None

    for i, t in enumerate(tickers):
        try:
            if CACHE_ONLY or hits_limited:
                df = load_cached_ohlcv_local(t, min_rows=260)
                if df is None:
                    cache_miss += 1
                    continue
                cache_used += 1
            else:
                df = fetch_ohlcv_equity(t)

            if df is None or df.empty:
                continue

            asof_row = pd.to_datetime(df["date"], errors="coerce").dropna().max()
            if pd.isna(asof_row):
                continue
            asof_row = asof_row.normalize()
            asof_str = asof_row.strftime("%Y-%m-%d")

            if asof_date_global is None or asof_row > asof_date_global:
                asof_date_global = asof_row

            if len(df) < 260:
                continue
            hist_ok += 1

            close_now = float(df["close"].iloc[-1])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
            sv20 = float(df["volume"].iloc[-20:].mean())

            if close_now < MIN_PX or close_now > MAX_PX:
                continue
            if dv20 < MIN_DV20 or dv20 > MAX_DV20:
                continue
            if sv20 < MIN_SV20:
                continue
            liq_ok += 1

            # compression
            bbw = compute_bb_width(df, 20)
            bbz = zscore(bbw, 120)
            atrp = compute_atr(df, 14) / df["close"]
            atr_win = atrp.iloc[-252:].dropna()
            if len(atr_win) < 80:
                continue

            bbz_last = float(bbz.iloc[-1])
            atr_last = float(atrp.iloc[-1])
            atr_pctl = float((atr_win <= atr_last).mean())

            if not (bbz_last < BBZ_GATE and atr_pctl < ATRPCTL_GATE):
                # store relax-candidate if close to passing compression
                if (bbz_last < (BBZ_GATE + 0.10)) or (atr_pctl < (ATRPCTL_GATE + 0.08)):
                    near_relax.append(("COMP", t, close_now, dv20, bbz_last, atr_pctl))
                continue
            comp_ok += 1

            # base
            ok, trig, lowb, dd, contr, win = base_scan(df)
            if not ok:
                near.append((t, close_now, dv20, bbz_last, "BASE"))
                continue
            base_ok += 1

            # dry-up
            dry = dryup_ratio(df)
            if (not np.isfinite(dry)) or (dry >= DRYUP_MAX):
                near.append((t, close_now, dv20, bbz_last, "DRY"))
                # still consider as relax candidate if only slightly above
                if np.isfinite(dry) and dry < (DRYUP_MAX + 0.12):
                    near_relax.append(("DRY", t, close_now, dv20, bbz_last, atr_pctl))
                continue
            dry_ok += 1

            # stop / R
            stop = lowb * 0.99
            R_pct = ((trig - stop) / trig * 100.0) if trig > 0 else np.nan
            if (not np.isfinite(R_pct)) or (R_pct < MIN_R_PCT):
                near.append((t, close_now, dv20, bbz_last, f"LOW_R({R_pct:.1f}%)"))
                if np.isfinite(R_pct) and R_pct >= (MIN_R_PCT - 1.5):
                    near_relax.append(("R", t, close_now, dv20, bbz_last, atr_pctl))
                continue

            dist_pct = ((trig - close_now) / close_now * 100.0) if close_now > 0 else np.nan
            overshoot_pct = ((close_now - trig) / trig * 100.0) if trig > 0 else 0.0
            overhead = overhead_supply_touches(df, trig)

            stats = recent_watch_stats(t)
            boost, stale_pen = watch_boost_and_stale_penalty(stats["dists"])

            sc = score_candidate(
                bbz_last, atr_pctl, dd, contr, dry, dv20,
                dist_pct if np.isfinite(dist_pct) else (DIST_LIMIT + 999.0),
                overhead, boost, stale_pen
            )

            # EXEC logic (EOD confirmed)
            open_now = float(df["open"].iloc[-1])
            close_prev = float(df["close"].iloc[-2])
            vol_now = float(df["volume"].iloc[-1])
            vol20 = float(df["volume"].iloc[-20:].mean())
            vol_mult = (vol_now / vol20) if vol20 > 0 else np.nan

            breakout_now = close_now > trig
            vol_confirm = (vol20 > 0) and (vol_now >= VOL_CONFIRM_MULT * vol20)
            no_big_gap = (close_prev > 0) and (open_now <= close_prev * MAX_GAP_UP)

            quality_ok = (bbz_last <= EXEC_BBZ_MAX) or (atr_pctl <= EXEC_ATRPCTL_MAX)
            not_too_extended = overshoot_pct <= EXEC_MAX_OVERSHOOT_PCT
            prev_above = float(df["close"].iloc[-2]) > trig

            if breakout_now and vol_confirm and no_big_gap and quality_ok and not_too_extended:
                sig = "EXEC_B" if prev_above else "EXEC_A"
                item = (
                    sc, t, close_now, dv20, bbz_last, atr_pctl,
                    trig, stop, overshoot_pct, R_pct, vol_mult, overhead, win, dist_pct
                )
                (execB if sig == "EXEC_B" else execA).append(item)

                append_signal_row({
                    "date": run_day,
                    "asof": asof_str,
                    "ticker": t,
                    "signal": sig,
                    "score": round(sc, 6),
                    "close": round(close_now, 6),
                    "trig": round(trig, 6),
                    "stop": round(stop, 6),
                    "dist_pct": round(float(dist_pct), 6) if np.isfinite(dist_pct) else "",
                    "dv20": round(dv20, 2),
                    "bbz": round(bbz_last, 6),
                    "atrpctl": round(atr_pctl, 6),
                    "dd": round(dd, 6),
                    "contr": round(contr, 6),
                    "dry": round(dry, 6),
                    "overhead_touches": overhead,
                    "vol_mult": round(float(vol_mult), 6) if np.isfinite(vol_mult) else "",
                    "overshoot_pct": round(float(overshoot_pct), 6) if np.isfinite(overshoot_pct) else "",
                    "R_pct": round(float(R_pct), 6) if np.isfinite(R_pct) else "",
                    "regime": mode,
                    "resolved": "0"
                })
            else:
                # WATCH (wider net)
                # - allow atr_pctl a bit higher than before
                # - allow some overhead but separate CLEAN vs OVER
                # - allow slight overshoot within limit
                if (
                    np.isfinite(dist_pct)
                    and dist_pct <= DIST_LIMIT
                    and dist_pct >= -EXEC_MAX_OVERSHOOT_PCT
                    and atr_pctl <= 0.42
                ):
                    item = (
                        sc, t, close_now, dv20, bbz_last, atr_pctl,
                        trig, stop, dist_pct, R_pct, overhead, win, boost, stale_pen
                    )
                    if overhead <= 5:
                        watch_clean.append(item)
                    else:
                        watch_over.append(item)

                    append_signal_row({
                        "date": run_day,
                        "asof": asof_str,
                        "ticker": t,
                        "signal": "WATCH",
                        "score": round(sc, 6),
                        "close": round(close_now, 6),
                        "trig": round(trig, 6),
                        "stop": round(stop, 6),
                        "dist_pct": round(float(dist_pct), 6) if np.isfinite(dist_pct) else "",
                        "dv20": round(dv20, 2),
                        "bbz": round(bbz_last, 6),
                        "atrpctl": round(atr_pctl, 6),
                        "dd": round(dd, 6),
                        "contr": round(contr, 6),
                        "dry": round(dry, 6),
                        "overhead_touches": overhead,
                        "vol_mult": round(float(vol_mult), 6) if np.isfinite(vol_mult) else "",
                        "overshoot_pct": round(float(overshoot_pct), 6) if np.isfinite(overshoot_pct) else "",
                        "R_pct": round(float(R_pct), 6) if np.isfinite(R_pct) else "",
                        "regime": mode,
                        "resolved": "0"
                    })
                else:
                    # store as relax candidate if it's close on dist/atr
                    if np.isfinite(dist_pct) and dist_pct <= (DIST_LIMIT + 6.0) and atr_pctl <= 0.50:
                        near_relax.append(("WATCH", t, close_now, dv20, bbz_last, atr_pctl))

        except Exception as e:
            s = str(e).upper()
            if "NO_DATA" in s:
                no_data += 1
            elif "HITS_LIMIT" in s:
                hits_limited = True
            pass

        if (i + 1) % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    # sort + cut
    execA.sort(key=lambda x: x[0], reverse=True)
    execB.sort(key=lambda x: x[0], reverse=True)
    watch_clean.sort(key=lambda x: x[0], reverse=True)
    watch_over.sort(key=lambda x: x[0], reverse=True)

    execA = execA[:12]
    execB = execB[:12]
    watch_clean = watch_clean[:7]
    watch_over = watch_over[:7]
    near = near[:10]

    emp = empirical_prob_by_score_bin(mode)
    asof_str_global = asof_date_global.strftime("%Y-%m-%d") if asof_date_global is not None else "—"

    # If too few proposals, build WATCH_RELAX from near_relax
    total_props = len(execA) + len(execB) + len(watch_clean) + len(watch_over)
    watch_relax_lines = []
    if total_props < MIN_DAILY_PROPOSALS and near_relax:
        # deterministic: sort by (reason, dv20 desc, ticker asc)
        near_relax_sorted = sorted(
            near_relax,
            key=lambda x: (str(x[0]), -float(x[3]) if np.isfinite(x[3]) else 0.0, str(x[1]))
        )[:WATCH_RELAX_TOP]

        watch_relax_lines.append(f"WATCH_RELAX (porque propostas<{MIN_DAILY_PROPOSALS}):")
        for reason, t, c, dv, bbz, atrp in near_relax_sorted:
            watch_relax_lines.append(
                f"- {t} | reason={reason} | close={c:.2f} | dv20={dv/1e6:.1f}M | BBz={bbz:.2f} | ATRpctl={atrp:.2f}"
            )

    msg = [f"[{now}] ### V7_2_PREOPEN_RELAX_DAILY ### Microcap scanner (RANKED)"]
    msg.append(f"ASOF={asof_str_global} (data de mercado)")
    msg.append(
        f"MODE={mode} | Eval={len(tickers)} | hist={hist_ok} liq={liq_ok} comp={comp_ok} base={base_ok} dry={dry_ok} | "
        f"EXEC_B={len(execB)} EXEC_A={len(execA)} WATCH_CLEAN={len(watch_clean)} WATCH_OVER={len(watch_over)} | nodata={no_data} | "
        f"PX<= {MAX_PX:.0f} DV20[{MIN_DV20/1e6:.1f}M..{MAX_DV20/1e6:.0f}M] | VOLx>={VOL_CONFIRM_MULT:.2f} | dist<={DIST_LIMIT:.0f}% | MIN_R%={MIN_R_PCT:.0f}"
    )
    msg.append(emp)
    msg.append(f"LEARNING: outcomes_updated={upd['updated']} | resolved_total={upd['resolved_total']}")
    if hits_limited or CACHE_ONLY:
        msg.append(f"NOTA: Stooq rate-limit ou CACHE_ONLY; cache_used={cache_used} cache_miss={cache_miss}.")
    msg.append("")

    if not execA and not execB:
        msg.append("EXECUTAR: (vazio)")
        msg.append("")
    else:
        if execB:
            msg.append("EXECUTAR_B (2-day confirmação; maior probabilidade):")
            for sc, t, c, dv, bbz, atrp, trig, stop, over, Rp, vm, oh, win, dist in execB:
                msg.append(
                    f"- {t} | score={sc:.2f} | close={c:.2f} | dist={dist:.1f}% | dv20={dv/1e6:.1f}M | "
                    f"BBz={bbz:.2f} ATRp={atrp:.2f} | trig={trig:.2f} | stop~{stop:.2f} | "
                    f"R%={Rp:.1f} | over={over:.1f}% | volx={vm:.2f} | overhead={oh} | base={win}d"
                )
            msg.append("")
        if execA:
            msg.append("EXECUTAR_A (1-day; mais sinais, menos robusto):")
            for sc, t, c, dv, bbz, atrp, trig, stop, over, Rp, vm, oh, win, dist in execA:
                msg.append(
                    f"- {t} | score={sc:.2f} | close={c:.2f} | dist={dist:.1f}% | dv20={dv/1e6:.1f}M | "
                    f"BBz={bbz:.2f} ATRp={atrp:.2f} | trig={trig:.2f} | stop~{stop:.2f} | "
                    f"R%={Rp:.1f} | over={over:.1f}% | volx={vm:.2f} | overhead={oh} | base={win}d"
                )
            msg.append("")

    if watch_clean:
        def _prio_key(x):
            sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale = x
            prioridade = 0
            if dist <= 4: prioridade += 1
            if oh <= 3: prioridade += 1
            if atrp <= 0.22: prioridade += 1
            if Rp >= 12: prioridade += 1
            return (prioridade, sc)

        watch_clean = sorted(watch_clean, key=_prio_key, reverse=True)

        msg.append("AGUARDAR_LIMPO (TOP 7; maior probabilidade):")
        for sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale in watch_clean:
            prioridade = 0
            if dist <= 4: prioridade += 1
            if oh <= 3: prioridade += 1
            if atrp <= 0.22: prioridade += 1
            if Rp >= 12: prioridade += 1
            emoji = "🔥" if prioridade == 4 else "✅" if prioridade == 3 else "⚠️" if prioridade == 2 else "❌"
            msg.append(
                f"- {emoji} {t} | PRIORIDADE={prioridade}/4 | score={sc:.2f} | close={c:.2f} | "
                f"dist={dist:.1f}% | dv20={dv/1e6:.1f}M | BBz={bbz:.2f} ATRp={atrp:.2f} | "
                f"trig={trig:.2f} | stop~{stop:.2f} | R%={Rp:.1f} | overhead={oh}"
            )
        msg.append("")
    else:
        msg.append("AGUARDAR_LIMPO: (vazio)")
        msg.append("")

    if watch_over:
        msg.append("AGUARDAR_COM_TETO (overhead>5; probabilidade menor):")
        for sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale in watch_over:
            msg.append(
                f"- {t} | score={sc:.2f} | close={c:.2f} | dist={dist:.1f}% | dv20={dv/1e6:.1f}M | "
                f"BBz={bbz:.2f} | trig={trig:.2f} | R%={Rp:.1f} | overhead={oh} [OVERHEAD]"
            )
        msg.append("")
    else:
        msg.append("AGUARDAR_COM_TETO: (vazio)")
        msg.append("")

    if watch_relax_lines:
        msg.extend(watch_relax_lines)
        msg.append("")

    if near:
        msg.append("QUASE (debug):")
        for t, c, dv, bbz, reason in near:
            msg.append(f"- {t} | {reason} | close={c:.2f} | dv20={dv/1e6:.1f}M | BBz={bbz:.2f}")
    else:
        msg.append("QUASE: (vazio)")

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()
