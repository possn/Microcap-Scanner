# scanner.py  (V7_1_PREOPEN_FULL_TOP7_IDEAL)
# EOD-only, public/free, cache-first, ranked, pre-open actionable.
# Adds: regime filter, overhead supply proxy w/ strong penalty tiers, MIN_R%,
# EXECUTAR_A/B, liquidity sweet spot, watch maturation/stale logic,
# empirical logging (P30/P50) using realized outcomes, and TOP7 watchlist with IDEAL flag.

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

MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "600"))
CANDIDATE_POOL = int(os.environ.get("CANDIDATE_POOL", "2600"))

# micro/small proxy (verifiable)
MIN_PX = float(os.environ.get("MIN_PX", "1.0"))
MAX_PX = float(os.environ.get("MAX_PX", "25"))
MIN_DV20 = float(os.environ.get("MIN_DV20", "3000000"))
MAX_DV20 = float(os.environ.get("MAX_DV20", "80000000"))
MIN_SV20 = float(os.environ.get("MIN_SV20", "600000"))

# compression gates
BBZ_GATE = float(os.environ.get("BBZ_GATE", "-0.7"))
ATRPCTL_GATE = float(os.environ.get("ATRPCTL_GATE", "0.45"))

# base/dry
BASE_DD_MAX = float(os.environ.get("BASE_DD_MAX", "0.55"))
CONTRACTION_MAX = float(os.environ.get("CONTRACTION_MAX", "0.85"))
DRYUP_MAX = float(os.environ.get("DRYUP_MAX", "0.95"))

# execution rules
VOL_CONFIRM_MULT_BASE = float(os.environ.get("VOL_CONFIRM_MULT", "1.15"))
MAX_GAP_UP = float(os.environ.get("MAX_GAP_UP", "1.12"))
DIST_MAX_PCT = float(os.environ.get("DIST_MAX_PCT", "12.0"))
EXEC_MAX_OVERSHOOT_PCT = float(os.environ.get("EXEC_MAX_OVERSHOOT_PCT", "6.0"))

# quality gates for EXECUTAR
EXEC_BBZ_MAX = float(os.environ.get("EXEC_BBZ_MAX", "-1.1"))
EXEC_ATRPCTL_MAX = float(os.environ.get("EXEC_ATRPCTL_MAX", "0.30"))

# payoff filter
MIN_R_PCT = float(os.environ.get("MIN_R_PCT", "8.0"))

# overhead supply proxy
OVERHEAD_WINDOW = int(os.environ.get("OVERHEAD_WINDOW", "200"))
OVERHEAD_BAND_PCT = float(os.environ.get("OVERHEAD_BAND_PCT", "8.0"))
OVERHEAD_MAX_TOUCHES = int(os.environ.get("OVERHEAD_MAX_TOUCHES", "10"))

# watch maturation/stale
WATCH_BOOST_MIN_DAYS = int(os.environ.get("WATCH_BOOST_MIN_DAYS", "3"))
WATCH_STALE_DAYS = int(os.environ.get("WATCH_STALE_DAYS", "10"))

# empirical learning
HORIZON = int(os.environ.get("HORIZON_SESS", "40"))
RET_WINDOWS = [5, 10, 20, 40]

# regime symbols (EOD)
REG_QQQ = os.environ.get("REG_QQQ", "QQQ")
REG_VIX = os.environ.get("REG_VIX", "VIX")  # may fail; fallback used

# sleep
SLEEP_EVERY = int(os.environ.get("SLEEP_EVERY", "60"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "1.0"))

# paths
CACHE_DIR = Path("cache")
OHLCV_DIR = CACHE_DIR / "ohlcv"
SIGNALS_CSV = CACHE_DIR / "signals.csv"
STOOQ_HITSLIMIT = False

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
def load_cached_ohlcv_local(ticker: str) -> pd.DataFrame | None:
    path = cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        need = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close","volume"])

        if len(df) < 260:
            return None

        return df

    except Exception:
        return None

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

def read_cached_dv20(ticker: str) -> float | None:
    path = cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not set(["close", "volume"]).issubset(df.columns):
            return None
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if len(df) < 30:
            return None
        dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
        if not np.isfinite(dv20):
            return None
        return dv20
    except Exception:
        return None

def fetch_ohlcv_equity(ticker: str) -> pd.DataFrame:
    """
    Fetch EOD OHLCV for ticker, with disk cache.
    - Tenta ir ao Stooq e atualizar o CSV local.
    - Se Stooq der HITS_LIMIT / falhar / BAD_FORMAT, usa o cache existente (se houver).
    - Se não houver cache e falhar, levanta erro.
    """
    ensure_dirs()
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(ticker)

    # 1) Lê cache (se existir)
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

    # 2) Tenta buscar dados frescos ao Stooq
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

        # numéricos
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

        # 3) Faz merge com cache (se existir)
        if cached is not None and len(cached) > 0:
            if all(c in cached.columns for c in need):
                merged = pd.concat([cached[need], df[need]], ignore_index=True)
                merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                df = merged

        # 4) Guarda sempre se o fetch correu bem
        df.to_csv(path, index=False)
        return df.reset_index(drop=True)

    except Exception:
        # 5) Fallback: se já houver cache, usa-o
        if cached is not None and len(cached) >= 120:
            for c in ["open", "high", "low", "close", "volume"]:
                if c in cached.columns:
                    cached[c] = pd.to_numeric(cached[c], errors="coerce")
            cached = cached.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
            return cached

        # 6) Sem cache -> falha mesmo
        raise
        
    df.to_csv(path, index=False)
    return df.reset_index(drop=True)

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
            df = df.dropna(subset=["close"]).reset_index(drop=True)
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
    if 5_000_000 <= dv20 <= 40_000_000:
        return 0.6
    if dv20 < 5_000_000:
        return -0.2
    return -0.25

def score_candidate(
    bbz: float, atrpctl: float, dd: float, contr: float, dry: float, dv20: float,
    dist_to_trig_pct: float, overhead_touches: int,
    watch_boost: float, stale_penalty: float
) -> float:
    s = 0.0
    s += (-bbz) * 2.2
    s += (0.60 - atrpctl) * 1.5
    s += (BASE_DD_MAX - dd) * 1.1
    s += (CONTRACTION_MAX - contr) * 1.0
    s += (DRYUP_MAX - dry) * 0.9
    s += liquidity_sweet_spot_bonus(dv20)

    if np.isfinite(dist_to_trig_pct):
        s += max(0.0, (DIST_MAX_PCT - dist_to_trig_pct)) * 0.10

    # Overhead supply penalty (strong tiers)
    if overhead_touches >= 60:
        s -= 1.5
    elif overhead_touches >= 30:
        s -= 0.8
    elif overhead_touches > OVERHEAD_MAX_TOUCHES:
        s -= 0.4

    s += watch_boost
    s -= stale_penalty
    return float(s)

# =========================
# Regime logic (EOD)
# =========================
def regime_snapshot() -> dict:
    out = {"mode": "TRANSITION", "vol_mult_adj": 0.0, "dist_adj": 0.0, "qqq_trend": None, "vix_trend": None}
    qqq = fetch_ohlcv_symbol_best_effort(REG_QQQ)
    if qqq is None or len(qqq) < 120:
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
    if vix is not None and len(vix) >= 60:
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
        out["vol_mult_adj"] = -0.03
        out["dist_adj"] = +2.0
    elif qqq_trend == "DOWN" and (out["vix_trend"] in ["UP"]):
        out["mode"] = "RISK_OFF"
        out["vol_mult_adj"] = +0.10
        out["dist_adj"] = -4.0
    else:
        out["mode"] = "TRANSITION"
        out["vol_mult_adj"] = +0.03
        out["dist_adj"] = -1.0

    return out

# =========================
# signals.csv helpers
# =========================
def signals_csv_init_if_needed() -> None:
    if SIGNALS_CSV.exists():
        return
    ensure_dirs()
    with SIGNALS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "date","ticker","signal","score","close","trig","stop","dist_pct","dv20",
            "bbz","atrpctl","dd","contr","dry","overhead_touches","vol_mult","overshoot_pct",
            "R_pct","regime",
            "ret_5","ret_10","ret_20","ret_40","hit_30","hit_50","resolved"
        ])

def append_signal_row(row: dict) -> None:
    signals_csv_init_if_needed()
    cols = [
        "date","ticker","signal","score","close","trig","stop","dist_pct","dv20",
        "bbz","atrpctl","dd","contr","dry","overhead_touches","vol_mult","overshoot_pct",
        "R_pct","regime",
        "ret_5","ret_10","ret_20","ret_40","hit_30","hit_50","resolved"
    ]
    with SIGNALS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writerow({c: row.get(c, "") for c in cols})

def recent_watch_stats(ticker: str, lookback_days: int = 15) -> dict:
    if not SIGNALS_CSV.exists():
        return {"days": 0, "dists": []}
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return {"days": 0, "dists": []}
    if df.empty or "ticker" not in df.columns:
        return {"days": 0, "dists": []}

    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        return {"days": 0, "dists": []}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]

    dists = []
    if "dist_pct" in df.columns:
        dd = pd.to_numeric(df["dist_pct"], errors="coerce").dropna()
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
            boost = 0.35

    if days >= WATCH_STALE_DAYS:
        tail = dist_series[-WATCH_STALE_DAYS:]
        if all(np.isfinite(x) for x in tail):
            if (tail[0] - tail[-1]) < 1.0:
                penalty = 0.45

    return (boost, penalty)

# =========================
# outcomes update
# =========================
def update_outcomes_using_cache() -> dict:
    if not SIGNALS_CSV.exists():
        return {"updated": 0, "resolved_total": 0}
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
            ticker = str(df.at[idx, "ticker"])
            sig_date = pd.to_datetime(df.at[idx, "date"], errors="coerce")
            if pd.isna(sig_date):
                continue
            p = cache_path(ticker)
            if not p.exists():
                continue
            o = pd.read_csv(p)
            o.columns = [c.strip().lower() for c in o.columns]
            if "date" not in o.columns or "close" not in o.columns:
                continue
            o["date"] = pd.to_datetime(o["date"], errors="coerce")
            o = o.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            o_dates = o["date"].dt.date
            target = sig_date.date()
            positions = np.where(o_dates.values == target)[0]
            if len(positions) == 0:
                continue
            pos = int(positions[0])
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
    if df.empty:
        return "EMP: sem histórico"

    df = df[df.get("resolved", "0").astype(str) == "1"].copy()
    if df.empty or "score" not in df.columns:
        return "EMP: sem histórico"

    if "signal" in df.columns:
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
        p30 = float(pd.to_numeric(g["hit_30"], errors="coerce").fillna(0).mean())
        p50 = float(pd.to_numeric(g["hit_50"], errors="coerce").fillna(0).mean())
        out.append((str(b), n, p30, p50))

    parts = [f"{i+1}:{n}|{p30*100:.0f}/{p50*100:.0f}" for i, (_, n, p30, p50) in enumerate(out)]
    return "EMP bins(n|P30/P50%): " + " ".join(parts)

# =========================
# MAIN
# =========================
def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ensure_dirs()

    # --- learning update (EOD-only outcomes from cache) ---
    upd = update_outcomes_using_cache()

    # --- regime ---
    reg = regime_snapshot()
    mode = reg["mode"]
    VOL_CONFIRM_MULT = max(1.05, VOL_CONFIRM_MULT_BASE + reg["vol_mult_adj"])
    DIST_LIMIT = max(6.0, DIST_MAX_PCT + reg["dist_adj"])

    # --- build universe (NO alphabetical bias) ---
    all_tickers = (
        get_universe_from_holdings(IWC_URL) +
        get_universe_from_holdings(IWM_URL) +
        get_universe_from_holdings(IJR_URL)
    )

    # remove obvious junk/dupes but DO NOT sort
    universe = list(set([t.strip().upper() for t in all_tickers if isinstance(t, str) and t.strip()]))

    # If you want to cap the *universe source* size (not scan order), do it after ranking logic.
    # We'll build a liquidity-ranked scan list using cached dv20 first.
    scored: list[tuple[str, float]] = []
    unscored: list[str] = []

    # Use the whole universe to form the candidate pool, but ranking by cached dv20 will prefer liquid names.
    # CANDIDATE_POOL limits the number we consider for ranking (helps CPU).
    pool = universe[: min(CANDIDATE_POOL, len(universe))]

    for t in pool:
        dv = read_cached_dv20(t)
        if dv is None or not np.isfinite(dv) or dv <= 0:
            unscored.append(t)
        else:
            scored.append((t, float(dv)))

    # Evaluate highest cached liquidity first -> better use of limited fetch budget
    scored.sort(key=lambda x: x[1], reverse=True)
    ordered = [t for t, _ in scored] + unscored

    tickers = ordered[:MAX_TICKERS]

    # --- containers ---
    execA: list[tuple] = []
    execB: list[tuple] = []
    watch_clean: list[tuple] = []
    watch_over: list[tuple] = []
    near: list[tuple] = []

    # --- diagnostics ---
    hits_limited = False
    no_data = 0
    hist_ok = liq_ok = comp_ok = base_ok = dry_ok = 0
    cache_used = 0
    cache_miss = 0

    # helper: load cached OHLCV if rate-limited (or if you want cache-first later)
    def load_cached_ohlcv_local(ticker: str) -> pd.DataFrame | None:
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
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
            return df
        except Exception:
            return None

    for i, t in enumerate(tickers):
        try:
            # If rate-limited: do NOT stop. Switch to cache-only evaluation.
            if hits_limited:
                df = load_cached_ohlcv_local(t)
                if df is None:
                    cache_miss += 1
                    continue
                cache_used += 1
            else:
                if CACHE_ONLY:
                    df = load_cached_ohlcv_local(t)
                    if df is None:
                     continue
                else:
                    df = fetch_ohlcv_equity(t)

            # numeric safety
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

            if len(df) < 260:
                continue
            hist_ok += 1

            close_now = float(df["close"].iloc[-1])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
            sv20 = float(df["volume"].iloc[-20:].mean())

            # liquidity / proxy caps
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
                continue
            dry_ok += 1

            # stop / R
            stop = lowb * 0.99
            R_pct = ((trig - stop) / trig * 100.0) if trig > 0 else np.nan
            if (not np.isfinite(R_pct)) or (R_pct < MIN_R_PCT):
                near.append((t, close_now, dv20, bbz_last, f"LOW_R({R_pct:.1f}%)"))
                continue

            # distances
            dist_pct = ((trig - close_now) / close_now * 100.0) if close_now > 0 else np.nan
            overshoot_pct = ((close_now - trig) / trig * 100.0) if trig > 0 else 0.0

            # overhead
            overhead = overhead_supply_touches(df, trig)

            # watch maturation/stale (from signals.csv)
            stats = recent_watch_stats(t)
            boost, stale_pen = watch_boost_and_stale_penalty(stats["dists"])

            # score
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
                    "date": now.split(" ")[0], "ticker": t, "signal": sig,
                    "score": round(sc, 6), "close": round(close_now, 6),
                    "trig": round(trig, 6), "stop": round(stop, 6),
                    "dist_pct": round(float(dist_pct), 6) if np.isfinite(dist_pct) else "",
                    "dv20": round(dv20, 2), "bbz": round(bbz_last, 6),
                    "atrpctl": round(atr_pctl, 6), "dd": round(dd, 6),
                    "contr": round(contr, 6), "dry": round(dry, 6),
                    "overhead_touches": overhead,
                    "vol_mult": round(float(vol_mult), 6) if np.isfinite(vol_mult) else "",
                    "overshoot_pct": round(float(overshoot_pct), 6) if np.isfinite(overshoot_pct) else "",
                    "R_pct": round(float(R_pct), 6) if np.isfinite(R_pct) else "",
                    "regime": mode, "resolved": "0"
                })

            else:
                # Pre-breakout watchlist
                if np.isfinite(dist_pct) and dist_pct <= DIST_LIMIT and dist_pct >= -EXEC_MAX_OVERSHOOT_PCT:
                    item = (
                        sc, t, close_now, dv20, bbz_last, atr_pctl,
                        trig, stop, dist_pct, R_pct, overhead, win, boost, stale_pen
                    )

                    # split clean vs overhead (key tweak)
                    if overhead <= 5:
                        watch_clean.append(item)
                    else:
                        watch_over.append(item)

                    append_signal_row({
                        "date": now.split(" ")[0], "ticker": t, "signal": "WATCH",
                        "score": round(sc, 6), "close": round(close_now, 6),
                        "trig": round(trig, 6), "stop": round(stop, 6),
                        "dist_pct": round(float(dist_pct), 6) if np.isfinite(dist_pct) else "",
                        "dv20": round(dv20, 2), "bbz": round(bbz_last, 6),
                        "atrpctl": round(atr_pctl, 6), "dd": round(dd, 6),
                        "contr": round(contr, 6), "dry": round(dry, 6),
                        "overhead_touches": overhead,
                        "vol_mult": round(float(vol_mult), 6) if np.isfinite(vol_mult) else "",
                        "overshoot_pct": round(float(overshoot_pct), 6) if np.isfinite(overshoot_pct) else "",
                        "R_pct": round(float(R_pct), 6) if np.isfinite(R_pct) else "",
                        "regime": mode, "resolved": "0"
                    })

        except Exception as e:
            s = str(e)
            if "NO_DATA" in s:
                no_data += 1
            elif "HITS_LIMIT" in s:
                # switch to cache-only; DO NOT break the loop
                hits_limited = True
            # swallow other errors to keep scanner running
            pass

        if (i + 1) % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    # --- sorting / tops ---
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

    # --- message ---
    msg = [f"[{now}] ### V7_1_PREOPEN_FULL_TOP7 ### Microcap scanner (RANKED)"]
    msg.append(
        f"MODE={mode} | Eval={len(tickers)} | hist={hist_ok} liq={liq_ok} comp={comp_ok} base={base_ok} dry={dry_ok} | "
        f"EXEC_B={len(execB)} EXEC_A={len(execA)} WATCH_CLEAN={len(watch_clean)} WATCH_OVER={len(watch_over)} | nodata={no_data} | "
        f"PX<={MAX_PX:.0f} DV20<={MAX_DV20/1e6:.0f}M | VOLx>={VOL_CONFIRM_MULT:.2f} | dist<={DIST_LIMIT:.0f}% | MIN_R%={MIN_R_PCT:.0f}"
    )
    msg.append(emp)
    msg.append(f"LEARNING: outcomes_updated={upd['updated']} | resolved_total={upd['resolved_total']}")
    if STOOQ_HITSLIMIT:
        hits_limited = True
    if hits_limited:
        msg.append(f"NOTA: Stooq rate-limit; modo CACHE_ONLY ativado (cache_used={cache_used} cache_miss={cache_miss}).")
    msg.append("")

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

    if not execA and not execB:
        msg.append("EXECUTAR: (vazio)")
        msg.append("")

    if watch_clean:

        # Função de prioridade para ordenação
        def _prio_key(x):
            sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale = x

            prioridade = 0
            if dist <= 4:
                prioridade += 1
            if oh <= 3:
                prioridade += 1
            if atrp <= 0.20:
                prioridade += 1
            if Rp >= 12:
                prioridade += 1

            return (prioridade, sc)

        # ORDENAR WATCH_CLEAN (prioridade primeiro, score depois)
        watch_clean = sorted(watch_clean, key=_prio_key, reverse=True)

        msg.append("AGUARDAR_LIMPO (TOP 7; maior probabilidade):")

        for sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale in watch_clean:

            prioridade = 0
            if dist <= 4:
                prioridade += 1
            if oh <= 3:
                prioridade += 1
            if atrp <= 0.20:
                prioridade += 1
            if Rp >= 12:
                prioridade += 1

            if prioridade == 4:
                emoji = "🔥"
            elif prioridade == 3:
                emoji = "✅"
            elif prioridade == 2:
                emoji = "⚠️"
            else:
                emoji = "❌"

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

    if near:
        msg.append("QUASE (debug):")
        for t, c, dv, bbz, reason in near:
            msg.append(f"- {t} | {reason} | close={c:.2f} | dv20={dv/1e6:.1f}M | BBz={bbz:.2f}")
    else:
        msg.append("QUASE: (vazio)")

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()
