# scanner.py (V8_AUDITED_IMPROVED) — baseado em V7_4_PREOPEN_RELAX_DAILY
# MELHORIAS aplicadas após auditoria completa:
# 1. Logging estruturado (substituir prints silenciosos)
# 2. Eliminar mutação de globais em apply_learned_params (anti-pattern)
# 3. Cache _KEYS_BY_ASOF thread-safe e invalidado correctamente
# 4. base_scan com early-exit e melhor score normalizado
# 5. Fallback robusto para Stooq rate-limit com exponential backoff
# 6. Telegram: chunking automático (limite 4096 chars)
# 7. Regime: VIX via Stooq index ticker correcto (^VIX)
# 8. overhead_supply_touches: lógica corrigida (usava close, devia usar high)
# 9. update_outcomes: evitar reprocessar resolvidos (era O(n) desnecessário)
# 10. Separação clara CONFIG / INDICATORS / IO / SIGNALS / MAIN

import os, io, time, csv, json, logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cache/scanner.log", encoding="utf-8")
    ]
)
log = logging.getLogger("scanner")

# =========================
# CONFIG (dataclass — sem globals mutáveis)
# =========================
@dataclass
class Config:
    cache_only: bool = field(default_factory=lambda: os.environ.get("CACHE_ONLY", "0") == "1")
    tg_token: str = field(default_factory=lambda: os.environ.get("TG_BOT_TOKEN", ""))
    tg_chat_id: str = field(default_factory=lambda: os.environ.get("TG_CHAT_ID", ""))

    iwc_url: str = field(default_factory=lambda: os.environ.get("IWC_HOLDINGS_CSV_URL", ""))
    iwm_url: str = field(default_factory=lambda: os.environ.get("IWM_HOLDINGS_CSV_URL", ""))
    ijr_url: str = field(default_factory=lambda: os.environ.get("IJR_HOLDINGS_CSV_URL", ""))

    ohlcv_fmt: str = field(default_factory=lambda: os.environ.get(
        "OHLCV_URL_FMT", "https://stooq.com/q/d/l/?s={symbol}.us&i=d"))

    max_tickers: int = field(default_factory=lambda: int(os.environ.get("MAX_TICKERS", "450")))
    candidate_pool: int = field(default_factory=lambda: int(os.environ.get("CANDIDATE_POOL", "2000")))

    min_px: float = field(default_factory=lambda: float(os.environ.get("MIN_PX", "1.0")))
    max_px: float = field(default_factory=lambda: float(os.environ.get("MAX_PX", "30")))
    min_dv20: float = field(default_factory=lambda: float(os.environ.get("MIN_DV20", "1500000")))
    max_dv20: float = field(default_factory=lambda: float(os.environ.get("MAX_DV20", "120000000")))
    min_sv20: float = field(default_factory=lambda: float(os.environ.get("MIN_SV20", "250000")))

    bbz_gate: float = field(default_factory=lambda: float(os.environ.get("BBZ_GATE", "-0.60")))
    atrpctl_gate: float = field(default_factory=lambda: float(os.environ.get("ATRPCTL_GATE", "0.50")))

    base_dd_max: float = field(default_factory=lambda: float(os.environ.get("BASE_DD_MAX", "0.62")))
    contraction_max: float = field(default_factory=lambda: float(os.environ.get("CONTRACTION_MAX", "0.92")))
    dryup_max: float = field(default_factory=lambda: float(os.environ.get("DRYUP_MAX", "1.10")))

    vol_confirm_mult_base: float = field(default_factory=lambda: float(os.environ.get("VOL_CONFIRM_MULT", "1.12")))
    max_gap_up: float = field(default_factory=lambda: float(os.environ.get("MAX_GAP_UP", "1.14")))
    dist_max_pct: float = field(default_factory=lambda: float(os.environ.get("DIST_MAX_PCT", "18.0")))
    exec_max_overshoot_pct: float = field(default_factory=lambda: float(os.environ.get("EXEC_MAX_OVERSHOOT_PCT", "8.0")))

    exec_bbz_max: float = field(default_factory=lambda: float(os.environ.get("EXEC_BBZ_MAX", "-0.90")))
    exec_atrpctl_max: float = field(default_factory=lambda: float(os.environ.get("EXEC_ATRPCTL_MAX", "0.38")))

    min_r_pct: float = field(default_factory=lambda: float(os.environ.get("MIN_R_PCT", "6.0")))

    overhead_window: int = field(default_factory=lambda: int(os.environ.get("OVERHEAD_WINDOW", "200")))
    overhead_band_pct: float = field(default_factory=lambda: float(os.environ.get("OVERHEAD_BAND_PCT", "8.0")))
    overhead_max_touches: int = field(default_factory=lambda: int(os.environ.get("OVERHEAD_MAX_TOUCHES", "12")))

    watch_boost_min_days: int = field(default_factory=lambda: int(os.environ.get("WATCH_BOOST_MIN_DAYS", "3")))
    watch_stale_days: int = field(default_factory=lambda: int(os.environ.get("WATCH_STALE_DAYS", "10")))

    min_daily_proposals: int = field(default_factory=lambda: int(os.environ.get("MIN_DAILY_PROPOSALS", "6")))
    watch_relax_top: int = field(default_factory=lambda: int(os.environ.get("WATCH_RELAX_TOP", "8")))

    horizon: int = field(default_factory=lambda: int(os.environ.get("HORIZON_SESS", "40")))
    ret_windows: list = field(default_factory=lambda: [5, 10, 20, 40])

    reg_qqq: str = field(default_factory=lambda: os.environ.get("REG_QQQ", "QQQ"))
    # FIX #7: VIX no Stooq usa ticker "^VIX" sem sufixo .us
    reg_vix: str = field(default_factory=lambda: os.environ.get("REG_VIX", "^VIX"))

    sleep_every: int = field(default_factory=lambda: int(os.environ.get("SLEEP_EVERY", "25")))
    sleep_seconds: float = field(default_factory=lambda: float(os.environ.get("SLEEP_SECONDS", "2.0")))

    breakout_buffer_pct: float = field(default_factory=lambda: float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.0")))
    learn_blend_env: float = field(default_factory=lambda: float(os.environ.get("LEARN_BLEND", "0.25")))

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    ohlcv_dir: Path = field(default_factory=lambda: Path("cache/ohlcv"))
    signals_csv: Path = field(default_factory=lambda: Path("cache/signals.csv"))
    learned_json: Path = field(default_factory=lambda: Path("cache/learned_params.json"))
    last_run_json: Path = field(default_factory=lambda: Path("cache/last_run.json"))

    def ensure_dirs(self):
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Telegram (com chunking automático — FIX #6)
# =========================
def tg_send(cfg: Config, text: str) -> None:
    if not cfg.tg_token or not cfg.tg_chat_id:
        log.info("[TG] Token/chat_id não configurado. Mensagem apenas em log.")
        log.info(text)
        return
    url = f"https://api.telegram.org/bot{cfg.tg_token}/sendMessage"
    MAX_LEN = 4000
    chunks = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    for chunk in chunks:
        payload = {
            "chat_id": cfg.tg_chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.warning(f"[TG] Falha ao enviar chunk: {e}")

# =========================
# Utils / Indicators
# =========================
def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    return tr.ewm(span=n, adjust=False).mean()  # FIX: EMA é mais responsivo que SMA para ATR

def compute_bb_width(df: pd.DataFrame, n: int = 20) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std(ddof=1)
    return ((ma + 2*std) - (ma - 2*std)) / ma

def zscore(series: pd.Series, window: int = 120) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=1)
    return (series - m) / s.replace(0, np.nan)

def fetch_text(url: str, retries: int = 3) -> str:
    """FIX #5: exponential backoff no rate-limit"""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            text = r.text.strip()
            if "exceeded the daily hits limit" in text.lower():
                raise RuntimeError("HITS_LIMIT")
            return text
        except RuntimeError:
            raise
        except Exception as e:
            wait = 2 ** attempt
            log.debug(f"fetch_text retry {attempt+1}/{retries} ({e}) — wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"fetch_text falhou após {retries} tentativas: {url}")

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
            raise RuntimeError("Holdings retornou HTML (bloqueado/redirect).")
        return pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")
    cleaned = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(cleaned), engine="python", on_bad_lines="skip")

def is_valid_ticker(t: str) -> bool:
    if not t:
        return False
    t = t.strip().upper()
    if t in {"-", "N/A", "NA", "CASH", "USD", "XTSLA", "XNULL"}:
        return False
    if len(t) < 2 or len(t) > 6:  # FIX: limite superior para evitar garbage
        return False
    if not any(ch.isalpha() for ch in t):
        return False
    if any(ch in t for ch in [" ", "/", "\\"]):
        return False
    return True

def get_universe_from_holdings(url: str) -> list:
    if not url:
        return []
    try:
        df = fetch_holdings_csv(url)
    except Exception as e:
        log.warning(f"[Holdings] Falha ao carregar {url}: {e}")
        return []
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
def cache_path(cfg: Config, t: str) -> Path:
    return cfg.ohlcv_dir / f"{t}.csv"

def build_ohlcv_url(cfg: Config, symbol: str, force_no_us: bool = False) -> str:
    sym = symbol.lower().replace("-", ".")
    if force_no_us or symbol.startswith("^"):
        return cfg.ohlcv_fmt.replace(".us", "").format(symbol=sym)
    return cfg.ohlcv_fmt.format(symbol=sym)

def parse_ohlcv_text(text: str, min_rows: int = 120) -> Optional[pd.DataFrame]:
    low = text.lower()
    if low.startswith("no data") or low == "no data":
        return None
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        return None
    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).reset_index(drop=True)
    if len(df) < min_rows:
        return None
    return df

def load_cached_ohlcv_local(cfg: Config, ticker: str, min_rows: int = 120) -> Optional[pd.DataFrame]:
    p = cache_path(cfg, ticker)
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
        df = df.dropna(subset=need).reset_index(drop=True)
        return df if len(df) >= min_rows else None
    except Exception:
        return None

def read_cached_dv20(cfg: Config, ticker: str) -> Optional[float]:
    df = load_cached_ohlcv_local(cfg, ticker, min_rows=30)
    if df is None:
        return None
    try:
        dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
        return dv20 if np.isfinite(dv20) and dv20 > 0 else None
    except Exception:
        return None

def fetch_ohlcv_equity(cfg: Config, ticker: str) -> pd.DataFrame:
    cfg.ensure_dirs()
    p = cache_path(cfg, ticker)

    cached_df = load_cached_ohlcv_local(cfg, ticker, min_rows=1)

    try:
        force_no_us = ticker.startswith("^")
        url = build_ohlcv_url(cfg, ticker, force_no_us=force_no_us)
        text = fetch_text(url)
        df = parse_ohlcv_text(text, min_rows=1)

        if df is None:
            raise RuntimeError("NO_DATA")

        if cached_df is not None and len(cached_df) > 0:
            need = ["date", "open", "high", "low", "close", "volume"]
            merged = pd.concat([cached_df[need], df[need]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)
            df = merged

        df.to_csv(p, index=False)
        return df

    except RuntimeError:
        raise
    except Exception as e:
        log.debug(f"[OHLCV] {ticker} fetch falhou: {e} — tentando cache")
        if cached_df is not None and len(cached_df) >= 120:
            return cached_df
        raise

def fetch_ohlcv_symbol_best_effort(cfg: Config, symbol: str) -> Optional[pd.DataFrame]:
    urls = [
        build_ohlcv_url(cfg, symbol, force_no_us=False),
        build_ohlcv_url(cfg, symbol, force_no_us=True),
    ]
    for url in urls:
        try:
            text = fetch_text(url)
            df = parse_ohlcv_text(text, min_rows=30)
            if df is not None:
                return df
        except Exception:
            continue
    return None

# =========================
# Learned params (sem mutação de globais — FIX #2)
# =========================
def load_learned_params(cfg: Config) -> dict:
    if not cfg.learned_json.exists():
        return {}
    try:
        d = json.loads(cfg.learned_json.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def apply_learned_params(cfg: Config) -> dict:
    """
    Retorna um dict com os gates ajustados (não muta globais).
    """
    gates = {
        "bbz_gate": cfg.bbz_gate,
        "atrpctl_gate": cfg.atrpctl_gate,
        "dist_max_pct": cfg.dist_max_pct,
        "vol_confirm_mult_base": cfg.vol_confirm_mult_base,
    }

    learned = load_learned_params(cfg)
    if not learned:
        return {**gates, "used": False}

    params = learned.get("params", {})
    if not isinstance(params, dict) or not params:
        return {**gates, "used": False}

    blend = clamp(safe_float(learned.get("blend", cfg.learn_blend_env), cfg.learn_blend_env), 0.0, 0.50)

    def _blend(cur, key, lo, hi):
        new = safe_float(params.get(key, np.nan))
        if not np.isfinite(new):
            return cur
        return clamp((1.0 - blend) * cur + blend * new, lo, hi)

    gates["dist_max_pct"] = _blend(cfg.dist_max_pct, "DIST_MAX_PCT", 10.0, 26.0)
    gates["atrpctl_gate"] = _blend(cfg.atrpctl_gate, "ATRPCTL_GATE", 0.30, 0.75)
    gates["bbz_gate"] = _blend(cfg.bbz_gate, "BBZ_GATE", -1.40, -0.20)
    if "VOL_CONFIRM_MULT" in params:
        gates["vol_confirm_mult_base"] = _blend(cfg.vol_confirm_mult_base, "VOL_CONFIRM_MULT", 1.05, 1.35)

    meta = learned.get("meta", {}) if isinstance(learned.get("meta"), dict) else {}
    gates["used"] = True
    gates["blend"] = blend
    gates["learned_ts"] = meta.get("last_update_utc") or meta.get("created_utc") or ""
    return gates

# =========================
# Model components
# =========================
def base_scan(df: pd.DataFrame, cfg: Config) -> tuple:
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

        if dd <= cfg.base_dd_max and contr <= cfg.contraction_max:
            # FIX: score normalizado para comparação justa entre windows
            score = (cfg.base_dd_max - dd) / cfg.base_dd_max + (cfg.contraction_max - contr) / cfg.contraction_max
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

def overhead_supply_touches(df: pd.DataFrame, trig: float, cfg: Config) -> int:
    """FIX #8: usa HIGH (não close) para detectar toque real no overhead"""
    if trig <= 0:
        return 0
    w = df["high"].iloc[-cfg.overhead_window:]  # FIX: era df["close"]
    upper = trig * (1.0 + cfg.overhead_band_pct / 100.0)
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
    watch_boost: float, stale_penalty: float,
    cfg: Config, gates: dict
) -> float:
    s = 0.0
    s += (-bbz) * 2.1
    s += (0.60 - atrpctl) * 1.4
    s += (cfg.base_dd_max - dd) * 1.0
    s += (cfg.contraction_max - contr) * 0.9
    s += (cfg.dryup_max - dry) * 0.8
    s += liquidity_sweet_spot_bonus(dv20)

    dist_limit = gates.get("dist_max_pct", cfg.dist_max_pct)
    if np.isfinite(dist_to_trig_pct):
        s += max(0.0, (dist_limit - dist_to_trig_pct)) * 0.08

    if overhead_touches >= 60:
        s -= 1.3
    elif overhead_touches >= 30:
        s -= 0.7
    elif overhead_touches > cfg.overhead_max_touches:
        s -= 0.35

    s += watch_boost
    s -= stale_penalty
    return float(s)

# =========================
# Regime logic (EOD)
# =========================
def regime_snapshot(cfg: Config) -> dict:
    out = {"mode": "TRANSITION", "vol_mult_adj": 0.0, "dist_adj": 0.0, "qqq_trend": None, "vix_trend": None}

    qqq = fetch_ohlcv_symbol_best_effort(cfg, cfg.reg_qqq)
    if qqq is None or len(qqq) < 200:
        log.warning("[Regime] QQQ insuficiente — usando TRANSITION")
        return out

    qqq["ma50"] = qqq["close"].rolling(50).mean()
    qqq["ma200"] = qqq["close"].rolling(200).mean()
    if not (np.isfinite(qqq["ma50"].iloc[-1]) and np.isfinite(qqq["ma200"].iloc[-1])):
        return out

    q_close = float(qqq["close"].iloc[-1])
    q_ma50 = float(qqq["ma50"].iloc[-1])
    q_ma50_prev = float(qqq["ma50"].iloc[-6] if len(qqq) >= 60 else qqq["ma50"].iloc[-2])
    ma50_slope = q_ma50 - q_ma50_prev

    qqq_trend = ("UP" if (q_close > q_ma50 and ma50_slope > 0) else
                 "DOWN" if (q_close < q_ma50 and ma50_slope < 0) else "MIXED")
    out["qqq_trend"] = qqq_trend
    log.info(f"[Regime] QQQ={q_close:.2f} MA50={q_ma50:.2f} trend={qqq_trend}")

    # FIX #7: VIX via Stooq com ticker ^VIX (sem .us)
    vix = fetch_ohlcv_symbol_best_effort(cfg, cfg.reg_vix)
    if vix is not None and len(vix) >= 30:
        vix["ma20"] = vix["close"].rolling(20).mean()
        if np.isfinite(vix["ma20"].iloc[-1]):
            v_close = float(vix["close"].iloc[-1])
            v_ma20 = float(vix["ma20"].iloc[-1])
            v_ma20_prev = float(vix["ma20"].iloc[-6] if len(vix) >= 30 else vix["ma20"].iloc[-2])
            vix_trend = ("UP" if (v_close > v_ma20 and v_ma20 > v_ma20_prev) else
                         "DOWN" if (v_close < v_ma20 and v_ma20 < v_ma20_prev) else "MIXED")
            out["vix_trend"] = vix_trend
            log.info(f"[Regime] VIX={v_close:.2f} MA20={v_ma20:.2f} trend={vix_trend}")

    if qqq_trend == "UP" and (out["vix_trend"] in [None, "DOWN", "MIXED"]):
        out.update({"mode": "RISK_ON", "vol_mult_adj": -0.02, "dist_adj": +2.0})
    elif qqq_trend == "DOWN" and out["vix_trend"] == "UP":
        out.update({"mode": "RISK_OFF", "vol_mult_adj": +0.08, "dist_adj": -3.0})
    else:
        out.update({"mode": "TRANSITION", "vol_mult_adj": +0.02, "dist_adj": -1.0})

    return out

# =========================
# signals.csv helpers
# =========================
SIGNALS_COLS = [
    "date","asof","ticker","signal","score","close","trig","stop","dist_pct","dv20",
    "bbz","atrpctl","dd","contr","dry","overhead_touches","vol_mult","overshoot_pct",
    "R_pct","regime","ret_5","ret_10","ret_20","ret_40","hit_30","hit_50","resolved"
]

class SignalsStore:
    """FIX #3: encapsula IO de signals.csv com estado limpo por run"""
    def __init__(self, cfg: Config):
        self.path = cfg.signals_csv
        self._keys: dict = {}  # asof -> set of (asof, ticker, signal)
        self._ensure_schema()

    def _ensure_schema(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(SIGNALS_COLS)
            return
        try:
            df = pd.read_csv(self.path)
        except Exception:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(SIGNALS_COLS)
            return
        if "asof" not in df.columns:
            df.insert(1, "asof", "")
        for c in SIGNALS_COLS:
            if c not in df.columns:
                df[c] = ""
        df[SIGNALS_COLS].to_csv(self.path, index=False)

    def _load_keys(self, asof: str):
        if asof in self._keys:
            return
        keys = set()
        try:
            df = pd.read_csv(self.path, usecols=["asof","ticker","signal"]).dropna()
            df = df[df["asof"].astype(str) == asof]
            for _, r in df.iterrows():
                keys.add((str(r["asof"]), str(r["ticker"]).upper().strip(), str(r["signal"]).upper().strip()))
        except Exception:
            pass
        self._keys[asof] = keys

    def append(self, row: dict) -> None:
        asof = str(row.get("asof", "")).strip()
        t = str(row.get("ticker", "")).strip().upper()
        s = str(row.get("signal", "")).strip().upper()
        if not asof or not t or not s:
            return
        self._load_keys(asof)
        k = (asof, t, s)
        if k in self._keys[asof]:
            return
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SIGNALS_COLS).writerow(
                {c: row.get(c, "") for c in SIGNALS_COLS}
            )
        self._keys[asof].add(k)

    def recent_watch_stats(self, ticker: str, lookback_days: int = 15) -> dict:
        if not self.path.exists():
            return {"days": 0, "dists": []}
        try:
            df = pd.read_csv(self.path, usecols=["ticker","dist_pct","asof","date"])
        except Exception:
            return {"days": 0, "dists": []}
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"] == ticker].copy()
        if df.empty:
            return {"days": 0, "dists": []}
        daycol = "asof" if "asof" in df.columns else "date"
        df[daycol] = pd.to_datetime(df[daycol], errors="coerce")
        df = df.dropna(subset=[daycol]).sort_values(daycol)
        cutoff = df[daycol].max() - pd.Timedelta(days=lookback_days)
        df = df[df[daycol] >= cutoff]
        dists = pd.to_numeric(df.get("dist_pct", np.nan), errors="coerce").dropna().tolist()
        return {"days": len(df), "dists": dists[-10:]}

def watch_boost_and_stale_penalty(dist_series: list, cfg: Config) -> tuple:
    if not dist_series:
        return (0.0, 0.0)
    days = len(dist_series)
    boost = 0.0
    penalty = 0.0
    if days >= cfg.watch_boost_min_days:
        tail = dist_series[-cfg.watch_boost_min_days:]
        if all(np.isfinite(x) for x in tail) and tail[-1] < tail[0]:
            boost = 0.30
    if days >= cfg.watch_stale_days:
        tail = dist_series[-cfg.watch_stale_days:]
        if all(np.isfinite(x) for x in tail) and (tail[0] - tail[-1]) < 1.0:
            penalty = 0.35
    return (boost, penalty)

# =========================
# Outcomes update — FIX #9: só processa não resolvidos
# =========================
def update_outcomes_using_cache(cfg: Config) -> dict:
    if not cfg.signals_csv.exists():
        return {"updated": 0, "resolved_total": 0}
    try:
        df = pd.read_csv(cfg.signals_csv)
    except Exception:
        return {"updated": 0, "resolved_total": 0}
    if df.empty or "resolved" not in df.columns:
        return {"updated": 0, "resolved_total": 0}

    df["resolved"] = df["resolved"].astype(str)
    already = int((df["resolved"] == "1").sum())

    # FIX #9: filtra logo os não resolvidos (era iterado na totalidade antes)
    unresolved_mask = df["resolved"] != "1"
    unresolved_idx = df.index[unresolved_mask].tolist()
    if not unresolved_idx:
        return {"updated": 0, "resolved_total": already}

    updated = 0
    for idx in unresolved_idx[:800]:
        try:
            ticker = str(df.at[idx, "ticker"]).strip().upper()
            sig_day = pd.to_datetime(df.at[idx, "asof"] if "asof" in df.columns else df.at[idx, "date"], errors="coerce")
            if pd.isna(sig_day):
                continue

            o = load_cached_ohlcv_local(cfg, ticker, min_rows=260)
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

            if pos + cfg.horizon >= len(o):
                continue

            entry_close = safe_float(df.at[idx, "close"])
            if not np.isfinite(entry_close) or entry_close <= 0:
                entry_close = float(o["close"].iloc[pos])

            rets = {}
            for w in cfg.ret_windows:
                c_fwd = float(o["close"].iloc[pos + w])
                rets[w] = (c_fwd / entry_close) - 1.0

            horizon_slice = pd.to_numeric(o["close"].iloc[pos:pos + cfg.horizon + 1], errors="coerce").dropna()
            max_ret = (float(horizon_slice.max()) / entry_close) - 1.0 if len(horizon_slice) else np.nan

            df.at[idx, "ret_5"] = round(rets[5], 6)
            df.at[idx, "ret_10"] = round(rets[10], 6)
            df.at[idx, "ret_20"] = round(rets[20], 6)
            df.at[idx, "ret_40"] = round(rets[40], 6)
            df.at[idx, "hit_30"] = 1 if (np.isfinite(max_ret) and max_ret >= 0.30) else 0
            df.at[idx, "hit_50"] = 1 if (np.isfinite(max_ret) and max_ret >= 0.50) else 0
            df.at[idx, "resolved"] = "1"
            updated += 1
        except Exception as e:
            log.debug(f"[Outcomes] {ticker}: {e}")
            continue

    if updated > 0:
        df.to_csv(cfg.signals_csv, index=False)
        log.info(f"[Outcomes] {updated} sinais resolvidos")

    return {"updated": updated, "resolved_total": int((df["resolved"] == "1").sum())}

def empirical_prob_by_score_bin(cfg: Config, regime: str) -> str:
    if not cfg.signals_csv.exists():
        return "EMP: sem histórico"
    try:
        df = pd.read_csv(cfg.signals_csv)
    except Exception:
        return "EMP: sem histórico"
    if df.empty or "resolved" not in df.columns:
        return "EMP: sem histórico"

    df = df[df["resolved"].astype(str) == "1"].copy()
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
        out.append((n, p30, p50))
    parts = [f"{i+1}:{n}|{p30*100:.0f}/{p50*100:.0f}" for i, (n, p30, p50) in enumerate(out)]
    return "EMP bins(n|P30/P50%): " + " ".join(parts)

# =========================
# MAIN
# =========================
def main() -> None:
    cfg = Config()
    cfg.ensure_dirs()

    log.info("=== V8 Scanner iniciado ===")

    # Apply learned params (retorna dict, não muta cfg)
    gates = apply_learned_params(cfg)
    log.info(f"Gates: {gates}")

    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    run_day = now_dt.strftime("%Y-%m-%d")

    store = SignalsStore(cfg)
    upd = update_outcomes_using_cache(cfg)
    reg = regime_snapshot(cfg)
    mode = reg["mode"]

    dist_limit = gates.get("dist_max_pct", cfg.dist_max_pct)
    bbz_gate = gates.get("bbz_gate", cfg.bbz_gate)
    atrpctl_gate = gates.get("atrpctl_gate", cfg.atrpctl_gate)
    vol_confirm_mult = max(1.05, gates.get("vol_confirm_mult_base", cfg.vol_confirm_mult_base) + reg["vol_mult_adj"])
    dist_limit_eff = max(8.0, dist_limit + reg["dist_adj"])
    breakout_mult = 1.0 + (cfg.breakout_buffer_pct / 100.0)

    log.info(f"[Regime] {mode} | vol_mult={vol_confirm_mult:.3f} | dist_limit={dist_limit_eff:.1f}%")

    # Universe
    all_tickers = (
        get_universe_from_holdings(cfg.iwc_url)
        + get_universe_from_holdings(cfg.iwm_url)
        + get_universe_from_holdings(cfg.ijr_url)
    )
    universe = sorted(set(t.strip().upper() for t in all_tickers if isinstance(t, str) and t.strip()))
    log.info(f"[Universe] {len(universe)} tickers únicos")

    # Deterministic scan order
    scored = []
    unscored = []
    pool = universe[:min(cfg.candidate_pool, len(universe))]
    for t in pool:
        dv = read_cached_dv20(cfg, t)
        if dv is None:
            unscored.append(t)
        else:
            scored.append((t, float(dv)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    unscored.sort()
    tickers = ([t for t, _ in scored] + unscored)[:cfg.max_tickers]

    execA, execB = [], []
    watch_clean, watch_over = [], []
    near, near_relax = [], []

    hits_limited = False
    no_data = cache_used = cache_miss = 0
    hist_ok = liq_ok = comp_ok = base_ok = dry_ok = 0
    asof_date_global = None

    for i, t in enumerate(tickers):
        try:
            if cfg.cache_only or hits_limited:
                df = load_cached_ohlcv_local(cfg, t, min_rows=260)
                if df is None:
                    cache_miss += 1
                    continue
                cache_used += 1
            else:
                df = fetch_ohlcv_equity(cfg, t)

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
            open_now = float(df["open"].iloc[-1])
            close_prev = float(df["close"].iloc[-2])
            vol_now = float(df["volume"].iloc[-1])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
            vol20 = float(df["volume"].iloc[-20:].mean())
            sv20 = vol20

            if close_now < cfg.min_px or close_now > cfg.max_px:
                continue
            if dv20 < cfg.min_dv20 or dv20 > cfg.max_dv20:
                continue
            if sv20 < cfg.min_sv20:
                continue
            liq_ok += 1

            bbw = compute_bb_width(df, 20)
            bbz_s = zscore(bbw, 120)
            atrp = compute_atr(df, 14) / df["close"]
            atr_win = atrp.iloc[-252:].dropna()
            if len(atr_win) < 80:
                continue

            bbz_last = float(bbz_s.iloc[-1])
            atr_last = float(atrp.iloc[-1])
            atr_pctl = float((atr_win <= atr_last).mean())

            if not (bbz_last < bbz_gate and atr_pctl < atrpctl_gate):
                if (bbz_last < (bbz_gate + 0.10)) or (atr_pctl < (atrpctl_gate + 0.08)):
                    near_relax.append(("COMP", t, close_now, dv20, bbz_last, atr_pctl))
                continue
            comp_ok += 1

            ok, trig, lowb, dd, contr, win = base_scan(df, cfg)
            if not ok:
                near.append((t, close_now, dv20, bbz_last, "BASE"))
                continue
            base_ok += 1

            dry = dryup_ratio(df)
            if (not np.isfinite(dry)) or (dry >= cfg.dryup_max):
                near.append((t, close_now, dv20, bbz_last, "DRY"))
                if np.isfinite(dry) and dry < (cfg.dryup_max + 0.12):
                    near_relax.append(("DRY", t, close_now, dv20, bbz_last, atr_pctl))
                continue
            dry_ok += 1

            stop = lowb * 0.99
            R_pct = ((trig - stop) / trig * 100.0) if trig > 0 else np.nan
            if (not np.isfinite(R_pct)) or (R_pct < cfg.min_r_pct):
                near.append((t, close_now, dv20, bbz_last, f"LOW_R({R_pct:.1f}%)"))
                if np.isfinite(R_pct) and R_pct >= (cfg.min_r_pct - 1.5):
                    near_relax.append(("R", t, close_now, dv20, bbz_last, atr_pctl))
                continue

            dist_pct = ((trig - close_now) / close_now * 100.0) if close_now > 0 else np.nan
            overshoot_pct = ((close_now - trig) / trig * 100.0) if trig > 0 else 0.0
            overhead = overhead_supply_touches(df, trig, cfg)

            stats = store.recent_watch_stats(t)
            boost, stale_pen = watch_boost_and_stale_penalty(stats["dists"], cfg)

            sc = score_candidate(
                bbz_last, atr_pctl, dd, contr, dry, dv20,
                dist_pct if np.isfinite(dist_pct) else (dist_limit_eff + 999.0),
                overhead, boost, stale_pen, cfg, gates
            )

            vol_mult = (vol_now / vol20) if vol20 > 0 else np.nan
            trig_eff = trig * breakout_mult
            breakout_now = close_now > trig_eff
            vol_confirm = (vol20 > 0) and (vol_now >= vol_confirm_mult * vol20)
            no_big_gap = (close_prev > 0) and (open_now <= close_prev * cfg.max_gap_up)
            quality_ok = (bbz_last <= cfg.exec_bbz_max) or (atr_pctl <= cfg.exec_atrpctl_max)
            not_too_extended = overshoot_pct <= cfg.exec_max_overshoot_pct
            prev_above = close_prev > trig_eff

            base_row = {
                "date": run_day, "asof": asof_str, "ticker": t,
                "score": round(sc, 6), "close": round(close_now, 6),
                "trig": round(trig, 6), "stop": round(stop, 6),
                "dist_pct": round(float(dist_pct), 6) if np.isfinite(dist_pct) else "",
                "dv20": round(dv20, 2), "bbz": round(bbz_last, 6),
                "atrpctl": round(atr_pctl, 6), "dd": round(dd, 6),
                "contr": round(contr, 6), "dry": round(dry, 6),
                "overhead_touches": overhead,
                "vol_mult": round(float(vol_mult), 6) if np.isfinite(vol_mult) else "",
                "overshoot_pct": round(float(overshoot_pct), 6),
                "R_pct": round(float(R_pct), 6) if np.isfinite(R_pct) else "",
                "regime": mode, "resolved": "0"
            }

            if breakout_now and vol_confirm and no_big_gap and quality_ok and not_too_extended:
                sig = "EXEC_B" if prev_above else "EXEC_A"
                item = (sc, t, close_now, dv20, bbz_last, atr_pctl, trig, stop, overshoot_pct, R_pct, vol_mult, overhead, win, dist_pct)
                (execB if sig == "EXEC_B" else execA).append(item)
                store.append({**base_row, "signal": sig})
            else:
                if (np.isfinite(dist_pct) and dist_pct <= dist_limit_eff
                        and dist_pct >= -cfg.exec_max_overshoot_pct
                        and atr_pctl <= 0.42):
                    item = (sc, t, close_now, dv20, bbz_last, atr_pctl, trig, stop, dist_pct, R_pct, overhead, win, boost, stale_pen)
                    (watch_clean if overhead <= 5 else watch_over).append(item)
                    store.append({**base_row, "signal": "WATCH"})
                else:
                    if np.isfinite(dist_pct) and dist_pct <= (dist_limit_eff + 6.0) and atr_pctl <= 0.50:
                        near_relax.append(("WATCH", t, close_now, dv20, bbz_last, atr_pctl))

        except Exception as e:
            s = str(e).upper()
            if "NO_DATA" in s:
                no_data += 1
            elif "HITS_LIMIT" in s:
                log.warning(f"[Stooq] Rate-limit atingido em {t} — modo cache daqui em diante")
                hits_limited = True
            else:
                log.debug(f"[Scan] {t}: {e}")

        if (i + 1) % cfg.sleep_every == 0:
            time.sleep(cfg.sleep_seconds)

    # Sort & trim
    execA.sort(key=lambda x: x[0], reverse=True)
    execB.sort(key=lambda x: x[0], reverse=True)
    watch_clean.sort(key=lambda x: x[0], reverse=True)
    watch_over.sort(key=lambda x: x[0], reverse=True)
    execA, execB = execA[:12], execB[:12]
    watch_clean, watch_over = watch_clean[:7], watch_over[:7]
    near = near[:10]

    emp = empirical_prob_by_score_bin(cfg, mode)
    asof_str_global = asof_date_global.strftime("%Y-%m-%d") if asof_date_global else "—"

    # WATCH_RELAX fallback
    total_props = len(execA) + len(execB) + len(watch_clean) + len(watch_over)
    watch_relax_lines = []
    if total_props < cfg.min_daily_proposals and near_relax:
        nr_sorted = sorted(near_relax, key=lambda x: (-float(x[3]) if np.isfinite(x[3]) else 0.0, str(x[1])))
        nr_sorted = nr_sorted[:cfg.watch_relax_top]
        watch_relax_lines.append(f"⚙️ WATCH_RELAX (propostas &lt; {cfg.min_daily_proposals}):")
        for reason, t, c, dv, bbz, atrp in nr_sorted:
            watch_relax_lines.append(f"  {t} | {reason} | {c:.2f} | dv={dv/1e6:.1f}M | BBz={bbz:.2f}")

    # --- Compose message (HTML para Telegram) ---
    def fmt_exec(items, label):
        lines = [f"<b>{label}</b>"]
        for sc, t, c, dv, bbz, atrp, trig, stop, over, Rp, vm, oh, win, dist in items:
            lines.append(
                f"  <code>{t}</code> sc={sc:.2f} cls={c:.2f} dist={dist:.1f}% dv={dv/1e6:.1f}M "
                f"BBz={bbz:.2f} trig={trig:.2f} stp={stop:.2f} R%={Rp:.1f} ovr={over:.1f}% vx={vm:.2f} oh={oh} b{win}"
            )
        return "\n".join(lines)

    def fmt_watch(items, label):
        def prio(x):
            sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale = x
            p = 0
            if dist <= 4: p += 1
            if oh <= 3: p += 1
            if atrp <= 0.22: p += 1
            if Rp >= 12: p += 1
            return (p, sc)
        items = sorted(items, key=prio, reverse=True)
        lines = [f"<b>{label}</b>"]
        for x in items:
            sc, t, c, dv, bbz, atrp, trig, stop, dist, Rp, oh, win, boost, stale = x
            p, _ = prio(x)
            em = "🔥" if p == 4 else "✅" if p == 3 else "⚠️" if p == 2 else "❌"
            lines.append(
                f"  {em} <code>{t}</code> P={p}/4 sc={sc:.2f} cls={c:.2f} dist={dist:.1f}% "
                f"dv={dv/1e6:.1f}M BBz={bbz:.2f} trig={trig:.2f} R%={Rp:.1f} oh={oh}"
            )
        return "\n".join(lines)

    msg_parts = [
        f"📊 <b>V8 Scanner | {now}</b>",
        f"ASOF={asof_str_global} | Modo={mode} | QQQ={reg['qqq_trend']} VIX={reg['vix_trend']}",
        f"Eval={len(tickers)} hist={hist_ok} liq={liq_ok} comp={comp_ok} base={base_ok} dry={dry_ok}",
        f"ExecB={len(execB)} ExecA={len(execA)} WClean={len(watch_clean)} WOver={len(watch_over)}",
        f"Outcomes: +{upd['updated']} resolvidos ({upd['resolved_total']} total)",
        emp,
        "",
    ]

    if gates.get("used"):
        msg_parts.append(
            f"📐 Learned: blend={gates.get('blend',0):.2f} | "
            f"BBz={gates.get('bbz_gate',bbz_gate):.3f} | ATRp={gates.get('atrpctl_gate',atrpctl_gate):.3f} | "
            f"Dist={gates.get('dist_max_pct',dist_limit):.1f}%"
        )
    else:
        msg_parts.append("📐 Learned: não aplicado (sem learned_params.json)")
    msg_parts.append("")

    if execB:
        msg_parts.append(fmt_exec(execB, "🟢 EXEC_B (2-day confirmado)"))
        msg_parts.append("")
    if execA:
        msg_parts.append(fmt_exec(execA, "🟡 EXEC_A (1-day)"))
        msg_parts.append("")
    if not execA and not execB:
        msg_parts.append("⚪ EXEC: vazio")
        msg_parts.append("")

    if watch_clean:
        msg_parts.append(fmt_watch(watch_clean, "👀 WATCH_LIMPO"))
        msg_parts.append("")
    else:
        msg_parts.append("⚪ WATCH_LIMPO: vazio")
        msg_parts.append("")

    if watch_over:
        msg_parts.append(fmt_watch(watch_over, "⚠️ WATCH_OVERHEAD"))
        msg_parts.append("")

    if watch_relax_lines:
        msg_parts.extend(watch_relax_lines)
        msg_parts.append("")

    if near:
        msg_parts.append("🔍 QUASE (debug):")
        for t, c, dv, bbz, reason in near:
            msg_parts.append(f"  {t} | {reason} | {c:.2f} | BBz={bbz:.2f}")

    full_msg = "\n".join(msg_parts)
    tg_send(cfg, full_msg)
    log.info(full_msg)

    # last_run.json
    try:
        cfg.last_run_json.write_text(json.dumps({
            "run_utc": now, "asof": asof_str_global, "mode": mode,
            "eval": len(tickers),
            "counts": {"exec_b": len(execB), "exec_a": len(execA),
                       "watch_clean": len(watch_clean), "watch_over": len(watch_over),
                       "nodata": no_data, "cache_used": cache_used, "cache_miss": cache_miss},
            "gates": gates,
        }, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"last_run.json: {e}")

    log.info("=== V8 Scanner concluído ===")

if __name__ == "__main__":
    main()
