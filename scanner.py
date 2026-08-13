"""Heartbeat Stage 2 Scanner.

Scanner NASDAQ (micro a mid cap, ~$0.08–$500, teto de capitalização
configurável até $10 mil milhões) desenhado para detetar bases de compressão
de volatilidade prolongada pouco depois de uma recuperação, com volume
confirmado, da média móvel simples de 150 dias, com a média móvel simples
diária de 50 dias confirmada a curvar para cima (inclinação positiva) como
gate adicional e independente.

Data sources (no paid API required):
- Nasdaq public stock screener: exchange membership, last price, market cap, volume.
- Yahoo chart endpoint, with Stooq fallback: daily OHLCV.

The scanner never pads the result set. A ticker is published only when every hard
technical criterion is satisfied. Float is reported only when available from market-data sources; no SEC or financial-risk analysis is performed automatically.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CACHE_DIR / "scanner.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("heartbeat-stage2")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    tg_token: str = field(default_factory=lambda: os.getenv("TG_BOT_TOKEN", ""))
    tg_chat_id: str = field(default_factory=lambda: os.getenv("TG_CHAT_ID", ""))

    max_price: float = field(default_factory=lambda: float(os.getenv("MAX_PRICE", "500.0")))
    min_price: float = field(default_factory=lambda: float(os.getenv("MIN_PRICE", "0.08")))
    min_history_sessions: int = field(
        default_factory=lambda: int(os.getenv("MIN_HISTORY_SESSIONS", "190"))
    )
    min_base_sessions: int = field(
        default_factory=lambda: int(os.getenv("MIN_BASE_SESSIONS", "84"))
    )
    preferred_base_sessions: int = field(
        default_factory=lambda: int(os.getenv("PREFERRED_BASE_SESSIONS", "126"))
    )
    max_base_sessions: int = field(
        default_factory=lambda: int(os.getenv("MAX_BASE_SESSIONS", "252"))
    )
    breakout_min_age: int = field(
        default_factory=lambda: int(os.getenv("BREAKOUT_MIN_AGE", "10"))
    )
    breakout_max_age: int = field(
        default_factory=lambda: int(os.getenv("BREAKOUT_MAX_AGE", "90"))
    )
    breakout_volume_mult: float = field(
        default_factory=lambda: float(os.getenv("BREAKOUT_VOLUME_MULT", "2.0"))
    )
    max_gain_since_breakout: float = field(
        default_factory=lambda: float(os.getenv("MAX_GAIN_SINCE_BREAKOUT", "0.50"))
    )
    max_distance_sma150: float = field(
        default_factory=lambda: float(os.getenv("MAX_DISTANCE_SMA150", "0.35"))
    )
    # SMA50 "a curvar para cima": inclinação positiva no curto prazo (lookback_a).
    # Por omissão NÃO exige aceleração (ver require_sma50_curl_accelerating) —
    # exigir a par o breakout confirmado da SMA150 E o instante exato da
    # inflexão da SMA50 no mesmo dia esvaziava o funil quase sempre: são dois
    # eventos raros que raramente coincidem no calendário. "Curvar para cima"
    # passa a significar inclinação positiva sustentada na janela, não o
    # ponto de inflexão isolado.
    sma50_curl_lookback: int = field(default_factory=lambda: int(os.getenv("SMA50_CURL_LOOKBACK", "10")))
    min_sma50_slope_pct: float = field(default_factory=lambda: float(os.getenv("MIN_SMA50_SLOPE_PCT", "0.0")))
    require_sma50_curl_accelerating: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_SMA50_CURL_ACCELERATING", "0") == "1"
    )
    # Pré-filtro de setor: só procura ações em setores cujo ETF representativo
    # tem a SMA50 diária a curvar para cima. Padrão ligado — é o pedido
    # original ("primeiro corre os ETFs... é aí que vamos à procura").
    require_sector_etf_curl: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_SECTOR_ETF_CURL", "1") == "1"
    )
    max_atr_contraction_ratio: float = field(
        default_factory=lambda: float(os.getenv("MAX_ATR_CONTRACTION_RATIO", "0.78"))
    )
    max_weekly_range_ratio: float = field(
        default_factory=lambda: float(os.getenv("MAX_WEEKLY_RANGE_RATIO", "0.82"))
    )
    max_base_drawdown: float = field(
        default_factory=lambda: float(os.getenv("MAX_BASE_DRAWDOWN", "0.62"))
    )
    max_recent_vertical_drop: float = field(
        default_factory=lambda: float(os.getenv("MAX_RECENT_VERTICAL_DROP", "0.45"))
    )
    min_avg_volume_20: int = field(
        default_factory=lambda: int(os.getenv("MIN_AVG_VOLUME_20", "100000"))
    )
    min_avg_dollar_volume_20: float = field(
        default_factory=lambda: float(os.getenv("MIN_AVG_DOLLAR_VOLUME_20", "75000"))
    )
    max_market_cap: float = field(
        default_factory=lambda: float(os.getenv("MAX_MARKET_CAP", "10000000000"))
    )
    preferred_float: float = field(
        default_factory=lambda: float(os.getenv("PREFERRED_FLOAT", "30000000"))
    )
    ideal_float: float = field(
        default_factory=lambda: float(os.getenv("IDEAL_FLOAT", "15000000"))
    )
    max_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESULTS", "10"))
    )
    max_candidates: int = field(
        default_factory=lambda: int(os.getenv("MAX_CANDIDATES", "0"))
    )
    breakout_volume_window: int = field(
        default_factory=lambda: int(os.getenv("BREAKOUT_VOLUME_WINDOW", "5"))
    )
    near_miss_limit: int = field(
        default_factory=lambda: int(os.getenv("NEAR_MISS_LIMIT", "5"))
    )
    request_pause: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_PAUSE", "0.12"))
    )
    cache_hours: int = field(
        default_factory=lambda: int(os.getenv("CACHE_HOURS", "18"))
    )
    strict_market_cap: bool = field(
        default_factory=lambda: os.getenv("STRICT_MARKET_CAP", "1") == "1"
    )
    strict_float: bool = field(
        default_factory=lambda: os.getenv("STRICT_FLOAT", "0") == "1"
    )
    min_quality_score: float = field(default_factory=lambda: float(os.getenv("MIN_QUALITY_SCORE", "68")))
    min_reward_risk: float = field(default_factory=lambda: float(os.getenv("MIN_REWARD_RISK", "2.0")))
    min_close_location: float = field(default_factory=lambda: float(os.getenv("MIN_CLOSE_LOCATION", "0.60")))
    max_failed_breakouts: int = field(default_factory=lambda: int(os.getenv("MAX_FAILED_BREAKOUTS", "3")))
    # MODO DESCOBERTA: o objetivo deixou de ser "só os melhores" e passou a
    # ser "candidatas plausíveis para avaliação manual". Sob este modo, os
    # critérios de QUALIDADE (força do setor, curvatura da própria SMA50,
    # CLV, relação potencial/risco, falsos breakouts) deixam de ELIMINAR a
    # candidata e passam a PENALIZAR o score, mantendo-a visível e sinalizada
    # — só quem decide manualmente é que corta. Os critérios ESTRUTURAIS
    # (histórico, liquidez, existência de um breakout e de uma base) mantêm-se
    # como filtros rígidos: sem eles não há sequer um setup para descrever.
    discovery_mode: bool = field(default_factory=lambda: os.getenv("DISCOVERY_MODE", "1") == "1")
    # Sob modo descoberta, este é o piso final (muito mais baixo que
    # MIN_QUALITY_SCORE) — continua a existir para não mostrar autêntico
    # lixo, mas deixou de ser o corte que apaga candidatas moderadas.
    discovery_min_score: float = field(default_factory=lambda: float(os.getenv("DISCOVERY_MIN_SCORE", "35")))

    results_json: Path = Path("cache/heartbeat_results.json")
    results_csv: Path = Path("cache/heartbeat_results.csv")
    last_run_json: Path = Path("cache/last_run.json")
    signal_journal_json: Path = Path("cache/signal_journal.json")
    ohlcv_dir: Path = Path("cache/ohlcv")
    sector_map_json: Path = Path("cache/universe_sectors.json")


@dataclass
class UniverseRow:
    ticker: str
    name: str
    price: float
    market_cap: Optional[float]
    reported_volume: Optional[float]
    sector: str = ""
    industry: str = ""


@dataclass
class MarketSnapshot:
    float_shares: Optional[float] = None


@dataclass
class TechnicalResult:
    ticker: str
    name: str
    price: float
    market_cap: Optional[float]
    float_shares: Optional[float]
    avg_volume_20: float
    avg_dollar_volume_20: float
    breakout_volume: float
    breakout_volume_multiple: float
    breakout_date: str
    breakout_age_sessions: int
    distance_sma150_pct: float
    gain_since_breakout_pct: float
    consolidation_sessions: int
    consolidation_months: float
    pattern: str
    support: float
    resistance: float
    ideal_entry_low: float
    ideal_entry_high: float
    invalidation: float
    confirmation: str
    atr_contraction_ratio: float
    weekly_range_ratio: float
    higher_lows_slope: float
    lower_highs_slope: float
    sma150_slope_pct_20d: float
    sma50_slope_pct: float
    sector_label: str
    sector_etf: str
    sector_etf_slope_pct: float
    soft_flags: list[str]
    close_location_value: float
    volume_dryup_ratio: float
    relative_strength_20d: float
    relative_strength_60d: float
    resistance_break_pct: float
    reward_risk: float
    failed_breakouts: int
    persistence_score: float
    market_regime: str
    setup_state: str
    probability_band: str
    market: MarketSnapshot
    catalysts: list[str]
    technical_score: float
    total_score: float


# ---------------------------------------------------------------------------
# HTTP and caching
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; HeartbeatStage2/1.2)",
        "Accept": "application/json,text/plain,*/*",
    }
)


def _fresh(path: Path, hours: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < hours * 3600


def get_json(url: str, *, headers: Optional[dict[str, str]] = None, retries: int = 3) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = SESSION.get(url, headers=headers, timeout=40)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Falha HTTP após {retries} tentativas: {url}: {last_error}")


def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text or text in {"N/A", "--", "-"}:
        return None
    multiplier = 1.0
    if text[-1:].upper() in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
        text = text[:-1]
    try:
        result = float(text) * multiplier
        return result if math.isfinite(result) else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Universe: Nasdaq-listed stocks only
# ---------------------------------------------------------------------------
NON_COMMON_NAME_RE = re.compile(
    r"\b(warrants?|rights?|units?|preferred|depositary|notes?)\b", re.IGNORECASE
)


def is_probably_common_stock(ticker: str, name: str) -> bool:
    """Exclude warrants/rights/units, abundant and toxic below $1.

    Nasdaq 5th-letter convention: W=warrant, R=right, U=unit. Name regex catches
    the rest (e.g. 4-letter warrant symbols listed with explicit names).
    """
    if len(ticker) == 5 and ticker[-1] in {"W", "R", "U"}:
        return False
    if NON_COMMON_NAME_RE.search(name or ""):
        return False
    return True


def fetch_nasdaq_universe(cfg: Config) -> list[UniverseRow]:
    url = (
        "https://api.nasdaq.com/api/screener/stocks"
        "?tableonly=true&limit=10000&offset=0&exchange=NASDAQ&download=true"
    )
    payload = get_json(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
        },
    )
    rows = (((payload or {}).get("data") or {}).get("rows") or [])
    universe: list[UniverseRow] = []
    for row in rows:
        ticker = str(row.get("symbol") or "").strip().upper()
        price = parse_money(row.get("lastsale"))
        cap = parse_money(row.get("marketCap"))
        volume = parse_money(row.get("volume"))
        if not ticker or price is None:
            continue
        if not is_probably_common_stock(ticker, str(row.get("name") or "")):
            continue
        if not (cfg.min_price <= price < cfg.max_price):
            continue
        if cfg.strict_market_cap and cap is not None and cap > cfg.max_market_cap:
            continue
        universe.append(
            UniverseRow(
                ticker=ticker,
                name=str(row.get("name") or ticker).strip(),
                price=price,
                market_cap=cap,
                reported_volume=volume,
                sector=str(row.get("sector") or "").strip(),
                industry=str(row.get("industry") or "").strip(),
            )
        )
    universe.sort(key=lambda item: (item.market_cap or float("inf"), -item.price))
    return universe if cfg.max_candidates <= 0 else universe[: cfg.max_candidates]


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------
def _normalise_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.rename(columns={c: c.lower() for c in df.columns}).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"]) & (df["volume"] >= 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df if len(df) >= 50 else None


def fetch_yahoo_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d&events=history"
    payload = get_json(url)
    result = (((payload or {}).get("chart") or {}).get("result") or [])
    if not result:
        return None
    block = result[0]
    timestamps = block.get("timestamp") or []
    quote = ((((block.get("indicators") or {}).get("quote") or [{}])[0]) or {})
    if not timestamps:
        return None
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    return _normalise_ohlcv(frame)


def fetch_stooq_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    symbol = ticker.lower().replace("-", ".")
    url = f"https://stooq.com/q/d/l/?s={symbol}.us&i=d"
    response = SESSION.get(url, timeout=40)
    response.raise_for_status()
    if "No data" in response.text or len(response.text) < 100:
        return None
    return _normalise_ohlcv(pd.read_csv(io.StringIO(response.text)))


def load_ohlcv(ticker: str, cfg: Config) -> Optional[pd.DataFrame]:
    cfg.ohlcv_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.ohlcv_dir / f"{ticker}.csv"
    if _fresh(path, cfg.cache_hours):
        try:
            return _normalise_ohlcv(pd.read_csv(path))
        except Exception:  # noqa: BLE001
            pass
    frame: Optional[pd.DataFrame] = None
    for loader in (fetch_yahoo_ohlcv, fetch_stooq_ohlcv):
        try:
            frame = loader(ticker)
            if frame is not None:
                break
        except Exception as exc:  # noqa: BLE001
            log.debug("%s OHLCV falhou em %s: %s", ticker, loader.__name__, exc)
    if frame is not None:
        frame.to_csv(path, index=False)
        return frame
    # Both remote sources failed: a stale cache beats no data at all.
    if path.exists():
        try:
            return _normalise_ohlcv(pd.read_csv(path))
        except Exception:  # noqa: BLE001
            pass
    return None


# ---------------------------------------------------------------------------
# Technical analysis
# ---------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def linear_slope(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 2 or np.all(~np.isfinite(arr)):
        return 0.0
    x = np.arange(len(arr), dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return 0.0
    return float(np.polyfit(x[mask], arr[mask], 1)[0])


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.set_index("date")
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    weekly["range_pct"] = (weekly["high"] - weekly["low"]) / weekly["close"].replace(0, np.nan)
    return weekly


def swing_slopes(base: pd.DataFrame) -> tuple[float, float]:
    # Compare successive 10-session extrema. Negative high slope + positive low
    # slope is the geometric signature of a triangle/VCP.
    blocks = max(4, min(10, len(base) // 10))
    tail = base.tail(blocks * 10)
    # Positional slicing, not np.array_split. np.array_split on a DataFrame
    # routes through the deprecated DataFrame.swapaxes and returns raw ndarrays
    # under pandas 3.x, which raises AttributeError here — an exception that the
    # caller's broad try/except swallows, turning every candidate into a silent
    # "error" and making a fully broken scan look like a legitimately empty one.
    bounds = np.linspace(0, len(tail), blocks + 1).astype(int)
    highs: list[float] = []
    lows: list[float] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        chunk = tail.iloc[start:stop]
        highs.append(float(chunk["high"].max()))
        lows.append(float(chunk["low"].min()))
    scale = max(float(base["close"].median()), 1e-6)
    return linear_slope(lows) / scale, linear_slope(highs) / scale


def detect_base(df: pd.DataFrame, breakout_idx: int, cfg: Config) -> Optional[dict[str, Any]]:
    end = max(0, breakout_idx - 1)
    best: Optional[dict[str, Any]] = None
    for sessions in [252, 210, 168, 147, 126, 105, 84]:
        if sessions < cfg.min_base_sessions or sessions > cfg.max_base_sessions:
            continue
        start = end - sessions + 1
        if start < 0:
            continue
        base = df.iloc[start : end + 1].copy()
        peak = float(base["high"].max())
        trough = float(base["low"].min())
        drawdown = 1 - trough / peak if peak else 1.0
        if drawdown > cfg.max_base_drawdown:
            continue

        atr = true_range(base).rolling(14).mean() / base["close"].replace(0, np.nan)
        if len(atr.dropna()) < 50:
            continue
        early_atr = float(atr.iloc[: max(20, len(atr) // 3)].median())
        late_atr = float(atr.iloc[-max(20, len(atr) // 3) :].median())
        atr_ratio = late_atr / early_atr if early_atr > 0 else 99.0

        weekly = resample_weekly(base)
        if len(weekly) < 12:
            continue
        split = max(4, len(weekly) // 3)
        early_weekly = float(weekly["range_pct"].iloc[:split].median())
        late_weekly = float(weekly["range_pct"].iloc[-split:].median())
        weekly_ratio = late_weekly / early_weekly if early_weekly > 0 else 99.0

        low_slope, high_slope = swing_slopes(base)
        compression_ok = (
            atr_ratio <= cfg.max_atr_contraction_ratio
            and weekly_ratio <= cfg.max_weekly_range_ratio
            and (low_slope > -0.0025)
            and (high_slope < 0.0025)
        )
        if not compression_ok:
            continue

        # Reject a recent waterfall masquerading as a base.
        recent_60 = base.tail(min(60, len(base)))
        recent_peak = float(recent_60["high"].max())
        recent_close = float(recent_60["close"].iloc[-1])
        vertical_drop = 1 - recent_close / recent_peak if recent_peak else 1.0
        if vertical_drop > cfg.max_recent_vertical_drop:
            continue

        if low_slope > 0.0008 and high_slope < -0.0008:
            pattern = "Triângulo simétrico / VCP"
        elif atr_ratio < 0.60 and weekly_ratio < 0.68:
            pattern = "Heartbeat / VCP apertado"
        elif abs(low_slope) < 0.0015 and abs(high_slope) < 0.0015:
            pattern = "Caixa horizontal comprimida"
        else:
            pattern = "Ondulação horizontal em compressão"

        support = float(base["low"].tail(30).quantile(0.20))
        resistance = float(base["high"].tail(60).quantile(0.90))
        quality = (1 - min(atr_ratio, 1)) + (1 - min(weekly_ratio, 1)) + min(sessions / 252, 1)
        candidate = {
            "sessions": sessions,
            "base": base,
            "atr_ratio": atr_ratio,
            "weekly_ratio": weekly_ratio,
            "low_slope": low_slope,
            "high_slope": high_slope,
            "pattern": pattern,
            "support": support,
            "resistance": resistance,
            "quality": quality,
        }
        if best is None or candidate["quality"] > best["quality"]:
            best = candidate
    return best


def find_sma150_breakout(df: pd.DataFrame, cfg: Config) -> tuple[Optional[dict[str, Any]], str]:
    """Find a recent SMA150 reclaim with volume confirmation near the crossing.

    The price crossing and the >=3x volume confirmation do not need to occur on
    exactly the same session. Institutional accumulation often appears shortly
    before or after the technical reclaim, so a configurable ±N-session window
    is used. The function returns both the result and a diagnostic reason.
    """
    work = df.copy()
    work["sma150"] = work["close"].rolling(150).mean()
    work["vol50"] = work["volume"].shift(1).rolling(50).mean()
    latest = len(work) - 1
    start = max(151, latest - cfg.breakout_max_age)
    end = latest - cfg.breakout_min_age
    if end < start:
        return None, "breakout_window"

    crosses: list[dict[str, Any]] = []
    for idx in range(start, end + 1):
        previous = work.iloc[idx - 1]
        row = work.iloc[idx]
        if not np.isfinite(row["sma150"]):
            continue
        crossed = previous["close"] <= previous["sma150"] and row["close"] > row["sma150"]
        if not crossed:
            continue

        lo = max(0, idx - cfg.breakout_volume_window)
        hi = min(latest, idx + cfg.breakout_volume_window)
        best_idx = None
        best_multiple = 0.0
        for vol_idx in range(lo, hi + 1):
            vol_row = work.iloc[vol_idx]
            if not np.isfinite(vol_row["vol50"]) or vol_row["vol50"] <= 0:
                continue
            multiple = float(vol_row["volume"] / vol_row["vol50"])
            if multiple > best_multiple:
                best_multiple = multiple
                best_idx = vol_idx
        crosses.append({"idx": idx, "volume_idx": best_idx, "volume_multiple": best_multiple})

    if not crosses:
        return None, "no_sma150_cross"

    volume_confirmed = [x for x in crosses if x["volume_multiple"] >= cfg.breakout_volume_mult]
    if not volume_confirmed:
        return None, "breakout_volume"

    chosen = volume_confirmed[-1]
    idx = chosen["idx"]
    volume_idx = chosen["volume_idx"] if chosen["volume_idx"] is not None else idx
    row = work.iloc[idx]
    volume_row = work.iloc[volume_idx]
    current = work.iloc[-1]
    if current["close"] <= current["sma150"]:
        return None, "below_sma150"
    gain = current["close"] / row["close"] - 1
    distance = current["close"] / current["sma150"] - 1
    if gain > cfg.max_gain_since_breakout:
        return None, "extended_gain"
    if distance > cfg.max_distance_sma150:
        return None, "extended_sma"
    slope20 = work["sma150"].iloc[-1] / work["sma150"].iloc[-21] - 1
    if slope20 < -0.035:
        return None, "falling_sma150"
    return {
        "idx": idx,
        "date": row["date"],
        "age": latest - idx,
        "price": float(row["close"]),
        "volume": float(volume_row["volume"]),
        "volume_idx": int(volume_idx),
        "volume_date": volume_row["date"],
        "volume_offset": int(volume_idx - idx),
        "volume_multiple": chosen["volume_multiple"],
        "gain": float(gain),
        "distance": float(distance),
        "sma150": float(current["sma150"]),
        "slope20": float(slope20),
    }, "ok"

BENCHMARKS: dict[str, pd.DataFrame] = {}
MARKET_REGIME: dict[str, Any] = {"label": "indeterminado", "score": 0.0}


def sma50_curling_up(df: pd.DataFrame, cfg: Config) -> Optional[dict[str, Any]]:
    """Critério adicional: a SMA50 diária está a curvar para cima.

    Definição operacional (não é só "a subir" — é "a curvar"):
    1. Inclinação recente positiva: SMA50 de hoje > SMA50 há `lookback` sessões.
    2. Aceleração: essa inclinação é maior do que a do segmento imediatamente
       anterior (mesma janela, deslocada). Sem este segundo teste, qualquer
       tendência de alta já madura passaria — "curvar" implica uma mudança de
       curvatura, não apenas coeficiente positivo.

    Devolve None se não há histórico suficiente (fail-safe: o chamador trata
    None como "não verificado" e rejeita, nunca como aprovação por omissão).
    """
    lookback = cfg.sma50_curl_lookback
    work = df.copy()
    work["sma50"] = work["close"].rolling(50).mean()
    if len(work) < 50 + 2 * lookback + 1:
        return None
    sma = work["sma50"]
    curr = float(sma.iloc[-1])
    prior = float(sma.iloc[-1 - lookback])
    earlier = float(sma.iloc[-1 - 2 * lookback])
    if not (np.isfinite(curr) and np.isfinite(prior) and np.isfinite(earlier)) or prior == 0 or earlier == 0:
        return None
    slope_recent = curr / prior - 1
    slope_prior = prior / earlier - 1
    accelerating = slope_recent > slope_prior
    curling_up = slope_recent > cfg.min_sma50_slope_pct and (
        accelerating if cfg.require_sma50_curl_accelerating else True
    )
    return {
        "sma50": curr,
        "slope_pct": slope_recent,
        "prior_slope_pct": slope_prior,
        "accelerating": accelerating,
        "curling_up": curling_up,
    }


# ---------------------------------------------------------------------------
# Filtro ETF de setor (pré-filtro): só se procura ações num setor cujo ETF
# representativo tem a SMA50 diária a curvar para cima. Primeiro corre-se o
# ETF, só depois as ações desse setor — na mesma definição de "curvar" já
# usada para ações individuais (sma50_curling_up), para não introduzir uma
# terceira noção divergente do mesmo conceito.
# ---------------------------------------------------------------------------
# (rótulo, ticker ETF, palavras-chave em minúsculas a procurar em "sector
# industry"). Primeira correspondência ganha — por isso categorias mais
# específicas (ex.: Semicondutores, Biotecnologia) vêm antes de categorias
# largas (ex.: Software/Tecnologia, Saúde) que as engoliriam.
SECTOR_ETF_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("Semicondutores", "SMH", ("semiconductor",)),
    ("Biotecnologia", "XBI", ("biotechnology", "biotech", "pharmaceutical", "drug manufactur", "therapeutic")),
    ("Petróleo e Gás", "XOP", ("oil", "petroleum", "drilling", "oil & gas", "oil and gas")),
    ("Água", "PHO", ("water",)),
    ("Utilities", "XLU", ("utilit",)),
    ("Energia", "XLE", ("energy", "coal")),
    ("Defesa e Aeroespacial", "ITA", ("defense", "aerospace", "military")),
    ("Robótica e Automação", "BOTZ", ("robot", "automation")),
    ("Data Centers e Cloud", "SKYY", ("data center", "cloud")),
    ("Computação Quântica", "QTUM", ("quantum",)),
    ("Software e Tecnologia", "IGV", ("software", "technology services", "internet", "computer services", "computer software")),
    ("Saúde", "XLV", ("health", "medical", "hospital")),
    ("Mineração e Metais", "XME", ("mining", "metal", "steel")),
    ("Financeiro", "XLF", ("bank", "financial", "insurance")),
    ("Imobiliário", "XLRE", ("real estate", "reit")),
    ("Telecomunicações", "IYZ", ("telecommunication", "wireless", "broadband")),
    ("Transportes", "XTN", ("transportation", "airline", "trucking", "shipping")),
    ("Consumo Discricionário", "XLY", ("retail", "apparel", "restaurant", "consumer discretionary", "auto")),
    ("Consumo Básico", "XLP", ("consumer non-durables", "food", "beverage", "household")),
    ("Industrial", "XLI", ("industrial", "machinery", "manufactur")),
]


def classify_sector(sector: Optional[str], industry: Optional[str]) -> Optional[tuple[str, str]]:
    """Devolve (rótulo, ticker ETF) para a primeira categoria cujas
    palavras-chave aparecem em "sector industry" (minúsculas). None se não
    houver correspondência — nesse caso não há ETF de referência para
    verificar, logo não há como aplicar o filtro a essa candidata."""
    text = f"{sector or ''} {industry or ''}".lower()
    if not text.strip():
        return None
    for label, etf, keywords in SECTOR_ETF_MAP:
        if any(kw in text for kw in keywords):
            return label, etf
    return None


SECTOR_STATUS: dict[str, dict[str, Any]] = {}


def compute_sector_status(cfg: Config) -> dict[str, dict[str, Any]]:
    """Corre sma50_curling_up sobre cada ETF de setor uma vez por execução.
    Chamado a partir de main() antes de varrer candidatas — os ETFs correm
    PRIMEIRO, exatamente como pedido."""
    status: dict[str, dict[str, Any]] = {}
    seen_etfs: dict[str, Optional[dict[str, Any]]] = {}
    for label, etf, _ in SECTOR_ETF_MAP:
        if etf not in seen_etfs:
            df = load_ohlcv(etf, cfg)
            seen_etfs[etf] = None if df is None else sma50_curling_up(df, cfg)
        curl = seen_etfs[etf]
        status[label] = {
            "etf": etf,
            "curling_up": bool(curl["curling_up"]) if curl else False,
            "slope_pct": 100 * curl["slope_pct"] if curl else None,
            "data_ok": curl is not None,
        }
    return status


def close_location(row: pd.Series) -> float:
    span = float(row["high"] - row["low"])
    return 0.5 if span <= 0 else float((row["close"] - row["low"]) / span)


def aligned_return(df: pd.DataFrame, benchmark: Optional[pd.DataFrame], sessions: int) -> float:
    if benchmark is None or len(df) < sessions + 1:
        return 0.0
    left = df[["date", "close"]].tail(sessions + 10).rename(columns={"close": "asset"})
    right = benchmark[["date", "close"]].tail(sessions + 10).rename(columns={"close": "bench"})
    joined = left.merge(right, on="date", how="inner").tail(sessions + 1)
    if len(joined) < max(10, sessions // 2):
        return 0.0
    ar = joined["asset"].iloc[-1] / joined["asset"].iloc[0] - 1
    br = joined["bench"].iloc[-1] / joined["bench"].iloc[0] - 1
    return float(ar - br)


def failed_breakout_count(df: pd.DataFrame, resistance: float, lookback: int = 126) -> int:
    w = df.tail(lookback).reset_index(drop=True)
    count = 0
    for i in range(1, max(1, len(w) - 3)):
        if w.loc[i, "high"] > resistance * 1.02 and w.loc[i, "close"] > resistance:
            future = w.loc[i + 1 : min(i + 3, len(w)-1), "close"]
            if len(future) and (future < resistance * 0.98).any():
                count += 1
    return count


def persistence_after_breakout(df: pd.DataFrame, idx: int) -> float:
    post = df.iloc[idx + 1 : min(len(df), idx + 6)]
    if post.empty:
        return 0.5
    sma = df["close"].rolling(150).mean().iloc[post.index]
    above = float((post["close"].values > sma.values).mean())
    giveback = max(0.0, 1 - float(post["close"].min() / max(df.iloc[idx]["close"], 1e-9)))
    return float(max(0.0, min(1.0, above - giveback)))


def compute_market_regime() -> dict[str, Any]:
    scores=[]
    labels=[]
    for ticker in ("QQQ", "IWM", "MDY"):
        df=BENCHMARKS.get(ticker)
        if df is None or len(df)<210:
            continue
        c=float(df["close"].iloc[-1]); sma50=float(df["close"].rolling(50).mean().iloc[-1]); sma200=float(df["close"].rolling(200).mean().iloc[-1])
        sc=(1 if c>sma50 else -1)+(1 if c>sma200 else -1)+(0.5 if sma50>sma200 else -0.5)
        scores.append(sc); labels.append(f"{ticker}:{'+' if sc>0 else '-'}")
    score=float(np.mean(scores)) if scores else 0.0
    label="favorável" if score>=1.0 else "adverso" if score<=-1.0 else "neutro"
    return {"label":label,"score":score,"detail":" ".join(labels)}


MID_CAP_RS_THRESHOLD = 2_000_000_000  # abaixo: IWM (small/micro); a partir daqui: MDY (S&P MidCap 400)


def choose_rs_benchmark(market_cap: Optional[float]) -> Optional[pd.DataFrame]:
    """IWM (Russell 2000) mede small/micro caps; comparar uma empresa de
    $8 mil milhões contra esse tape é a régua errada. Acima do limiar, usa
    MDY (S&P MidCap 400) — a referência correta para essa faixa."""
    if market_cap is not None and market_cap >= MID_CAP_RS_THRESHOLD:
        mid = BENCHMARKS.get("MDY")
        if mid is not None:
            return mid
    small = BENCHMARKS.get("IWM")
    if small is not None:
        return small
    return BENCHMARKS.get("QQQ")


def enhanced_metrics(base: dict[str, Any], breakout: dict[str, Any], df: pd.DataFrame, market_cap: Optional[float] = None) -> dict[str, Any]:
    vi=int(breakout.get("volume_idx", breakout["idx"]))
    volrow=df.iloc[vi]
    clv=close_location(volrow)
    pre=df.iloc[max(0, breakout["idx"]-20):breakout["idx"]]
    dry=float(pre["volume"].tail(10).mean()/max(pre["volume"].tail(50).mean(),1)) if len(pre)>=10 else 1.0
    rs_bench = choose_rs_benchmark(market_cap)
    rs20=aligned_return(df, rs_bench, 20)
    rs60=aligned_return(df, rs_bench, 60)
    resistance=float(base["resistance"]); current=float(df["close"].iloc[-1])
    rbreak=current/resistance-1 if resistance else 0.0
    support=min(float(base["support"]), float(breakout["sma150"]))
    invalid=min(support*0.97, float(breakout["sma150"])*0.96)
    risk=max(current-invalid, current*0.03)
    next_target=max(resistance*1.25, current*1.20)
    rr=max(0.0,(next_target-current)/risk)
    fails=failed_breakout_count(df.iloc[:breakout["idx"]+1], resistance)
    persist=persistence_after_breakout(df, breakout["idx"])
    state="RETEST" if current>=resistance*0.97 and current<=resistance*1.08 and breakout["age"]>=5 else "BREAKOUT"
    return {"clv":clv,"dry":dry,"rs20":rs20,"rs60":rs60,"rbreak":rbreak,"rr":rr,"fails":fails,"persist":persist,"state":state}


def technical_score(base: dict[str, Any], breakout: dict[str, Any], df: pd.DataFrame, cfg: Config, extra: Optional[dict[str, Any]]=None, market_cap: Optional[float]=None) -> float:
    extra=extra or enhanced_metrics(base, breakout, df, market_cap)
    sessions = base["sessions"]
    score=0.0
    score += 12 + 6 * min(max((sessions-cfg.min_base_sessions)/126,0),1)
    score += 10 * max(0,1-base["atr_ratio"]/cfg.max_atr_contraction_ratio)
    score += 7 * max(0,1-base["weekly_ratio"]/cfg.max_weekly_range_ratio)
    score += 9 * min(1,max(0,base["low_slope"]*180)+max(0,-base["high_slope"]*180)+0.25)
    score += 12 * min(1,max(0,(breakout["volume_multiple"]-2.5)/3.5))
    score += 7 if breakout["slope20"]>=0 else max(0,7+breakout["slope20"]*180)
    score += 6 * max(0,1-breakout["distance"]/cfg.max_distance_sma150)
    score += 7 * max(0,min(1,(extra["clv"]-0.45)/0.45))
    score += 5 * max(0,min(1,(1.15-extra["dry"])/0.55))
    score += 7 * max(0,min(1,(extra["rs20"]+0.05)/0.25))
    score += 5 * max(0,min(1,(extra["rs60"]+0.05)/0.35))
    score += 5 * max(0,min(1,extra["rr"]/3.0))
    score += 4 * extra["persist"]
    score -= min(9,extra["fails"]*3)
    score += 3 if MARKET_REGIME.get("label")=="favorável" else -3 if MARKET_REGIME.get("label")=="adverso" else 0
    return round(max(0.0,min(100.0,score)),1)


# ---------------------------------------------------------------------------
# Catalysts and final ranking
# ---------------------------------------------------------------------------
def infer_catalysts(company: UniverseRow) -> list[str]:
    text = f"{company.name} {company.sector} {company.industry}".lower()
    mapping = [
        (("biotech", "pharmaceutical", "therapeutic", "medical"), "Biotecnologia/FDA e dados clínicos"),
        (("semiconductor", "chip"), "Semicondutores"),
        (("artificial intelligence", " ai ", "software"), "IA/software"),
        (("defense", "aerospace"), "Defesa/aeroespacial"),
        (("energy", "oil", "gas", "solar", "battery"), "Energia"),
        (("quantum",), "Computação quântica"),
        (("robot", "automation"), "Robótica/automação"),
        (("data center", "cloud"), "Data centers/cloud"),
    ]
    catalysts = [label for keys, label in mapping if any(key in text for key in keys)]
    return catalysts or ["Resultados, contratos ou atualização estratégica da empresa"]


def market_adjustment(float_shares: Optional[float], market_cap: Optional[float], cfg: Config) -> float:
    score = 0.0
    if float_shares is not None:
        if float_shares <= cfg.ideal_float:
            score += 6
        elif float_shares <= cfg.preferred_float:
            score += 3
        elif cfg.strict_float:
            score -= 20
    if market_cap is not None:
        # O ponto de quebra é 1/3 do teto configurado, não um valor absoluto:
        # com o teto antigo de $150M isso dava exatamente $50M (o valor
        # histórico); com o teto agora em $10 mil milhões, um breakpoint fixo
        # em $50M penalizaria sistematicamente TODAS as mid caps, tornando o
        # alargamento inútil na prática (entravam no universo mas perdiam
        # sempre no ranking).
        small_tier = cfg.max_market_cap / 3
        score += 4 if market_cap <= small_tier else 2 if market_cap <= cfg.max_market_cap else -8
    return score


def analyse_candidate(company: UniverseRow, cfg: Config) -> tuple[Optional[TechnicalResult], str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"ticker": company.ticker, "name": company.name}
    df = load_ohlcv(company.ticker, cfg)
    if df is None: return None, "no_history", diagnostics
    diagnostics["history_sessions"] = len(df)
    if len(df) < cfg.min_history_sessions: return None, "short_history", diagnostics
    latest_price=float(df["close"].iloc[-1]); diagnostics["latest_price"]=latest_price
    if not (cfg.min_price <= latest_price < cfg.max_price): return None,"price_changed",diagnostics
    avg_volume_20=float(df["volume"].tail(20).mean()); avg_dollar_volume_20=float((df["close"]*df["volume"]).tail(20).mean())
    diagnostics.update(avg_volume_20=avg_volume_20,avg_dollar_volume_20=avg_dollar_volume_20)
    if avg_volume_20<cfg.min_avg_volume_20 or avg_dollar_volume_20<cfg.min_avg_dollar_volume_20: return None,"liquidity",diagnostics

    # A partir daqui: histórico, preço e liquidez já são estruturais e
    # continuam a eliminar (não há candidata sem eles). Os critérios abaixo
    # são de QUALIDADE — em modo descoberta, penalizam o score e ficam
    # sinalizados em soft_flags; a candidata continua visível.
    penalty = 0.0
    soft_flags: list[str] = []

    sector_info = classify_sector(company.sector, company.industry)
    if sector_info is not None:
        diagnostics["sector_label"], diagnostics["sector_etf"] = sector_info
    if cfg.require_sector_etf_curl:
        if sector_info is None:
            if not cfg.discovery_mode:
                return None, "sector_unmapped", diagnostics
            penalty += 10; soft_flags.append("Setor não classificado (sem ETF de referência)")
        else:
            label, etf = sector_info
            sector_status = SECTOR_STATUS.get(label)
            if sector_status is not None:
                diagnostics["sector_etf_slope_pct"] = sector_status.get("slope_pct")
            if sector_status is None or not sector_status.get("data_ok"):
                if not cfg.discovery_mode:
                    return None, "sector_etf_no_data", diagnostics
                penalty += 8; soft_flags.append("Sem dados do ETF de setor")
            elif not sector_status.get("curling_up"):
                if not cfg.discovery_mode:
                    return None, "sector_etf_not_curling", diagnostics
                penalty += 15; soft_flags.append(f"ETF do setor ({etf}) sem SMA50 a curvar para cima")

    breakout,reason=find_sma150_breakout(df,cfg)
    if breakout is None: return None,reason,diagnostics
    diagnostics.update(breakout_age=breakout["age"],breakout_volume_multiple=breakout["volume_multiple"],distance_sma150_pct=100*breakout["distance"],gain_since_breakout_pct=100*breakout["gain"])

    curl=sma50_curling_up(df,cfg)
    if curl is None:
        if not cfg.discovery_mode:
            return None,"sma50_no_history",diagnostics
        curl = {"slope_pct": 0.0, "prior_slope_pct": 0.0, "curling_up": False, "accelerating": False}
        penalty += 8; soft_flags.append("Histórico insuficiente para confirmar a SMA50")
    diagnostics.update(sma50_slope_pct=100*curl["slope_pct"],sma50_prior_slope_pct=100*curl["prior_slope_pct"])
    if not curl["curling_up"]:
        if not cfg.discovery_mode:
            return None,"sma50_not_curling_up",diagnostics
        penalty += 12; soft_flags.append("SMA50 da própria ação não está a curvar para cima")

    base=detect_base(df,breakout["idx"],cfg)
    if base is None: return None,"base_compression",diagnostics
    extra=enhanced_metrics(base,breakout,df,company.market_cap)
    diagnostics.update(base_sessions=base["sessions"],atr_ratio=base["atr_ratio"],weekly_ratio=base["weekly_ratio"],close_location=extra["clv"],relative_strength_20d=extra["rs20"],reward_risk=extra["rr"],failed_breakouts=extra["fails"])

    if extra["clv"] < cfg.min_close_location:
        if not cfg.discovery_mode:
            return None,"weak_close",diagnostics
        penalty += round(15 * (cfg.min_close_location - extra["clv"]) / cfg.min_close_location, 1)
        soft_flags.append(f"Fecho fraco no impulso (CLV {extra['clv']:.2f} < {cfg.min_close_location:.2f})")
    if extra["rr"] < cfg.min_reward_risk:
        if not cfg.discovery_mode:
            return None,"poor_reward_risk",diagnostics
        penalty += round(15 * max(0.0, (cfg.min_reward_risk - extra["rr"]) / cfg.min_reward_risk), 1)
        soft_flags.append(f"Assimetria risco/retorno abaixo do ideal (RR {extra['rr']:.1f}x < {cfg.min_reward_risk:.1f}x)")
    if extra["fails"] > cfg.max_failed_breakouts:
        if not cfg.discovery_mode:
            return None,"repeated_failures",diagnostics
        penalty += min(20, 6 * (extra["fails"] - cfg.max_failed_breakouts))
        soft_flags.append(f"{extra['fails']} falsos breakouts na base (limite {cfg.max_failed_breakouts})")

    snapshot=MarketSnapshot()
    tech_score=technical_score(base,breakout,df,cfg,extra)
    total_score=round(max(0,min(100,tech_score+market_adjustment(snapshot.float_shares,company.market_cap,cfg)-penalty)),1)
    min_score = cfg.discovery_min_score if cfg.discovery_mode else cfg.min_quality_score
    if total_score < min_score: return None,"quality_score",{**diagnostics,"quality_score":total_score}
    resistance=max(base["resistance"],breakout["price"]); support=min(base["support"],breakout["sma150"]); ideal_low=max(breakout["sma150"],resistance*0.97); ideal_high=resistance*1.04; invalidation=min(support*0.97,breakout["sma150"]*0.96)
    band=("Alta" if total_score>=82 and MARKET_REGIME.get("label")!="adverso" and not soft_flags
          else "Moderada-alta" if total_score>=74 and len(soft_flags)<=1
          else "Moderada" if total_score>=55
          else "Especulativa")
    confirmation=f"Fecho acima de ${resistance:.3f} com CLV ≥0,60, volume ≥2x, SMA50 a curvar para cima e manutenção acima da SMA150."
    return TechnicalResult(ticker=company.ticker,name=company.name,price=latest_price,market_cap=company.market_cap,float_shares=snapshot.float_shares,avg_volume_20=avg_volume_20,avg_dollar_volume_20=avg_dollar_volume_20,breakout_volume=breakout["volume"],breakout_volume_multiple=breakout["volume_multiple"],breakout_date=pd.Timestamp(breakout["date"]).strftime("%Y-%m-%d"),breakout_age_sessions=breakout["age"],distance_sma150_pct=100*breakout["distance"],gain_since_breakout_pct=100*breakout["gain"],consolidation_sessions=base["sessions"],consolidation_months=round(base["sessions"]/21,1),pattern=base["pattern"],support=support,resistance=resistance,ideal_entry_low=ideal_low,ideal_entry_high=ideal_high,invalidation=invalidation,confirmation=confirmation,atr_contraction_ratio=base["atr_ratio"],weekly_range_ratio=base["weekly_ratio"],higher_lows_slope=base["low_slope"],lower_highs_slope=base["high_slope"],sma150_slope_pct_20d=100*breakout["slope20"],sma50_slope_pct=100*curl["slope_pct"],sector_label=diagnostics.get("sector_label") or "Não classificado",sector_etf=diagnostics.get("sector_etf") or "n/d",sector_etf_slope_pct=diagnostics.get("sector_etf_slope_pct") if diagnostics.get("sector_etf_slope_pct") is not None else 0.0,soft_flags=soft_flags,close_location_value=extra["clv"],volume_dryup_ratio=extra["dry"],relative_strength_20d=100*extra["rs20"],relative_strength_60d=100*extra["rs60"],resistance_break_pct=100*extra["rbreak"],reward_risk=extra["rr"],failed_breakouts=extra["fails"],persistence_score=extra["persist"],market_regime=MARKET_REGIME.get("label","indeterminado"),setup_state=extra["state"],probability_band=band,market=snapshot,catalysts=infer_catalysts(company),technical_score=tech_score,total_score=total_score),"qualified",diagnostics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def human_number(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/d"
    for divisor, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(value) >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:.0f}"


def telegram_message(results: list[TechnicalResult], funnel: dict[str, int], near_misses: list[dict[str, Any]], cfg: Config, calibration: Optional[str] = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reason_labels = {
        "no_history": "sem histórico/dados",
        "short_history": "histórico insuficiente",
        "price_changed": "preço fora do intervalo",
        "liquidity": "liquidez insuficiente",
        "sector_unmapped": "setor não classificado (sem ETF de referência)",
        "sector_etf_no_data": "sem dados do ETF de setor",
        "sector_etf_not_curling": "ETF do setor sem SMA50 a curvar para cima",
        "breakout_window": "janela de breakout inválida",
        "no_sma150_cross": "sem cruzamento recente da SMA150",
        "breakout_volume": "volume <3x perto do breakout",
        "below_sma150": "voltou abaixo da SMA150",
        "extended_gain": "subida >50% desde breakout",
        "extended_sma": "demasiado afastada da SMA150",
        "falling_sma150": "SMA150 ainda muito descendente",
        "sma50_no_history": "histórico insuficiente para SMA50",
        "sma50_not_curling_up": "SMA50 não está a curvar para cima",
        "base_compression": "base/compressão insuficiente",
        "float": "float acima do limite",
        "weak_close": "fecho fraco no impulso",
        "poor_reward_risk": "assimetria risco/retorno insuficiente",
        "repeated_failures": "demasiados falsos breakouts",
        "quality_score": "score probabilístico insuficiente",
        "error": "erro técnico",
    }
    header = [
        "🫀 HEARTBEAT STAGE 2",
        f"Execução: {stamp}",
    ]
    if cfg.discovery_mode:
        header.append("Modo: DESCOBERTA — candidatas para avaliação manual, não recomendações. Sinalizadas com o que está fraco.")

    if cfg.require_sector_etf_curl and SECTOR_STATUS:
        approved = sorted(
            ((label, s) for label, s in SECTOR_STATUS.items() if s["curling_up"]),
            key=lambda x: x[1]["slope_pct"] or 0, reverse=True,
        )
        rejected = [(label, s) for label, s in SECTOR_STATUS.items() if not s["curling_up"]]
        header.extend(["", f"ETFs DE SETOR — {len(approved)}/{len(SECTOR_STATUS)} aprovados (SMA50 a curvar para cima)"])
        for label, s in approved:
            slope = f"{s['slope_pct']:+.2f}%" if s["slope_pct"] is not None else "n/d"
            header.append(f"✅ {label} ({s['etf']}): {slope}")
        if rejected:
            names = ", ".join(f"{label} ({s['etf']})" for label, s in rejected)
            header.append(f"❌ Sem sinal: {names}")

    header.extend([
        "",
        "FUNIL DE AUDITORIA",
        f"Universo NASDAQ elegível: {funnel.get('universe', 0)}",
        f"Empresas efetivamente processadas: {funnel.get('scanned', 0)}",
        f"Histórico válido: {funnel.get('history_ok', 0)}",
        f"Liquidez aprovada: {funnel.get('liquidity_ok', 0)}",
        f"Setor com ETF a curvar: {funnel.get('sector_ok', 0)}",
        f"Breakout SMA150 aprovado: {funnel.get('breakout_ok', 0)}",
        f"SMA50 a curvar para cima: {funnel.get('sma50_curl_ok', 0)}",
        f"Base/compressão aprovada: {funnel.get('base_ok', 0)}",
        f"Qualificadas: {len(results)}",
        f"Regime: {MARKET_REGIME.get('label','indeterminado')} ({MARKET_REGIME.get('detail','')})",
    ])
    exclusions = [(k, v) for k, v in funnel.items() if k.startswith("reason:") and k != "reason:qualified" and v]
    if exclusions:
        exclusions.sort(key=lambda item: item[1], reverse=True)
        header.append("")
        header.append("PRINCIPAIS EXCLUSÕES")
        for key, count in exclusions[:6]:
            reason = key.split(":", 1)[1]
            header.append(f"• {reason_labels.get(reason, reason)}: {count}")

    if not results:
        header.extend(["", "Nenhuma empresa cumpriu TODOS os critérios obrigatórios."])
    else:
        header.extend(["", f"RESULTADOS — {len(results)} setup(s), agrupados por setor"])
        by_sector: dict[str, list[TechnicalResult]] = {}
        for result in results:
            by_sector.setdefault(result.sector_label, []).append(result)
        # Setores ordenados pela MELHOR candidata dentro do grupo — em modo
        # descoberta um setor sem ETF a curvar ainda pode ter a ação mais
        # forte do dia (penalizada, não eliminada), e não deve ficar
        # enterrada no fim só por causa da ordenação do ETF.
        sector_order = sorted(
            by_sector.keys(),
            key=lambda label: max(r.total_score for r in by_sector[label]),
            reverse=True,
        )
        rank = 0
        for label in sector_order:
            group = by_sector[label]
            status = SECTOR_STATUS.get(label)
            etf_line = f"{group[0].sector_etf} {status['slope_pct']:+.2f}%" if status and status.get("slope_pct") is not None else group[0].sector_etf
            header.append(f"\n▶ SETOR: {label} (ETF {etf_line})")
            for result in group:
                rank += 1
                flags_line = ("\n⚠ " + " | ".join(result.soft_flags)) if result.soft_flags else ""
                header.append(
                    f"\n#{rank} {result.ticker} — {result.name}\n"
                    f"Score: {result.total_score:.1f}/100 | Probabilidade: {result.probability_band} | Estado: {result.setup_state}\n"
                    f"Preço: ${result.price:.3f} | Cap: ${human_number(result.market_cap)} | Float: {human_number(result.float_shares)}\n"
                    f"Vol. médio 20d: {human_number(result.avg_volume_20)} | Confirmação de volume: {human_number(result.breakout_volume)} ({result.breakout_volume_multiple:.1f}x)\n"
                    f"SMA150: breakout {result.breakout_date}, há {result.breakout_age_sessions} sessões | Distância {result.distance_sma150_pct:+.1f}%\n"
                    f"SMA50 (curvatura, {cfg.sma50_curl_lookback}d): {result.sma50_slope_pct:+.2f}%\n"
                    f"Base: {result.consolidation_months:.1f} meses | {result.pattern}\n"
                    f"RS20: {result.relative_strength_20d:+.1f}% | RS60: {result.relative_strength_60d:+.1f}% | CLV: {result.close_location_value:.2f} | RR: {result.reward_risk:.1f}x\n"
                    f"Suporte: ${result.support:.3f} | Resistência: ${result.resistance:.3f}\n"
                    f"Entrada: ${result.ideal_entry_low:.3f}–${result.ideal_entry_high:.3f} | Invalidação: ${result.invalidation:.3f}"
                    f"{flags_line}"
                )

    if near_misses:
        header.extend(["", "QUASE APROVADAS — para revisão, não são recomendações"])
        for item in near_misses:
            metrics = []
            if "breakout_volume_multiple" in item:
                metrics.append(f"vol {item['breakout_volume_multiple']:.1f}x")
            if "distance_sma150_pct" in item:
                metrics.append(f"dist SMA150 {item['distance_sma150_pct']:+.1f}%")
            if "sma50_slope_pct" in item:
                metrics.append(f"SMA50 {item['sma50_slope_pct']:+.2f}%")
            if "base_sessions" in item:
                metrics.append(f"base {item['base_sessions']}d")
            suffix = f" | {', '.join(metrics)}" if metrics else ""
            header.append(f"• {item['ticker']}: falhou {reason_labels.get(item['reason'], item['reason'])}{suffix}")

    if calibration:
        header.extend(["", calibration])
    header.append("\nA análise SEC/diluição continua manual.")
    header.append("Score = ranking técnico. Só é probabilidade depois de calibrado com amostra suficiente.")
    return "\n".join(header)

def tg_send(cfg: Config, text: str) -> None:
    if not cfg.tg_token or not cfg.tg_chat_id:
        log.info("Telegram não configurado.\n%s", text)
        return
    url = f"https://api.telegram.org/bot{cfg.tg_token}/sendMessage"
    for start in range(0, len(text), 3900):
        chunk = text[start : start + 3900]
        response = SESSION.post(
            url,
            json={"chat_id": cfg.tg_chat_id, "text": chunk, "disable_web_page_preview": True},
            timeout=30,
        )
        response.raise_for_status()


def serialise_result(result: TechnicalResult) -> dict[str, Any]:
    data = asdict(result)
    return data


def save_outputs(results: list[TechnicalResult], cfg: Config, universe_size: int, scanned: int, funnel: dict[str, int], near_misses: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": universe_size,
        "scanned": scanned,
        "qualified": len(results),
        "criteria": {
            "exchange": "NASDAQ",
            "price": f"{cfg.min_price} <= price < {cfg.max_price}",
            "minimum_base_sessions": cfg.min_base_sessions,
            "breakout_age_sessions": [cfg.breakout_min_age, cfg.breakout_max_age],
            "minimum_breakout_volume_multiple": cfg.breakout_volume_mult,
            "maximum_gain_since_breakout": cfg.max_gain_since_breakout,
        },
        "funnel": funnel,
        "near_misses": near_misses,
        "results": [serialise_result(result) for result in results],
    }
    cfg.results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for result in results:
        row = serialise_result(result)
        market = row.pop("market")
        row.update({f"market_{key}": value for key, value in market.items()})
        row["catalysts"] = "; ".join(row["catalysts"])
        row["soft_flags"] = "; ".join(row["soft_flags"])
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(cfg.results_csv, index=False)
    cfg.last_run_json.write_text(
        json.dumps(
            {
                "generated_at": payload["generated_at"],
                "universe_size": universe_size,
                "scanned": scanned,
                "qualified": len(results),
                "status": "ok",
            },
            indent=2,
        ),
        encoding="utf-8",
    )



JOURNAL_HORIZONS = (5, 10, 20, 40, 60)
CONTROL_REASONS = {"sma50_not_curling_up", "weak_close", "poor_reward_risk", "repeated_failures", "quality_score"}


def update_signal_journal(
    results: list[TechnicalResult],
    cfg: Config,
    near_misses: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Persist signals and evaluate realised 5/10/20/40/60-session outcomes.

    Statistical integrity rules:
    1. Returns are computed against the ADJUSTED close on the signal date taken
       from the same freshly fetched series ("entry_price_adjusted"), never the
       stored nominal price. Reverse splits — endemic below $1 — retroactively
       rescale the whole adjusted history; comparing a stale nominal price with
       adjusted future prices fabricates phantom gains and corrupts calibration.
    2. Signals whose data stream dies (delisting/suspension) are explicitly
       flagged "data_missing" instead of silently dropped; ignoring them is
       survivorship bias that inflates every hit rate.
    3. Final-gate rejects are logged as an unpublished control group so the
       lift of the quality gates can actually be measured.
    """
    try:
        journal = json.loads(cfg.signal_journal_json.read_text(encoding="utf-8")) if cfg.signal_journal_json.exists() else []
    except Exception:
        journal = []
    today_date = datetime.now(timezone.utc).date()
    today = today_date.isoformat()
    keys = {(x.get("ticker"), x.get("signal_date")) for x in journal}
    for r in results:
        if (r.ticker, today) not in keys:
            journal.append({
                "ticker": r.ticker, "signal_date": today, "signal_price": r.price,
                "score": r.total_score, "band": r.probability_band, "state": r.setup_state,
                "published": True, "outcomes": {},
            })
            keys.add((r.ticker, today))
    for item in near_misses or []:
        if item.get("reason") not in CONTROL_REASONS:
            continue
        ticker = item.get("ticker")
        price = item.get("latest_price")
        if not ticker or price is None or (ticker, today) in keys:
            continue
        journal.append({
            "ticker": ticker, "signal_date": today, "signal_price": float(price),
            "score": item.get("quality_score"), "band": None, "state": None,
            "published": False, "reject_reason": item["reason"], "outcomes": {},
        })
        keys.add((ticker, today))

    def _pending(entry: dict[str, Any]) -> bool:
        outcomes = entry.get("outcomes", {})
        return any(str(h) not in outcomes for h in JOURNAL_HORIZONS)

    unresolved = [x for x in journal if _pending(x)]
    for item in unresolved[-200:]:
        sig = datetime.fromisoformat(item["signal_date"]).date()
        age_days = (today_date - sig).days
        df = load_ohlcv(item["ticker"], cfg)
        outcomes = item.setdefault("outcomes", {})
        if df is None:
            if age_days > 120:
                for h in JOURNAL_HORIZONS:
                    outcomes.setdefault(str(h), {"data_missing": True})
                item["data_missing_note"] = "sem dados >120d — possível delisting/suspensão"
            continue
        dates = pd.to_datetime(df["date"]).dt.date
        idxs = np.where(dates >= sig)[0]
        if len(idxs) == 0:
            continue
        i = int(idxs[0])
        entry = float(df.iloc[i]["close"])  # adjusted series: reverse-split safe
        item["entry_price_adjusted"] = round(entry, 6)
        last_date = dates.iloc[-1]
        stream_dead = (today_date - last_date).days > 21
        for h in JOURNAL_HORIZONS:
            key = str(h)
            if key in outcomes:
                continue
            if i + h < len(df):
                segment = df.iloc[i + 1 : i + h + 1]  # post-entry window only
                outcomes[key] = {
                    "return_pct": round(100 * (float(df.iloc[i + h]["close"]) / entry - 1), 2),
                    "max_gain_pct": round(100 * (float(segment["high"].max()) / entry - 1), 2),
                    "max_drawdown_pct": round(100 * (float(segment["low"].min()) / entry - 1), 2),
                }
            elif stream_dead:
                outcomes[key] = {
                    "data_missing": True,
                    "truncated_return_pct": round(100 * (float(df.iloc[-1]["close"]) / entry - 1), 2),
                }
    cfg.signal_journal_json.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct for small samples, unlike the normal
    approximation, which is invalid at the sample sizes this journal will hold
    for months."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def calibration_summary(cfg: Config, horizon: int = 20, min_n: int = 20) -> Optional[str]:
    """Empirical hit rates (published vs control) with Wilson 95% CIs.

    data_missing signals count as failures: assuming the worst for dead data
    streams is the only assumption that does not flatter the model.
    """
    try:
        journal = json.loads(cfg.signal_journal_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    groups: dict[str, list[float]] = {"publicados": [], "controlo": []}
    missing = {"publicados": 0, "controlo": 0}
    for item in journal:
        outcome = (item.get("outcomes") or {}).get(str(horizon))
        if not outcome:
            continue
        group = "publicados" if item.get("published", True) else "controlo"
        if outcome.get("data_missing"):
            missing[group] += 1
            groups[group].append(-100.0)  # worst-case assumption
        else:
            groups[group].append(float(outcome["return_pct"]))
    if len(groups["publicados"]) < min_n:
        resolved = len(groups["publicados"])
        return (
            f"CALIBRAÇÃO ({horizon} sessões): amostra insuficiente "
            f"({resolved}/{min_n} sinais resolvidos). O score continua a ser um ranking, não uma probabilidade."
        )
    lines = [f"CALIBRAÇÃO EMPÍRICA — retorno a {horizon} sessões"]
    for group, values in groups.items():
        if not values:
            continue
        n = len(values)
        wins = sum(1 for v in values if v > 0)
        lo, hi = wilson_ci(wins, n)
        median = float(np.median(values))
        note = f", {missing[group]} sem dados (contados como perda)" if missing[group] else ""
        lines.append(
            f"• {group}: n={n}, %positivos {100*wins/n:.0f}% (IC95 {100*lo:.0f}–{100*hi:.0f}%), mediana {median:+.1f}%{note}"
        )
    return "\n".join(lines)

def main() -> None:
    global MARKET_REGIME, SECTOR_STATUS
    cfg = Config()
    cfg.ohlcv_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in ("QQQ", "IWM", "MDY"):
        try:
            BENCHMARKS[benchmark] = load_ohlcv(benchmark, cfg)
        except Exception as exc:
            log.warning("Benchmark %s indisponível: %s", benchmark, exc)
    MARKET_REGIME = compute_market_regime()
    log.info("Regime de mercado: %s", MARKET_REGIME)
    if cfg.require_sector_etf_curl:
        log.info("A correr ETFs de setor primeiro…")
        SECTOR_STATUS = compute_sector_status(cfg)
        approved = [f"{label} ({s['etf']} {s['slope_pct']:+.2f}%)" for label, s in SECTOR_STATUS.items() if s["curling_up"]]
        log.info("Setores aprovados (%d/%d): %s", len(approved), len(SECTOR_STATUS), ", ".join(approved) or "nenhum")
    log.info("A construir universo NASDAQ $%.2f–$%.2f, cap ≤ $%.0f…", cfg.min_price, cfg.max_price, cfg.max_market_cap)
    universe = fetch_nasdaq_universe(cfg)
    log.info("Universo inicial: %d empresas", len(universe))
    try:
        sector_map = {row.ticker: {"sector": row.sector, "industry": row.industry} for row in universe}
        cfg.sector_map_json.write_text(json.dumps(sector_map, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("Não foi possível gravar cache/universe_sectors.json: %s", exc)

    funnel: dict[str, int] = {"universe": len(universe), "scanned": 0}
    results: list[TechnicalResult] = []
    near_misses: list[dict[str, Any]] = []
    scanned = 0
    for company in universe:
        scanned += 1
        funnel["scanned"] = scanned
        try:
            result, reason, diagnostics = analyse_candidate(company, cfg)
            funnel[f"reason:{reason}"] = funnel.get(f"reason:{reason}", 0) + 1
            if reason not in {"no_history", "short_history", "error"}:
                funnel["history_ok"] = funnel.get("history_ok", 0) + 1
            if reason not in {"no_history", "short_history", "price_changed", "liquidity", "error"}:
                funnel["liquidity_ok"] = funnel.get("liquidity_ok", 0) + 1
            post_sector = {"sma50_no_history", "sma50_not_curling_up", "base_compression", "float",
                            "weak_close", "poor_reward_risk", "repeated_failures", "quality_score", "qualified",
                            "breakout_window", "no_sma150_cross", "breakout_volume", "below_sma150",
                            "extended_gain", "extended_sma", "falling_sma150"}
            post_breakout = {"sma50_no_history", "sma50_not_curling_up", "base_compression", "float",
                              "weak_close", "poor_reward_risk", "repeated_failures", "quality_score", "qualified"}
            post_sma50_curl = {"base_compression", "float", "weak_close", "poor_reward_risk",
                                "repeated_failures", "quality_score", "qualified"}
            post_base = {"float", "weak_close", "poor_reward_risk", "repeated_failures", "quality_score", "qualified"}
            if reason in post_sector:
                funnel["sector_ok"] = funnel.get("sector_ok", 0) + 1
            if reason in post_breakout:
                funnel["breakout_ok"] = funnel.get("breakout_ok", 0) + 1
            if reason in post_sma50_curl:
                funnel["sma50_curl_ok"] = funnel.get("sma50_curl_ok", 0) + 1
            if reason in post_base:
                funnel["base_ok"] = funnel.get("base_ok", 0) + 1
            if result is not None:
                results.append(result)
                log.info("QUALIFICADA %s — %.1f/100", result.ticker, result.total_score)
            elif reason in {"breakout_volume", "below_sma150", "extended_gain", "extended_sma", "falling_sma150", "sma50_not_curling_up", "base_compression", "weak_close", "poor_reward_risk", "repeated_failures", "quality_score"}:
                diagnostics["reason"] = reason
                # Near misses are ranked by how far they progressed in the funnel.
                diagnostics["progress"] = {
                    "breakout_volume": 3,
                    "below_sma150": 4,
                    "extended_gain": 4,
                    "extended_sma": 4,
                    "falling_sma150": 4,
                    "sma50_not_curling_up": 5,
                    "base_compression": 6,
                    "weak_close": 7, "poor_reward_risk": 7, "repeated_failures": 7, "quality_score": 8,
                }.get(reason, 0)
                near_misses.append(diagnostics)
        except Exception as exc:  # noqa: BLE001
            funnel["reason:error"] = funnel.get("reason:error", 0) + 1
            log.warning("%s falhou: %s", company.ticker, exc)
        if cfg.request_pause:
            time.sleep(cfg.request_pause)

    # A broken scan and an empty scan produce the same Telegram message unless
    # the error rate is checked. Fail loudly instead of implying "no setups".
    errors = funnel.get("reason:error", 0)
    error_rate = errors / max(scanned, 1)
    if scanned >= 20 and error_rate > 0.25:
        alert = (
            "🚨 HEARTBEAT — VARRIMENTO INVÁLIDO\n"
            f"{errors}/{scanned} candidatas falharam com erro técnico ({100*error_rate:.0f}%).\n"
            "O resultado de hoje NÃO é 'nenhum setup': é um varrimento corrompido. Ver cache/scanner.log."
        )
        log.error("Taxa de erro %.0f%% — varrimento inválido.", 100 * error_rate)
        tg_send(cfg, alert)

    results.sort(key=lambda item: (item.total_score, item.technical_score), reverse=True)
    results = results[: cfg.max_results]
    near_misses.sort(key=lambda x: (x.get("progress", 0), x.get("breakout_volume_multiple", 0), x.get("base_sessions", 0)), reverse=True)
    control_pool = list(near_misses)  # full reject pool: the control group must not be truncated
    near_misses = near_misses[: cfg.near_miss_limit]
    save_outputs(results, cfg, len(universe), scanned, funnel, near_misses)
    update_signal_journal(results, cfg, control_pool)
    calibration = calibration_summary(cfg)
    message = telegram_message(results, funnel, near_misses, cfg, calibration)
    tg_send(cfg, message)
    log.info("Concluído: %d/%d qualificadas", len(results), scanned)


if __name__ == "__main__":
    main()
