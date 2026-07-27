"""Heartbeat Stage 2 Scanner.

Strict NASDAQ sub-$1 scanner designed to detect prolonged volatility contraction
bases shortly after a high-volume reclaim of the 150-day simple moving average.

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

    max_price: float = field(default_factory=lambda: float(os.getenv("MAX_PRICE", "1.0")))
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
        default_factory=lambda: int(os.getenv("BREAKOUT_MAX_AGE", "40"))
    )
    breakout_volume_mult: float = field(
        default_factory=lambda: float(os.getenv("BREAKOUT_VOLUME_MULT", "3.0"))
    )
    max_gain_since_breakout: float = field(
        default_factory=lambda: float(os.getenv("MAX_GAIN_SINCE_BREAKOUT", "0.50"))
    )
    max_distance_sma150: float = field(
        default_factory=lambda: float(os.getenv("MAX_DISTANCE_SMA150", "0.35"))
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
        default_factory=lambda: float(os.getenv("MAX_MARKET_CAP", "150000000"))
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
        default_factory=lambda: int(os.getenv("MAX_CANDIDATES", "500"))
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

    results_json: Path = Path("cache/heartbeat_results.json")
    results_csv: Path = Path("cache/heartbeat_results.csv")
    last_run_json: Path = Path("cache/last_run.json")
    ohlcv_dir: Path = Path("cache/ohlcv")


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
        "User-Agent": "Mozilla/5.0 (compatible; HeartbeatStage2/1.0)",
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
    return universe[: cfg.max_candidates]


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
    chunks = np.array_split(base.tail(blocks * 10), blocks)
    highs = [chunk["high"].max() for chunk in chunks if not chunk.empty]
    lows = [chunk["low"].min() for chunk in chunks if not chunk.empty]
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


def find_sma150_breakout(df: pd.DataFrame, cfg: Config) -> Optional[dict[str, Any]]:
    work = df.copy()
    work["sma150"] = work["close"].rolling(150).mean()
    work["vol50"] = work["volume"].shift(1).rolling(50).mean()
    latest = len(work) - 1
    start = max(151, latest - cfg.breakout_max_age)
    end = latest - cfg.breakout_min_age
    if end < start:
        return None

    possibilities: list[dict[str, Any]] = []
    for idx in range(start, end + 1):
        previous = work.iloc[idx - 1]
        row = work.iloc[idx]
        if not np.isfinite(row["sma150"]) or not np.isfinite(row["vol50"]):
            continue
        crossed = previous["close"] <= previous["sma150"] and row["close"] > row["sma150"]
        volume_multiple = row["volume"] / row["vol50"] if row["vol50"] > 0 else 0
        if crossed and volume_multiple >= cfg.breakout_volume_mult:
            possibilities.append({"idx": idx, "volume_multiple": float(volume_multiple)})
    if not possibilities:
        return None
    # Prefer the most recent qualifying reclaim.
    chosen = possibilities[-1]
    idx = chosen["idx"]
    row = work.iloc[idx]
    current = work.iloc[-1]
    if current["close"] <= current["sma150"]:
        return None
    gain = current["close"] / row["close"] - 1
    distance = current["close"] / current["sma150"] - 1
    if gain > cfg.max_gain_since_breakout or distance > cfg.max_distance_sma150:
        return None
    slope20 = work["sma150"].iloc[-1] / work["sma150"].iloc[-21] - 1
    if slope20 < -0.035:  # sharply falling SMA150 is not Stage 2
        return None
    return {
        "idx": idx,
        "date": row["date"],
        "age": latest - idx,
        "price": float(row["close"]),
        "volume": float(row["volume"]),
        "volume_multiple": chosen["volume_multiple"],
        "gain": float(gain),
        "distance": float(distance),
        "sma150": float(current["sma150"]),
        "slope20": float(slope20),
    }


def technical_score(base: dict[str, Any], breakout: dict[str, Any], df: pd.DataFrame, cfg: Config) -> float:
    sessions = base["sessions"]
    base_points = 12 + 8 * min(max((sessions - cfg.min_base_sessions) / 126, 0), 1)
    atr_points = 15 * max(0, 1 - base["atr_ratio"] / cfg.max_atr_contraction_ratio)
    weekly_points = 10 * max(0, 1 - base["weekly_ratio"] / cfg.max_weekly_range_ratio)
    geometry_points = 10 * min(1, max(0, base["low_slope"] * 180) + max(0, -base["high_slope"] * 180) + 0.25)
    vol_points = 15 * min((breakout["volume_multiple"] - 3) / 4 + 0.55, 1)
    sma_points = 8 if breakout["slope20"] >= 0 else max(0, 8 + breakout["slope20"] * 200)
    proximity_points = 10 * max(0, 1 - breakout["distance"] / cfg.max_distance_sma150)
    avg_volume = float(df["volume"].tail(20).mean())
    liquidity_points = 5 * min(avg_volume / 1_000_000, 1)
    return round(min(100.0, base_points + atr_points + weekly_points + geometry_points + vol_points + sma_points + proximity_points + liquidity_points), 1)


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
        score += 4 if market_cap <= 50_000_000 else 2 if market_cap <= cfg.max_market_cap else -8
    return score


def analyse_candidate(company: UniverseRow, cfg: Config) -> Optional[TechnicalResult]:
    df = load_ohlcv(company.ticker, cfg)
    if df is None or len(df) < cfg.min_history_sessions:
        return None
    latest_price = float(df["close"].iloc[-1])
    if not (cfg.min_price <= latest_price < cfg.max_price):
        return None

    avg_volume_20 = float(df["volume"].tail(20).mean())
    avg_dollar_volume_20 = float((df["close"] * df["volume"]).tail(20).mean())
    if avg_volume_20 < cfg.min_avg_volume_20 or avg_dollar_volume_20 < cfg.min_avg_dollar_volume_20:
        return None

    breakout = find_sma150_breakout(df, cfg)
    if breakout is None:
        return None
    base = detect_base(df, breakout["idx"], cfg)
    if base is None:
        return None

    snapshot = MarketSnapshot()
    if cfg.strict_float and snapshot.float_shares is not None and snapshot.float_shares > cfg.preferred_float:
        return None

    tech_score = technical_score(base, breakout, df, cfg)
    total_score = round(max(0.0, min(100.0, tech_score + market_adjustment(snapshot.float_shares, company.market_cap, cfg))), 1)

    resistance = max(base["resistance"], breakout["price"])
    support = min(base["support"], breakout["sma150"])
    ideal_low = max(breakout["sma150"], resistance * 0.97)
    ideal_high = resistance * 1.04
    invalidation = min(support * 0.97, breakout["sma150"] * 0.96)
    confirmation = f"Fecho acima de ${resistance:.3f} com volume ≥2x média de 50 sessões e manutenção acima da SMA150."

    return TechnicalResult(
        ticker=company.ticker,
        name=company.name,
        price=latest_price,
        market_cap=company.market_cap,
        float_shares=snapshot.float_shares,
        avg_volume_20=avg_volume_20,
        avg_dollar_volume_20=avg_dollar_volume_20,
        breakout_volume=breakout["volume"],
        breakout_volume_multiple=breakout["volume_multiple"],
        breakout_date=pd.Timestamp(breakout["date"]).strftime("%Y-%m-%d"),
        breakout_age_sessions=breakout["age"],
        distance_sma150_pct=100 * breakout["distance"],
        gain_since_breakout_pct=100 * breakout["gain"],
        consolidation_sessions=base["sessions"],
        consolidation_months=round(base["sessions"] / 21, 1),
        pattern=base["pattern"],
        support=support,
        resistance=resistance,
        ideal_entry_low=ideal_low,
        ideal_entry_high=ideal_high,
        invalidation=invalidation,
        confirmation=confirmation,
        atr_contraction_ratio=base["atr_ratio"],
        weekly_range_ratio=base["weekly_ratio"],
        higher_lows_slope=base["low_slope"],
        lower_highs_slope=base["high_slope"],
        sma150_slope_pct_20d=100 * breakout["slope20"],
        market=snapshot,
        catalysts=infer_catalysts(company),
        technical_score=tech_score,
        total_score=total_score,
    )


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


def telegram_message(results: list[TechnicalResult]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not results:
        return (
            "🫀 HEARTBEAT STAGE 2\n"
            f"Execução: {stamp}\n\n"
            "Nenhuma empresa cumpriu TODOS os critérios obrigatórios. "
            "O scanner não adicionou candidatos fracos para preencher a lista."
        )
    blocks = [f"🫀 HEARTBEAT STAGE 2 — {len(results)} setup(s)\nExecução: {stamp}"]
    for rank, result in enumerate(results, 1):
        m = result.market
        blocks.append(
            f"\n#{rank} {result.ticker} — {result.name}\n"
            f"Score: {result.total_score:.1f}/100 (técnico {result.technical_score:.1f})\n"
            f"Preço: ${result.price:.3f} | Cap: ${human_number(result.market_cap)} | Float: {human_number(result.float_shares)}\n"
            f"Vol. médio 20d: {human_number(result.avg_volume_20)} | Breakout: {human_number(result.breakout_volume)} ({result.breakout_volume_multiple:.1f}x)\n"
            f"SMA150: breakout {result.breakout_date}, há {result.breakout_age_sessions} sessões | Distância {result.distance_sma150_pct:+.1f}%\n"
            f"Base: {result.consolidation_months:.1f} meses | {result.pattern}\n"
            f"ATR compressão: {result.atr_contraction_ratio:.2f} | Range semanal: {result.weekly_range_ratio:.2f}\n"
            f"Suporte: ${result.support:.3f} | Resistência: ${result.resistance:.3f}\n"
            f"Entrada ideal: ${result.ideal_entry_low:.3f}–${result.ideal_entry_high:.3f} | Invalidação: ${result.invalidation:.3f}\n"
            f"Catalisadores: {', '.join(result.catalysts)}\n"
            f"Confirmação: {result.confirmation}"
        )
    blocks.append("\nAnálise financeira, diluição e filings SEC não são avaliados automaticamente.")
    return "\n".join(blocks)


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


def save_outputs(results: list[TechnicalResult], cfg: Config, universe_size: int, scanned: int) -> None:
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
        "results": [serialise_result(result) for result in results],
    }
    cfg.results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for result in results:
        row = serialise_result(result)
        market = row.pop("market")
        row.update({f"market_{key}": value for key, value in market.items()})
        row["catalysts"] = "; ".join(row["catalysts"])
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


def main() -> None:
    cfg = Config()
    cfg.ohlcv_dir.mkdir(parents=True, exist_ok=True)
    log.info("A construir universo NASDAQ sub-$%.2f…", cfg.max_price)
    universe = fetch_nasdaq_universe(cfg)
    log.info("Universo inicial: %d empresas", len(universe))

    results: list[TechnicalResult] = []
    scanned = 0
    for company in universe:
        scanned += 1
        try:
            result = analyse_candidate(company, cfg)
            if result is not None:
                results.append(result)
                log.info("QUALIFICADA %s — %.1f/100", result.ticker, result.total_score)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s falhou: %s", company.ticker, exc)
        if cfg.request_pause:
            time.sleep(cfg.request_pause)

    results.sort(key=lambda item: (item.total_score, item.technical_score), reverse=True)
    results = results[: cfg.max_results]
    save_outputs(results, cfg, len(universe), scanned)
    message = telegram_message(results)
    tg_send(cfg, message)
    log.info("Concluído: %d/%d qualificadas", len(results), scanned)


if __name__ == "__main__":
    main()
