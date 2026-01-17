import os, io, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

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
MAX_PX = float(os.environ.get("MAX_PX", "35"))
MIN_DV20 = float(os.environ.get("MIN_DV20", "3000000"))
MAX_DV20 = float(os.environ.get("MAX_DV20", "120000000"))
MIN_SV20 = float(os.environ.get("MIN_SV20", "600000"))

BBZ_GATE = float(os.environ.get("BBZ_GATE", "-0.7"))
ATRPCTL_GATE = float(os.environ.get("ATRPCTL_GATE", "0.45"))
BASE_DD_MAX = float(os.environ.get("BASE_DD_MAX", "0.55"))
CONTRACTION_MAX = float(os.environ.get("CONTRACTION_MAX", "0.85"))
DRYUP_MAX = float(os.environ.get("DRYUP_MAX", "0.95"))

VOL_CONFIRM_MULT = float(os.environ.get("VOL_CONFIRM_MULT", "1.3"))
MAX_GAP_UP = float(os.environ.get("MAX_GAP_UP", "1.12"))

SLEEP_EVERY = int(os.environ.get("SLEEP_EVERY", "60"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "1.0"))


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


def ensure_dirs() -> None:
    os.makedirs("cache/ohlcv", exist_ok=True)


def cache_path(t: str) -> str:
    return f"cache/ohlcv/{t}.csv"


def build_ohlcv_url(ticker: str) -> str:
    sym = ticker.lower().replace("-", ".")
    if ".us" in OHLCV_FMT.lower():
        return OHLCV_FMT.format(symbol=sym)
    return OHLCV_FMT.format(symbol=f"{sym}.us")


def read_cached_dv20(ticker: str) -> float | None:
    path = cache_path(ticker)
    if not os.path.exists(path):
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


def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    ensure_dirs()
    path = cache_path(ticker)

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
    need = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        raise RuntimeError("BAD_FORMAT")

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


def base_scan(df: pd.DataFrame) -> tuple[bool, float, float, float, float]:
    # returns (ok, high_base, low_base, dd, contr)
    if len(df) < 260:
        return (False, np.nan, np.nan, np.nan, np.nan)

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
            cand = (score, highb, lowb, dd, contr)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return (False, np.nan, np.nan, np.nan, np.nan)

    _, highb, lowb, dd, contr = best
    return (True, float(highb), float(lowb), float(dd), float(contr))


def dryup_ratio(df: pd.DataFrame) -> float:
    v10 = float(df["volume"].iloc[-10:].mean())
    v60 = float(df["volume"].iloc[-60:].mean())
    if v60 <= 0:
        return np.nan
    return v10 / v60


def score_candidate(bbz: float, atrpctl: float, dd: float, contr: float, dry: float, dv20: float) -> float:
    # more negative bbz = better; lower atrpctl = better; lower dd/contr/dry = better; some liquidity preference
    # all inputs verifiable
    s = 0.0
    s += (-bbz) * 2.0
    s += (0.6 - atrpctl) * 1.5
    s += (BASE_DD_MAX - dd) * 1.2
    s += (CONTRACTION_MAX - contr) * 1.0
    s += (DRYUP_MAX - dry) * 0.8
    s += min(dv20 / 50_000_000.0, 1.0) * 0.6
    return float(s)


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ensure_dirs()

    universe = list(set(get_universe_from_holdings(IWC_URL) + get_universe_from_holdings(IWM_URL) + get_universe_from_holdings(IJR_URL)))

    # cache-first pre-ranking by dv20 (no network)
    pool = universe[:min(CANDIDATE_POOL, len(universe))]
    scored = []
    unscored = []
    for t in pool:
        dv = read_cached_dv20(t)
        if dv is None:
            unscored.append(t)
        else:
            scored.append((t, dv))

    scored.sort(key=lambda x: x[1], reverse=True)
    ordered = [t for t, _ in scored] + unscored
    tickers = ordered[:MAX_TICKERS]

    hits_limited = False
    no_data = 0

    watch = []
    near = []
    execs = []

    hist_ok = liq_ok = comp_ok = base_ok = dry_ok = 0

    for i, t in enumerate(tickers):
        if hits_limited:
            break
        try:
            df = fetch_ohlcv(t)

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

            if len(df) < 260:
                continue
            hist_ok += 1

            px = float(df["close"].iloc[-1])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())
            sv20 = float(df["volume"].iloc[-20:].mean())

            if px < MIN_PX or px > MAX_PX:
                continue
            if dv20 < MIN_DV20 or dv20 > MAX_DV20:
                continue
            if sv20 < MIN_SV20:
                continue
            liq_ok += 1

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

            ok, trig, lowb, dd, contr = base_scan(df)
            if not ok:
                near.append((0.0, t, px, dv20, bbz_last, "BASE"))
                continue
            base_ok += 1

            dry = dryup_ratio(df)
            if not np.isfinite(dry) or dry >= DRYUP_MAX:
                near.append((0.0, t, px, dv20, bbz_last, "DRY"))
                continue
            dry_ok += 1

            stop = lowb * 0.99

            close_today = float(df["close"].iloc[-1])
            open_today = float(df["open"].iloc[-1])
            close_prev = float(df["close"].iloc[-2])
            vol_today = float(df["volume"].iloc[-1])
            vol20 = float(df["volume"].iloc[-20:].mean())

            breakout = close_today > trig
            vol_confirm = (vol20 > 0) and (vol_today >= VOL_CONFIRM_MULT * vol20)
            no_big_gap = (close_prev > 0) and (open_today <= close_prev * MAX_GAP_UP)

            sc = score_candidate(bbz_last, atr_pctl, dd, contr, dry, dv20)
            item = (sc, t, px, dv20, bbz_last, trig, stop)

            if breakout and vol_confirm and no_big_gap:
                execs.append(item)
            else:
                watch.append(item)

        except Exception as e:
            s = str(e)
            if "NO_DATA" in s:
                no_data += 1
            elif "HITS_LIMIT" in s:
                hits_limited = True
            pass

        if (i + 1) % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    # Rank by score desc (NOT alphabetical)
    execs.sort(key=lambda x: x[0], reverse=True)
    watch.sort(key=lambda x: x[0], reverse=True)
    near.sort(key=lambda x: (x[0], -x[3]), reverse=True)

    execs = execs[:8]
    watch = watch[:10]
    near = near[:10]

    msg = [f"[{now}] ### V4_RUNNING ### Microcap scanner (RANKED)"]
    msg.append(f"Eval={len(tickers)} | hist={hist_ok} liq={liq_ok} comp={comp_ok} base={base_ok} dry={dry_ok} | watch={len(watch)} exec={len(execs)} | nodata={no_data}")
    msg.append("")

    if execs:
        msg.append("EXECUTAR (EOD):")
        for sc, t, px, dv20, bbz, trig, stop in execs:
            msg.append(f"- {t} | score={sc:.2f} | {px:.2f} | dv20={dv20/1e6:.1f}M | BBz={bbz:.2f} | trig>={trig:.2f} | stop~{stop:.2f}")
        msg.append("")
    else:
        msg.append("EXECUTAR: (vazio)")
        msg.append("")

    if watch:
        msg.append("AGUARDAR (TOP 10):")
        for sc, t, px, dv20, bbz, trig, stop in watch:
            msg.append(f"- {t} | score={sc:.2f} | {px:.2f} | dv20={dv20/1e6:.1f}M | BBz={bbz:.2f} | trig>={trig:.2f} | stop~{stop:.2f}")
        msg.append("")
    else:
        msg.append("AGUARDAR: (vazio)")
        msg.append("")

    if near:
        msg.append("QUASE (TOP 10):")
        for sc, t, px, dv20, bbz, reason in near:
            msg.append(f"- {t} | {reason} | {px:.2f} | dv20={dv20/1e6:.1f}M | BBz={bbz:.2f}")
    else:
        msg.append("QUASE: (vazio)")

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
