# weekly_eval.py  (QUANT v6 - baseline random + regime segmentation + rolling-4)
# - Baseline aleatório (controle): amostra tickers do cache/ohlcv com filtros básicos
#   e simula "trig" usando a distribuição real de DIST% dos WATCH da semana.
# - Segmenta métricas WATCH por regime (RISK_ON / TRANSITION / RISK_OFF).
# - Rolling-4: inclui edge vs baseline e por-regime.
#
# Nota: Não mexe no scanner.py. Só avaliação.

import os
import io
import csv
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# -------------------------
# ENV
# -------------------------
TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
OHLCV_FMT = os.getenv("OHLCV_URL_FMT", "https://stooq.com/q/d/l/?s={symbol}.us&i=d")

SIGNALS_CSV = Path("cache/signals.csv")
CACHE_OHLCV_DIR = Path("cache/ohlcv")
SUMMARY_CSV = Path("cache/weekly_summary.csv")

HORIZON_SESS = int(os.getenv("WEEKLY_HORIZON_SESS", "5"))
WATCH_OVERHEAD_CLEAN_MAX = int(os.getenv("WATCH_OVERHEAD_CLEAN_MAX", "5"))

# Baseline sample sizing
BASELINE_MULT = float(os.getenv("BASELINE_MULT", "2.0"))  # tenta amostrar 2x para compensar falhas
BASELINE_SEED = os.getenv("BASELINE_SEED", "")           # opcional (string)

# Filtros básicos baseline (para ser "justo" vs universo alvo)
BASE_MIN_PX = float(os.getenv("BASE_MIN_PX", "1.0"))
BASE_MAX_PX = float(os.getenv("BASE_MAX_PX", "25.0"))
BASE_MIN_DV20 = float(os.getenv("BASE_MIN_DV20", "3000000"))
BASE_MAX_DV20 = float(os.getenv("BASE_MAX_DV20", "80000000"))

# sanity thresholds
SANITY_MIN_COVERAGE_PCT = float(os.getenv("SANITY_MIN_COVERAGE_PCT", "0.65"))
SANITY_MAX_WEEKLY_JUMP_PP = float(os.getenv("SANITY_MAX_WEEKLY_JUMP_PP", "20"))

ROLL_MIN_WEEKS_STRUCTURAL = int(os.getenv("ROLL_MIN_WEEKS_STRUCTURAL", "4"))

# -------------------------
# Telegram
# -------------------------
def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
    except Exception:
        print(text)

# -------------------------
# Dates helper
# -------------------------
def last_n_business_days(end_dt: datetime, n: int = 5) -> list[datetime.date]:
    d = end_dt.date()
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = (datetime.combine(d, datetime.min.time()) - timedelta(days=1)).date()
    return sorted(out)

# -------------------------
# OHLCV cache-first
# -------------------------
def _load_cached_ohlcv(ticker: str) -> pd.DataFrame | None:
    p = CACHE_OHLCV_DIR / f"{ticker.upper()}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        need = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["high","low","close","volume"]).reset_index(drop=True)
        if len(df) >= 30:
            return df
    except Exception:
        return None
    return None

def _build_symbol_for_stooq(ticker: str) -> str:
    return ticker.lower().replace("-", ".")

def _candidate_urls(ticker: str) -> list[str]:
    sym = _build_symbol_for_stooq(ticker)
    if ".us" in OHLCV_FMT.lower():
        return [OHLCV_FMT.format(symbol=sym), OHLCV_FMT.format(symbol=f"{sym}.us")]
    return [OHLCV_FMT.format(symbol=sym), OHLCV_FMT.format(symbol=f"{sym}.us")]

def fetch_ohlcv_equity(ticker: str) -> pd.DataFrame | None:
    cached = _load_cached_ohlcv(ticker)
    if cached is not None and not cached.empty:
        return cached

    # best-effort web fetch (raramente usado no weekly; preferimos cache)
    for url in _candidate_urls(ticker):
        try:
            text = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text.strip()
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
            df = df.dropna(subset=["high","low","close","volume"]).reset_index(drop=True)
            if len(df) >= 30:
                return df
        except Exception:
            continue
    return None

# -------------------------
# Align: T -> T-1 -> T+1
# -------------------------
def _align_index_T_Tm1_Tp1(o: pd.DataFrame, target_date: datetime.date) -> int | None:
    o_dates = o["date"].dt.date.values

    idxs = np.where(o_dates == target_date)[0]
    if len(idxs) > 0:
        return int(idxs[0])

    t_minus = target_date - timedelta(days=1)
    idxs = np.where(o_dates == t_minus)[0]
    if len(idxs) > 0:
        return int(idxs[0])

    after = np.where(o_dates > target_date)[0]
    if len(after) > 0:
        return int(after[0])

    return None

# -------------------------
# Formatting
# -------------------------
def pct(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.0f}%"

def retfmt(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.1f}%"

# -------------------------
# Metrics core
# -------------------------
def compute_metrics_with_trig(df_sig: pd.DataFrame, horizon: int) -> dict:
    """
    Espera colunas: ticker, date, close(entry), trig.
    Mede sucesso = any(close > trig) em <= horizon sessões.
    Mede H1/FF usando closes após primeiro breakout.
    MFE/MAE usando high/low no horizonte.
    """
    res = {
        "n": int(len(df_sig)),
        "coverage": 0,
        "success_rate": np.nan,
        "hold1_rate": np.nan,
        "fail_fast_rate": np.nan,
        "mfe_mean": np.nan,
        "mfe_median": np.nan,
        "mae_mean": np.nan,
    }
    if df_sig.empty:
        return res

    success, hold1, fail_fast, mfe, mae = [], [], [], [], []

    for _, r in df_sig.iterrows():
        t = str(r.get("ticker", "")).strip().upper()
        if not t:
            continue

        dt = pd.to_datetime(r.get("date", None), errors="coerce")
        trig = pd.to_numeric(r.get("trig", np.nan), errors="coerce")
        entry = pd.to_numeric(r.get("close", np.nan), errors="coerce")

        if pd.isna(dt) or (not np.isfinite(trig)) or trig <= 0 or (not np.isfinite(entry)) or entry <= 0:
            continue

        o = fetch_ohlcv_equity(t)
        if o is None or o.empty:
            continue

        i0 = _align_index_T_Tm1_Tp1(o, dt.date())
        if i0 is None:
            continue

        i1 = min(i0 + horizon, len(o) - 1)
        sl = o.iloc[i0:i1 + 1].copy()
        if sl.empty:
            continue

        res["coverage"] += 1

        closes = sl["close"].values
        above = closes > trig

        s = bool(np.any(above))
        success.append(s)

        h = False
        ff = False
        if s:
            first = int(np.argmax(above))
            if first + 1 < len(closes):
                h = bool(closes[first + 1] > trig)
            j2 = min(first + 2, len(closes) - 1)
            if np.any(closes[first:j2 + 1] < trig):
                ff = True

        hold1.append(h)
        fail_fast.append(ff)

        mx = float(sl["high"].max())
        mn = float(sl["low"].min())
        mfe.append((mx / entry) - 1.0)
        mae.append((mn / entry) - 1.0)

    if success:
        res["success_rate"] = float(np.mean(success))
    if hold1:
        res["hold1_rate"] = float(np.mean(hold1))
    if fail_fast:
        res["fail_fast_rate"] = float(np.mean(fail_fast))
    if mfe:
        res["mfe_mean"] = float(np.mean(mfe))
        res["mfe_median"] = float(np.median(mfe))
    if mae:
        res["mae_mean"] = float(np.mean(mae))

    return res

# -------------------------
# Baseline: random matched by DIST%
# -------------------------
def _cached_tickers_list() -> list[str]:
    if not CACHE_OHLCV_DIR.exists():
        return []
    out = []
    for p in CACHE_OHLCV_DIR.glob("*.csv"):
        out.append(p.stem.upper())
    return list(set(out))

def _passes_baseline_filters(o: pd.DataFrame) -> bool:
    if o is None or len(o) < 60:
        return False
    close_now = float(o["close"].iloc[-1])
    if close_now < BASE_MIN_PX or close_now > BASE_MAX_PX:
        return False
    dv20 = float((o["close"].iloc[-20:] * o["volume"].iloc[-20:]).mean())
    if dv20 < BASE_MIN_DV20 or dv20 > BASE_MAX_DV20:
        return False
    return True

def build_baseline_df(watch_df: pd.DataFrame, horizon: int, window_end: datetime.date) -> pd.DataFrame:
    """
    Cria baseline com n == len(watch_df), usando tickers do cache.
    Matching: dist% é amostrado da distribuição da semana; trig = entry*(1+dist/100).
    Date: usa o mesmo window_end (alinhamento T/T-1/T+1 trata).
    """
    n_target = int(len(watch_df))
    if n_target <= 0:
        return pd.DataFrame()

    # dist distribution (só valores finitos)
    dist_vals = pd.to_numeric(watch_df.get("dist_pct", np.nan), errors="coerce").dropna().tolist()
    if not dist_vals:
        # fallback: assume 4%
        dist_vals = [4.0] * n_target

    # seed deterministic opcional
    if BASELINE_SEED:
        random.seed(BASELINE_SEED)
    else:
        random.seed(str(window_end))

    candidates = _cached_tickers_list()
    random.shuffle(candidates)

    # tenta amostrar >n para compensar falhas
    attempt = int(max(n_target, n_target * BASELINE_MULT))
    picks = candidates[:attempt] if len(candidates) >= attempt else candidates[:]

    rows = []
    dt = datetime.combine(window_end, datetime.min.time()).replace(tzinfo=ZoneInfo("Europe/Lisbon"))

    for t in picks:
        o = _load_cached_ohlcv(t)
        if o is None:
            continue
        if not _passes_baseline_filters(o):
            continue

        # entry = close em T (ou T-1/T+1 via alinhamento)
        i0 = _align_index_T_Tm1_Tp1(o, dt.date())
        if i0 is None:
            continue

        entry = float(o["close"].iloc[i0])
        if not np.isfinite(entry) or entry <= 0:
            continue

        dist = float(random.choice(dist_vals))
        # se dist negativa (overshoot), permitimos; trig pode ficar abaixo do entry
        trig = entry * (1.0 + dist / 100.0)
        if not np.isfinite(trig) or trig <= 0:
            continue

        rows.append({"date": dt.date().isoformat(), "ticker": t, "close": entry, "trig": trig})

        if len(rows) >= n_target:
            break

    return pd.DataFrame(rows)

# -------------------------
# Buckets (compact)
# -------------------------
def bucket_series(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True)

def bucket_report_watch(df: pd.DataFrame, horizon: int) -> list[str]:
    out = []
    if df.empty:
        return out

    for col in ["overhead_touches", "dist_pct", "bbz", "atrpctl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def line(g: pd.DataFrame, name: str) -> str:
        # usa trig real já existente no df
        m = compute_metrics_with_trig(g, horizon)
        return f"- {name}: n={m['n']} cov={m['coverage']} S={pct(m['success_rate'])} MFE_med={retfmt(m['mfe_median'])} MAE={retfmt(m['mae_mean'])}"

    out.append("Buckets WATCH (monitorizar):")

    if "overhead_touches" in df.columns:
        oh = df["overhead_touches"].fillna(-1)
        b = bucket_series(oh, [-1, 3, 8, 15, 30, 10_000], ["0-3", "4-8", "9-15", "16-30", "31+"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics_with_trig(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"OH {name}"))

    if "dist_pct" in df.columns:
        d = df["dist_pct"]
        b = bucket_series(d, [-999, 2, 4, 8, 12, 999], ["≤2", "2-4", "4-8", "8-12", ">12"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics_with_trig(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"DIST {name}%"))

    if "bbz" in df.columns:
        z = df["bbz"]
        b = bucket_series(z, [-999, -1.4, -1.2, -1.0, -0.8, 999], ["≤-1.4", "-1.4..-1.2", "-1.2..-1.0", "-1.0..-0.8", ">-0.8"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics_with_trig(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"BBZ {name}"))

    if "atrpctl" in df.columns:
        a = df["atrpctl"]
        b = bucket_series(a, [-999, 0.15, 0.20, 0.25, 0.30, 999], ["≤0.15", "0.15..0.20", "0.20..0.25", "0.25..0.30", ">0.30"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics_with_trig(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"ATRp {name}"))

    return out

# -------------------------
# Weekly summary persistence
# -------------------------
def summary_init_if_needed() -> None:
    if SUMMARY_CSV.exists():
        return
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "week_end",
            "window_start", "window_end",
            "horizon",
            "signals_total",
            "exec_total", "watch_total",
            "watch_cov_pct",
            "watch_success",
            "watch_clean_success",
            "watch_over_success",
            "watch_mae",
            "watch_mfe_med",
            # baseline + edge
            "baseline_cov_pct",
            "baseline_success",
            "baseline_mfe_med",
            "baseline_mae",
            "edge_pp",
            # regime segmented
            "watch_success_risk_on",
            "watch_success_transition",
            "watch_success_risk_off",
        ])

def append_week_summary(row: dict) -> None:
    summary_init_if_needed()
    cols = [
        "week_end",
        "window_start", "window_end",
        "horizon",
        "signals_total",
        "exec_total", "watch_total",
        "watch_cov_pct",
        "watch_success",
        "watch_clean_success",
        "watch_over_success",
        "watch_mae",
        "watch_mfe_med",
        "baseline_cov_pct",
        "baseline_success",
        "baseline_mfe_med",
        "baseline_mae",
        "edge_pp",
        "watch_success_risk_on",
        "watch_success_transition",
        "watch_success_risk_off",
    ]
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writerow({c: row.get(c, "") for c in cols})

def load_summary_df() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    try:
        s = pd.read_csv(SUMMARY_CSV)
        return s if not s.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# -------------------------
# SANITY
# -------------------------
def sanity_checks(total: int, watch_m: dict, prev_week: dict | None) -> list[str]:
    fails = []
    if total <= 0:
        fails.append("Sinais=0 (weekly sem dados)")

    if watch_m["n"] > 0:
        cov_pct = (watch_m["coverage"] / max(1, watch_m["n"]))
        if cov_pct < SANITY_MIN_COVERAGE_PCT:
            fails.append(f"Coverage baixo em WATCH ({cov_pct*100:.0f}%) — provável cache/alinhamento")

    if prev_week is not None:
        try:
            prev = float(prev_week.get("watch_success", "nan"))
            cur = float(watch_m["success_rate"]) if np.isfinite(watch_m["success_rate"]) else np.nan
            if np.isfinite(prev) and np.isfinite(cur):
                diff_pp = abs((cur - prev) * 100.0)
                if diff_pp >= SANITY_MAX_WEEKLY_JUMP_PP:
                    fails.append(f"Salto anómalo na success WATCH vs semana anterior ({diff_pp:.0f}pp) — verificar")
        except Exception:
            pass

    return fails

# -------------------------
# Rolling-4 blocks
# -------------------------
def _mean_tail(sdf: pd.DataFrame, col: str, k: int = 4) -> float:
    if sdf.empty or col not in sdf.columns:
        return np.nan
    v = pd.to_numeric(sdf[col], errors="coerce").dropna()
    if v.empty:
        return np.nan
    return float(v.tail(min(k, len(v))).mean())

def rolling4_block(sdf: pd.DataFrame) -> list[str]:
    out = []
    if sdf.empty:
        out.append("📈 ROLLING-4: sem histórico (weekly_summary.csv vazio).")
        return out

    # sort
    if "week_end" in sdf.columns:
        sdf["week_end_dt"] = pd.to_datetime(sdf["week_end"], errors="coerce")
        sdf = sdf.dropna(subset=["week_end_dt"]).sort_values("week_end_dt")

    n_weeks = len(sdf)
    if n_weeks < ROLL_MIN_WEEKS_STRUCTURAL:
        out.append(f"📈 ROLLING-4: em construção ({n_weeks}/{ROLL_MIN_WEEKS_STRUCTURAL} semanas).")
    else:
        out.append("📈 ROLLING-4 (últimas 4 semanas):")

    r_cov = _mean_tail(sdf, "watch_cov_pct", 4)
    r_all = _mean_tail(sdf, "watch_success", 4)
    r_clean = _mean_tail(sdf, "watch_clean_success", 4)
    r_over = _mean_tail(sdf, "watch_over_success", 4)
    r_mae = _mean_tail(sdf, "watch_mae", 4)
    r_mfe = _mean_tail(sdf, "watch_mfe_med", 4)

    rb_all = _mean_tail(sdf, "baseline_success", 4)
    rb_mae = _mean_tail(sdf, "baseline_mae", 4)
    rb_mfe = _mean_tail(sdf, "baseline_mfe_med", 4)
    redge = _mean_tail(sdf, "edge_pp", 4)

    r_on = _mean_tail(sdf, "watch_success_risk_on", 4)
    r_tr = _mean_tail(sdf, "watch_success_transition", 4)
    r_off = _mean_tail(sdf, "watch_success_risk_off", 4)

    out.append(f"- WATCH cov%: {pct(r_cov)}")
    out.append(f"- WATCH success: {pct(r_all)} | LIMPO: {pct(r_clean)} | TETO: {pct(r_over)}")
    out.append(f"- WATCH MFE_med: {retfmt(r_mfe)} | MAE: {retfmt(r_mae)}")

    out.append(f"- BASELINE success: {pct(rb_all)} | MFE_med: {retfmt(rb_mfe)} | MAE: {retfmt(rb_mae)}")
    if np.isfinite(redge):
        out.append(f"- EDGE (Modelo - Baseline): {redge:+.0f}pp (rolling-4)")
    else:
        out.append("- EDGE (Modelo - Baseline): —")

    out.append(f"- Regime success (rolling-4): RISK_ON {pct(r_on)} | TRANSITION {pct(r_tr)} | RISK_OFF {pct(r_off)}")
    out.append("Critério p/ mudar estruturalmente (guia): EDGE ≥ +8pp (rolling-4) e MAE não piorar >~1pp, consistente.")

    return out

# -------------------------
# MAIN
# -------------------------
def main() -> None:
    now = datetime.now(ZoneInfo("Europe/Lisbon"))
    days = last_n_business_days(now, 5)
    start = days[0]
    end = days[-1]

    if not SIGNALS_CSV.exists():
        tg_send("❌ WEEKLY: falta cache/signals.csv (daily não commitou ou cache não restaurada).")
        return

    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        tg_send("⚠ WEEKLY: signals.csv vazio (daily não está a escrever sinais).")
        return

    df["date"] = pd.to_datetime(df.get("date", None), errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["day"] = df["date"].dt.date
    wdf = df[(df["day"] >= start) & (df["day"] <= end)].copy()

    total = int(len(wdf))
    if "signal" not in wdf.columns:
        wdf["signal"] = ""

    exec_df = wdf[wdf["signal"].astype(str).isin(["EXEC_A", "EXEC_B"])].copy()
    watch_df = wdf[wdf["signal"].astype(str) == "WATCH"].copy()

    # ensure trig exists for metrics
    for col in ["close", "trig"]:
        if col in watch_df.columns:
            watch_df[col] = pd.to_numeric(watch_df[col], errors="coerce")
    watch_df = watch_df.dropna(subset=["close","trig"]).copy()

    # clean vs overhead
    watch_df["overhead_touches"] = pd.to_numeric(watch_df.get("overhead_touches", np.nan), errors="coerce")
    watch_clean = watch_df[watch_df["overhead_touches"] <= WATCH_OVERHEAD_CLEAN_MAX].copy()
    watch_over = watch_df[watch_df["overhead_touches"] > WATCH_OVERHEAD_CLEAN_MAX].copy()

    # metrics
    m_exec = compute_metrics_with_trig(exec_df, HORIZON_SESS) if not exec_df.empty else {
        "n": 0, "coverage": 0, "success_rate": np.nan, "hold1_rate": np.nan, "fail_fast_rate": np.nan,
        "mfe_mean": np.nan, "mfe_median": np.nan, "mae_mean": np.nan
    }
    m_watch_all = compute_metrics_with_trig(watch_df, HORIZON_SESS)
    m_watch_clean = compute_metrics_with_trig(watch_clean, HORIZON_SESS)
    m_watch_over = compute_metrics_with_trig(watch_over, HORIZON_SESS)

    watch_cov_pct = (m_watch_all["coverage"] / max(1, m_watch_all["n"])) if m_watch_all["n"] > 0 else np.nan

    # regime segmentation (WATCH)
    if "regime" in watch_df.columns:
        w_on = watch_df[watch_df["regime"].astype(str) == "RISK_ON"].copy()
        w_tr = watch_df[watch_df["regime"].astype(str) == "TRANSITION"].copy()
        w_off = watch_df[watch_df["regime"].astype(str) == "RISK_OFF"].copy()
        m_on = compute_metrics_with_trig(w_on, HORIZON_SESS) if not w_on.empty else {"success_rate": np.nan}
        m_tr = compute_metrics_with_trig(w_tr, HORIZON_SESS) if not w_tr.empty else {"success_rate": np.nan}
        m_off = compute_metrics_with_trig(w_off, HORIZON_SESS) if not w_off.empty else {"success_rate": np.nan}
    else:
        m_on = {"success_rate": np.nan}
        m_tr = {"success_rate": np.nan}
        m_off = {"success_rate": np.nan}

    # baseline build + metrics
    baseline_df = build_baseline_df(watch_df, HORIZON_SESS, end)
    m_base = compute_metrics_with_trig(baseline_df, HORIZON_SESS) if not baseline_df.empty else {
        "n": 0, "coverage": 0, "success_rate": np.nan, "hold1_rate": np.nan, "fail_fast_rate": np.nan,
        "mfe_mean": np.nan, "mfe_median": np.nan, "mae_mean": np.nan
    }
    base_cov_pct = (m_base["coverage"] / max(1, m_base["n"])) if m_base["n"] > 0 else np.nan

    edge_pp = np.nan
    if np.isfinite(m_watch_all["success_rate"]) and np.isfinite(m_base["success_rate"]):
        edge_pp = (m_watch_all["success_rate"] - m_base["success_rate"]) * 100.0

    # persist summary
    append_week_summary({
        "week_end": str(end),
        "window_start": str(start),
        "window_end": str(end),
        "horizon": HORIZON_SESS,
        "signals_total": total,
        "exec_total": int(len(exec_df)),
        "watch_total": int(len(watch_df)),
        "watch_cov_pct": round(float(watch_cov_pct), 6) if np.isfinite(watch_cov_pct) else "",
        "watch_success": round(float(m_watch_all["success_rate"]), 6) if np.isfinite(m_watch_all["success_rate"]) else "",
        "watch_clean_success": round(float(m_watch_clean["success_rate"]), 6) if np.isfinite(m_watch_clean["success_rate"]) else "",
        "watch_over_success": round(float(m_watch_over["success_rate"]), 6) if np.isfinite(m_watch_over["success_rate"]) else "",
        "watch_mae": round(float(m_watch_all["mae_mean"]), 6) if np.isfinite(m_watch_all["mae_mean"]) else "",
        "watch_mfe_med": round(float(m_watch_all["mfe_median"]), 6) if np.isfinite(m_watch_all["mfe_median"]) else "",
        "baseline_cov_pct": round(float(base_cov_pct), 6) if np.isfinite(base_cov_pct) else "",
        "baseline_success": round(float(m_base["success_rate"]), 6) if np.isfinite(m_base["success_rate"]) else "",
        "baseline_mfe_med": round(float(m_base["mfe_median"]), 6) if np.isfinite(m_base["mfe_median"]) else "",
        "baseline_mae": round(float(m_base["mae_mean"]), 6) if np.isfinite(m_base["mae_mean"]) else "",
        "edge_pp": round(float(edge_pp), 3) if np.isfinite(edge_pp) else "",
        "watch_success_risk_on": round(float(m_on["success_rate"]), 6) if np.isfinite(m_on["success_rate"]) else "",
        "watch_success_transition": round(float(m_tr["success_rate"]), 6) if np.isfinite(m_tr["success_rate"]) else "",
        "watch_success_risk_off": round(float(m_off["success_rate"]), 6) if np.isfinite(m_off["success_rate"]) else "",
    })

    sdf = load_summary_df()
    prev_week = None
    if len(sdf) >= 2:
        prev_week = sdf.iloc[-2].to_dict()

    # SANITY
    sanity_fails = sanity_checks(total, m_watch_all, prev_week)

    # Message
    msg = []
    msg.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v6)")
    msg.append(f"Janela (últimos 5 dias úteis): {start} → {end}")
    msg.append(f"Horizonte métricas: {HORIZON_SESS} sessões")
    msg.append("")
    msg.append(f"Sinais na janela: {total}")
    msg.append(f"• EXEC: {len(exec_df)} | WATCH: {len(watch_df)} (LIMPO: {len(watch_clean)} | TETO: {len(watch_over)})")
    msg.append("")

    msg.append(f"EXEC:  cov={m_exec['coverage']}/{m_exec['n']}  S={pct(m_exec['success_rate'])} H1={pct(m_exec['hold1_rate'])} FF={pct(m_exec['fail_fast_rate'])}  MFE_med={retfmt(m_exec['mfe_median'])}  MAE={retfmt(m_exec['mae_mean'])}")
    msg.append(f"WATCH: cov={m_watch_all['coverage']}/{m_watch_all['n']}  S={pct(m_watch_all['success_rate'])} H1={pct(m_watch_all['hold1_rate'])} FF={pct(m_watch_all['fail_fast_rate'])}  MFE_med={retfmt(m_watch_all['mfe_median'])}  MAE={retfmt(m_watch_all['mae_mean'])}")
    msg.append(f"  LIMPO(≤{WATCH_OVERHEAD_CLEAN_MAX}): cov={m_watch_clean['coverage']}/{m_watch_clean['n']}  S={pct(m_watch_clean['success_rate'])}  MFE_med={retfmt(m_watch_clean['mfe_median'])}  MAE={retfmt(m_watch_clean['mae_mean'])}")
    msg.append(f"  TETO(>{WATCH_OVERHEAD_CLEAN_MAX}):  cov={m_watch_over['coverage']}/{m_watch_over['n']}  S={pct(m_watch_over['success_rate'])}  MFE_med={retfmt(m_watch_over['mfe_median'])}  MAE={retfmt(m_watch_over['mae_mean'])}")
    msg.append("")

    # Baseline block
    msg.append("🧪 BASELINE (random matched DIST%):")
    msg.append(f"BASE:  cov={m_base['coverage']}/{m_base['n']}  S={pct(m_base['success_rate'])}  MFE_med={retfmt(m_base['mfe_median'])}  MAE={retfmt(m_base['mae_mean'])}")
    if np.isfinite(edge_pp):
        msg.append(f"EDGE:  Modelo - Baseline = {edge_pp:+.0f}pp")
    else:
        msg.append("EDGE:  —")
    msg.append("")

    # Regime block
    msg.append("🧭 REGIME (WATCH success):")
    msg.append(f"RISK_ON: {pct(m_on.get('success_rate', np.nan))} | TRANSITION: {pct(m_tr.get('success_rate', np.nan))} | RISK_OFF: {pct(m_off.get('success_rate', np.nan))}")
    msg.append("")

    # SANITY
    if sanity_fails:
        msg.append("🧯 SANITY: FAIL (corrigir já)")
        for f in sanity_fails:
            msg.append(f"- {f}")
    else:
        msg.append("✅ SANITY: OK (sem bugs críticos detectados)")
    msg.append("")

    # Rolling-4
    msg.extend(rolling4_block(sdf))
    msg.append("")

    # Buckets
    if len(watch_df) > 0:
        msg.extend(bucket_report_watch(watch_df, HORIZON_SESS))
        msg.append("")

    msg.append("Ações:")
    msg.append("• Bugs: corrigir imediatamente se SANITY=FAIL.")
    msg.append(f"• Modelo: alterações estruturais só com evidência em ROLLING-4 (≥{ROLL_MIN_WEEKS_STRUCTURAL} semanas).")
    msg.append("• Histórico: weekly_summary.csv actualizado (base para decisões).")

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()
