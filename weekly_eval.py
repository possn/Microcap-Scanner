# weekly_eval.py — QUANT v7 (breakout-correct: HIGH/LOW + regime segmentation + learned_params.json)
# Reads cache/signals.csv + cache/ohlcv/*.csv
# Weekly window is LAGGED (end=max(eval_day)-HORIZON BDays) to avoid cov=0
# Computes:
# - WATCH/EXEC success using HIGH (intraday breakout) with buffer
# - MFE/MAE using HIGH/LOW (not close)
# - Regime segmentation (RISK_ON/TRANSITION/RISK_OFF)
# Writes:
# - cache/weekly_summary.csv (schema fixed)
# - cache/learned_params.json (conservative parameter nudges, only if enough coverage)
# Sends Telegram summary

import os, csv, json, random
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from pandas.tseries.offsets import BDay

# =========================
# ENV
# =========================
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")

OHLCV_DIR = Path("cache") / "ohlcv"
SIGNALS_CSV = Path("cache") / "signals.csv"
SUMMARY_CSV = Path("cache") / "weekly_summary.csv"
LEARNED_JSON = Path("cache") / "learned_params.json"

HORIZON = int(os.environ.get("WEEKLY_HORIZON_SESS", "5"))  # sessions forward for weekly metrics
CLEAN_MAX_OH = int(os.environ.get("WATCH_OVERHEAD_CLEAN_MAX", "5"))
WINDOW_DAYS = 5

# breakout definition (intraday)
BREAKOUT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.5"))  # 0.5% default

# learning controls
LEARN_MIN_COV = int(os.environ.get("LEARN_MIN_COV", "30"))   # minimum covered signals to consider nudges
LEARN_BLEND = float(os.environ.get("LEARN_BLEND", "0.25"))   # how strongly daily blends learned params
LEARN_STEP_DIST = float(os.environ.get("LEARN_STEP_DIST", "1.0"))    # +/-
LEARN_STEP_ATR = float(os.environ.get("LEARN_STEP_ATR", "0.02"))     # +/-
LEARN_STEP_BBZ = float(os.environ.get("LEARN_STEP_BBZ", "0.05"))     # +/-

# =========================
# TELEGRAM
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
# IO helpers
# =========================
def load_ohlcv(ticker: str) -> pd.DataFrame | None:
    p = OHLCV_DIR / f"{ticker}.csv"
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
        df = df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)
        return df
    except Exception:
        return None

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# =========================
# weekly_summary schema FIXO
# =========================
SUMMARY_HEADER = [
    "week_end","window_start","window_end","horizon",
    "signals_total","exec_count","watch_count",
    "watch_clean","watch_over",
    "watch_cov","watch_success","watch_h1","watch_ff",
    "watch_mfe_med","watch_mae_mean",
    "clean_success","clean_mfe_med","clean_mae_mean",
    "over_success","over_mfe_med","over_mae_mean",
    "baseline_cov","baseline_success","baseline_mfe_med","baseline_mae_mean","edge_pp",
    "regime_mode",
    "risk_on_succ","transition_succ","risk_off_succ",
    "learn_action","learn_notes"
]

def append_weekly_summary(row: dict) -> None:
    ensure_parent(SUMMARY_CSV)
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        if not exists:
            w.writeheader()
        clean = {k: row.get(k, "") for k in SUMMARY_HEADER}
        w.writerow(clean)

def read_weekly_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SUMMARY_CSV)
        if df.empty:
            return df
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end"]).sort_values("week_end").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

# =========================
# Formatting helpers
# =========================
def pct(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.0f}%"

def fmt_pct_from_ret(x: float | None, nd=1) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.{nd}f}%"

# =========================
# Date helpers
# =========================
def pick_target_date_from_row(row: pd.Series) -> pd.Timestamp | None:
    if "asof" in row.index:
        d = pd.to_datetime(row.get("asof", None), errors="coerce")
        if pd.notna(d):
            return d.normalize()
    d0 = pd.to_datetime(row.get("date", None), errors="coerce")
    if pd.isna(d0):
        return None
    return d0.normalize()

def find_pos_best_effort(df: pd.DataFrame, target: pd.Timestamp) -> int | None:
    dates = df["date"].dt.normalize().values
    t64 = target.to_datetime64()

    idx = np.where(dates == t64)[0]
    if len(idx) > 0:
        return int(idx[0])

    idx2 = np.where(dates <= t64)[0]
    if len(idx2) > 0:
        return int(idx2[-1])

    return None

# =========================
# MFE/MAE using HIGH/LOW
# =========================
def mfe_mae_from_entry(df: pd.DataFrame, pos: int, horizon: int, entry: float) -> tuple[float | None, float | None]:
    if entry <= 0:
        return (None, None)
    end = min(len(df) - 1, pos + horizon)
    sl_hi = df.iloc[pos:end+1]["high"]
    sl_lo = df.iloc[pos:end+1]["low"]
    if sl_hi.empty or sl_lo.empty:
        return (None, None)
    mfe = float((sl_hi.max() / entry) - 1.0)
    mae = float((sl_lo.min() / entry) - 1.0)
    return (mfe, mae)

# =========================
# Evaluation (intraday breakout)
# =========================
def eval_signal_row(row: pd.Series, horizon: int) -> dict:
    """
    covered=1 only if full horizon exists
    success: any HIGH >= trig*(1+buffer)
    h1: next-day HIGH >= trig*(1+buffer)
    fail-fast: LOW < stop within next 2 sessions
    """
    out = {"covered": 0, "success": 0, "h1": 0, "ff": 0, "mfe": None, "mae": None}

    t = str(row.get("ticker", "")).strip().upper()
    trig = float(pd.to_numeric(row.get("trig", np.nan), errors="coerce"))
    stop = float(pd.to_numeric(row.get("stop", np.nan), errors="coerce"))
    entry_close = float(pd.to_numeric(row.get("close", np.nan), errors="coerce"))
    target = pick_target_date_from_row(row)

    if (not t) or pd.isna(target) or (not np.isfinite(trig)) or (not np.isfinite(entry_close)) or entry_close <= 0:
        return out

    df = load_ohlcv(t)
    if df is None or df.empty:
        return out

    pos = find_pos_best_effort(df, target)
    if pos is None:
        return out

    if pos + horizon >= len(df):
        return out

    out["covered"] = 1
    end = pos + horizon

    trig_hit = trig * (1.0 + BREAKOUT_BUFFER_PCT / 100.0)

    future_high = df.iloc[pos:end+1]["high"].values
    out["success"] = 1 if np.any(future_high >= trig_hit) else 0

    out["h1"] = 1 if float(df["high"].iloc[pos+1]) >= trig_hit else 0

    ff = 0
    if np.isfinite(stop):
        for j in [pos+1, pos+2]:
            if j < len(df) and float(df["low"].iloc[j]) < stop:
                ff = 1
                break
    out["ff"] = ff

    mfe, mae = mfe_mae_from_entry(df, pos, horizon, entry_close)
    out["mfe"] = mfe
    out["mae"] = mae
    return out

def aggregate_metrics(rows_eval: list[dict]) -> dict:
    if not rows_eval:
        return {"cov": 0, "succ": None, "h1": None, "ff": None, "mfe_med": None, "mae_mean": None}

    cov = sum(r["covered"] for r in rows_eval)
    if cov == 0:
        return {"cov": 0, "succ": None, "h1": None, "ff": None, "mfe_med": None, "mae_mean": None}

    suc = float(np.mean([r["success"] for r in rows_eval if r["covered"]]))
    h1 = float(np.mean([r["h1"] for r in rows_eval if r["covered"]]))
    ff = float(np.mean([r["ff"] for r in rows_eval if r["covered"]]))

    mfes = [r["mfe"] for r in rows_eval if r["covered"] and r["mfe"] is not None and np.isfinite(r["mfe"])]
    maes = [r["mae"] for r in rows_eval if r["covered"] and r["mae"] is not None and np.isfinite(r["mae"])]

    mfe_med = float(np.median(mfes)) if mfes else None
    mae_mean = float(np.mean(maes)) if maes else None

    return {"cov": cov, "succ": suc, "h1": h1, "ff": ff, "mfe_med": mfe_med, "mae_mean": mae_mean}

# =========================
# BASELINE (aligned with breakout)
# =========================
def baseline_sample(dist_targets: list[float], window_end: pd.Timestamp, horizon: int, k: int) -> list[dict]:
    """
    Baseline cache-only:
    - pick random tickers from cache/ohlcv
    - trig = max(HIGH last 20 sessions before window_end)
    - success = any future HIGH >= trig*(1+buffer)
    - dist% computed similarly
    - optional matching by dist quartiles when enough targets exist
    """
    if k <= 0:
        return []

    if not OHLCV_DIR.exists():
        return []
    files = [p for p in OHLCV_DIR.iterdir() if p.is_file() and p.name.endswith(".csv")]
    if not files:
        return []
    random.shuffle(files)

    tt = [x for x in dist_targets if np.isfinite(x)]
    use_matching = len(tt) >= 6

    bucket = None
    want = None
    got = None
    if use_matching:
        qs = np.quantile(tt, [0.25, 0.50, 0.75])

        def _bucket(x: float) -> int:
            if x <= qs[0]:
                return 0
            if x <= qs[1]:
                return 1
            if x <= qs[2]:
                return 2
            return 3

        bucket = _bucket
        target_buckets = [_bucket(x) for x in tt]
        want = {b: target_buckets.count(b) for b in [0, 1, 2, 3]}
        got = {0: 0, 1: 0, 2: 0, 3: 0}

    trig_hit_mult = (1.0 + BREAKOUT_BUFFER_PCT / 100.0)

    picked: list[dict] = []
    tries = 0

    for p in files:
        if len(picked) >= k:
            break
        tries += 1
        if tries > 2500:
            break

        t = p.stem.upper()
        df = load_ohlcv(t)
        if df is None or len(df) < 60:
            continue

        pos = find_pos_best_effort(df, window_end.normalize())
        if pos is None or pos < 25:
            continue
        if pos + horizon >= len(df):
            continue

        close0 = float(df["close"].iloc[pos])
        if close0 <= 0:
            continue

        trig = float(df["high"].iloc[pos-20:pos].max())
        if trig <= 0:
            continue

        dist = (trig - close0) / close0 * 100.0
        if not np.isfinite(dist):
            continue

        if use_matching:
            b = bucket(dist)  # type: ignore[misc]
            if got[b] >= want.get(b, 0):  # type: ignore[union-attr]
                continue

        end = pos + horizon
        future_hi = df["high"].iloc[pos:end+1].values
        success = 1 if np.any(future_hi >= trig * trig_hit_mult) else 0
        mfe, mae = mfe_mae_from_entry(df, pos, horizon, close0)

        picked.append({"covered": 1, "success": success, "mfe": mfe, "mae": mae})

        if use_matching:
            got[b] += 1  # type: ignore[index]

    return picked

# =========================
# Learning (conservative nudges)
# =========================
def propose_learned_params(m_watch: dict, m_base: dict, cov: int, current: dict) -> tuple[str, str, dict]:
    """
    Produce a small adjustment suggestion.
    Goal: increase success edge vs baseline while keeping quality.
    Rule-based, conservative:
      - If success is low and worse than baseline: tighten a bit (reduce dist, tighten ATR gate, require more compression)
      - If success is decent but cov is too low: relax slightly (increase dist, relax ATR, allow slightly less negative BBZ)
      - Else: keep
    Returns: (learn_action, learn_notes, params_dict)
    """
    # current gates expected in "current"
    dist = float(current.get("DIST_MAX_PCT", 18.0))
    atr = float(current.get("ATRPCTL_GATE", 0.50))
    bbz = float(current.get("BBZ_GATE", -0.60))

    succ = m_watch.get("succ", None)
    base_succ = m_base.get("succ", None)

    if succ is None or base_succ is None or cov < LEARN_MIN_COV:
        return ("HOLD", f"cov<{LEARN_MIN_COV} ou métricas insuficientes", {})

    edge = succ - base_succ

    # Heuristic targets
    # - want positive edge
    # - want not-too-low coverage; if cov small, allow a bit more ideas
    if edge < -0.02:
        # underperforming baseline -> tighten
        new = {
            "DIST_MAX_PCT": max(10.0, dist - LEARN_STEP_DIST),
            "ATRPCTL_GATE": max(0.30, atr - LEARN_STEP_ATR),
            "BBZ_GATE": min(-0.40, bbz - LEARN_STEP_BBZ),  # more negative => stricter compression
        }
        return ("TIGHTEN", f"edge={edge*100:.1f}pp vs baseline; tighten gates", new)

    if edge > 0.01 and cov < (LEARN_MIN_COV + 15):
        # doing ok but cov still low -> relax a hair to get more samples
        new = {
            "DIST_MAX_PCT": min(25.0, dist + LEARN_STEP_DIST),
            "ATRPCTL_GATE": min(0.65, atr + LEARN_STEP_ATR),
            "BBZ_GATE": max(-1.20, bbz + LEARN_STEP_BBZ),  # less negative => slightly more permissive
        }
        return ("RELAX", f"edge={edge*100:.1f}pp, mas cov baixo; relax ligeiro", new)

    return ("HOLD", f"edge={edge*100:.1f}pp; sem ajuste", {})

def write_learned_params(asof: str, params: dict, action: str, notes: str) -> None:
    ensure_parent(LEARNED_JSON)
    if not params:
        # still write a marker so you know weekly ran (optional)
        try:
            LEARNED_JSON.write_text(json.dumps({
                "version": 1,
                "asof": asof,
                "blend": float(max(0.0, min(0.50, LEARN_BLEND))),
                "params": {},
                "action": action,
                "notes": notes
            }, indent=2), encoding="utf-8")
        except Exception:
            pass
        return

    try:
        LEARNED_JSON.write_text(json.dumps({
            "version": 1,
            "asof": asof,
            "blend": float(max(0.0, min(0.50, LEARN_BLEND))),
            "params": params,
            "action": action,
            "notes": notes
        }, indent=2), encoding="utf-8")
    except Exception:
        pass

# =========================
# MAIN
# =========================
def main():
    if not SIGNALS_CSV.exists():
        tg_send("WEEKLY REVIEW: FAIL — cache/signals.csv não existe.")
        return

    try:
        sig = pd.read_csv(SIGNALS_CSV)
    except Exception:
        tg_send("WEEKLY REVIEW: FAIL — não consegui ler cache/signals.csv.")
        return

    if sig.empty:
        tg_send("WEEKLY REVIEW: FAIL — signals.csv vazio.")
        return

    sig.columns = [c.strip().lower() for c in sig.columns]

    need_cols = {"date", "ticker", "signal", "close", "trig", "stop", "dist_pct", "overhead_touches", "regime"}
    missing = [c for c in need_cols if c not in sig.columns]
    if missing:
        tg_send(f"WEEKLY REVIEW: FAIL — signals.csv sem colunas: {missing}")
        return

    sig["date"] = pd.to_datetime(sig["date"], errors="coerce").dt.normalize()
    sig = sig.dropna(subset=["date"]).copy()

    if "asof" in sig.columns:
        sig["asof"] = pd.to_datetime(sig["asof"], errors="coerce").dt.normalize()
        sig["eval_day"] = sig["asof"].where(sig["asof"].notna(), sig["date"])
    else:
        sig["eval_day"] = sig["date"]

    max_day = pd.to_datetime(sig["eval_day"], errors="coerce").dropna().max()
    if pd.isna(max_day):
        tg_send("WEEKLY REVIEW: FAIL — sem datas válidas.")
        return
    max_day = max_day.normalize()

    lag_end = (max_day - BDay(HORIZON)).normalize()
    lag_start = (lag_end - BDay(WINDOW_DAYS - 1)).normalize()

    w = sig[(sig["eval_day"] >= lag_start) & (sig["eval_day"] <= lag_end)].copy()

    if w.empty:
        u = pd.to_datetime(sig["eval_day"], errors="coerce").dropna().dt.normalize().unique()
        u = sorted(u)
        if not u:
            tg_send("WEEKLY REVIEW: FAIL — sem datas válidas.")
            return
        last_days = [pd.Timestamp(x) for x in u[-WINDOW_DAYS:]]
        lag_start, lag_end = last_days[0], last_days[-1]
        w = sig[(sig["eval_day"] >= lag_start) & (sig["eval_day"] <= lag_end)].copy()

    window_start = lag_start.strftime("%Y-%m-%d")
    window_end = lag_end.strftime("%Y-%m-%d")
    week_end = window_end

    w_exec = w[w["signal"].astype(str).str.contains("EXEC", na=False)].copy()
    w_watch = w[w["signal"].astype(str).str.upper().eq("WATCH")].copy()

    w_watch["overhead_touches"] = pd.to_numeric(w_watch["overhead_touches"], errors="coerce")
    w_watch["dist_pct"] = pd.to_numeric(w_watch["dist_pct"], errors="coerce")

    w_watch_clean = w_watch[w_watch["overhead_touches"] <= CLEAN_MAX_OH].copy()
    w_watch_over = w_watch[w_watch["overhead_touches"] > CLEAN_MAX_OH].copy()

    regime_mode = "—"
    if not w_watch.empty:
        vc = w_watch["regime"].astype(str).value_counts()
        if len(vc) > 0:
            regime_mode = str(vc.index[0])

    eval_watch = [eval_signal_row(r, HORIZON) for _, r in w_watch.iterrows()]
    eval_clean = [eval_signal_row(r, HORIZON) for _, r in w_watch_clean.iterrows()]
    eval_over = [eval_signal_row(r, HORIZON) for _, r in w_watch_over.iterrows()]
    eval_exec = [eval_signal_row(r, HORIZON) for _, r in w_exec.iterrows()]

    m_watch = aggregate_metrics(eval_watch)
    m_clean = aggregate_metrics(eval_clean)
    m_over = aggregate_metrics(eval_over)
    m_exec = aggregate_metrics(eval_exec)

    # baseline aligned
    dist_targets = w_watch["dist_pct"].dropna().tolist()
    baseline = baseline_sample(dist_targets, lag_end, HORIZON, k=len(w_watch))
    m_base = aggregate_metrics(baseline)

    edge_pp = None
    if m_base["cov"] and m_watch["cov"] and (m_base["succ"] is not None) and (m_watch["succ"] is not None):
        edge_pp = (m_watch["succ"] - m_base["succ"]) * 100.0

    # regime segmentation
    def succ_by_regime(tag: str) -> float | None:
        ww = w_watch[w_watch["regime"].astype(str) == tag].copy()
        if ww.empty:
            return None
        ev = [eval_signal_row(r, HORIZON) for _, r in ww.iterrows()]
        mm = aggregate_metrics(ev)
        return mm["succ"]

    risk_on_s = succ_by_regime("RISK_ON")
    trans_s = succ_by_regime("TRANSITION")
    risk_off_s = succ_by_regime("RISK_OFF")

    # rolling-4
    hist = read_weekly_summary()
    rolling_txt = "📈 ROLLING-4: sem histórico (weekly_summary.csv vazio)."
    if hist is not None and (not hist.empty):
        h4 = hist.tail(4).copy()
        try:
            cov_total = int(pd.to_numeric(h4["watch_cov"], errors="coerce").fillna(0).sum())
            succ = pd.to_numeric(h4["watch_success"], errors="coerce")
            mae = pd.to_numeric(h4["watch_mae_mean"], errors="coerce")
            mfe = pd.to_numeric(h4["watch_mfe_med"], errors="coerce")

            succ_m = float(np.nanmean(succ.values)) if len(succ.dropna()) else None
            mae_m = float(np.nanmean(mae.values)) if len(mae.dropna()) else None
            mfe_m = float(np.nanmean(mfe.values)) if len(mfe.dropna()) else None

            rolling_txt = (
                f"📈 ROLLING-4 (últimas {len(h4)} semanas): "
                f"WATCH success {pct(succ_m)} | "
                f"MFE_med {fmt_pct_from_ret(mfe_m,1)} | "
                f"MAE {fmt_pct_from_ret(mae_m,1)} | cov_total={cov_total}"
            )
        except Exception:
            rolling_txt = "📈 ROLLING-4: histórico existe mas parsing falhou (ver CSV)."

    sanity = "OK"
    if w_watch.empty and w_exec.empty:
        sanity = "WARN (sem sinais WATCH/EXEC na janela)"

    # learning: derive current from latest daily run gates (best effort)
    # We cannot read the scanner env here, so we infer "current" from last learned, else defaults.
    current = {"DIST_MAX_PCT": 18.0, "ATRPCTL_GATE": 0.50, "BBZ_GATE": -0.60}
    if LEARNED_JSON.exists():
        try:
            prev = json.loads(LEARNED_JSON.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and isinstance(prev.get("params", {}), dict):
                # last suggested params are *targets*; treat them as current estimate
                pp = prev.get("params", {})
                for k in ["DIST_MAX_PCT", "ATRPCTL_GATE", "BBZ_GATE"]:
                    if k in pp and np.isfinite(float(pp[k])):
                        current[k] = float(pp[k])
        except Exception:
            pass

    learn_action, learn_notes, learn_params = propose_learned_params(m_watch, m_base, int(m_watch["cov"]), current)
    write_learned_params(week_end, learn_params, learn_action, learn_notes)

    # write weekly summary row
    append_weekly_summary({
        "week_end": week_end,
        "window_start": window_start,
        "window_end": window_end,
        "horizon": HORIZON,

        "signals_total": int(len(w_exec) + len(w_watch)),
        "exec_count": int(len(w_exec)),
        "watch_count": int(len(w_watch)),
        "watch_clean": int(len(w_watch_clean)),
        "watch_over": int(len(w_watch_over)),

        "watch_cov": int(m_watch["cov"]),
        "watch_success": (m_watch["succ"] if m_watch["succ"] is not None else ""),
        "watch_h1": (m_watch["h1"] if m_watch["h1"] is not None else ""),
        "watch_ff": (m_watch["ff"] if m_watch["ff"] is not None else ""),
        "watch_mfe_med": (m_watch["mfe_med"] if m_watch["mfe_med"] is not None else ""),
        "watch_mae_mean": (m_watch["mae_mean"] if m_watch["mae_mean"] is not None else ""),

        "clean_success": (m_clean["succ"] if m_clean["succ"] is not None else ""),
        "clean_mfe_med": (m_clean["mfe_med"] if m_clean["mfe_med"] is not None else ""),
        "clean_mae_mean": (m_clean["mae_mean"] if m_clean["mae_mean"] is not None else ""),

        "over_success": (m_over["succ"] if m_over["succ"] is not None else ""),
        "over_mfe_med": (m_over["mfe_med"] if m_over["mfe_med"] is not None else ""),
        "over_mae_mean": (m_over["mae_mean"] if m_over["mae_mean"] is not None else ""),

        "baseline_cov": int(m_base["cov"]),
        "baseline_success": (m_base["succ"] if m_base["succ"] is not None else ""),
        "baseline_mfe_med": (m_base["mfe_med"] if m_base["mfe_med"] is not None else ""),
        "baseline_mae_mean": (m_base["mae_mean"] if m_base["mae_mean"] is not None else ""),
        "edge_pp": (edge_pp if edge_pp is not None else ""),

        "regime_mode": regime_mode,
        "risk_on_succ": (risk_on_s if risk_on_s is not None else ""),
        "transition_succ": (trans_s if trans_s is not None else ""),
        "risk_off_succ": (risk_off_s if risk_off_s is not None else ""),

        "learn_action": learn_action,
        "learn_notes": learn_notes
    })

    pending_exec = max(0, len(w_exec) - m_exec["cov"])
    pending_watch = max(0, len(w_watch) - m_watch["cov"])

    # Telegram message
    lines = []
    lines.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v7)")
    lines.append(f"Janela (lagged): {window_start} → {window_end}  (end=max(eval_day)-{HORIZON}d úteis)")
    lines.append(f"Horizonte métricas: {HORIZON} sessões | Breakout=HIGH ≥ trig*(1+{BREAKOUT_BUFFER_PCT:.1f}%)")
    lines.append("")
    lines.append(f"Sinais na janela: {len(w_exec) + len(w_watch)}")
    lines.append(f"• EXEC: {len(w_exec)} | WATCH: {len(w_watch)} (LIMPO: {len(w_watch_clean)} | TETO: {len(w_watch_over)})")
    if (pending_exec + pending_watch) > 0:
        lines.append(f"• PENDING (sem futuro suficiente no OHLCV p/ horizonte): EXEC={pending_exec} WATCH={pending_watch}")
    lines.append("")

    lines.append(
        f"EXEC:  cov={m_exec['cov']}/{len(w_exec)}  "
        f"S={pct(m_exec['succ'])} H1={pct(m_exec['h1'])} FF={pct(m_exec['ff'])}  "
        f"MFE_med={fmt_pct_from_ret(m_exec['mfe_med'],1)}  "
        f"MAE={fmt_pct_from_ret(m_exec['mae_mean'],1)}"
    )
    lines.append(
        f"WATCH: cov={m_watch['cov']}/{len(w_watch)}  "
        f"S={pct(m_watch['succ'])} H1={pct(m_watch['h1'])} FF={pct(m_watch['ff'])}  "
        f"MFE_med={fmt_pct_from_ret(m_watch['mfe_med'],1)}  "
        f"MAE={fmt_pct_from_ret(m_watch['mae_mean'],1)}"
    )
    lines.append(
        f"  LIMPO(≤{CLEAN_MAX_OH}): cov={m_clean['cov']}/{len(w_watch_clean)}  "
        f"S={pct(m_clean['succ'])}  "
        f"MFE_med={fmt_pct_from_ret(m_clean['mfe_med'],1)}  "
        f"MAE={fmt_pct_from_ret(m_clean['mae_mean'],1)}"
    )
    lines.append(
        f"  TETO(>{CLEAN_MAX_OH}):  cov={m_over['cov']}/{len(w_watch_over)}  "
        f"S={pct(m_over['succ'])}  "
        f"MFE_med={fmt_pct_from_ret(m_over['mfe_med'],1)}  "
        f"MAE={fmt_pct_from_ret(m_over['mae_mean'],1)}"
    )

    lines.append("")
    lines.append("🧪 BASELINE (cache-only; matched se possível):")
    if len(w_watch) == 0:
        lines.append("BASE:  cov=0/0  S=—  MFE_med=—  MAE=—")
        lines.append("EDGE:  —")
    elif m_base["cov"] == 0:
        lines.append(f"BASE:  cov=0/{len(w_watch)}  S=—  MFE_med=—  MAE=—")
        lines.append("EDGE:  —")
    else:
        lines.append(
            f"BASE:  cov={m_base['cov']}/{len(w_watch)}  "
            f"S={pct(m_base['succ'])}  "
            f"MFE_med={fmt_pct_from_ret(m_base['mfe_med'],1)}  "
            f"MAE={fmt_pct_from_ret(m_base['mae_mean'],1)}"
        )
        lines.append(f"EDGE:  {edge_pp:+.0f}pp" if edge_pp is not None else "EDGE: —")

    lines.append("")
    lines.append("🧭 REGIME (WATCH success):")
    lines.append(f"RISK_ON: {pct(risk_on_s)} | TRANSITION: {pct(trans_s)} | RISK_OFF: {pct(risk_off_s)}")
    lines.append("")
    lines.append(f"✅ SANITY: {sanity}")
    lines.append("")
    lines.append(rolling_txt)
    lines.append("")
    lines.append("🧠 LEARNING:")
    lines.append(f"• action={learn_action} | {learn_notes}")
    if learn_params:
        lines.append(f"• learned_params.json: {learn_params}")
    else:
        lines.append("• learned_params.json: (sem alterações)")

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()
