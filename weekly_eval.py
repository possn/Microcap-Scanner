# weekly_eval.py — QUANT v8.0
# MELHORIAS vs v7.3:
# 1. Telegram chunking (mesmo fix do scanner_v8)
# 2. baseline_sample: sem random.shuffle (determinístico para audit trail)
# 3. Edge: usa MFE em vez de só success rate (mais informativo para trend-following)
# 4. eval_watch_row: captura MFE/MAE com HIGH/LOW (não só close)
# 5. propose_learning: limiar edge_pp dinâmico (não hardcoded 0/5)
# 6. Logging estruturado
# 7. Regime-stratified reporting (RISK_ON / RISK_OFF / TRANSITION separados)
# 8. Hold time estimate: quantos dias em média até ao trigger
# 9. Schema v8: adiciona colunas edge_mfe_pp e regime_breakdown sem quebrar retrocompat
# 10. Separação limpa de IO, métricas, learning, output

import os, csv, json, logging
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from pandas.tseries.offsets import BDay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cache/weekly_eval.log", encoding="utf-8")
    ]
)
log = logging.getLogger("weekly_eval")

# =========================
# ENV / CONFIG
# =========================
TG_TOKEN    = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID  = os.environ.get("TG_CHAT_ID", "")

OHLCV_DIR    = Path("cache") / "ohlcv"
SIGNALS_CSV  = Path("cache") / "signals.csv"
SUMMARY_CSV  = Path("cache") / "weekly_summary.csv"
LEARNED_JSON = Path("cache") / "learned_params.json"

HORIZON       = int(os.environ.get("WEEKLY_HORIZON_SESS", "5"))
CLEAN_MAX_OH  = int(os.environ.get("WATCH_OVERHEAD_CLEAN_MAX", "5"))
WINDOW_DAYS   = 5

BREAKOUT_BUFFER_PCT  = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.5"))
LEARN_MIN_COV        = int(os.environ.get("LEARN_MIN_COV", "30"))
LEARN_BLEND          = float(os.environ.get("LEARN_BLEND", "0.25"))
LEARN_STEP_DIST      = float(os.environ.get("LEARN_STEP_DIST", "1.0"))
LEARN_STEP_ATR       = float(os.environ.get("LEARN_STEP_ATR", "0.02"))
LEARN_STEP_BBZ       = float(os.environ.get("LEARN_STEP_BBZ", "0.05"))

# FIX #5: limiares dinâmicos para aprendizagem
LEARN_TIGHTEN_THRESHOLD = float(os.environ.get("LEARN_TIGHTEN_THRESHOLD", "0.0"))   # edge_pp < 0 → apertar
LEARN_RELAX_THRESHOLD   = float(os.environ.get("LEARN_RELAX_THRESHOLD", "5.0"))    # edge_pp > 5 → relaxar

BASELINE_MIN_COV_RATIO       = 0.80
BASELINE_NOFALLBACK_REQUIRED = True

DEFAULT_PARAMS = {
    "DIST_MAX_PCT":    18.0,
    "ATRPCTL_GATE":    0.50,
    "BBZ_GATE":       -0.60,
    "VOL_CONFIRM_MULT": 1.12,
}

# =========================
# TELEGRAM (FIX #1: chunking)
# =========================
def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        log.info("[TG] não configurado\n" + text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        try:
            r = requests.post(url, json={
                "chat_id": TG_CHAT_ID, "text": chunk,
                "disable_web_page_preview": True
            }, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.warning(f"[TG] chunk falhou: {e}")

# =========================
# IO helpers
# =========================
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def load_ohlcv(ticker: str) -> pd.DataFrame | None:
    p = OHLCV_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        need = ["date","open","high","low","close","volume"]
        if not all(c in df.columns for c in need):
            return None
        df = df[need].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)
    except Exception:
        return None

# =========================
# Schema v8 (retrocompat: colunas novas no fim)
# =========================
SUMMARY_HEADER = [
    # --- v7 original ---
    "week_end","window_start","window_end","horizon",
    "signals_total","exec_count","watch_count",
    "watch_clean","watch_over",
    "watch_cov","watch_success","watch_h1","watch_ff",
    "watch_mfe_med","watch_mae_mean",
    "clean_success","clean_mfe_med","clean_mae_mean",
    "over_success","over_mfe_med","over_mae_mean",
    "baseline_cov","baseline_success","baseline_mfe_med","baseline_mae_mean","edge_pp",
    "baseline_files","baseline_picked","baseline_fallback",
    "regime_mode",
    "learn_action","learn_edge_pp","learn_cov_used",
    # --- v8 new ---
    "edge_mfe_pp",          # FIX #3: edge em MFE (não só success)
    "watch_hit_trig_rate",  # % de WATCH que tocou trigger nos 5 dias
    "days_to_trig_med",     # FIX #8: mediana de dias até trigger
    "regime_risk_on_succ",  # FIX #7: success por regime
    "regime_risk_off_succ",
    "regime_transition_succ",
    "watch_clean_ff",       # FIX: false fast por categoria
    "watch_over_ff",
]

def _read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.readline().strip()

def weekly_summary_ensure_schema() -> None:
    ensure_parent(SUMMARY_CSV)
    expected = ",".join(SUMMARY_HEADER)
    if not SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SUMMARY_HEADER)
        return
    first = _read_first_line(SUMMARY_CSV)
    if first == expected:
        return
    # retrocompat: v7 header — migra
    v7_cols = first.split(",")
    if set(v7_cols).issubset(set(SUMMARY_HEADER)):
        log.info("[Schema] Migrando weekly_summary.csv de v7 para v8 (adicionar colunas)")
        df_old = pd.read_csv(SUMMARY_CSV)
        for c in SUMMARY_HEADER:
            if c not in df_old.columns:
                df_old[c] = ""
        df_old[SUMMARY_HEADER].to_csv(SUMMARY_CSV, index=False)
    else:
        msg = (f"WEEKLY REVIEW: FAIL — header mismatch irreparável.\n"
               f"Expected: {expected}\nFound: {first}")
        tg_send(msg)
        raise RuntimeError(msg)

def append_weekly_summary(row: dict) -> None:
    weekly_summary_ensure_schema()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writerow({c: row.get(c, "") for c in SUMMARY_HEADER})

def read_weekly_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SUMMARY_CSV)
        if df.empty:
            return df
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        return df.dropna(subset=["week_end"]).sort_values("week_end").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# =========================
# Formatting
# =========================
def pct(x) -> str:
    if x is None or not np.isfinite(float(x) if x is not None else float("nan")):
        return "—"
    return f"{float(x)*100:.0f}%"

def fpct(x, nd=1) -> str:
    if x is None or not np.isfinite(float(x) if x is not None else float("nan")):
        return "—"
    return f"{float(x)*100:.{nd}f}%"

# =========================
# Date helpers
# =========================
def pick_eval_day(row: pd.Series) -> pd.Timestamp | None:
    for col in ["asof", "date"]:
        d = pd.to_datetime(row.get(col, None), errors="coerce")
        if pd.notna(d):
            return d.normalize()
    return None

def find_pos(df: pd.DataFrame, target: pd.Timestamp) -> int | None:
    dates = df["date"].dt.normalize().values
    t64 = target.to_datetime64()
    idx = np.where(dates == t64)[0]
    if len(idx):
        return int(idx[0])
    idx2 = np.where(dates <= t64)[0]
    return int(idx2[-1]) if len(idx2) else None

# =========================
# MFE/MAE — FIX #4: usa HIGH e LOW reais
# =========================
def mfe_mae_hl(df: pd.DataFrame, pos: int, horizon: int, entry: float) -> tuple:
    """
    MFE com HIGH (melhor caso intraday), MAE com LOW (pior caso intraday).
    Mais realista que usar só close.
    """
    if entry <= 0:
        return (None, None)
    end = min(len(df) - 1, pos + horizon)
    highs = df["high"].iloc[pos:end+1].values
    lows  = df["low"].iloc[pos:end+1].values
    highs = pd.to_numeric(highs, errors="coerce")
    lows  = pd.to_numeric(lows, errors="coerce")
    mfe = float(np.nanmax(highs) / entry - 1.0) if not np.all(np.isnan(highs)) else None
    mae = float(np.nanmin(lows)  / entry - 1.0) if not np.all(np.isnan(lows))  else None
    return (mfe, mae)

def days_to_trigger(df: pd.DataFrame, pos: int, horizon: int, trig_eff: float) -> int | None:
    """FIX #8: quantos dias (sessões) até o HIGH superar o trigger."""
    for d in range(1, min(horizon + 1, len(df) - pos)):
        if float(pd.to_numeric(df["high"].iloc[pos + d], errors="coerce")) >= trig_eff:
            return d
    return None

# =========================
# Evaluation row
# =========================
def eval_row(row: pd.Series, horizon: int, buf_pct: float) -> dict:
    out = {"covered": 0, "success": 0, "h1": 0, "ff": 0,
           "mfe": None, "mae": None, "days_trig": None, "hit_trig": 0}

    t = str(row.get("ticker", "")).strip().upper()
    trig  = float(pd.to_numeric(row.get("trig",  np.nan), errors="coerce"))
    stop  = float(pd.to_numeric(row.get("stop",  np.nan), errors="coerce"))
    entry = float(pd.to_numeric(row.get("close", np.nan), errors="coerce"))
    target = pick_eval_day(row)

    if not t or pd.isna(target) or not np.isfinite(trig) or not np.isfinite(entry):
        return out

    df = load_ohlcv(t)
    if df is None or df.empty:
        return out

    pos = find_pos(df, target)
    if pos is None or pos + horizon >= len(df):
        return out

    out["covered"] = 1
    trig_eff = trig * (1.0 + buf_pct / 100.0)

    # success via HIGH (mais justo — preço pode tocar intraday)
    highs_fwd = df["high"].iloc[pos+1:pos+horizon+1]
    highs_fwd = pd.to_numeric(highs_fwd, errors="coerce")
    hit = bool(np.any(highs_fwd >= trig_eff))
    out["success"] = int(hit)
    out["hit_trig"] = int(hit)

    # H1: HIGH do dia seguinte >= trig
    if pos + 1 < len(df):
        h1_high = float(pd.to_numeric(df["high"].iloc[pos+1], errors="coerce"))
        out["h1"] = int(h1_high >= trig_eff)

    # FF: stop hit via LOW nos primeiros 2 dias
    if np.isfinite(stop):
        for j in [pos+1, pos+2]:
            if j < len(df):
                low_j = float(pd.to_numeric(df["low"].iloc[j], errors="coerce"))
                if np.isfinite(low_j) and low_j < stop:
                    out["ff"] = 1
                    break

    # FIX #4: MFE/MAE com HIGH/LOW
    mfe, mae = mfe_mae_hl(df, pos, horizon, entry)
    out["mfe"] = mfe
    out["mae"] = mae

    # FIX #8: dias até trigger
    dtrig = days_to_trigger(df, pos, horizon, trig_eff)
    out["days_trig"] = dtrig

    return out

def aggregate(evals: list[dict]) -> dict:
    cov = [e for e in evals if e["covered"]]
    if not cov:
        return {"cov": 0, "succ": None, "h1": None, "ff": None,
                "mfe_med": None, "mae_mean": None, "hit_trig": None, "days_trig_med": None}
    n = len(cov)
    succ   = float(np.mean([e["success"] for e in cov]))
    h1     = float(np.mean([e["h1"] for e in cov]))
    ff     = float(np.mean([e["ff"] for e in cov]))
    hit    = float(np.mean([e["hit_trig"] for e in cov]))
    mfes   = [e["mfe"] for e in cov if e["mfe"] is not None and np.isfinite(e["mfe"])]
    maes   = [e["mae"] for e in cov if e["mae"] is not None and np.isfinite(e["mae"])]
    dts    = [e["days_trig"] for e in cov if e["days_trig"] is not None]
    return {
        "cov": n,
        "succ": succ,
        "h1": h1,
        "ff": ff,
        "hit_trig": hit,
        "mfe_med": float(np.median(mfes)) if mfes else None,
        "mae_mean": float(np.mean(maes)) if maes else None,
        "days_trig_med": float(np.median(dts)) if dts else None,
    }

# =========================
# Regime breakdown — FIX #7
# =========================
def regime_success(evals: list[dict], regimes: list[str], target: str) -> float | None:
    sub = [e for e, r in zip(evals, regimes) if r == target and e["covered"]]
    if not sub:
        return None
    return float(np.mean([e["success"] for e in sub]))

# =========================
# BASELINE — FIX #2: determinístico (sem random.shuffle)
# =========================
def baseline_sample(dist_targets: list, window_end: pd.Timestamp,
                    horizon: int, k: int, buf_pct: float) -> tuple:
    dbg = {"files": 0, "picked": 0, "fallback": 0}
    if k <= 0 or not OHLCV_DIR.exists():
        return ([], dbg)

    # FIX #2: ordenado por nome (determinístico e reproduzível)
    files = sorted([p for p in OHLCV_DIR.iterdir() if p.suffix == ".csv"])
    dbg["files"] = len(files)
    if not files:
        return ([], dbg)

    tt = [x for x in dist_targets if np.isfinite(x)]
    use_matching = len(tt) >= 6
    want, got = {}, {}
    qs = None

    if use_matching:
        qs = np.quantile(tt, [0.25, 0.50, 0.75])
        def _b(x):
            if x <= qs[0]: return 0
            if x <= qs[1]: return 1
            if x <= qs[2]: return 2
            return 3
        tb = [_b(x) for x in tt]
        want = {b: tb.count(b) for b in [0,1,2,3]}
        got  = {0:0, 1:0, 2:0, 3:0}
    else:
        _b = None

    def _try_pick(df, enforce_match):
        if df is None or len(df) < horizon + 30:
            return None
        pos_cap = len(df) - horizon - 1
        if pos_cap < 25:
            return None
        pos0 = find_pos(df, window_end.normalize())
        if pos0 is None:
            return None
        pos = int(min(pos0, pos_cap))
        if pos < 25:
            return None
        c0 = float(df["close"].iloc[pos])
        if c0 <= 0:
            return None
        trig = float(df["close"].iloc[max(0,pos-20):pos].max())
        if trig <= 0:
            return None
        dist = (trig - c0) / c0 * 100.0
        if not np.isfinite(dist):
            return None
        if enforce_match and use_matching and _b:
            b = _b(dist)
            if got[b] >= want.get(b, 0):
                return None
        else:
            b = None
        trig_eff = trig * (1.0 + buf_pct / 100.0)
        highs_fwd = df["high"].iloc[pos+1:pos+horizon+1]
        highs_fwd = pd.to_numeric(highs_fwd, errors="coerce")
        hit = bool(np.any(highs_fwd >= trig_eff))
        h1_high = float(pd.to_numeric(df["high"].iloc[pos+1], errors="coerce")) if pos+1 < len(df) else 0.0
        h1 = int(h1_high >= trig_eff)
        stop = float(df["low"].iloc[max(0,pos-10):pos].min()) * 0.95
        ff = 0
        for j in [pos+1, pos+2]:
            if j < len(df):
                lj = float(pd.to_numeric(df["low"].iloc[j], errors="coerce"))
                if np.isfinite(lj) and lj < stop:
                    ff = 1
                    break
        mfe, mae = mfe_mae_hl(df, pos, horizon, c0)
        if enforce_match and use_matching and _b and b is not None:
            got[b] += 1
        return {"covered": 1, "success": int(hit), "h1": h1, "ff": ff,
                "mfe": mfe, "mae": mae, "hit_trig": int(hit), "days_trig": None}

    def _run(enforce):
        picked = []
        for p in files:
            if len(picked) >= k:
                break
            df = load_ohlcv(p.stem.upper())
            item = _try_pick(df, enforce)
            if item:
                picked.append(item)
        return picked

    picked = _run(True)
    if len(picked) < k and use_matching:
        dbg["fallback"] = 1
        rem = k - len(picked)
        picked.extend(_run(False)[:rem])

    dbg["picked"] = len(picked)
    return (picked, dbg)

# =========================
# Learning — FIX #5: limiares configuráveis
# =========================
def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))

def load_learned() -> dict:
    if not LEARNED_JSON.exists():
        return {}
    try:
        d = json.loads(LEARNED_JSON.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def save_learned(version, asof, blend, params, meta) -> None:
    ensure_parent(LEARNED_JSON)
    LEARNED_JSON.write_text(json.dumps({
        "version": int(version), "asof": str(asof),
        "blend": float(blend), "params": params, "meta": meta,
    }, indent=2), encoding="utf-8")

def propose_update(edge_pp: float, edge_mfe_pp: float | None, cov: int, cur: dict) -> tuple:
    """
    FIX #5: decision usa edge_pp E edge_mfe_pp combinados.
    Só aperta se AMBOS negativos; só relaxa se edge_pp > threshold E mfe também melhor.
    """
    if not np.isfinite(edge_pp) or cov < LEARN_MIN_COV:
        return ("NO_LEARN", cur)

    dist = _clamp(cur.get("DIST_MAX_PCT",    DEFAULT_PARAMS["DIST_MAX_PCT"]),    10, 30)
    atrg = _clamp(cur.get("ATRPCTL_GATE",    DEFAULT_PARAMS["ATRPCTL_GATE"]),   0.25, 0.65)
    bbzg = _clamp(cur.get("BBZ_GATE",        DEFAULT_PARAMS["BBZ_GATE"]),       -2.5, -0.1)
    volm = _clamp(cur.get("VOL_CONFIRM_MULT",DEFAULT_PARAMS["VOL_CONFIRM_MULT"]),1.05, 1.40)

    mfe_signal = (edge_mfe_pp is not None and np.isfinite(edge_mfe_pp))

    if edge_pp < LEARN_TIGHTEN_THRESHOLD and (not mfe_signal or edge_mfe_pp < 0):
        return ("TIGHTEN", {
            "DIST_MAX_PCT":    _clamp(dist - LEARN_STEP_DIST, 10, 30),
            "ATRPCTL_GATE":    _clamp(atrg - LEARN_STEP_ATR,  0.25, 0.65),
            "BBZ_GATE":        _clamp(bbzg - LEARN_STEP_BBZ,  -2.5, -0.1),
            "VOL_CONFIRM_MULT":_clamp(volm + 0.01,             1.05, 1.40),
        })

    if edge_pp > LEARN_RELAX_THRESHOLD and (not mfe_signal or edge_mfe_pp > 0):
        return ("RELAX", {
            "DIST_MAX_PCT":    _clamp(dist + LEARN_STEP_DIST, 10, 30),
            "ATRPCTL_GATE":    _clamp(atrg + LEARN_STEP_ATR,  0.25, 0.65),
            "BBZ_GATE":        _clamp(bbzg + LEARN_STEP_BBZ,  -2.5, -0.1),
            "VOL_CONFIRM_MULT":_clamp(volm - 0.01,             1.05, 1.40),
        })

    return ("KEEP", cur)

def blend_params(old: dict, new: dict, blend: float) -> dict:
    b = _clamp(blend, 0.0, 0.50)
    return {k: (1.0-b)*float(old.get(k, DEFAULT_PARAMS[k])) + b*float(new.get(k, old.get(k, DEFAULT_PARAMS[k])))
            for k in DEFAULT_PARAMS}

# =========================
# MAIN
# =========================
def main() -> None:
    log.info("=== weekly_eval v8.0 iniciado ===")

    if not SIGNALS_CSV.exists():
        tg_send("WEEKLY REVIEW: FAIL — signals.csv não existe.")
        raise RuntimeError("signals.csv missing")

    try:
        sig = pd.read_csv(SIGNALS_CSV)
    except Exception as e:
        tg_send(f"WEEKLY REVIEW: FAIL — {e}")
        raise

    if sig.empty:
        tg_send("WEEKLY REVIEW: FAIL — signals.csv vazio.")
        raise RuntimeError("signals.csv empty")

    sig.columns = [c.strip().lower() for c in sig.columns]

    need_cols = {"date","ticker","signal","close","trig","stop","dist_pct","overhead_touches","regime"}
    missing = [c for c in need_cols if c not in sig.columns]
    if missing:
        tg_send(f"WEEKLY REVIEW: FAIL — colunas em falta: {missing}")
        raise RuntimeError(f"missing cols: {missing}")

    sig["date"] = pd.to_datetime(sig["date"], errors="coerce").dt.normalize()
    sig = sig.dropna(subset=["date"]).copy()
    if "asof" in sig.columns:
        sig["asof"] = pd.to_datetime(sig["asof"], errors="coerce").dt.normalize()
        sig["eval_day"] = sig["asof"].where(sig["asof"].notna(), sig["date"])
    else:
        sig["eval_day"] = sig["date"]

    max_day = pd.to_datetime(sig["eval_day"], errors="coerce").dropna().max().normalize()
    lag_end   = (max_day - BDay(HORIZON)).normalize()
    lag_start = (lag_end  - BDay(WINDOW_DAYS - 1)).normalize()

    w = sig[(sig["eval_day"] >= lag_start) & (sig["eval_day"] <= lag_end)].copy()
    if w.empty:
        u = sorted(pd.to_datetime(sig["eval_day"], errors="coerce").dropna().dt.normalize().unique())
        last = [pd.Timestamp(x) for x in u[-WINDOW_DAYS:]]
        lag_start, lag_end = last[0], last[-1]
        w = sig[(sig["eval_day"] >= lag_start) & (sig["eval_day"] <= lag_end)].copy()

    week_end     = lag_end.strftime("%Y-%m-%d")
    window_start = lag_start.strftime("%Y-%m-%d")
    window_end   = lag_end.strftime("%Y-%m-%d")

    w_exec  = w[w["signal"].astype(str).str.contains("EXEC", na=False)].copy()
    w_watch = w[w["signal"].astype(str).str.upper().eq("WATCH")].copy()
    w_watch["overhead_touches"] = pd.to_numeric(w_watch["overhead_touches"], errors="coerce")
    w_watch["dist_pct"]         = pd.to_numeric(w_watch["dist_pct"], errors="coerce")

    w_clean = w_watch[w_watch["overhead_touches"] <= CLEAN_MAX_OH].copy()
    w_over  = w_watch[w_watch["overhead_touches"]  > CLEAN_MAX_OH].copy()

    regime_mode = "—"
    if not w_watch.empty:
        regime_mode = str(w_watch["regime"].astype(str).value_counts().index[0])

    # Evaluate
    eval_all   = [eval_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_watch.iterrows()]
    eval_clean = [eval_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_clean.iterrows()]
    eval_over  = [eval_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_over.iterrows()]
    eval_exec  = [eval_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_exec.iterrows()]

    m_all   = aggregate(eval_all)
    m_clean = aggregate(eval_clean)
    m_over  = aggregate(eval_over)
    m_exec  = aggregate(eval_exec)

    # FIX #7: regime breakdown
    regimes_all = w_watch["regime"].astype(str).tolist()
    succ_on   = regime_success(eval_all, regimes_all, "RISK_ON")
    succ_off  = regime_success(eval_all, regimes_all, "RISK_OFF")
    succ_tr   = regime_success(eval_all, regimes_all, "TRANSITION")

    # Baseline
    dist_targets = w_watch["dist_pct"].dropna().tolist()
    baseline_rows, base_dbg = baseline_sample(
        dist_targets, lag_end, HORIZON, len(w_watch), BREAKOUT_BUFFER_PCT
    )
    m_base = aggregate(baseline_rows)

    # Edge (success e MFE)
    edge_pp = None
    edge_mfe_pp = None
    if m_base["cov"] and m_all["cov"] and m_base["succ"] is not None and m_all["succ"] is not None:
        edge_pp = (m_all["succ"] - m_base["succ"]) * 100.0
    if m_base["cov"] and m_all["cov"] and m_base["mfe_med"] is not None and m_all["mfe_med"] is not None:
        edge_mfe_pp = (m_all["mfe_med"] - m_base["mfe_med"]) * 100.0

    # Rolling-4
    hist = read_weekly_summary()
    rolling_txt = "📈 ROLLING-4: sem histórico."
    if hist is not None and not hist.empty:
        h4 = hist.tail(4)
        try:
            succ_r = pd.to_numeric(h4.get("watch_success", np.nan), errors="coerce")
            mfe_r  = pd.to_numeric(h4.get("watch_mfe_med",  np.nan), errors="coerce")
            mae_r  = pd.to_numeric(h4.get("watch_mae_mean", np.nan), errors="coerce")
            edge_r = pd.to_numeric(h4.get("edge_pp",        np.nan), errors="coerce")
            cov_t  = int(pd.to_numeric(h4.get("watch_cov", 0), errors="coerce").fillna(0).sum())
            rolling_txt = (
                f"📈 ROLLING-4 ({len(h4)} semanas): "
                f"succ={pct(succ_r.mean() if len(succ_r.dropna()) else None)} | "
                f"MFE_med={fpct(mfe_r.mean() if len(mfe_r.dropna()) else None)} | "
                f"MAE={fpct(mae_r.mean() if len(mae_r.dropna()) else None)} | "
                f"edge={edge_r.mean():+.1f}pp avg | cov={cov_t}"
            )
        except Exception:
            rolling_txt = "📈 ROLLING-4: parsing falhou."

    # Learning
    learned = load_learned()
    cur_params = DEFAULT_PARAMS.copy()
    if isinstance(learned.get("params"), dict):
        for k in DEFAULT_PARAMS:
            try:
                cur_params[k] = float(learned["params"].get(k, cur_params[k]))
            except Exception:
                pass

    watch_n = len(w_watch)
    base_cov = int(m_base["cov"])
    base_cov_ratio = (base_cov / watch_n) if watch_n > 0 else 0.0
    base_ok = ((watch_n == 0) or (base_cov_ratio >= BASELINE_MIN_COV_RATIO))
    if BASELINE_NOFALLBACK_REQUIRED and base_dbg.get("fallback", 0) == 1:
        base_ok = False

    edge_val     = float(edge_pp) if edge_pp is not None and np.isfinite(edge_pp) else None
    edge_mfe_val = float(edge_mfe_pp) if edge_mfe_pp is not None and np.isfinite(edge_mfe_pp) else None

    if not base_ok or edge_val is None:
        action, blended = "NO_LEARN", cur_params
    else:
        action0, proposed = propose_update(edge_val, edge_mfe_val, int(m_all["cov"]), cur_params)
        if action0 in ("TIGHTEN","RELAX"):
            blended = blend_params(cur_params, proposed, LEARN_BLEND)
            action = action0
        else:
            blended, action = cur_params, action0

    save_learned(
        version=2, asof=week_end, blend=_clamp(LEARN_BLEND, 0, 0.5),
        params=blended,
        meta={
            "action": action, "edge_pp": edge_val, "edge_mfe_pp": edge_mfe_val,
            "cov_used": int(m_all["cov"]), "watch_count": watch_n,
            "baseline_cov": base_cov, "baseline_cov_ratio": round(base_cov_ratio,4),
            "baseline_fallback": int(base_dbg.get("fallback",0)),
            "breakout_buffer_pct": BREAKOUT_BUFFER_PCT,
        }
    )

    append_weekly_summary({
        "week_end": week_end, "window_start": window_start, "window_end": window_end,
        "horizon": HORIZON,
        "signals_total": len(w_exec) + len(w_watch),
        "exec_count": len(w_exec), "watch_count": len(w_watch),
        "watch_clean": len(w_clean), "watch_over": len(w_over),
        "watch_cov": int(m_all["cov"]),
        "watch_success": m_all["succ"] or "",
        "watch_h1": m_all["h1"] or "",
        "watch_ff": m_all["ff"] or "",
        "watch_mfe_med": m_all["mfe_med"] or "",
        "watch_mae_mean": m_all["mae_mean"] or "",
        "clean_success": m_clean["succ"] or "",
        "clean_mfe_med": m_clean["mfe_med"] or "",
        "clean_mae_mean": m_clean["mae_mean"] or "",
        "over_success": m_over["succ"] or "",
        "over_mfe_med": m_over["mfe_med"] or "",
        "over_mae_mean": m_over["mae_mean"] or "",
        "baseline_cov": base_cov,
        "baseline_success": m_base["succ"] or "",
        "baseline_mfe_med": m_base["mfe_med"] or "",
        "baseline_mae_mean": m_base["mae_mean"] or "",
        "edge_pp": edge_val or "",
        "baseline_files": int(base_dbg.get("files",0)),
        "baseline_picked": int(base_dbg.get("picked",0)),
        "baseline_fallback": int(base_dbg.get("fallback",0)),
        "regime_mode": regime_mode,
        "learn_action": action,
        "learn_edge_pp": edge_val or "",
        "learn_cov_used": int(m_all["cov"]),
        # v8 new
        "edge_mfe_pp": edge_mfe_val or "",
        "watch_hit_trig_rate": m_all["hit_trig"] or "",
        "days_to_trig_med": m_all["days_trig_med"] or "",
        "regime_risk_on_succ": succ_on or "",
        "regime_risk_off_succ": succ_off or "",
        "regime_transition_succ": succ_tr or "",
        "watch_clean_ff": m_clean["ff"] or "",
        "watch_over_ff": m_over["ff"] or "",
    })

    # === MENSAGEM ===
    pending_w = max(0, len(w_watch) - int(m_all["cov"]))
    pending_e = max(0, len(w_exec)  - int(m_exec["cov"]))
    edge_txt  = "—" if edge_val is None else f"{edge_val:+.1f}"
    emfe_txt  = "—" if edge_mfe_val is None else f"{edge_mfe_val:+.1f}"

    lines = [
        "📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v8.0)",
        f"Janela: {window_start} → {window_end}  (lag={HORIZON}d úteis)",
        f"Horizonte: {HORIZON} sessões | buf={BREAKOUT_BUFFER_PCT:.1f}% | MFE/MAE via HIGH/LOW",
        "",
        f"Sinais: {len(w_exec)+len(w_watch)} total  |  EXEC={len(w_exec)}  WATCH={len(w_watch)} (L={len(w_clean)} T={len(w_over)})",
    ]
    if pending_e + pending_w:
        lines.append(f"⚠️ Pending (sem futuro): EXEC={pending_e} WATCH={pending_w}")

    lines += [
        "",
        f"EXEC:   cov={m_exec['cov']}/{len(w_exec)}  S={pct(m_exec['succ'])} H1={pct(m_exec['h1'])} FF={pct(m_exec['ff'])}  MFE={fpct(m_exec['mfe_med'])}  MAE={fpct(m_exec['mae_mean'])}",
        f"WATCH:  cov={m_all['cov']}/{len(w_watch)}  S={pct(m_all['succ'])} H1={pct(m_all['h1'])} FF={pct(m_all['ff'])}  MFE={fpct(m_all['mfe_med'])}  MAE={fpct(m_all['mae_mean'])}",
        f"  LIMPO:  S={pct(m_clean['succ'])} FF={pct(m_clean['ff'])} MFE={fpct(m_clean['mfe_med'])} MAE={fpct(m_clean['mae_mean'])}  (n={len(w_clean)})",
        f"  TETO:   S={pct(m_over['succ'])}  FF={pct(m_over['ff'])} MFE={fpct(m_over['mfe_med'])}  MAE={fpct(m_over['mae_mean'])}   (n={len(w_over)})",
        "",
        f"⏱ Dias até trigger (mediana): {m_all['days_trig_med'] if m_all['days_trig_med'] else '—'}",
        "",
        "🌍 REGIME BREAKDOWN:",
        f"  RISK_ON:     S={pct(succ_on)}",
        f"  RISK_OFF:    S={pct(succ_off)}",
        f"  TRANSITION:  S={pct(succ_tr)}",
        "",
        "🧪 BASELINE:",
        f"  S={pct(m_base['succ'])}  MFE={fpct(m_base['mfe_med'])}  MAE={fpct(m_base['mae_mean'])}  cov={base_cov}/{watch_n}",
        f"  files={base_dbg.get('files',0)} picked={base_dbg.get('picked',0)} fallback={base_dbg.get('fallback',0)}",
        f"EDGE (success): {edge_txt} pp  |  EDGE (MFE): {emfe_txt} pp",
        "",
        "🧠 LEARNING:",
    ]

    if not base_ok or edge_val is None:
        why = []
        if not base_ok:   why.append("baseline inválido")
        if edge_val is None: why.append("edge indisponível")
        lines.append(f"  Action=NO_LEARN ({', '.join(why)}) | cov={int(m_all['cov'])}")
    else:
        lines.append(f"  Action={action} | cov={int(m_all['cov'])} | edge={edge_txt}pp | edge_mfe={emfe_txt}pp")

    lines.append(
        f"  Params → DIST={blended['DIST_MAX_PCT']:.1f} | ATRp={blended['ATRPCTL_GATE']:.2f} | "
        f"BBz={blended['BBZ_GATE']:.2f} | VOLx={blended['VOL_CONFIRM_MULT']:.2f}"
    )
    lines += ["", rolling_txt, ""]

    tg_send("\n".join(lines))
    log.info("=== weekly_eval v8.0 concluído ===")

if __name__ == "__main__":
    main()
