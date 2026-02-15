# weekly_eval.py — QUANT v7 (schema FIXO + rolling-4 + AUTO-LEARNING)
# Lê cache/signals.csv + cache/ohlcv/*.csv
# Calcula métricas semanais em janela "lagged" para evitar cov=0 por falta de futuro
# Escreve cache/weekly_summary.csv com colunas fixas (sem corrupção)
# Actualiza cache/learned_params.json (auto-learning, auditável)
# Envia resumo para Telegram
#
# Fixes:
# - Robustez total: elimina KeyError (ex: 'h1') com rows incompletos
# - Baseline não precisa de h1/ff
#
# Learning:
# - só ajusta parâmetros se cov >= LEARN_MIN_COV
# - usa EDGE (watch_success - baseline_success) como sinal
# - updates pequenos, com blend, e clamps
# - grava histórico em learned_params.json

import os, csv, json, random
from pathlib import Path
from datetime import datetime, timezone
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

HORIZON = int(os.environ.get("WEEKLY_HORIZON_SESS", "5"))  # sessões úteis
CLEAN_MAX_OH = int(os.environ.get("WATCH_OVERHEAD_CLEAN_MAX", "5"))
WINDOW_DAYS = 5

# Breakout buffer (ex: 0.5 => sucesso só se close > trig*1.005)
BREAKOUT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.0"))

# Learning knobs (seguros)
LEARN_MIN_COV = int(os.environ.get("LEARN_MIN_COV", "30"))
LEARN_BLEND = float(os.environ.get("LEARN_BLEND", "0.25"))  # 0..1 (quanto “mexe” por semana)
LEARN_STEP_DIST = float(os.environ.get("LEARN_STEP_DIST", "1.0"))   # pontos percentuais
LEARN_STEP_ATR = float(os.environ.get("LEARN_STEP_ATR", "0.02"))     # unidades (0..1)
LEARN_STEP_BBZ = float(os.environ.get("LEARN_STEP_BBZ", "0.05"))     # unidades (BBZ gate)

# Defaults (se learned_params ainda não existir)
DEFAULTS = {
    "DIST_MAX_PCT": float(os.environ.get("DIST_MAX_PCT", "18.0")),
    "ATRPCTL_GATE": float(os.environ.get("ATRPCTL_GATE", "0.50")),
    "BBZ_GATE": float(os.environ.get("BBZ_GATE", "-0.60")),
}

# clamps (para não descontrolar)
CLAMPS = {
    "DIST_MAX_PCT": (10.0, 26.0),
    "ATRPCTL_GATE": (0.30, 0.75),
    "BBZ_GATE": (-1.40, -0.20),  # mais negativo = mais exigente em compressão
}

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
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

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
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df
    except Exception:
        return None

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
    "breakout_buffer_pct",
    "learn_cov_ok","learn_edge_pp","learn_action",
    "learn_dist_max_pct","learn_atrpctl_gate","learn_bbz_gate"
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
# MFE/MAE
# =========================
def mfe_mae_from_entry(df: pd.DataFrame, pos: int, horizon: int, entry: float) -> tuple[float | None, float | None]:
    if entry <= 0:
        return (None, None)
    end = min(len(df) - 1, pos + horizon)
    sl = df.iloc[pos:end+1]["close"]
    if sl.empty:
        return (None, None)
    rets = (sl / entry) - 1.0
    mfe = float(np.nanmax(rets.values))
    mae = float(np.nanmin(rets.values))
    return (mfe, mae)

# =========================
# Evaluation
# =========================
def _empty_eval() -> dict:
    return {"covered": 0, "success": 0, "h1": 0, "ff": 0, "mfe": None, "mae": None}

def eval_watch_row(row: pd.Series, horizon: int, breakout_buffer_pct: float) -> dict:
    """
    covered=1 apenas se existir futuro suficiente (pos + horizon < len(df))
    Sucesso: algum close > trig*(1+buffer)
    H1: close_{+1} > trig*(1+buffer)
    Fail-fast: close < stop em <=2 sessões
    """
    out = _empty_eval()

    t = str(row.get("ticker", "")).strip().upper()
    trig = float(pd.to_numeric(row.get("trig", np.nan), errors="coerce"))
    stop = float(pd.to_numeric(row.get("stop", np.nan), errors="coerce"))
    entry_close = float(pd.to_numeric(row.get("close", np.nan), errors="coerce"))
    target = pick_target_date_from_row(row)

    if (not t) or pd.isna(target) or (not np.isfinite(trig)) or (not np.isfinite(entry_close)):
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

    buf = 1.0 + (breakout_buffer_pct / 100.0)
    trig_eff = trig * buf

    end = pos + horizon
    future = pd.to_numeric(df.iloc[pos:end+1]["close"], errors="coerce").dropna().values
    if len(future) == 0:
        return out

    out["success"] = 1 if np.any(future > trig_eff) else 0
    out["h1"] = 1 if float(df["close"].iloc[pos+1]) > trig_eff else 0

    ff = 0
    if np.isfinite(stop):
        for j in [pos+1, pos+2]:
            if j < len(df) and float(df["close"].iloc[j]) < stop:
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

    cov_rows = [r for r in rows_eval if int(r.get("covered", 0)) == 1]
    cov = len(cov_rows)
    if cov == 0:
        return {"cov": 0, "succ": None, "h1": None, "ff": None, "mfe_med": None, "mae_mean": None}

    succ = float(np.mean([int(r.get("success", 0)) for r in cov_rows]))
    h1 = float(np.mean([int(r.get("h1", 0)) for r in cov_rows]))
    ff = float(np.mean([int(r.get("ff", 0)) for r in cov_rows]))

    mfes = [r.get("mfe") for r in cov_rows]
    mfes = [float(x) for x in mfes if x is not None and np.isfinite(x)]
    maes = [r.get("mae") for r in cov_rows]
    maes = [float(x) for x in maes if x is not None and np.isfinite(x)]

    mfe_med = float(np.median(mfes)) if mfes else None
    mae_mean = float(np.mean(maes)) if maes else None

    return {"cov": cov, "succ": succ, "h1": h1, "ff": ff, "mfe_med": mfe_med, "mae_mean": mae_mean}

# =========================
# BASELINE (cache-only; robusto)
# =========================
def baseline_sample(dist_targets: list[float], window_end: pd.Timestamp, horizon: int, k: int, breakout_buffer_pct: float) -> list[dict]:
    """
    Baseline aproximado (cache-only):
    - escolhe tickers aleatórios do cache/ohlcv
    - trig_baseline = max(close últimos 20 dias antes do window_end)
    - dist% = (trig - close)/close
    - matching por buckets se dist_targets >= 6
    - devolve dicts com: covered, success, mfe, mae (sem h1/ff -> aggregate é robusto)
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

    picked: list[dict] = []
    tries = 0
    buf = 1.0 + (breakout_buffer_pct / 100.0)

    for p in files:
        if len(picked) >= k:
            break
        tries += 1
        if tries > 2500:
            break

        t = p.stem.upper()
        df = load_ohlcv(t)
        if df is None or len(df) < 80:
            continue

        pos = find_pos_best_effort(df, window_end.normalize())
        if pos is None or pos < 25:
            continue

        if pos + horizon >= len(df):
            continue

        close0 = float(df["close"].iloc[pos])
        if close0 <= 0:
            continue

        trig = float(df["close"].iloc[pos-20:pos].max())
        if trig <= 0:
            continue

        dist = (trig - close0) / close0 * 100.0
        if not np.isfinite(dist):
            continue

        if use_matching:
            b = bucket(dist)  # type: ignore[misc]
            if got[b] >= want.get(b, 0):  # type: ignore[union-attr]
                continue

        trig_eff = trig * buf
        end = pos + horizon
        future = df["close"].iloc[pos:end+1].values
        success = 1 if np.any(future > trig_eff) else 0
        mfe, mae = mfe_mae_from_entry(df, pos, horizon, close0)

        picked.append({"covered": 1, "success": success, "mfe": mfe, "mae": mae})

        if use_matching:
            got[b] += 1  # type: ignore[index]

    return picked

# =========================
# LEARNING (auditável)
# =========================
def _clamp(name: str, x: float) -> float:
    lo, hi = CLAMPS[name]
    return float(min(hi, max(lo, x)))

def load_learned_params() -> dict:
    ensure_parent(LEARNED_JSON)
    if not LEARNED_JSON.exists():
        return {
            "params": dict(DEFAULTS),
            "history": [],
            "meta": {"created_utc": datetime.now(timezone.utc).isoformat()}
        }
    try:
        d = json.loads(LEARNED_JSON.read_text(encoding="utf-8"))
        if "params" not in d or not isinstance(d["params"], dict):
            raise ValueError("bad learned schema")
        for k in DEFAULTS:
            if k not in d["params"] or not np.isfinite(float(d["params"][k])):
                d["params"][k] = DEFAULTS[k]
        if "history" not in d or not isinstance(d["history"], list):
            d["history"] = []
        if "meta" not in d or not isinstance(d["meta"], dict):
            d["meta"] = {}
        return d
    except Exception:
        return {
            "params": dict(DEFAULTS),
            "history": [{"ts_utc": datetime.now(timezone.utc).isoformat(), "note": "reset_invalid_json"}],
            "meta": {"reset_utc": datetime.now(timezone.utc).isoformat()}
        }

def save_learned_params(obj: dict) -> None:
    ensure_parent(LEARNED_JSON)
    LEARNED_JSON.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def learning_update(week_end: str, regime_mode: str, m_watch: dict, m_base: dict) -> dict:
    """
    Decide e aplica update. Retorna: {did_update, action, params, edge_pp, cov_ok}
    """
    learned = load_learned_params()
    params = learned["params"]

    cov_watch = int(m_watch.get("cov", 0) or 0)
    cov_base = int(m_base.get("cov", 0) or 0)
    cov_ok = (cov_watch >= LEARN_MIN_COV) and (cov_base >= max(10, int(0.6 * LEARN_MIN_COV)))

    succ_w = m_watch.get("succ", None)
    succ_b = m_base.get("succ", None)
    edge_pp = None
    if succ_w is not None and succ_b is not None and np.isfinite(succ_w) and np.isfinite(succ_b):
        edge_pp = (float(succ_w) - float(succ_b)) * 100.0

    action = "NOOP"
    did = False

    # sem cov suficiente ou sem edge -> não mexe
    if (not cov_ok) or (edge_pp is None) or (not np.isfinite(edge_pp)):
        learned["history"].append({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "week_end": week_end,
            "regime_mode": regime_mode,
            "cov_watch": cov_watch,
            "cov_base": cov_base,
            "edge_pp": edge_pp,
            "action": "NOOP_cov_or_edge"
        })
        save_learned_params(learned)
        return {
            "did_update": False,
            "action": "NOOP_cov_or_edge",
            "params": dict(params),
            "edge_pp": edge_pp,
            "cov_ok": cov_ok
        }

    # regra direccional (simples e estável)
    # edge positivo => relaxa ligeiramente; edge negativo => aperta ligeiramente
    if edge_pp > 0.0:
        direction = +1
        action = "RELAX"
    elif edge_pp < 0.0:
        direction = -1
        action = "TIGHTEN"
    else:
        direction = 0
        action = "NOOP_edge_zero"

    if direction == 0:
        learned["history"].append({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "week_end": week_end,
            "regime_mode": regime_mode,
            "cov_watch": cov_watch,
            "cov_base": cov_base,
            "edge_pp": edge_pp,
            "action": action
        })
        save_learned_params(learned)
        return {
            "did_update": False,
            "action": action,
            "params": dict(params),
            "edge_pp": edge_pp,
            "cov_ok": cov_ok
        }

    # current
    dist = float(params.get("DIST_MAX_PCT", DEFAULTS["DIST_MAX_PCT"]))
    atrg = float(params.get("ATRPCTL_GATE", DEFAULTS["ATRPCTL_GATE"]))
    bbzg = float(params.get("BBZ_GATE", DEFAULTS["BBZ_GATE"]))

    # targets (um passo)
    # RELAX: dist ↑ ; atr_gate ↑ ; bbz_gate ↑ (menos negativo, mais permissivo)
    # TIGHTEN: dist ↓ ; atr_gate ↓ ; bbz_gate ↓ (mais negativo, mais exigente)
    dist_t = dist + direction * LEARN_STEP_DIST
    atrg_t = atrg + direction * LEARN_STEP_ATR
    bbzg_t = bbzg + direction * LEARN_STEP_BBZ

    # blend + clamp
    dist_n = _clamp("DIST_MAX_PCT", dist * (1 - LEARN_BLEND) + dist_t * LEARN_BLEND)
    atrg_n = _clamp("ATRPCTL_GATE", atrg * (1 - LEARN_BLEND) + atrg_t * LEARN_BLEND)
    bbzg_n = _clamp("BBZ_GATE", bbzg * (1 - LEARN_BLEND) + bbzg_t * LEARN_BLEND)

    params["DIST_MAX_PCT"] = float(dist_n)
    params["ATRPCTL_GATE"] = float(atrg_n)
    params["BBZ_GATE"] = float(bbzg_n)

    learned["params"] = params
    learned["history"].append({
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "week_end": week_end,
        "regime_mode": regime_mode,
        "cov_watch": cov_watch,
        "cov_base": cov_base,
        "edge_pp": float(edge_pp),
        "action": action,
        "delta": {
            "DIST_MAX_PCT": float(dist_n - dist),
            "ATRPCTL_GATE": float(atrg_n - atrg),
            "BBZ_GATE": float(bbzg_n - bbzg)
        },
        "new_params": dict(params)
    })
    learned["meta"]["last_update_utc"] = datetime.now(timezone.utc).isoformat()
    learned["meta"]["last_week_end"] = week_end

    save_learned_params(learned)
    did = True

    return {
        "did_update": did,
        "action": action,
        "params": dict(params),
        "edge_pp": float(edge_pp),
        "cov_ok": cov_ok
    }

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

    # -------------------------
    # Janela "lagged":
    # lag_end = max(eval_day) - HORIZON BDays
    # window = [lag_end - (WINDOW_DAYS-1) BDays, lag_end]
    # fallback: últimos 5 dias únicos
    # -------------------------
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

    eval_all = [eval_watch_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_watch.iterrows()]
    eval_clean = [eval_watch_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_watch_clean.iterrows()]
    eval_over = [eval_watch_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_watch_over.iterrows()]
    eval_exec = [eval_watch_row(r, HORIZON, BREAKOUT_BUFFER_PCT) for _, r in w_exec.iterrows()]

    m_all = aggregate_metrics(eval_all)
    m_clean = aggregate_metrics(eval_clean)
    m_over = aggregate_metrics(eval_over)
    m_exec = aggregate_metrics(eval_exec)

    # baseline
    dist_targets = w_watch["dist_pct"].dropna().tolist()
    baseline = baseline_sample(dist_targets, lag_end, HORIZON, k=len(w_watch), breakout_buffer_pct=BREAKOUT_BUFFER_PCT)
    m_base = aggregate_metrics(baseline)

    edge_pp = None
    if m_base["cov"] and m_all["cov"] and (m_base["succ"] is not None) and (m_all["succ"] is not None):
        edge_pp = (float(m_all["succ"]) - float(m_base["succ"])) * 100.0

    # auto-learning (update learned_params.json)
    learn = learning_update(week_end=week_end, regime_mode=regime_mode, m_watch=m_all, m_base=m_base)

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

    # SANITY
    sanity = "OK"
    if w_watch.empty and w_exec.empty:
        sanity = "WARN (sem sinais WATCH/EXEC na janela)"

    # escrever summary row (schema fixo)
    lp = learn["params"]
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

        "watch_cov": int(m_all["cov"]),
        "watch_success": (m_all["succ"] if m_all["succ"] is not None else ""),
        "watch_h1": (m_all["h1"] if m_all["h1"] is not None else ""),
        "watch_ff": (m_all["ff"] if m_all["ff"] is not None else ""),
        "watch_mfe_med": (m_all["mfe_med"] if m_all["mfe_med"] is not None else ""),
        "watch_mae_mean": (m_all["mae_mean"] if m_all["mae_mean"] is not None else ""),

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
        "breakout_buffer_pct": BREAKOUT_BUFFER_PCT,

        "learn_cov_ok": (1 if learn["cov_ok"] else 0),
        "learn_edge_pp": (learn["edge_pp"] if learn["edge_pp"] is not None else ""),
        "learn_action": learn["action"],
        "learn_dist_max_pct": lp.get("DIST_MAX_PCT", ""),
        "learn_atrpctl_gate": lp.get("ATRPCTL_GATE", ""),
        "learn_bbz_gate": lp.get("BBZ_GATE", "")
    })

    # PENDING (sem futuro suficiente)
    pending_exec = max(0, len(w_exec) - int(m_exec["cov"] or 0))
    pending_watch = max(0, len(w_watch) - int(m_all["cov"] or 0))

    # mensagem
    lines = []
    lines.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v7)")
    lines.append(f"Janela (lagged): {window_start} → {window_end}  (end=max(eval_day)-{HORIZON}d úteis)")
    lines.append(f"Horizonte métricas: {HORIZON} sessões | breakout_buffer={BREAKOUT_BUFFER_PCT:.2f}%")
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
        f"WATCH: cov={m_all['cov']}/{len(w_watch)}  "
        f"S={pct(m_all['succ'])} H1={pct(m_all['h1'])} FF={pct(m_all['ff'])}  "
        f"MFE_med={fmt_pct_from_ret(m_all['mfe_med'],1)}  "
        f"MAE={fmt_pct_from_ret(m_all['mae_mean'],1)}"
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
    lines.append("🤖 LEARNING:")
    lines.append(
        f"cov_ok={1 if learn['cov_ok'] else 0} | edge_pp={('—' if learn['edge_pp'] is None else f'{learn['edge_pp']:+.1f}')} | action={learn['action']}"
    )
    lines.append(
        f"params -> DIST_MAX_PCT={lp.get('DIST_MAX_PCT', DEFAULTS['DIST_MAX_PCT']):.2f} | "
        f"ATRPCTL_GATE={lp.get('ATRPCTL_GATE', DEFAULTS['ATRPCTL_GATE']):.3f} | "
        f"BBZ_GATE={lp.get('BBZ_GATE', DEFAULTS['BBZ_GATE']):.3f}"
    )

    lines.append("")
    lines.append("🧭 REGIME (WATCH success):")
    lines.append(f"{regime_mode}: {pct(m_all['succ'])}")
    lines.append("")
    lines.append(f"✅ SANITY: {sanity}")
    lines.append("")
    lines.append(rolling_txt)
    lines.append("")
    lines.append("Ações:")
    lines.append("• weekly_summary.csv actualizado (base para decisões).")
    lines.append("• learned_params.json actualizado (se cov_ok=1).")
    lines.append("• Nota: o daily só melhora se o scanner.py ler learned_params.json (overrides).")

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()
