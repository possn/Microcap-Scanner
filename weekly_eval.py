# weekly_eval.py — QUANT v7.2 (SCHEMA FIXO + LAGGED WINDOW + BASELINE ROBUSTO + AUTO-LEARNING COM GUARDRAILS)
# - Lê cache/signals.csv + cache/ohlcv/*.csv
# - Calcula métricas semanais em janela "lagged" (evita cov=0 por falta de futuro)
# - Escreve cache/weekly_summary.csv com colunas fixas (e valida header; LOUD FAIL se mismatch)
# - Atualiza cache/learned_params.json (aprendizagem conservadora, só se baseline válido)
# - Envia resumo para Telegram
#
# Política: se weekly_summary.csv existir e tiver header diferente => FAIL (job vermelho).
# Isto evita corrupção silenciosa do histórico.

import os
import csv
import json
import random
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

HORIZON = int(os.environ.get("WEEKLY_HORIZON_SESS", "5"))  # sessões úteis para métricas
CLEAN_MAX_OH = int(os.environ.get("WATCH_OVERHEAD_CLEAN_MAX", "5"))
WINDOW_DAYS = 5  # sempre 5 dias úteis

# learning knobs
BREAKOUT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.5"))  # buffer p/ trig (evita falsos)
LEARN_MIN_COV = int(os.environ.get("LEARN_MIN_COV", "30"))                 # cobertura mínima para aprender
LEARN_BLEND = float(os.environ.get("LEARN_BLEND", "0.25"))                 # 0..0.5 (conservador)
LEARN_STEP_DIST = float(os.environ.get("LEARN_STEP_DIST", "1.0"))          # pontos percentuais
LEARN_STEP_ATR = float(os.environ.get("LEARN_STEP_ATR", "0.02"))           # 0.xx
LEARN_STEP_BBZ = float(os.environ.get("LEARN_STEP_BBZ", "0.05"))           # z-score units

# defaults (devem bater com o daily)
DEFAULT_PARAMS = {
    "DIST_MAX_PCT": 18.0,
    "ATRPCTL_GATE": 0.50,
    "BBZ_GATE": -0.60,
    "VOL_CONFIRM_MULT": 1.12,
}

# baseline guardrails
BASELINE_MIN_COV_RATIO = 0.80  # baseline_cov tem de cobrir pelo menos 80% de WATCH count
BASELINE_NOFALLBACK_REQUIRED = True  # se houve fallback, não aprende

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
# weekly_summary schema FIXO (v7.2)
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
    "baseline_files","baseline_picked","baseline_fallback",
    "regime_mode",
    "learn_action","learn_edge_pp","learn_cov_used"
]

def _read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.readline().strip("\n").strip("\r").strip()

def weekly_summary_ensure_schema_loud_fail() -> None:
    """
    Se o CSV existir e o header não for EXACTAMENTE o esperado, falha o job.
    Nunca reescreve em silêncio.
    """
    ensure_parent(SUMMARY_CSV)
    if not SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SUMMARY_HEADER)
        return

    first = _read_first_line(SUMMARY_CSV)
    expected = ",".join(SUMMARY_HEADER)

    if first != expected:
        tg_send(
            "WEEKLY REVIEW: FAIL — weekly_summary.csv header mismatch.\n"
            "Ação: apagar/renomear cache/weekly_summary.csv e re-run.\n"
            f"Expected: {expected}\n"
            f"Found:    {first}"
        )
        raise RuntimeError("weekly_summary.csv header mismatch (LOUD FAIL)")

def append_weekly_summary(row: dict) -> None:
    weekly_summary_ensure_schema_loud_fail()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
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
# Formatting
# =========================
def pct(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.0f}%"

def fmt_pct_from_ret(x: float | None, nd: int = 1) -> str:
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
# Evaluation row
# =========================
def eval_watch_row(row: pd.Series, horizon: int, breakout_buffer_pct: float) -> dict:
    """
    covered=1 apenas se existir futuro suficiente (pos + horizon < len(df))
    Sucesso: algum close > trig*(1+buffer) no horizonte
    H1: close_{+1} > trig*(1+buffer)
    Fail-fast: close < stop em <=2 sessões
    """
    out = {"covered": 0, "success": 0, "h1": 0, "ff": 0, "mfe": None, "mae": None}

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

    trig_eff = trig * (1.0 + breakout_buffer_pct / 100.0)

    end = pos + horizon
    future = df.iloc[pos:end+1]["close"].values
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

    cov = sum(1 for r in rows_eval if int(r.get("covered", 0)) == 1)
    if cov == 0:
        return {"cov": 0, "succ": None, "h1": None, "ff": None, "mfe_med": None, "mae_mean": None}

    succ = float(np.mean([float(r.get("success", 0)) for r in rows_eval if int(r.get("covered", 0)) == 1]))
    h1 = float(np.mean([float(r.get("h1", 0)) for r in rows_eval if int(r.get("covered", 0)) == 1]))
    ff = float(np.mean([float(r.get("ff", 0)) for r in rows_eval if int(r.get("covered", 0)) == 1]))

    mfes = [r.get("mfe") for r in rows_eval if int(r.get("covered", 0)) == 1]
    maes = [r.get("mae") for r in rows_eval if int(r.get("covered", 0)) == 1]
    mfes = [float(x) for x in mfes if x is not None and np.isfinite(x)]
    maes = [float(x) for x in maes if x is not None and np.isfinite(x)]

    mfe_med = float(np.median(mfes)) if mfes else None
    mae_mean = float(np.mean(maes)) if maes else None

    return {"cov": cov, "succ": succ, "h1": h1, "ff": ff, "mfe_med": mfe_med, "mae_mean": mae_mean}

# =========================
# BASELINE (cache-only; matching por dist% se possível; fallback robusto)
# =========================
def baseline_sample(
    dist_targets: list[float],
    window_end: pd.Timestamp,
    horizon: int,
    k: int,
    breakout_buffer_pct: float
) -> tuple[list[dict], dict]:
    """
    Retorna (picked_rows, dbg):
      dbg = {files, picked, fallback}
    Baseline aproximado (cache-only):
      - escolhe tickers aleatórios do cache/ohlcv
      - trig_baseline = max(close últimos 20 dias antes do window_end)
      - dist% = (trig - close)/close
      - matching por buckets se dist_targets>=6
      - se matching falhar (poucos picked), faz fallback sem matching para completar k
      - métricas iguais a WATCH (success/h1/ff/mfe/mae)
    """
    dbg = {"files": 0, "picked": 0, "fallback": 0}

    if k <= 0 or (not OHLCV_DIR.exists()):
        return ([], dbg)

    files = [p for p in OHLCV_DIR.iterdir() if p.is_file() and p.name.endswith(".csv")]
    dbg["files"] = len(files)
    if not files:
        return ([], dbg)
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

    def _try_pick(files_iter, enforce_matching: bool) -> list[dict]:
        picked: list[dict] = []
        tries = 0

        for p in files_iter:
            if len(picked) >= k:
                break
            tries += 1
            if tries > 3000:
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

            if enforce_matching and use_matching and bucket and want and got:
                b = bucket(dist)
                if got[b] >= want.get(b, 0):
                    continue

            trig_eff = trig * (1.0 + breakout_buffer_pct / 100.0)
            end = pos + horizon
            future = df.iloc[pos:end+1]["close"].values

            success = 1 if np.any(future > trig_eff) else 0
            h1 = 1 if float(df["close"].iloc[pos+1]) > trig_eff else 0

            ff = 0
            stop = float(df["close"].iloc[pos-10:pos].min()) * 0.95  # stop baseline genérico
            for j in [pos+1, pos+2]:
                if j < len(df) and float(df["close"].iloc[j]) < stop:
                    ff = 1
                    break

            mfe, mae = mfe_mae_from_entry(df, pos, horizon, close0)

            picked.append({"covered": 1, "success": success, "h1": h1, "ff": ff, "mfe": mfe, "mae": mae})

            if enforce_matching and use_matching and got is not None:
                got[b] += 1

        return picked

    # Passo 1: matching (se possível)
    picked = _try_pick(files, enforce_matching=True) if use_matching else _try_pick(files, enforce_matching=False)

    # Fallback: se matching não completou k, tenta completar sem matching
    if len(picked) < k and use_matching:
        dbg["fallback"] = 1
        remaining = k - len(picked)
        # reset got para não bloquear
        picked2 = _try_pick(files, enforce_matching=False)
        # junta sem duplicar “covered rows” (não temos ticker no dict; aceitamos aproximação)
        picked.extend(picked2[:remaining])

    dbg["picked"] = len(picked)
    return (picked, dbg)

# =========================
# Learning params helpers
# =========================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def load_learned_params() -> dict:
    if not LEARNED_JSON.exists():
        return {}
    try:
        d = json.loads(LEARNED_JSON.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def save_learned_params(version: int, asof: str, blend: float, params: dict, meta: dict) -> None:
    ensure_parent(LEARNED_JSON)
    payload = {
        "version": int(version),
        "asof": str(asof),
        "blend": float(blend),
        "params": params,
        "meta": meta,
    }
    LEARNED_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def propose_learning_update(edge_pp: float | None, cov: int, current_params: dict) -> tuple[str, dict]:
    """
    Política conservadora:
    - Só propõe mudança se cov >= LEARN_MIN_COV e edge_pp válido.
    - Se edge_pp < 0: tornar gates ligeiramente MAIS estritos.
    - Se edge_pp > +5: relaxar ligeiramente.
    - Caso contrário: KEEP.
    """
    if edge_pp is None or (not np.isfinite(edge_pp)) or cov < LEARN_MIN_COV:
        return ("NO_LEARN", current_params)

    dist = float(current_params.get("DIST_MAX_PCT", DEFAULT_PARAMS["DIST_MAX_PCT"]))
    atrg = float(current_params.get("ATRPCTL_GATE", DEFAULT_PARAMS["ATRPCTL_GATE"]))
    bbzg = float(current_params.get("BBZ_GATE", DEFAULT_PARAMS["BBZ_GATE"]))
    volm = float(current_params.get("VOL_CONFIRM_MULT", DEFAULT_PARAMS["VOL_CONFIRM_MULT"]))

    if edge_pp < 0:
        dist = _clamp(dist - LEARN_STEP_DIST, 10.0, 30.0)
        atrg = _clamp(atrg - LEARN_STEP_ATR, 0.25, 0.65)
        bbzg = _clamp(bbzg - LEARN_STEP_BBZ, -2.50, -0.10)  # mais negativo = mais estrito
        volm = _clamp(volm + 0.01, 1.05, 1.40)
        return ("TIGHTEN", {"DIST_MAX_PCT": dist, "ATRPCTL_GATE": atrg, "BBZ_GATE": bbzg, "VOL_CONFIRM_MULT": volm})

    if edge_pp > 5:
        dist = _clamp(dist + LEARN_STEP_DIST, 10.0, 30.0)
        atrg = _clamp(atrg + LEARN_STEP_ATR, 0.25, 0.65)
        bbzg = _clamp(bbzg + LEARN_STEP_BBZ, -2.50, -0.10)  # menos negativo = mais permissivo
        volm = _clamp(volm - 0.01, 1.05, 1.40)
        return ("RELAX", {"DIST_MAX_PCT": dist, "ATRPCTL_GATE": atrg, "BBZ_GATE": bbzg, "VOL_CONFIRM_MULT": volm})

    return ("KEEP", current_params)

def blend_params(old_params: dict, new_params: dict, blend: float) -> dict:
    b = _clamp(float(blend), 0.0, 0.50)

    def _mix(k: str) -> float:
        a = float(old_params.get(k, DEFAULT_PARAMS[k]))
        n = float(new_params.get(k, a))
        return (1.0 - b) * a + b * n

    return {
        "DIST_MAX_PCT": _mix("DIST_MAX_PCT"),
        "ATRPCTL_GATE": _mix("ATRPCTL_GATE"),
        "BBZ_GATE": _mix("BBZ_GATE"),
        "VOL_CONFIRM_MULT": _mix("VOL_CONFIRM_MULT"),
    }

# =========================
# MAIN
# =========================
def main() -> None:
    if not SIGNALS_CSV.exists():
        tg_send("WEEKLY REVIEW: FAIL — cache/signals.csv não existe.")
        raise RuntimeError("signals.csv missing")

    try:
        sig = pd.read_csv(SIGNALS_CSV)
    except Exception:
        tg_send("WEEKLY REVIEW: FAIL — não consegui ler cache/signals.csv.")
        raise

    if sig.empty:
        tg_send("WEEKLY REVIEW: FAIL — signals.csv vazio.")
        raise RuntimeError("signals.csv empty")

    sig.columns = [c.strip().lower() for c in sig.columns]

    need_cols = {"date", "ticker", "signal", "close", "trig", "stop", "dist_pct", "overhead_touches", "regime"}
    missing = [c for c in need_cols if c not in sig.columns]
    if missing:
        tg_send(f"WEEKLY REVIEW: FAIL — signals.csv sem colunas: {missing}")
        raise RuntimeError(f"signals.csv missing cols: {missing}")

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
        raise RuntimeError("no valid dates")

    max_day = max_day.normalize()
    lag_end = (max_day - BDay(HORIZON)).normalize()
    lag_start = (lag_end - BDay(WINDOW_DAYS - 1)).normalize()

    w = sig[(sig["eval_day"] >= lag_start) & (sig["eval_day"] <= lag_end)].copy()

    if w.empty:
        # fallback: últimos 5 dias únicos disponíveis
        u = pd.to_datetime(sig["eval_day"], errors="coerce").dropna().dt.normalize().unique()
        u = sorted(u)
        if not u:
            tg_send("WEEKLY REVIEW: FAIL — sem datas válidas.")
            raise RuntimeError("no valid eval_day uniques")
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

    # baseline (+ dbg)
    dist_targets = w_watch["dist_pct"].dropna().tolist()
    baseline_rows, base_dbg = baseline_sample(
        dist_targets=dist_targets,
        window_end=lag_end,
        horizon=HORIZON,
        k=len(w_watch),
        breakout_buffer_pct=BREAKOUT_BUFFER_PCT
    )
    m_base = aggregate_metrics(baseline_rows)

    edge_pp = None
    if (int(m_base["cov"]) and int(m_all["cov"]) and (m_base["succ"] is not None) and (m_all["succ"] is not None)):
        edge_pp = (float(m_all["succ"]) - float(m_base["succ"])) * 100.0

    # rolling-4
    hist = read_weekly_summary()
    rolling_txt = "📈 ROLLING-4: sem histórico (weekly_summary.csv vazio)."
    if hist is not None and (not hist.empty):
        h4 = hist.tail(4).copy()
        try:
            cov_total = int(pd.to_numeric(h4.get("watch_cov", 0), errors="coerce").fillna(0).sum())
            succ = pd.to_numeric(h4.get("watch_success", np.nan), errors="coerce")
            mae = pd.to_numeric(h4.get("watch_mae_mean", np.nan), errors="coerce")
            mfe = pd.to_numeric(h4.get("watch_mfe_med", np.nan), errors="coerce")

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

    # =========================
    # AUTO-LEARNING (com guardrails)
    # =========================
    learned = load_learned_params()
    cur_params = DEFAULT_PARAMS.copy()
    if isinstance(learned.get("params", None), dict):
        for k in DEFAULT_PARAMS:
            try:
                cur_params[k] = float(learned["params"].get(k, cur_params[k]))
            except Exception:
                pass

    # baseline validity
    watch_n = int(len(w_watch))
    base_cov = int(m_base["cov"])
    base_cov_ratio = (base_cov / watch_n) if watch_n > 0 else 0.0
    base_ok = (watch_n == 0) or (base_cov_ratio >= BASELINE_MIN_COV_RATIO)
    if BASELINE_NOFALLBACK_REQUIRED and int(base_dbg.get("fallback", 0)) == 1:
        base_ok = False

    edge_val = None if edge_pp is None or (not np.isfinite(edge_pp)) else float(edge_pp)

    if (not base_ok) or (edge_val is None):
        action = "NO_LEARN"
        blended = cur_params
    else:
        action0, proposed = propose_learning_update(edge_pp=edge_val, cov=int(m_all["cov"]), current_params=cur_params)
        if action0 in ("TIGHTEN", "RELAX"):
            blended = blend_params(cur_params, proposed, LEARN_BLEND)
            action = action0
        else:
            blended = cur_params
            action = action0  # KEEP ou NO_LEARN

    save_learned_params(
        version=1,
        asof=week_end,
        blend=float(_clamp(LEARN_BLEND, 0.0, 0.50)),
        params=blended,
        meta={
            "action": action,
            "edge_pp": edge_val,
            "cov_used": int(m_all["cov"]),
            "watch_count": watch_n,
            "baseline_cov": base_cov,
            "baseline_cov_ratio": round(base_cov_ratio, 4),
            "baseline_fallback": int(base_dbg.get("fallback", 0)),
            "breakout_buffer_pct": BREAKOUT_BUFFER_PCT,
            "notes": "Weekly learning with baseline validity guardrails; daily reads params and blends gates."
        }
    )

    # =========================
    # persist weekly summary (LOUD header check inside)
    # =========================
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
        "edge_pp": (edge_val if edge_val is not None else ""),

        "baseline_files": int(base_dbg.get("files", 0)),
        "baseline_picked": int(base_dbg.get("picked", 0)),
        "baseline_fallback": int(base_dbg.get("fallback", 0)),

        "regime_mode": regime_mode,

        "learn_action": action,
        "learn_edge_pp": (edge_val if edge_val is not None else ""),
        "learn_cov_used": int(m_all["cov"]),
    })

    # PENDING
    pending_exec = max(0, len(w_exec) - int(m_exec["cov"]))
    pending_watch = max(0, len(w_watch) - int(m_all["cov"]))

    edge_txt = "—" if edge_val is None else f"{edge_val:+.1f}"

    # baseline string
    if len(w_watch) == 0:
        base_line = "BASE: cov=0/0  S=—  MFE_med=—  MAE=—"
        edge_line = "EDGE: —"
    elif int(m_base["cov"]) == 0:
        base_line = f"BASE: cov=0/{len(w_watch)}  S=—  MFE_med=—  MAE=—"
        edge_line = "EDGE: —"
    else:
        base_line = (
            f"BASE: cov={m_base['cov']}/{len(w_watch)}  "
            f"S={pct(m_base['succ'])}  "
            f"MFE_med={fmt_pct_from_ret(m_base['mfe_med'],1)}  "
            f"MAE={fmt_pct_from_ret(m_base['mae_mean'],1)}"
        )
        edge_line = f"EDGE: {edge_txt} pp"

    # mensagem
    lines: list[str] = []
    lines.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v7.2)")
    lines.append(f"Janela (lagged): {window_start} → {window_end}  (end=max(eval_day)-{HORIZON}d úteis)")
    lines.append(f"Horizonte métricas: {HORIZON} sessões | breakout_buffer={BREAKOUT_BUFFER_PCT:.1f}%")
    lines.append("")
    lines.append(f"Sinais na janela: {len(w_exec) + len(w_watch)}")
    lines.append(f"• EXEC: {len(w_exec)} | WATCH: {len(w_watch)} (LIMPO: {len(w_watch_clean)} | TETO: {len(w_watch_over)})")
    if (pending_exec + pending_watch) > 0:
        lines.append(f"• PENDING (sem futuro suficiente p/ horizonte): EXEC={pending_exec} WATCH={pending_watch}")
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
    lines.append("🧪 BASELINE (cache-only; matched se possível; fallback se necessário):")
    lines.append(base_line)
    lines.append(edge_line)
    lines.append(f"BASE_DBG: files={int(base_dbg.get('files',0))} picked={int(base_dbg.get('picked',0))} fallback={int(base_dbg.get('fallback',0))}")

    lines.append("")
    lines.append("🧠 LEARNING:")
    if not base_ok or edge_val is None:
        why = []
        if not base_ok:
            why.append("baseline inválido")
        if edge_val is None:
            why.append("edge indisponível")
        lines.append(f"Action=NO_LEARN ({', '.join(why)}) | cov_used={int(m_all['cov'])} | edge_pp={edge_txt}")
    else:
        lines.append(f"Action={action} | cov_used={int(m_all['cov'])} | edge_pp={edge_txt}")

    lines.append(
        f"learned_params.json → DIST_MAX_PCT={blended['DIST_MAX_PCT']:.1f} | "
        f"ATRPCTL_GATE={blended['ATRPCTL_GATE']:.2f} | BBZ_GATE={blended['BBZ_GATE']:.2f} | "
        f"VOL_CONFIRM_MULT={blended['VOL_CONFIRM_MULT']:.2f}"
    )

    lines.append("")
    lines.append(f"✅ SANITY: {sanity}")
    lines.append("")
    lines.append(rolling_txt)
    lines.append("")
    lines.append("Notas:")
    lines.append("• weekly_summary.csv: schema fixo e LOUD FAIL se header não bater (evita corrupção silenciosa).")
    lines.append("• Aprendizagem só mexe nos gates se houver cobertura suficiente e baseline válido (sem fallback).")
    lines.append("• Ajustes são conservadores (LEARN_BLEND<=0.50); o daily ainda faz blending adicional.")

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()
