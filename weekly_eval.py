# weekly_eval.py — QUANT v7.1 (SCHEMA FIXO + REPAIR + UPSERT + LAGGED WINDOW + BASELINE + AUTO-LEARNING)
# - Repara cache/weekly_summary.csv se estiver corrompido (colunas inconsistentes)
# - Garante schema fixo
# - Faz upsert por week_end (evita duplicados)
# - Lê cache/signals.csv + cache/ohlcv/*.csv
# - Calcula métricas semanais em janela "lagged"
# - Baseline cache-only
# - Atualiza cache/learned_params.json (aprendizagem conservadora)
# - Envia resumo Telegram

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

CACHE_DIR = Path("cache")
OHLCV_DIR = CACHE_DIR / "ohlcv"
SIGNALS_CSV = CACHE_DIR / "signals.csv"
SUMMARY_CSV = CACHE_DIR / "weekly_summary.csv"
LEARNED_JSON = CACHE_DIR / "learned_params.json"

HORIZON = int(os.environ.get("WEEKLY_HORIZON_SESS", "5"))          # sessões úteis
CLEAN_MAX_OH = int(os.environ.get("WATCH_OVERHEAD_CLEAN_MAX", "5"))
WINDOW_DAYS = 5                                                    # sempre 5 dias úteis

# learning knobs
BREAKOUT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.5"))
LEARN_MIN_COV = int(os.environ.get("LEARN_MIN_COV", "30"))
LEARN_BLEND = float(os.environ.get("LEARN_BLEND", "0.25"))
LEARN_STEP_DIST = float(os.environ.get("LEARN_STEP_DIST", "1.0"))
LEARN_STEP_ATR = float(os.environ.get("LEARN_STEP_ATR", "0.02"))
LEARN_STEP_BBZ = float(os.environ.get("LEARN_STEP_BBZ", "0.05"))

DEFAULT_PARAMS = {
    "DIST_MAX_PCT": 18.0,
    "ATRPCTL_GATE": 0.50,
    "BBZ_GATE": -0.60,
    "VOL_CONFIRM_MULT": 1.12,
}

# =========================
# weekly_summary schema FIXO
# =========================
SUMMARY_HEADER = [
    "week_end", "window_start", "window_end", "horizon",
    "signals_total", "exec_count", "watch_count",
    "watch_clean", "watch_over",
    "watch_cov", "watch_success", "watch_h1", "watch_ff",
    "watch_mfe_med", "watch_mae_mean",
    "clean_success", "clean_mfe_med", "clean_mae_mean",
    "over_success", "over_mfe_med", "over_mae_mean",
    "baseline_cov", "baseline_success", "baseline_mfe_med", "baseline_mae_mean", "edge_pp",
    "regime_mode",
    "learn_action", "learn_edge_pp", "learn_cov_used"
]

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
# Utils
# =========================
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct(x) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if not np.isfinite(x):
            return "—"
        return f"{x*100:.0f}%"
    except Exception:
        return "—"

def fmt_pct_from_ret(x, nd=1) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if not np.isfinite(x):
            return "—"
        return f"{x*100:.{nd}f}%"
    except Exception:
        return "—"

# =========================
# REPAIR weekly_summary.csv
# =========================
def repair_weekly_summary_schema() -> None:
    """
    Normaliza cache/weekly_summary.csv para SUMMARY_HEADER.
    - Se não existir: cria com header correcto.
    - Se existir com header diferente ou linhas com nº de colunas variável:
      reescreve com SUMMARY_HEADER, mapeando o que conseguir pelo nome.
    """
    ensure_parent(SUMMARY_CSV)

    if not SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SUMMARY_HEADER)
        return

    # lê raw (sem pandas, porque pode estar "corrompido")
    try:
        with SUMMARY_CSV.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except Exception:
        # se nem lê, recria
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SUMMARY_HEADER)
        return

    if not rows:
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SUMMARY_HEADER)
        return

    old_header = [c.strip() for c in rows[0]]
    data_rows = rows[1:]

    # Se já estiver certo e todas as linhas tiverem o mesmo nº de colunas, não mexe
    if old_header == SUMMARY_HEADER:
        ok = True
        for r in data_rows:
            if len(r) != len(SUMMARY_HEADER):
                ok = False
                break
        if ok:
            return

    # mapeamento por nome (o que existir)
    old_idx = {name: i for i, name in enumerate(old_header) if name}
    new_rows = []

    for r in data_rows:
        out = [""] * len(SUMMARY_HEADER)
        for j, name in enumerate(SUMMARY_HEADER):
            if name in old_idx:
                i = old_idx[name]
                if i < len(r):
                    out[j] = r[i]
        new_rows.append(out)

    # reescreve limpo
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        w.writerows(new_rows)

def read_weekly_summary_df() -> pd.DataFrame:
    repair_weekly_summary_schema()
    try:
        df = pd.read_csv(SUMMARY_CSV)
        for c in SUMMARY_HEADER:
            if c not in df.columns:
                df[c] = ""
        df = df[SUMMARY_HEADER].copy()
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end"]).sort_values("week_end").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=SUMMARY_HEADER)

def upsert_weekly_summary(row: dict) -> None:
    """
    Substitui a linha da mesma week_end (se existir), senão adiciona.
    Depois escreve o CSV inteiro com schema fixo.
    """
    repair_weekly_summary_schema()

    week_end = str(row.get("week_end", "")).strip()
    if not week_end:
        return

    df = read_weekly_summary_df()

    # remove a semana se já existe
    if not df.empty:
        df["week_end_str"] = df["week_end"].dt.strftime("%Y-%m-%d")
        df = df[df["week_end_str"] != week_end].copy()
        df = df.drop(columns=["week_end_str"], errors="ignore")

    # adiciona nova linha
    clean = {k: row.get(k, "") for k in SUMMARY_HEADER}
    df2 = pd.DataFrame([clean])
    df = pd.concat([df, df2], ignore_index=True)

    # ordena e escreve
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df = df.dropna(subset=["week_end"]).sort_values("week_end").reset_index(drop=True)

    # escreve com header fixo (valores como string)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        for _, rr in df.iterrows():
            out = []
            for c in SUMMARY_HEADER:
                v = rr.get(c, "")
                if pd.isna(v):
                    v = ""
                if isinstance(v, pd.Timestamp):
                    v = v.strftime("%Y-%m-%d")
                out.append(str(v))
            w.writerow(out)

# =========================
# OHLCV
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
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df
    except Exception:
        return None

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
# BASELINE (cache-only)
# =========================
def baseline_sample(dist_targets: list[float], window_end: pd.Timestamp, horizon: int, k: int, breakout_buffer_pct: float) -> list[dict]:
    if k <= 0 or (not OHLCV_DIR.exists()):
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
    max_tries = 3000

    for p in files:
        if len(picked) >= k:
            break
        tries += 1
        if tries > max_tries:
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

        if use_matching and bucket and want and got:
            b = bucket(dist)
            if got[b] >= want.get(b, 0):
                continue

        trig_eff = trig * (1.0 + breakout_buffer_pct / 100.0)
        end = pos + horizon
        future = df.iloc[pos:end+1]["close"].values

        success = 1 if np.any(future > trig_eff) else 0
        h1 = 1 if float(df["close"].iloc[pos+1]) > trig_eff else 0

        ff = 0
        stop = float(df["close"].iloc[max(0, pos-10):pos].min()) * 0.95
        for j in [pos+1, pos+2]:
            if j < len(df) and float(df["close"].iloc[j]) < stop:
                ff = 1
                break

        mfe, mae = mfe_mae_from_entry(df, pos, horizon, close0)

        picked.append({"covered": 1, "success": success, "h1": h1, "ff": ff, "mfe": mfe, "mae": mae})

        if use_matching and got is not None:
            got[b] += 1

    return picked

# =========================
# Learning params
# =========================
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
    if edge_pp is None or (not np.isfinite(edge_pp)) or cov < LEARN_MIN_COV:
        return ("NO_LEARN", current_params)

    dist = float(current_params.get("DIST_MAX_PCT", DEFAULT_PARAMS["DIST_MAX_PCT"]))
    atrg = float(current_params.get("ATRPCTL_GATE", DEFAULT_PARAMS["ATRPCTL_GATE"]))
    bbzg = float(current_params.get("BBZ_GATE", DEFAULT_PARAMS["BBZ_GATE"]))
    volm = float(current_params.get("VOL_CONFIRM_MULT", DEFAULT_PARAMS["VOL_CONFIRM_MULT"]))

    if edge_pp < 0:
        dist = _clamp(dist - LEARN_STEP_DIST, 10.0, 30.0)
        atrg = _clamp(atrg - LEARN_STEP_ATR, 0.25, 0.65)
        bbzg = _clamp(bbzg - LEARN_STEP_BBZ, -2.50, -0.10)
        volm = _clamp(volm + 0.01, 1.05, 1.40)
        return ("TIGHTEN", {"DIST_MAX_PCT": dist, "ATRPCTL_GATE": atrg, "BBZ_GATE": bbzg, "VOL_CONFIRM_MULT": volm})

    if edge_pp > 5:
        dist = _clamp(dist + LEARN_STEP_DIST, 10.0, 30.0)
        atrg = _clamp(atrg + LEARN_STEP_ATR, 0.25, 0.65)
        bbzg = _clamp(bbzg + LEARN_STEP_BBZ, -2.50, -0.10)
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
    # 1) Garantir que weekly_summary está saudável antes de qualquer coisa
    repair_weekly_summary_schema()

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

    # lagged window
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
    if (m_base["cov"] and m_all["cov"] and (m_base["succ"] is not None) and (m_all["succ"] is not None)):
        edge_pp = (float(m_all["succ"]) - float(m_base["succ"])) * 100.0

    # rolling-4 (agora deve funcionar porque o CSV fica consistente)
    hist = read_weekly_summary_df()
    rolling_txt = "📈 ROLLING-4: sem histórico."
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
            rolling_txt = "📈 ROLLING-4: parsing falhou (ver weekly_summary.csv)."

    sanity = "OK"
    if w_watch.empty and w_exec.empty:
        sanity = "WARN (sem sinais WATCH/EXEC na janela)"

    # =========================
    # AUTO-LEARNING
    # =========================
    learned = load_learned_params()
    cur_params = DEFAULT_PARAMS.copy()
    if isinstance(learned.get("params", None), dict):
        for k in DEFAULT_PARAMS:
            try:
                cur_params[k] = float(learned["params"].get(k, cur_params[k]))
            except Exception:
                pass

    action, proposed = propose_learning_update(edge_pp=edge_pp, cov=int(m_all["cov"]), current_params=cur_params)
    if action in ("TIGHTEN", "RELAX"):
        blended = blend_params(cur_params, proposed, LEARN_BLEND)
    else:
        blended = cur_params

    edge_val = None if edge_pp is None or (not np.isfinite(edge_pp)) else float(edge_pp)
    save_learned_params(
        version=1,
        asof=week_end,
        blend=LEARN_BLEND,
        params=blended,
        meta={
            "action": action,
            "edge_pp": edge_val,
            "cov_used": int(m_all["cov"]),
            "baseline_cov": int(m_base["cov"]),
            "breakout_buffer_pct": BREAKOUT_BUFFER_PCT,
            "notes": "Conservative weekly learning; daily reads params and blends gates."
        }
    )

    # persist weekly summary (UPSERT)
    upsert_weekly_summary({
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

        "learn_action": action,
        "learn_edge_pp": (edge_pp if edge_pp is not None else ""),
        "learn_cov_used": int(m_all["cov"]),
    })

    # pending
    pending_exec = max(0, len(w_exec) - int(m_exec["cov"]))
    pending_watch = max(0, len(w_watch) - int(m_all["cov"]))

    edge_txt = "—"
    if edge_pp is not None and np.isfinite(edge_pp):
        edge_txt = f"{float(edge_pp):+.1f}"

    lines = []
    lines.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v7.1)")
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
    lines.append("🧪 BASELINE (cache-only; matched se possível):")
    if len(w_watch) == 0:
        lines.append("BASE: cov=0/0  S=—  MFE_med=—  MAE=—")
        lines.append("EDGE: —")
    elif int(m_base["cov"]) == 0:
        lines.append(f"BASE: cov=0/{len(w_watch)}  S=—  MFE_med=—  MAE=—")
        lines.append("EDGE: —")
    else:
        lines.append(
            f"BASE: cov={m_base['cov']}/{len(w_watch)}  "
            f"S={pct(m_base['succ'])}  "
            f"MFE_med={fmt_pct_from_ret(m_base['mfe_med'],1)}  "
            f"MAE={fmt_pct_from_ret(m_base['mae_mean'],1)}"
        )
        lines.append(f"EDGE: {edge_txt} pp")
    lines.append("")
    lines.append("🧠 LEARNING:")
    lines.append(f"Action={action} | cov_used={int(m_all['cov'])} | edge_pp={edge_txt}")
    lines.append(
        "learned_params.json → "
        f"DIST_MAX_PCT={blended['DIST_MAX_PCT']:.1f} | "
        f"ATRPCTL_GATE={blended['ATRPCTL_GATE']:.2f} | "
        f"BBZ_GATE={blended['BBZ_GATE']:.2f} | "
        f"VOL_CONFIRM_MULT={blended['VOL_CONFIRM_MULT']:.2f}"
    )
    lines.append("")
    lines.append(f"✅ SANITY: {sanity}")
    lines.append("")
    lines.append(rolling_txt)

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()
