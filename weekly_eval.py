# weekly_eval.py  (QUANT v5 - rolling 4 semanas)
# - Mantém v4: sanity, alinhamento T/T-1/T+1, buckets, persistência weekly_summary.csv
# - NOVO: calcula ROLLING-4 (média móvel 4 semanas) e imprime bloco para decisão estrutural

import os
import io
import csv
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
OHLCV_FMT = os.getenv("OHLCV_URL_FMT", "https://stooq.com/q/d/l/?s={symbol}.us&i=d")

SIGNALS_CSV = Path("cache/signals.csv")
CACHE_OHLCV_DIR = Path("cache/ohlcv")
SUMMARY_CSV = Path("cache/weekly_summary.csv")

HORIZON_SESS = int(os.getenv("WEEKLY_HORIZON_SESS", "5"))
WATCH_OVERHEAD_CLEAN_MAX = int(os.getenv("WATCH_OVERHEAD_CLEAN_MAX", "5"))

# sanity thresholds
SANITY_MIN_COVERAGE_PCT = float(os.getenv("SANITY_MIN_COVERAGE_PCT", "0.65"))  # 65%
SANITY_MAX_WEEKLY_JUMP_PP = float(os.getenv("SANITY_MAX_WEEKLY_JUMP_PP", "20")) # 20pp jump as "suspect"

# rolling decision thresholds (apenas informativos)
ROLL_MIN_WEEKS_STRUCTURAL = int(os.getenv("ROLL_MIN_WEEKS_STRUCTURAL", "4"))
ROLL_SUCCESS_UP_PP = float(os.getenv("ROLL_SUCCESS_UP_PP", "10"))  # +10pp vs baseline
ROLL_MAE_WORSEN_PP = float(os.getenv("ROLL_MAE_WORSEN_PP", "1"))    # não piorar mais que 1pp

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
def _build_symbol_for_stooq(ticker: str) -> str:
    return ticker.lower().replace("-", ".")

def _candidate_urls(ticker: str) -> list[str]:
    sym = _build_symbol_for_stooq(ticker)
    if ".us" in OHLCV_FMT.lower():
        return [OHLCV_FMT.format(symbol=sym), OHLCV_FMT.format(symbol=f"{sym}.us")]
    return [OHLCV_FMT.format(symbol=sym), OHLCV_FMT.format(symbol=f"{sym}.us")]

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
        df = df.dropna(subset=["high","low","close"]).reset_index(drop=True)
        if len(df) >= 30:
            return df
    except Exception:
        return None
    return None

def fetch_ohlcv_equity(ticker: str) -> pd.DataFrame | None:
    cached = _load_cached_ohlcv(ticker)
    if cached is not None and not cached.empty:
        return cached

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

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)

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

    # T
    idxs = np.where(o_dates == target_date)[0]
    if len(idxs) > 0:
        return int(idxs[0])

    # T-1
    t_minus = target_date - timedelta(days=1)
    idxs = np.where(o_dates == t_minus)[0]
    if len(idxs) > 0:
        return int(idxs[0])

    # T+1
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
# Metric computation
# -------------------------
def compute_metrics(df_sig: pd.DataFrame, horizon: int) -> dict:
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
# Buckets (compact)
# -------------------------
def bucket_series(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True)

def bucket_report(df: pd.DataFrame, horizon: int, title: str) -> list[str]:
    out = []
    if df.empty:
        return out

    for col in ["overhead_touches", "dist_pct", "bbz", "atrpctl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def line(g: pd.DataFrame, name: str) -> str:
        m = compute_metrics(g, horizon)
        return f"- {name}: n={m['n']} cov={m['coverage']} S={pct(m['success_rate'])} MFE_med={retfmt(m['mfe_median'])} MAE={retfmt(m['mae_mean'])}"

    out.append(title)

    if "overhead_touches" in df.columns:
        oh = df["overhead_touches"].fillna(-1)
        b = bucket_series(oh, [-1, 3, 8, 15, 30, 10_000], ["0-3", "4-8", "9-15", "16-30", "31+"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"OH {name}"))

    if "dist_pct" in df.columns:
        d = df["dist_pct"]
        b = bucket_series(d, [-999, 2, 4, 8, 12, 999], ["≤2", "2-4", "4-8", "8-12", ">12"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"DIST {name}%"))

    if "bbz" in df.columns:
        z = df["bbz"]
        b = bucket_series(z, [-999, -1.4, -1.2, -1.0, -0.8, 999], ["≤-1.4", "-1.4..-1.2", "-1.2..-1.0", "-1.0..-0.8", ">-0.8"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics(g, horizon)
            if m["coverage"] >= 5:
                out.append(line(g, f"BBZ {name}"))

    if "atrpctl" in df.columns:
        a = df["atrpctl"]
        b = bucket_series(a, [-999, 0.15, 0.20, 0.25, 0.30, 999], ["≤0.15", "0.15..0.20", "0.20..0.25", "0.25..0.30", ">0.30"])
        for name, g in df.groupby(b, dropna=False):
            if name is None:
                continue
            m = compute_metrics(g, horizon)
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
    ]
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writerow({c: row.get(c, "") for c in cols})

def load_summary_df() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    try:
        s = pd.read_csv(SUMMARY_CSV)
        if s.empty:
            return pd.DataFrame()
        return s
    except Exception:
        return pd.DataFrame()

# -------------------------
# SANITY + ROLLING
# -------------------------
def sanity_checks(total: int, watch_m: dict, prev_week: dict | None) -> list[str]:
    fails = []
    if total <= 0:
        fails.append("Sinais=0 (weekly sem dados)")

    if watch_m["n"] > 0:
        cov_pct = (watch_m["coverage"] / max(1, watch_m["n"]))
        if cov_pct < SANITY_MIN_COVERAGE_PCT:
            fails.append(f"Coverage baixo em WATCH ({cov_pct*100:.0f}%) — provável cache/OHLCV/alinhamento")

    if prev_week is not None:
        try:
            prev = float(prev_week.get("watch_success", "nan"))
            cur = float(watch_m["success_rate"]) if np.isfinite(watch_m["success_rate"]) else np.nan
            if np.isfinite(prev) and np.isfinite(cur):
                diff_pp = abs((cur - prev) * 100.0)
                if diff_pp >= SANITY_MAX_WEEKLY_JUMP_PP:
                    fails.append(f"Salto anómalo na success WATCH vs semana anterior ({diff_pp:.0f}pp) — verificar alinhamento e cache")
        except Exception:
            pass

    return fails

def _to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def rolling4_block(sdf: pd.DataFrame) -> list[str]:
    """
    Mostra rolling-4 (média) das métricas chave.
    Se <4 semanas, mostra "em construção".
    """
    out = []
    if sdf.empty:
        out.append("📈 ROLLING-4: sem histórico (weekly_summary.csv vazio).")
        return out

    # sort by week_end
    try:
        sdf["week_end_dt"] = pd.to_datetime(sdf["week_end"], errors="coerce")
        sdf = sdf.dropna(subset=["week_end_dt"]).sort_values("week_end_dt")
    except Exception:
        pass

    n_weeks = len(sdf)
    tail = sdf.tail(min(ROLL_MIN_WEEKS_STRUCTURAL, n_weeks)).copy()

    # numeric columns
    for c in ["watch_success","watch_clean_success","watch_over_success","watch_mae","watch_mfe_med","watch_cov_pct"]:
        if c in tail.columns:
            tail[c] = pd.to_numeric(tail[c], errors="coerce")

    if n_weeks < ROLL_MIN_WEEKS_STRUCTURAL:
        out.append(f"📈 ROLLING-4: em construção ({n_weeks}/{ROLL_MIN_WEEKS_STRUCTURAL} semanas).")
    else:
        out.append("📈 ROLLING-4 (últimas 4 semanas):")

    def mean_col(c):
        if c not in tail.columns:
            return np.nan
        v = tail[c].dropna()
        return float(v.mean()) if len(v) else np.nan

    r_cov = mean_col("watch_cov_pct")
    r_all = mean_col("watch_success")
    r_clean = mean_col("watch_clean_success")
    r_over = mean_col("watch_over_success")
    r_mae = mean_col("watch_mae")
    r_mfe = mean_col("watch_mfe_med")

    out.append(f"- WATCH cov%: {pct(r_cov)}")
    out.append(f"- WATCH success: {pct(r_all)} | LIMPO: {pct(r_clean)} | TETO: {pct(r_over)}")
    out.append(f"- WATCH MFE_med: {retfmt(r_mfe)} | MAE: {retfmt(r_mae)}")

    # Optional: eligibility hint (não altera modelo, apenas “semáforo”)
    if n_weeks >= ROLL_MIN_WEEKS_STRUCTURAL and np.isfinite(r_all):
        # comparar LIMPO vs TETO se ambos existem
        if np.isfinite(r_clean) and np.isfinite(r_over):
            diff_pp = (r_over - r_clean) * 100.0
            out.append(f"- Δ(TETO-LIMPO) success: {diff_pp:+.0f}pp (rolling-4)")
        out.append("Critério p/ mudar estruturalmente (guia): efeito ≥ +10pp e MAE não piorar >1pp, consistente.")

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
        tg_send("❌ WEEKLY: falta cache/signals.csv (BUG: daily não commitou ou cache não restaurada).")
        return

    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        tg_send("⚠ WEEKLY: signals.csv vazio (BUG: daily não está a escrever sinais).")
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

    watch_df["overhead_touches"] = pd.to_numeric(watch_df.get("overhead_touches", np.nan), errors="coerce")
    watch_clean = watch_df[watch_df["overhead_touches"] <= WATCH_OVERHEAD_CLEAN_MAX].copy()
    watch_over = watch_df[watch_df["overhead_touches"] > WATCH_OVERHEAD_CLEAN_MAX].copy()

    # Metrics
    m_exec = compute_metrics(exec_df, HORIZON_SESS)
    m_watch_all = compute_metrics(watch_df, HORIZON_SESS)
    m_watch_clean = compute_metrics(watch_clean, HORIZON_SESS)
    m_watch_over = compute_metrics(watch_over, HORIZON_SESS)

    # Persist summary
    cov_pct = (m_watch_all["coverage"] / max(1, m_watch_all["n"])) if m_watch_all["n"] > 0 else np.nan

    append_week_summary({
        "week_end": str(end),
        "window_start": str(start),
        "window_end": str(end),
        "horizon": HORIZON_SESS,
        "signals_total": total,
        "exec_total": int(len(exec_df)),
        "watch_total": int(len(watch_df)),
        "watch_cov_pct": round(float(cov_pct), 4) if np.isfinite(cov_pct) else "",
        "watch_success": round(float(m_watch_all["success_rate"]), 6) if np.isfinite(m_watch_all["success_rate"]) else "",
        "watch_clean_success": round(float(m_watch_clean["success_rate"]), 6) if np.isfinite(m_watch_clean["success_rate"]) else "",
        "watch_over_success": round(float(m_watch_over["success_rate"]), 6) if np.isfinite(m_watch_over["success_rate"]) else "",
        "watch_mae": round(float(m_watch_all["mae_mean"]), 6) if np.isfinite(m_watch_all["mae_mean"]) else "",
        "watch_mfe_med": round(float(m_watch_all["mfe_median"]), 6) if np.isfinite(m_watch_all["mfe_median"]) else "",
    })

    sdf = load_summary_df()
    prev_week = None
    if len(sdf) >= 2:
        prev_week = sdf.iloc[-2].to_dict()

    # SANITY
    sanity_fails = sanity_checks(total, m_watch_all, prev_week)

    # Build message
    msg = []
    msg.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v5)")
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

    if sanity_fails:
        msg.append("🧯 SANITY: FAIL (corrigir já)")
        for f in sanity_fails:
            msg.append(f"- {f}")
        msg.append("Ação típica: verificar cache restore, rate-limit Stooq, e alinhamento de datas.")
    else:
        msg.append("✅ SANITY: OK (sem bugs críticos detectados)")

    msg.append("")
    msg.extend(rolling4_block(sdf))
    msg.append("")

    # Buckets
    if len(watch_df) > 0:
        msg.extend(bucket_report(watch_df, HORIZON_SESS, "Buckets WATCH (monitorizar):"))
        msg.append("")
    if len(exec_df) > 0:
        msg.extend(bucket_report(exec_df, HORIZON_SESS, "Buckets EXEC (monitorizar):"))
        msg.append("")

    msg.append("Ações:")
    msg.append("• Bugs: corrigir imediatamente se SANITY=FAIL.")
    msg.append(f"• Modelo: sem alterações estruturais até acumular ≥{ROLL_MIN_WEEKS_STRUCTURAL} semanas (usar ROLLING-4).")
    msg.append("• Histórico: weekly_summary.csv actualizado (base para decisões).")

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()
