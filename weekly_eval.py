import os
import io
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

HORIZON_SESS = int(os.getenv("WEEKLY_HORIZON_SESS", "5"))
WATCH_OVERHEAD_CLEAN_MAX = int(os.getenv("WATCH_OVERHEAD_CLEAN_MAX", "5"))

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

def _align_index_by_date(o: pd.DataFrame, target_date: datetime.date) -> int | None:
    # exact match first
    o_dates = o["date"].dt.date.values
    idxs = np.where(o_dates == target_date)[0]
    if len(idxs) > 0:
        return int(idxs[0])

    # fallback: next trading day (T+1) if exact date absent
    # (ex: signals date logged in UTC vs local; or missing candle)
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
# Core metric computation
# -------------------------
def compute_metrics(df_sig: pd.DataFrame, horizon: int, success_def: str) -> dict:
    """
    success_def:
      - "EXEC": success = exists close > trig (breakout confirmation) within horizon
      - "WATCH_B": success = exists close > trig within horizon (your definition B)
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

        i0 = _align_index_by_date(o, dt.date())
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
            # fail-fast proxy: falls back below trig within 2 days after first success
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
# Bucketing analysis
# -------------------------
def bucket_series(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    # edges must be len(labels)+1
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True)

def bucket_report(df: pd.DataFrame, horizon: int, title: str, success_def: str) -> list[str]:
    """
    Produz linhas compactas: bucket -> n, cov, success%, hold1%, MFE med, MAE mean
    """
    out = []
    if df.empty:
        return out

    def _one(group: pd.DataFrame, name: str) -> str:
        m = compute_metrics(group, horizon, success_def)
        return (
            f"- {name}: n={m['n']} cov={m['coverage']} "
            f"S={pct(m['success_rate'])} H1={pct(m['hold1_rate'])} "
            f"MFE_med={retfmt(m['mfe_median'])} MAE={retfmt(m['mae_mean'])}"
        )

    # Ensure numeric
    for col in ["overhead_touches", "dist_pct", "bbz", "atrpctl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out.append(title)

    # Overhead buckets
    if "overhead_touches" in df.columns:
        oh = df["overhead_touches"].copy()
        b = bucket_series(
            oh.fillna(-1),
            edges=[-1, 3, 8, 15, 30, 10_000],
            labels=["0-3", "4-8", "9-15", "16-30", "31+"],
        )
        for name, g in df.groupby(b, dropna=False):
            if name is not None:
                out.append(_one(g, f"OH {name}"))

    # Dist buckets
    if "dist_pct" in df.columns:
        d = df["dist_pct"].copy()
        b = bucket_series(
            d,
            edges=[-999, 2, 4, 8, 12, 999],
            labels=["≤2", "2-4", "4-8", "8-12", ">12"],
        )
        for name, g in df.groupby(b, dropna=False):
            if name is not None:
                out.append(_one(g, f"DIST {name}%"))

    # BBZ buckets (more negative = tighter)
    if "bbz" in df.columns:
        z = df["bbz"].copy()
        b = bucket_series(
            z,
            edges=[-999, -1.4, -1.2, -1.0, -0.8, 999],
            labels=["≤-1.4", "-1.4..-1.2", "-1.2..-1.0", "-1.0..-0.8", ">-0.8"],
        )
        for name, g in df.groupby(b, dropna=False):
            if name is not None:
                out.append(_one(g, f"BBZ {name}"))

    # ATRpctl buckets (lower = more compressed)
    if "atrpctl" in df.columns:
        a = df["atrpctl"].copy()
        b = bucket_series(
            a,
            edges=[-999, 0.15, 0.20, 0.25, 0.30, 999],
            labels=["≤0.15", "0.15..0.20", "0.20..0.25", "0.25..0.30", ">0.30"],
        )
        for name, g in df.groupby(b, dropna=False):
            if name is not None:
                out.append(_one(g, f"ATRp {name}"))

    return out

# -------------------------
# Recommendations engine
# -------------------------
def recommend(execA: dict, execB: dict, w_clean: dict, w_over: dict) -> list[str]:
    rec = ["Recomendações (baseadas em dados desta semana):"]

    # If EXEC is persistently zero, suggest a minimal relaxation path (not automatic change)
    # Here we only comment if EXEC n=0.
    if execA["n"] == 0 and execB["n"] == 0:
        rec.append("• EXEC=0: não há breakouts confirmados. Antes de mexer no modelo, acumular 2–4 semanas. Se persistir:")
        rec.append("  - rever VOL_CONFIRM_MULT (p.ex. -0.05) OU permitir EXEC_A com qualidade extrema (BBZ<=-1.3 AND ATRpctl<=0.20).")
        rec.append("  - rever MAX_GAP_UP (p.ex. 1.12→1.15) apenas se o gap filter estiver a cortar execuções reais.")
    else:
        rec.append("• EXEC presente: usar EXEC_B como base; medir diferença EXEC_B vs EXEC_A e ajustar gates por bins (score/BBZ/ATRpctl).")

    # Overhead vs clean for WATCH
    if np.isfinite(w_clean["success_rate"]) and np.isfinite(w_over["success_rate"]):
        diff = w_over["success_rate"] - w_clean["success_rate"]
        if diff >= 0.12:
            rec.append("• WATCH_TETO supera WATCH_LIMPO de forma material: não apertar overhead agora; considerar reduzir penalização de overhead em setups de compressão extrema.")
        elif diff <= -0.12:
            rec.append("• WATCH_LIMPO supera WATCH_TETO de forma material: reforçar penalização/filtragem de overhead (ou reduzir OVERHEAD_BAND/OVERHEAD_WINDOW).")
        else:
            rec.append("• WATCH_LIMPO ~ WATCH_TETO: overhead proxy neutro; manter parâmetros e aumentar amostra.")
    else:
        rec.append("• Comparação LIMPO vs TETO: amostra/coverage insuficiente; manter e acumular.")

    # Risk envelope hint (MAE)
    if np.isfinite(w_clean["mae_mean"]) and np.isfinite(w_over["mae_mean"]):
        if w_over["mae_mean"] < w_clean["mae_mean"] - 0.01:
            rec.append("• Nota risco: TETO teve MAE mais profunda; operacionalmente, stops/posição mais conservadores em TETO.")
        elif w_clean["mae_mean"] < w_over["mae_mean"] - 0.01:
            rec.append("• Nota risco: LIMPO teve MAE mais profunda; rever dist/qualidade nos LIMPO.")
        else:
            rec.append("• Nota risco: MAE semelhante; risco comparável nesta semana.")

    return rec

# -------------------------
# MAIN
# -------------------------
def main() -> None:
    now = datetime.now(ZoneInfo("Europe/Lisbon"))
    days = last_n_business_days(now, 5)
    start = days[0]
    end = days[-1]

    if not SIGNALS_CSV.exists():
        tg_send(
            "❌ MICROCAP WEEKLY REVIEW\n"
            "Não existe cache/signals.csv no repositório.\n"
            "O daily tem de fazer commit de cache/signals.csv."
        )
        return

    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        tg_send("⚠ MICROCAP WEEKLY REVIEW\nsignals.csv está vazio.")
        return

    df["date"] = pd.to_datetime(df.get("date", None), errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["day"] = df["date"].dt.date

    wdf = df[(df["day"] >= start) & (df["day"] <= end)].copy()
    total = int(len(wdf))

    # Normalise columns
    if "signal" not in wdf.columns:
        wdf["signal"] = ""

    execB_df = wdf[wdf["signal"].astype(str) == "EXEC_B"].copy()
    execA_df = wdf[wdf["signal"].astype(str) == "EXEC_A"].copy()
    watch_df = wdf[wdf["signal"].astype(str) == "WATCH"].copy()

    # WATCH split by overhead
    watch_df["overhead_touches"] = pd.to_numeric(watch_df.get("overhead_touches", np.nan), errors="coerce")
    watch_clean = watch_df[watch_df["overhead_touches"] <= WATCH_OVERHEAD_CLEAN_MAX].copy()
    watch_over = watch_df[watch_df["overhead_touches"] > WATCH_OVERHEAD_CLEAN_MAX].copy()

    # Base summary metrics
    m_execB = compute_metrics(execB_df, HORIZON_SESS, "EXEC")
    m_execA = compute_metrics(execA_df, HORIZON_SESS, "EXEC")
    m_watch_all = compute_metrics(watch_df, HORIZON_SESS, "WATCH_B")
    m_watch_clean = compute_metrics(watch_clean, HORIZON_SESS, "WATCH_B")
    m_watch_over = compute_metrics(watch_over, HORIZON_SESS, "WATCH_B")

    msg = []
    msg.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v3)")
    msg.append(f"Janela (últimos 5 dias úteis): {start} → {end}")
    msg.append(f"Horizonte métricas: {HORIZON_SESS} sessões")
    msg.append("")
    msg.append(f"Sinais na janela: {total}")
    msg.append(f"• EXEC_B: {len(execB_df)} | EXEC_A: {len(execA_df)} | WATCH: {len(watch_df)} (LIMPO: {len(watch_clean)} | TETO: {len(watch_over)})")
    msg.append("")

    # EXEC summary
    msg.append(f"EXEC_B: cov={m_execB['coverage']}/{m_execB['n']}  S={pct(m_execB['success_rate'])} H1={pct(m_execB['hold1_rate'])} FF={pct(m_execB['fail_fast_rate'])}  MFE_med={retfmt(m_execB['mfe_median'])}  MAE={retfmt(m_execB['mae_mean'])}")
    msg.append(f"EXEC_A: cov={m_execA['coverage']}/{m_execA['n']}  S={pct(m_execA['success_rate'])} H1={pct(m_execA['hold1_rate'])} FF={pct(m_execA['fail_fast_rate'])}  MFE_med={retfmt(m_execA['mfe_median'])}  MAE={retfmt(m_execA['mae_mean'])}")
    msg.append("")

    # WATCH summary
    msg.append("WATCH (definição B: sucesso = close > trig em ≤ horizonte)")
    msg.append(f"ALL:   cov={m_watch_all['coverage']}/{m_watch_all['n']}  S={pct(m_watch_all['success_rate'])} H1={pct(m_watch_all['hold1_rate'])} FF={pct(m_watch_all['fail_fast_rate'])}  MFE_med={retfmt(m_watch_all['mfe_median'])}  MAE={retfmt(m_watch_all['mae_mean'])}")
    msg.append(f"LIMPO: cov={m_watch_clean['coverage']}/{m_watch_clean['n']}  S={pct(m_watch_clean['success_rate'])} H1={pct(m_watch_clean['hold1_rate'])} FF={pct(m_watch_clean['fail_fast_rate'])}  MFE_med={retfmt(m_watch_clean['mfe_median'])}  MAE={retfmt(m_watch_clean['mae_mean'])}")
    msg.append(f"TETO:  cov={m_watch_over['coverage']}/{m_watch_over['n']}  S={pct(m_watch_over['success_rate'])} H1={pct(m_watch_over['hold1_rate'])} FF={pct(m_watch_over['fail_fast_rate'])}  MFE_med={retfmt(m_watch_over['mfe_median'])}  MAE={retfmt(m_watch_over['mae_mean'])}")
    msg.append("")

    # Buckets (compact but informative)
    # limit output length by focusing on WATCH and EXEC separately
    if len(watch_df) > 0:
        msg.extend(bucket_report(watch_df, HORIZON_SESS, "Buckets WATCH (OH / DIST / BBZ / ATRpctl):", "WATCH_B"))
        msg.append("")
    if len(execA_df) + len(execB_df) > 0:
        exec_all = pd.concat([execB_df, execA_df], ignore_index=True)
        msg.extend(bucket_report(exec_all, HORIZON_SESS, "Buckets EXEC (OH / DIST / BBZ / ATRpctl):", "EXEC"))
        msg.append("")

    # Recommendations
    msg.extend(recommend(m_execA, m_execB, m_watch_clean, m_watch_over))

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()
