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

HORIZON_SESS = int(os.getenv("WEEKLY_HORIZON_SESS", "5"))  # horizonte default = 5 sessões
WATCH_OVERHEAD_CLEAN_MAX = int(os.getenv("WATCH_OVERHEAD_CLEAN_MAX", "5"))  # LIMPO <=5 (alinha com scanner)


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
# OHLCV fetch (best effort)
# -------------------------
def _build_symbol_for_stooq(ticker: str) -> str:
    return ticker.lower().replace("-", ".")

def _candidate_urls(ticker: str) -> list[str]:
    sym = _build_symbol_for_stooq(ticker)
    if ".us" in OHLCV_FMT.lower():
        return [OHLCV_FMT.format(symbol=sym), OHLCV_FMT.format(symbol=f"{sym}.us")]
    return [OHLCV_FMT.format(symbol=sym), OHLCV_FMT.format(symbol=f"{sym}.us")]

def fetch_ohlcv_equity(ticker: str) -> pd.DataFrame | None:
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

            if len(df) >= 120:
                return df
        except Exception:
            continue
    return None


# -------------------------
# Helpers
# -------------------------
def last_n_business_days(end_date: datetime, n: int = 5) -> list[datetime.date]:
    d = end_date.date()
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = (datetime.combine(d, datetime.min.time()) - timedelta(days=1)).date()
    return sorted(out)

def pct(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.0f}%"

def retfmt(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.1f}%"


# -------------------------
# Metrics engines
# -------------------------
def _align_index_by_date(o: pd.DataFrame, target_date: datetime.date) -> int | None:
    o_dates = o["date"].dt.date.values
    idxs = np.where(o_dates == target_date)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0])

def compute_exec_metrics(exec_df: pd.DataFrame, horizon: int) -> dict:
    res = {
        "n": int(len(exec_df)),
        "coverage": 0,
        "breakout_rate": np.nan,
        "hold1_rate": np.nan,
        "fail_fast_rate": np.nan,
        "mfe_mean": np.nan,
        "mfe_median": np.nan,
        "mae_mean": np.nan,
    }
    if exec_df.empty:
        return res

    breakout, hold1, fail_fast, mfe, mae = [], [], [], [], []

    for _, r in exec_df.iterrows():
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
        b = bool(np.any(above))
        breakout.append(b)

        h = False
        ff = False
        if b:
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

    if breakout:
        res["breakout_rate"] = float(np.mean(breakout))
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

def compute_watch_metrics_B(watch_df: pd.DataFrame, horizon: int) -> dict:
    """
    Definição B:
      sucesso = existe pelo menos 1 sessão com close > trig no horizonte.
    """
    res = {
        "n": int(len(watch_df)),
        "coverage": 0,
        "success_rate": np.nan,
        "hold1_rate": np.nan,
        "fail_fast_rate": np.nan,
        "mfe_mean": np.nan,
        "mfe_median": np.nan,
        "mae_mean": np.nan,
    }
    if watch_df.empty:
        return res

    success, hold1, fail_fast, mfe, mae = [], [], [], [], []

    for _, r in watch_df.iterrows():
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
            "O daily tem de fazer commit do ficheiro cache/signals.csv."
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
    execB = wdf[wdf.get("signal", "").astype(str) == "EXEC_B"].copy()
    execA = wdf[wdf.get("signal", "").astype(str) == "EXEC_A"].copy()
    watch = wdf[wdf.get("signal", "").astype(str) == "WATCH"].copy()

    # WATCH split por overhead_touches
    watch["overhead_touches"] = pd.to_numeric(watch.get("overhead_touches", np.nan), errors="coerce")
    watch_clean = watch[watch["overhead_touches"] <= WATCH_OVERHEAD_CLEAN_MAX]
    watch_over = watch[watch["overhead_touches"] > WATCH_OVERHEAD_CLEAN_MAX]

    # métricas EXEC
    mB = compute_exec_metrics(execB, HORIZON_SESS)
    mA = compute_exec_metrics(execA, HORIZON_SESS)

    # métricas WATCH (B)
    wAll = compute_watch_metrics_B(watch, HORIZON_SESS)
    wC = compute_watch_metrics_B(watch_clean, HORIZON_SESS)
    wO = compute_watch_metrics_B(watch_over, HORIZON_SESS)

    msg = []
    msg.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v2)")
    msg.append(f"Janela (últimos 5 dias úteis): {start} → {end}")
    msg.append(f"Horizonte métricas: {HORIZON_SESS} sessões")
    msg.append("")

    msg.append(f"Sinais na janela: {total}")
    msg.append(
        f"• EXEC_B: {len(execB)} | EXEC_A: {len(execA)} | WATCH: {len(watch)} "
        f"(LIMPO: {len(watch_clean)} | TETO: {len(watch_over)})"
    )
    msg.append("")

    # EXEC
    msg.append(f"EXEC_B (coverage={mB['coverage']}/{mB['n']}): Breakout {pct(mB['breakout_rate'])} | Hold+1 {pct(mB['hold1_rate'])} | Fail-fast {pct(mB['fail_fast_rate'])}")
    msg.append(f"      MFE mean/med {retfmt(mB['mfe_mean'])}/{retfmt(mB['mfe_median'])} | MAE mean {retfmt(mB['mae_mean'])}")
    msg.append(f"EXEC_A (coverage={mA['coverage']}/{mA['n']}): Breakout {pct(mA['breakout_rate'])} | Hold+1 {pct(mA['hold1_rate'])} | Fail-fast {pct(mA['fail_fast_rate'])}")
    msg.append(f"      MFE mean/med {retfmt(mA['mfe_mean'])}/{retfmt(mA['mfe_median'])} | MAE mean {retfmt(mA['mae_mean'])}")
    msg.append("")

    # WATCH (B)
    msg.append("WATCH (definição B: sucesso = close > trig em ≤ horizonte)")
    msg.append(f"• WATCH_ALL   (coverage={wAll['coverage']}/{wAll['n']}): Success {pct(wAll['success_rate'])} | Hold+1 {pct(wAll['hold1_rate'])} | Fail-fast {pct(wAll['fail_fast_rate'])}")
    msg.append(f"             MFE mean/med {retfmt(wAll['mfe_mean'])}/{retfmt(wAll['mfe_median'])} | MAE mean {retfmt(wAll['mae_mean'])}")
    msg.append(f"• WATCH_LIMPO (≤{WATCH_OVERHEAD_CLEAN_MAX}) (coverage={wC['coverage']}/{wC['n']}): Success {pct(wC['success_rate'])} | Hold+1 {pct(wC['hold1_rate'])} | Fail-fast {pct(wC['fail_fast_rate'])}")
    msg.append(f"             MFE mean/med {retfmt(wC['mfe_mean'])}/{retfmt(wC['mfe_median'])} | MAE mean {retfmt(wC['mae_mean'])}")
    msg.append(f"• WATCH_TETO  (>{WATCH_OVERHEAD_CLEAN_MAX}) (coverage={wO['coverage']}/{wO['n']}): Success {pct(wO['success_rate'])} | Hold+1 {pct(wO['hold1_rate'])} | Fail-fast {pct(wO['fail_fast_rate'])}")
    msg.append(f"             MFE mean/med {retfmt(wO['mfe_mean'])}/{retfmt(wO['mfe_median'])} | MAE mean {retfmt(wO['mae_mean'])}")
    msg.append("")

    # Diagnóstico automático mínimo
    msg.append("Diagnóstico automático:")
    if len(watch) >= 10 and np.isfinite(wC["success_rate"]) and np.isfinite(wO["success_rate"]):
        if wC["success_rate"] >= wO["success_rate"] + 0.10:
            msg.append("• LIMPO supera TETO de forma material → reforçar penalização/filtragem de overhead faz sentido.")
        elif wO["success_rate"] >= wC["success_rate"] + 0.10:
            msg.append("• TETO supera LIMPO de forma material → overhead proxy pode estar a penalizar setups bons (rever banda/janela).")
        else:
            msg.append("• LIMPO ~ TETO → overhead proxy está neutro nesta semana (manter, acumular amostra).")
    else:
        msg.append("• Amostra insuficiente para comparar LIMPO vs TETO com confiança.")
    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
