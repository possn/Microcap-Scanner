import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import requests

# --- paths (iguais ao teu projecto) ---
CACHE_DIR = Path("cache")
OHLCV_DIR = CACHE_DIR / "ohlcv"
SIGNALS_CSV = CACHE_DIR / "signals.csv"

# --- Telegram ---
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")

HORIZON = int(os.environ.get("HORIZON_SESS", "40"))  # default 40 sessões

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

def load_ohlcv_from_cache(ticker: str) -> pd.DataFrame | None:
    p = OHLCV_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        need = {"date","open","high","low","close","volume"}
        if not need.issubset(df.columns):
            return None
        df = df[list(need)].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)
        return df
    except Exception:
        return None

def max_runup_and_stop_hit(df: pd.DataFrame, entry_idx: int, entry_price: float, stop: float, horizon: int) -> tuple[float|None, bool|None]:
    """
    - max_runup: máximo retorno (em %) dentro do horizonte, usando HIGH
    - stop_hit: True se LOW tocar/romper stop dentro do horizonte
    """
    end = entry_idx + horizon
    if end >= len(df):
        return (None, None)

    window = df.iloc[entry_idx: end + 1].copy()
    if window.empty:
        return (None, None)

    max_high = float(window["high"].max())
    min_low  = float(window["low"].min())

    if entry_price <= 0:
        return (None, None)

    max_runup = (max_high / entry_price) - 1.0
    stop_hit = (min_low <= stop) if np.isfinite(stop) else None
    return (float(max_runup), bool(stop_hit))

def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not SIGNALS_CSV.exists():
        tg_send(f"[{now}] WEEKLY EVAL: não existe cache/signals.csv")
        return

    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        tg_send(f"[{now}] WEEKLY EVAL: signals.csv vazio")
        return

    # só EXEC
    df["signal"] = df.get("signal", "").astype(str)
    df_exec = df[df["signal"].isin(["EXEC_A","EXEC_B"])].copy()
    if df_exec.empty:
        tg_send(f"[{now}] WEEKLY EVAL: sem EXEC_A/EXEC_B no signals.csv")
        return

    # normalizar campos
    df_exec["date"] = pd.to_datetime(df_exec.get("date", ""), errors="coerce")
    df_exec["ticker"] = df_exec.get("ticker", "").astype(str)
    df_exec["close"] = pd.to_numeric(df_exec.get("close", np.nan), errors="coerce")
    df_exec["stop"] = pd.to_numeric(df_exec.get("stop", np.nan), errors="coerce")
    df_exec["regime"] = df_exec.get("regime", "").astype(str)

    df_exec = df_exec.dropna(subset=["date","ticker","close"]).copy()
    if df_exec.empty:
        tg_send(f"[{now}] WEEKLY EVAL: EXEC sem campos suficientes (date/ticker/close)")
        return

    # métricas por linha (usando apenas cache)
    rows = []
    cache_found = 0
    cache_missing = 0
    resolved = 0

    for _, r in df_exec.iterrows():
        t = r["ticker"].strip().upper()
        d = r["date"]
        entry_close = float(r["close"])
        stop = float(r["stop"]) if np.isfinite(r["stop"]) else np.nan

        ohlcv = load_ohlcv_from_cache(t)
        if ohlcv is None or ohlcv.empty:
            cache_missing += 1
            continue
        cache_found += 1

        # localizar o índice do dia do sinal (match por data)
        dates = ohlcv["date"].dt.date.values
        target = d.date()
        idxs = np.where(dates == target)[0]
        if len(idxs) == 0:
            continue
        entry_idx = int(idxs[0])

        max_runup, stop_hit = max_runup_and_stop_hit(ohlcv, entry_idx, entry_close, stop, HORIZON)
        if max_runup is None:
            continue

        resolved += 1
        hit30 = 1 if max_runup >= 0.30 else 0
        hit50 = 1 if max_runup >= 0.50 else 0

        rows.append({
            "ticker": t,
            "signal": r["signal"],
            "date": d,
            "regime": r.get("regime",""),
            "max_runup": max_runup,
            "hit30": hit30,
            "hit50": hit50,
            "stop_hit": 1 if stop_hit else 0
        })

    if not rows:
        tg_send(f"[{now}] WEEKLY EVAL: sem EXEC resolvidos (faltam sessões no cache) | cache_ok={cache_found} cache_miss={cache_missing}")
        return

    rep = pd.DataFrame(rows)

    def pct(x): 
        return 100.0 * float(np.mean(x)) if len(x) else float("nan")

    # resumo global
    n = len(rep)
    p30 = pct(rep["hit30"])
    p50 = pct(rep["hit50"])
    pstop = pct(rep["stop_hit"])
    avg_runup = 100.0 * float(np.mean(rep["max_runup"]))

    # por tipo de sinal
    by_sig = rep.groupby("signal").agg(
        n=("ticker","count"),
        p30=("hit30","mean"),
        p50=("hit50","mean"),
        pstop=("stop_hit","mean"),
        avg_runup=("max_runup","mean")
    ).reset_index()

    # por regime (se existir)
    by_reg = None
    if rep["regime"].astype(str).str.len().sum() > 0:
        by_reg = rep.groupby("regime").agg(
            n=("ticker","count"),
            p30=("hit30","mean"),
            p50=("hit50","mean"),
            pstop=("stop_hit","mean"),
            avg_runup=("max_runup","mean")
        ).reset_index()

    # mensagem telegram (curta e útil)
    msg = []
    msg.append(f"[{now}] 📊 WEEKLY EVAL (EXEC apenas) | horizonte={HORIZON} sessões")
    msg.append(f"EXEC resolvidos: n={n} | hit30={p30:.0f}% | hit50={p50:.0f}% | stop_hit={pstop:.0f}% | avg_max_runup={avg_runup:.1f}%")
    msg.append(f"Cache: ok={cache_found} miss={cache_missing} | resolvidos={resolved}")
    msg.append("")
    msg.append("Por tipo:")
    for _, r in by_sig.iterrows():
        msg.append(f"- {r['signal']}: n={int(r['n'])} | hit30={100*r['p30']:.0f}% | hit50={100*r['p50']:.0f}% | stop={100*r['pstop']:.0f}% | avg={100*r['avg_runup']:.1f}%")

    if by_reg is not None and len(by_reg) > 0:
        msg.append("")
        msg.append("Por regime:")
        for _, r in by_reg.iterrows():
            reg = str(r["regime"]) if str(r["regime"]).strip() else "NA"
            msg.append(f"- {reg}: n={int(r['n'])} | hit30={100*r['p30']:.0f}% | hit50={100*r['p50']:.0f}% | stop={100*r['pstop']:.0f}% | avg={100*r['avg_runup']:.1f}%")

    tg_send("\n".join(msg))

    # opcional: guardar relatório semanal em csv (fica no repo se quiseres commit)
    out = CACHE_DIR / "weekly_eval_exec.csv"
    rep.sort_values(["date","ticker"]).to_csv(out, index=False)

if __name__ == "__main__":
    main()
