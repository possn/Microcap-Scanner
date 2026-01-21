import os
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# =========================
# ENV
# =========================
TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
OHLCV_FMT = os.getenv("OHLCV_URL_FMT", "https://stooq.com/q/d/l/?s={symbol}.us&i=d")

SIGNALS_CSV = Path("cache/signals.csv")

# horizonte para métricas semanais (curto prazo)
HORIZON_SESS = int(os.getenv("WEEKLY_HORIZON_SESS", "5"))  # 5 sessões por defeito


# =========================
# Telegram
# =========================
def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram secrets missing; printing only.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print("Telegram send failed:", e)
        print(text)


# =========================
# OHLCV fetch (EOD)
# =========================
def _build_symbol_for_stooq(ticker: str) -> str:
    # no teu scanner: t.replace("-", ".") antes do format
    return ticker.lower().replace("-", ".")

def _candidate_urls(ticker: str) -> list[str]:
    sym = _build_symbol_for_stooq(ticker)
    # tenta as duas variações para maximizar robustez
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


# =========================
# Date helpers
# =========================
def last_n_business_days(end_date: datetime, n: int = 5) -> list[datetime.date]:
    # considera dias úteis; suficiente para o teu objectivo semanal (proxy)
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

def pct1(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.1f}%"

def retfmt(x: float | None) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.1f}%"


# =========================
# Metrics for EXEC_A / EXEC_B
# =========================
def compute_exec_metrics(exec_df: pd.DataFrame, horizon: int) -> dict:
    """
    Métricas:
      - breakout_rate: houve algum close > trig nos próximos 'horizon' dias (inclui dia do sinal)
      - hold1_rate: após o 1º close > trig, o close do dia seguinte também > trig
      - fail_fast_rate: após 1º close > trig, existe close < trig dentro de 2 dias (proxy de falso breakout)
      - mfe/mae: max(high)/entry - 1 ; min(low)/entry - 1 no horizonte
    """
    res = {
        "n": int(len(exec_df)),
        "breakout_rate": np.nan,
        "hold1_rate": np.nan,
        "fail_fast_rate": np.nan,
        "mfe_mean": np.nan,
        "mfe_median": np.nan,
        "mae_mean": np.nan,
        "coverage": 0,  # quantos conseguiram OHLCV+matching date
    }
    if exec_df.empty:
        return res

    breakout = []
    hold1 = []
    fail_fast = []
    mfe = []
    mae = []

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

        # alinhar com a data do sinal
        o_dates = o["date"].dt.date.values
        target = dt.date()
        idxs = np.where(o_dates == target)[0]
        if len(idxs) == 0:
            continue

        i0 = int(idxs[0])
        i1 = min(i0 + horizon, len(o) - 1)
        sl = o.iloc[i0:i1 + 1].copy()
        if sl.empty:
            continue

        res["coverage"] += 1

        closes = sl["close"].values
        above = closes > trig
        b = bool(np.any(above))
        breakout.append(b)

        # hold+1
        h = False
        ff = False
        if b:
            first = int(np.argmax(above))
            if first + 1 < len(closes):
                h = bool(closes[first + 1] > trig)

            # fail-fast: fecha abaixo do trig em <=2 dias após o 1º acima
            j2 = min(first + 2, len(closes) - 1)
            if np.any(closes[first:j2 + 1] < trig):
                ff = True

        hold1.append(h)
        fail_fast.append(ff)

        # MFE/MAE vs entry close do sinal
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


# =========================
# MAIN weekly eval
# =========================
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

    # normalizar
    df["date"] = pd.to_datetime(df.get("date", None), errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["day"] = df["date"].dt.date

    wdf = df[(df["day"] >= start) & (df["day"] <= end)].copy()

    # contagens
    total = int(len(wdf))
    execB = wdf[wdf.get("signal", "").astype(str) == "EXEC_B"].copy()
    execA = wdf[wdf.get("signal", "").astype(str) == "EXEC_A"].copy()
    watch = wdf[wdf.get("signal", "").astype(str) == "WATCH"].copy()

    # WATCH split: por overhead_touches (já logado pelo scanner)
    watch["overhead_touches"] = pd.to_numeric(watch.get("overhead_touches", np.nan), errors="coerce")
    watch_clean = watch[watch["overhead_touches"] <= 5]
    watch_over = watch[watch["overhead_touches"] > 5]

    # métricas EXEC
    mB = compute_exec_metrics(execB, HORIZON_SESS)
    mA = compute_exec_metrics(execA, HORIZON_SESS)

    # mensagem
    msg = []
    msg.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT)")
    msg.append(f"Janela (últimos 5 dias úteis): {start} → {end}")
    msg.append(f"Horizonte métricas: {HORIZON_SESS} sessões")
    msg.append("")
    msg.append(f"Sinais na janela: {total}")
    msg.append(
        f"• EXEC_B: {len(execB)} | EXEC_A: {len(execA)} | WATCH: {len(watch)} "
        f"(LIMPO: {len(watch_clean)} | TETO: {len(watch_over)})"
    )
    msg.append("")

    msg.append(f"EXEC_B (coverage={mB['coverage']}/{mB['n']} com OHLCV+data):")
    msg.append(f"• Breakout: {pct(mB['breakout_rate'])} | Hold+1: {pct(mB['hold1_rate'])} | Fail-fast: {pct(mB['fail_fast_rate'])}")
    msg.append(f"• MFE mean/med: {retfmt(mB['mfe_mean'])} / {retfmt(mB['mfe_median'])} | MAE mean: {retfmt(mB['mae_mean'])}")
    msg.append("")

    msg.append(f"EXEC_A (coverage={mA['coverage']}/{mA['n']} com OHLCV+data):")
    msg.append(f"• Breakout: {pct(mA['breakout_rate'])} | Hold+1: {pct(mA['hold1_rate'])} | Fail-fast: {pct(mA['fail_fast_rate'])}")
    msg.append(f"• MFE mean/med: {retfmt(mA['mfe_mean'])} / {retfmt(mA['mfe_median'])} | MAE mean: {retfmt(mA['mae_mean'])}")
    msg.append("")

    # diagnóstico operacional simples
    msg.append("Diagnóstico automático:")
    if np.isfinite(mB["fail_fast_rate"]) and mB["fail_fast_rate"] >= 0.35:
        msg.append("• Fail-fast elevado em EXEC_B → overhead supply / distância / gaps demasiado permissivos.")
    if np.isfinite(mB["breakout_rate"]) and mB["breakout_rate"] <= 0.40 and len(execB) >= 5:
        msg.append("• Breakout baixo em EXEC_B → triggers demasiado altos ou regime desfavorável.")
    if len(watch_over) > len(watch_clean) and (len(watch) >= 6):
        msg.append("• WATCH_TETO domina → filtro overhead (ou scoring) está frouxo.")
    if (mB["coverage"] + mA["coverage"]) == 0 and (mB["n"] + mA["n"]) > 0:
        msg.append("• Sem coverage OHLCV → Stooq/URL indisponível ou datas não alinham.")
    if msg[-1] == "Diagnóstico automático:":
        msg.append("• Sem alertas críticos na semana.")

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
