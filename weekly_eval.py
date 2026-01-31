# weekly_eval.py — QUANT v6 (schema FIXO + rolling-4)
# Lê cache/signals.csv + cache/ohlcv/*.csv
# Calcula métricas semanais (últimos 5 dias úteis de sinais)
# Escreve cache/weekly_summary.csv com colunas fixas (sem corrupção)
# Envia resumo para Telegram

import os, csv, io, random
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

# =========================
# ENV
# =========================
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
OHLCV_DIR = Path("cache") / "ohlcv"
SIGNALS_CSV = Path("cache") / "signals.csv"
SUMMARY_CSV = Path("cache") / "weekly_summary.csv"

HORIZON = int(os.environ.get("WEEKLY_HORIZON_SESS", "5"))  # horizonte de métricas (sessões)
CLEAN_MAX_OH = int(os.environ.get("WATCH_OVERHEAD_CLEAN_MAX", "5"))

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
        need = ["date","open","high","low","close","volume"]
        if not all(c in df.columns for c in need):
            return None
        df = df[need].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
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
    "regime_mode"
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
# metrics
# =========================
def pct(x: float) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x*100:.0f}%"

def fmt(x: float, nd=1) -> str:
    if x is None or (not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}%"

def business_last_n(dates: pd.Series, n: int = 5) -> list[pd.Timestamp]:
    # devolve as últimas n datas únicas (assumindo que já são dias úteis)
    u = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().unique()
    u = sorted(u)
    return [pd.Timestamp(x) for x in u[-n:]]

def mfe_mae_from_entry(df: pd.DataFrame, pos: int, horizon: int, entry: float) -> tuple[float|None, float|None]:
    # MFE = max retorno no horizonte; MAE = min retorno no horizonte
    if entry <= 0:
        return (None, None)
    end = min(len(df)-1, pos + horizon)
    sl = df.iloc[pos:end+1]["close"]
    if sl.empty:
        return (None, None)
    rets = (sl / entry) - 1.0
    mfe = float(np.nanmax(rets.values))
    mae = float(np.nanmin(rets.values))
    return (mfe, mae)

def eval_watch_row(row, horizon: int) -> dict:
    """
    Sucesso (def B): close > trig em <= horizonte
    H1: close_{+1} > trig
    Fail-fast: close < stop em <=2 sessões (pos+1 ou pos+2)
    """
    out = {"covered": 0, "success": 0, "h1": 0, "ff": 0, "mfe": None, "mae": None}
    t = str(row["ticker"]).strip().upper()
    trig = float(pd.to_numeric(row.get("trig", np.nan), errors="coerce"))
    stop = float(pd.to_numeric(row.get("stop", np.nan), errors="coerce"))
    entry_close = float(pd.to_numeric(row.get("close", np.nan), errors="coerce"))
    d0 = pd.to_datetime(row.get("date", None), errors="coerce")
    if not np.isfinite(trig) or not np.isfinite(entry_close) or pd.isna(d0):
        return out

    df = load_ohlcv(t)
    if df is None or df.empty:
        return out

    # localizar o dia do sinal
    dates = df["date"].dt.normalize()
    target = d0.normalize()
    idx = np.where(dates.values == target.to_datetime64())[0]
    if len(idx) == 0:
        return out
    pos = int(idx[0])

    # precisa de horizonte suficiente
    if pos + 1 >= len(df):
        return out

    out["covered"] = 1

    # sucesso: algum close > trig no horizonte
    end = min(len(df)-1, pos + horizon)
    future = df.iloc[pos:end+1]["close"].values
    out["success"] = 1 if np.any(future > trig) else 0

    # H1: dia seguinte > trig
    if pos + 1 < len(df):
        out["h1"] = 1 if float(df["close"].iloc[pos+1]) > trig else 0

    # fail-fast: < stop em <=2 sessões
    ff = 0
    for j in [pos+1, pos+2]:
        if j < len(df) and np.isfinite(stop):
            if float(df["close"].iloc[j]) < stop:
                ff = 1
                break
    out["ff"] = ff

    # mfe/mae
    mfe, mae = mfe_mae_from_entry(df, pos, horizon, entry_close)
    out["mfe"] = mfe
    out["mae"] = mae
    return out

def aggregate_metrics(rows_eval: list[dict]) -> dict:
    if not rows_eval:
        return {"cov":0,"succ":None,"h1":None,"ff":None,"mfe_med":None,"mae_mean":None}
    cov = sum(r["covered"] for r in rows_eval)
    if cov == 0:
        return {"cov":0,"succ":None,"h1":None,"ff":None,"mfe_med":None,"mae_mean":None}

    suc = np.mean([r["success"] for r in rows_eval if r["covered"]])
    h1 = np.mean([r["h1"] for r in rows_eval if r["covered"]])
    ff = np.mean([r["ff"] for r in rows_eval if r["covered"]])

    mfes = [r["mfe"] for r in rows_eval if r["covered"] and r["mfe"] is not None and np.isfinite(r["mfe"])]
    maes = [r["mae"] for r in rows_eval if r["covered"] and r["mae"] is not None and np.isfinite(r["mae"])]

    mfe_med = float(np.median(mfes)) if mfes else None
    mae_mean = float(np.mean(maes)) if maes else None

    return {"cov":cov,"succ":float(suc),"h1":float(h1),"ff":float(ff),"mfe_med":mfe_med,"mae_mean":mae_mean}

# =========================
# BASELINE (aprox. "random matched DIST%")
# =========================
def baseline_sample(dist_targets: list[float], window_end: pd.Timestamp, horizon: int, k: int) -> list[dict]:
    """
    Baseline aproximado (sem APIs):
    - escolhe tickers aleatórios do cache/ohlcv
    - define trig_baseline = max(close últimos 20 dias antes do window_end)
    - dist% = (trig - close)/close
    - aceita ticker se dist% cair no mesmo bucket do alvo (bucket por quartis simples)
    - avalia sucesso close>trig em <= horizonte
    """
    if k <= 0:
        return []

    # buckets pelos targets do modelo
    tt = [x for x in dist_targets if np.isfinite(x)]
    if len(tt) < 6:
        return []  # sem dist suficiente para match

    qs = np.quantile(tt, [0.25, 0.50, 0.75])
    def bucket(x):
        if x <= qs[0]: return 0
        if x <= qs[1]: return 1
        if x <= qs[2]: return 2
        return 3

    target_buckets = [bucket(x) for x in tt]
    # distribuição desejada
    want = {b: target_buckets.count(b) for b in [0,1,2,3]}

    # universo baseline = ficheiros no cache
    if not OHLCV_DIR.exists():
        return []
    all_files = [p for p in OHLCV_DIR.iterdir() if p.is_file() and p.name.endswith(".csv")]
    random.shuffle(all_files)

    picked = []
    got = {0:0,1:0,2:0,3:0}

    # tenta até 1500 tickers para preencher
    tries = 0
    for p in all_files:
        if len(picked) >= k:
            break
        tries += 1
        if tries > 1500:
            break

        t = p.stem.upper()
        df = load_ohlcv(t)
        if df is None or len(df) < 60:
            continue

        # encontrar pos do window_end (ou último antes)
        dates = df["date"].dt.normalize()
        # usa o último dia <= window_end
        mask = dates <= window_end.normalize()
        if not mask.any():
            continue
        pos = int(np.where(mask.values)[0][-1])

        if pos < 25 or pos + 1 >= len(df):
            continue

        close0 = float(df["close"].iloc[pos])
        if close0 <= 0:
            continue

        # trigger baseline: máximo close dos 20 dias anteriores
        trig = float(df["close"].iloc[pos-20:pos].max())
        if trig <= 0:
            continue

        dist = (trig - close0) / close0 * 100.0
        if not np.isfinite(dist):
            continue

        b = bucket(dist)
        if got[b] >= want.get(b, 0):
            continue

        # avaliar sucesso no horizonte (close>trig)
        end = min(len(df)-1, pos + horizon)
        future = df["close"].iloc[pos:end+1].values
        success = 1 if np.any(future > trig) else 0

        mfe, mae = mfe_mae_from_entry(df, pos, horizon, close0)

        picked.append({"covered":1,"success":success,"mfe":mfe,"mae":mae})
        got[b] += 1

    return picked

# =========================
# MAIN
# =========================
def main():
    # Sanity: signals.csv
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

    # normalizar
    sig.columns = [c.strip().lower() for c in sig.columns]
    need_cols = {"date","ticker","signal","close","trig","stop","dist_pct","overhead_touches","regime"}
    missing = [c for c in need_cols if c not in sig.columns]
    if missing:
        tg_send(f"WEEKLY REVIEW: FAIL — signals.csv sem colunas: {missing}")
        return

    sig["date"] = pd.to_datetime(sig["date"], errors="coerce")
    sig = sig.dropna(subset=["date"]).copy()
    sig["date"] = sig["date"].dt.normalize()

    # janela: últimos 5 dias de sinais
    last5 = business_last_n(sig["date"], 5)
    if len(last5) == 0:
        tg_send("WEEKLY REVIEW: FAIL — sem datas válidas.")
        return

    window_start = last5[0].strftime("%Y-%m-%d")
    window_end = last5[-1].strftime("%Y-%m-%d")
    week_end = window_end  # simplificação: week_end = último dia da janela

    w = sig[(sig["date"] >= last5[0]) & (sig["date"] <= last5[-1])].copy()

    # focar apenas WATCH/EXEC
    w_exec = w[w["signal"].astype(str).str.contains("EXEC", na=False)].copy()
    w_watch = w[w["signal"].astype(str).str.upper().eq("WATCH")].copy()

    # split limpo/teto para WATCH (overhead)
    w_watch["overhead_touches"] = pd.to_numeric(w_watch["overhead_touches"], errors="coerce")
    w_watch["dist_pct"] = pd.to_numeric(w_watch["dist_pct"], errors="coerce")
    w_watch_clean = w_watch[w_watch["overhead_touches"] <= CLEAN_MAX_OH].copy()
    w_watch_over = w_watch[w_watch["overhead_touches"] > CLEAN_MAX_OH].copy()

    # regime dominante no WATCH (modo)
    regime_mode = "—"
    if not w_watch.empty:
        vc = w_watch["regime"].astype(str).value_counts()
        if len(vc) > 0:
            regime_mode = str(vc.index[0])

    # avaliar WATCH
    eval_all = [eval_watch_row(r, HORIZON) for _, r in w_watch.iterrows()]
    eval_clean = [eval_watch_row(r, HORIZON) for _, r in w_watch_clean.iterrows()]
    eval_over = [eval_watch_row(r, HORIZON) for _, r in w_watch_over.iterrows()]

    m_all = aggregate_metrics(eval_all)
    m_clean = aggregate_metrics(eval_clean)
    m_over = aggregate_metrics(eval_over)

    # avaliar EXEC (opcional; em muitos períodos dá 0)
    eval_exec = [eval_watch_row(r, HORIZON) for _, r in w_exec.iterrows()]
    m_exec = aggregate_metrics(eval_exec)

    # baseline (random matched dist%)
    dist_targets = w_watch["dist_pct"].dropna().tolist()
    baseline = baseline_sample(dist_targets, pd.to_datetime(window_end), HORIZON, k=len(w_watch))
    m_base = aggregate_metrics(baseline)

    edge_pp = None
    if m_base["cov"] and m_all["cov"] and m_base["succ"] is not None and m_all["succ"] is not None:
        edge_pp = (m_all["succ"] - m_base["succ"]) * 100.0

    # rolling-4 (se houver histórico válido)
    hist = read_weekly_summary()
    rolling_txt = "📈 ROLLING-4: sem histórico (weekly_summary.csv vazio)."
    if hist is not None and (not hist.empty):
        # últimas 4 linhas
        h4 = hist.tail(4).copy()
        # usar colunas existentes
        try:
            cov = pd.to_numeric(h4["watch_cov"], errors="coerce").fillna(0).sum()
            succ = pd.to_numeric(h4["watch_success"], errors="coerce")
            mae = pd.to_numeric(h4["watch_mae_mean"], errors="coerce")
            mfe = pd.to_numeric(h4["watch_mfe_med"], errors="coerce")
            # métricas agregadas (média de semanas, não ponderada)
            succ_m = float(np.nanmean(succ.values)) if len(succ.dropna()) else None
            mae_m = float(np.nanmean(mae.values)) if len(mae.dropna()) else None
            mfe_m = float(np.nanmean(mfe.values)) if len(mfe.dropna()) else None
            rolling_txt = (
                f"📈 ROLLING-4 (últimas {len(h4)} semanas): "
                f"WATCH success {pct(succ_m)} | MFE_med {fmt(mfe_m*100 if mfe_m is not None else None,1)} | "
                f"MAE {fmt(mae_m*100 if mae_m is not None else None,1)} | cov_total={int(cov)}"
            )
        except Exception:
            rolling_txt = "📈 ROLLING-4: histórico existe mas parsing falhou (ver CSV)."

    # SANITY
    sanity = "OK"
    if w_watch.empty and w_exec.empty:
        sanity = "WARN (sem sinais WATCH/EXEC na janela)"

    # escrever summary row (schema fixo)
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

        "regime_mode": regime_mode
    })

    # construir mensagem
    lines = []
    lines.append("📊 MICROCAP BREAKOUT — WEEKLY REVIEW (QUANT v6)")
    lines.append(f"Janela (últimos 5 dias úteis): {window_start} → {window_end}")
    lines.append(f"Horizonte métricas: {HORIZON} sessões")
    lines.append("")
    lines.append(f"Sinais na janela: {len(w_exec)+len(w_watch)}")
    lines.append(f"• EXEC: {len(w_exec)} | WATCH: {len(w_watch)} (LIMPO: {len(w_watch_clean)} | TETO: {len(w_watch_over)})")
    lines.append("")
    lines.append(
        f"EXEC:  cov={m_exec['cov']}/{len(w_exec)}  "
        f"S={pct(m_exec['succ'])} H1={pct(m_exec['h1'])} FF={pct(m_exec['ff'])}  "
        f"MFE_med={fmt((m_exec['mfe_med']*100) if m_exec['mfe_med'] is not None else None,1)}  "
        f"MAE={fmt((m_exec['mae_mean']*100) if m_exec['mae_mean'] is not None else None,1)}"
    )
    lines.append(
        f"WATCH: cov={m_all['cov']}/{len(w_watch)}  "
        f"S={pct(m_all['succ'])} H1={pct(m_all['h1'])} FF={pct(m_all['ff'])}  "
        f"MFE_med={fmt((m_all['mfe_med']*100) if m_all['mfe_med'] is not None else None,1)}  "
        f"MAE={fmt((m_all['mae_mean']*100) if m_all['mae_mean'] is not None else None,1)}"
    )
    lines.append(
        f"  LIMPO(≤{CLEAN_MAX_OH}): cov={m_clean['cov']}/{len(w_watch_clean)}  "
        f"S={pct(m_clean['succ'])}  "
        f"MFE_med={fmt((m_clean['mfe_med']*100) if m_clean['mfe_med'] is not None else None,1)}  "
        f"MAE={fmt((m_clean['mae_mean']*100) if m_clean['mae_mean'] is not None else None,1)}"
    )
    lines.append(
        f"  TETO(>{CLEAN_MAX_OH}):  cov={m_over['cov']}/{len(w_watch_over)}  "
        f"S={pct(m_over['succ'])}  "
        f"MFE_med={fmt((m_over['mfe_med']*100) if m_over['mfe_med'] is not None else None,1)}  "
        f"MAE={fmt((m_over['mae_mean']*100) if m_over['mae_mean'] is not None else None,1)}"
    )
    lines.append("")
    lines.append("🧪 BASELINE (random matched DIST% — aproximado):")
    if m_base["cov"] == 0:
        lines.append("BASE:  cov=0/0  S=—  MFE_med=—  MAE=—")
        lines.append("EDGE:  —")
    else:
        lines.append(
            f"BASE:  cov={m_base['cov']}/{len(w_watch)}  "
            f"S={pct(m_base['succ'])}  "
            f"MFE_med={fmt((m_base['mfe_med']*100) if m_base['mfe_med'] is not None else None,1)}  "
            f"MAE={fmt((m_base['mae_mean']*100) if m_base['mae_mean'] is not None else None,1)}"
        )
        lines.append(f"EDGE:  {edge_pp:+.0f}pp" if edge_pp is not None else "EDGE: —")

    lines.append("")
    lines.append("🧭 REGIME (WATCH success):")
    lines.append(f"{regime_mode}: {pct(m_all['succ'])} | TRANSITION: — | RISK_OFF: —")
    lines.append("")
    lines.append(f"✅ SANITY: {sanity}")
    lines.append("")
    lines.append(rolling_txt)
    lines.append("")
    lines.append("Ações:")
    lines.append("• Bugs: corrigir imediatamente se SANITY=FAIL.")
    lines.append("• Modelo: alterações estruturais só com evidência em ROLLING-4 (≥4 semanas).")
    lines.append("• Histórico: weekly_summary.csv actualizado (base para decisões).")

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()
