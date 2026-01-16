import os
import io
import time
import requests
import pandas as pd
from datetime import datetime, timezone

TG_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]

def tg_send(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def fetch_csv(url: str, holdings: bool = False) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text

    if not holdings:
        return pd.read_csv(io.StringIO(text))

    # HOLDINGS CSV (iShares): tem metadados antes da tabela
    lines = text.splitlines()

    header_idx = None
    for i, ln in enumerate(lines[:200]):
        # procura header real
        if ln.strip().lower().startswith("ticker,"):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError("Não encontrei o header 'Ticker,' no holdings CSV.")

    cleaned = "\n".join(lines[header_idx:])

    return pd.read_csv(io.StringIO(cleaned), engine="python", on_bad_lines="skip")

def get_universe(holdings_url: str):
    df = fetch_csv(holdings_url, holdings=True)
    if "Ticker" not in df.columns:
        raise RuntimeError("Holdings CSV não tem coluna 'Ticker'")
    tickers = df["Ticker"].astype(str).str.strip()
    tickers = tickers[tickers.str.len() > 0].unique().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers

def fetch_ohlcv(ticker: str, fmt: str):
    url = fmt.format(symbol=ticker.lower())
    df = fetch_csv(url)

    cols = {c.lower(): c for c in df.columns}
    need = ["date","open","high","low","close","volume"]
    if not all(k in cols for k in need):
        raise RuntimeError("Formato OHLCV inválido")

    df = df[[cols["date"],cols["open"],cols["high"],cols["low"],cols["close"],cols["volume"]]]
    df.columns = ["date","open","high","low","close","volume"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    return df.reset_index(drop=True)

def main():

    holdings_url = os.environ["IWC_HOLDINGS_CSV_URL"]
    ohlcv_fmt = os.environ["OHLCV_URL_FMT"]
    max_n = int(os.environ.get("MAX_TICKERS","300"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tickers = get_universe(holdings_url)[:max_n]

    results = []

    for i,t in enumerate(tickers):
        try:
            df = fetch_ohlcv(t, ohlcv_fmt)

            if len(df) < 60:
                continue

            px = float(df.iloc[-1]["close"])
            dv20 = float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean())

            if px >= 1 and dv20 >= 3_000_000:
                results.append((t,px,dv20))

        except:
            pass

        if (i+1) % 50 == 0:
            time.sleep(1)

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:15]

    msg = [f"[{now}] Microcap scanner (bootstrap online)"]
    msg.append(f"Universo avaliado: {len(tickers)}")
    msg.append("")
    msg.append("Top liquidez (proxy universo):")

    for t,px,dv in top:
        msg.append(f"- {t} | close={px:.2f} | dv20=${dv/1e6:.1f}M")

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()
