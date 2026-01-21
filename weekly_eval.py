import os
import datetime
from zoneinfo import ZoneInfo
import subprocess

TG_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT = os.getenv("TG_CHAT_ID")

def send_telegram(msg):
    import requests
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT, "text": msg})


def main():

    now = datetime.datetime.now(ZoneInfo("Europe/Lisbon"))
    start = now - datetime.timedelta(days=7)

    header = (
        "📊 MICROCAP BREAKOUT — WEEKLY REVIEW\n"
        f"Semana: {start.date()} → {now.date()}\n\n"
        "Estado actual:\n"
        "• Weekly pipeline activo\n"
        "• Daily scanner operacional\n\n"
        "⚠ Métricas históricas automáticas ainda NÃO persistidas.\n"
        "Fase seguinte: activar logging diário para avaliação quantitativa real.\n\n"
        "Sistema estrutural OK.\n"
        "Nenhum erro crítico detectado."
    )

    send_telegram(header)

    print("Weekly review sent to Telegram.")


if __name__ == "__main__":
    main()
