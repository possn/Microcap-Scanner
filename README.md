# Heartbeat Stage 2 — Probability Engine v1.2.0

Scanner diário de ações NASDAQ abaixo de 1 USD. Não promete probabilidades certas: procura setups com melhor assimetria e cria um histórico para calibrar empiricamente o modelo.

## Motor técnico

Mantém os critérios nucleares: base ≥4 meses, compressão ATR/semanal, recuperação recente da SMA150, volume ≥3x, preço não esticado, liquidez e market cap.

Acrescenta:

- força relativa a 20 e 60 sessões contra QQQ;
- regime de mercado através de QQQ e IWM;
- qualidade do fecho da sessão de impulso (CLV);
- volume seco antes do breakout;
- resistência real da base e estado BREAKOUT/RETEST;
- persistência acima da SMA150 após a quebra;
- contagem de falsos breakouts anteriores;
- relação potencial/risco mínima;
- score probabilístico multifatorial 0–100;
- diário de sinais com retornos a 5, 10, 20, 40 e 60 sessões.

## Telegram

Secrets necessários: `TG_BOT_TOKEN` e `TG_CHAT_ID`.

## Limitação importante

O score é um ranking técnico, não uma probabilidade matemática validada enquanto o ficheiro `cache/signal_journal.json` não acumular amostra suficiente. A análise de SEC, caixa e diluição continua manual.
