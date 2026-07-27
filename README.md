# Heartbeat Stage 2 Scanner

Scanner técnico diário para ações NASDAQ abaixo de 1 USD, com foco em consolidações prolongadas, compressão de volatilidade e breakout recente da SMA150 com volume anormal.

## Dados analisados automaticamente

- preço e histórico OHLCV;
- SMA150 e data do breakout;
- volume relativo no breakout;
- duração e geometria da consolidação;
- compressão de ATR e amplitude semanal;
- liquidez;
- market cap;
- float apenas quando disponível na fonte de mercado;
- suporte, resistência, entrada, invalidação e confirmação;
- ranking técnico de 0 a 100.

O scanner não consulta a SEC e não avalia automaticamente caixa, runway, ofertas, diluição, reverse splits ou risco de permanência no Nasdaq. Essa análise é feita manualmente apenas para os tickers selecionados.

## Secrets obrigatórios no GitHub

- `TG_BOT_TOKEN`
- `TG_CHAT_ID`

Não é necessário `SEC_USER_AGENT`.

## Execução

O workflow `.github/workflows/daily.yml` corre de segunda a sexta-feira após o fecho regular dos EUA e também pode ser iniciado manualmente em **Actions**.
