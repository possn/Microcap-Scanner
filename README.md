# Heartbeat Stage 2 — Probability Engine v1.3.0

Scanner diário de ações NASDAQ abaixo de 1 USD.

## O que este scanner é — e o que não é

É um **filtro de assimetria**: procura setups onde a relação potencial/risco é favorável e regista tudo para calibração empírica posterior.

**Não** entrega "empresas com grande probabilidade de subir". Essa afirmação exige uma probabilidade medida, e uma probabilidade medida exige amostra fora de treino. Enquanto `cache/backtest_report.json` e `cache/signal_journal.json` não tiverem amostra suficiente, o score de 0–100 é um **ranking ordinal**, não uma probabilidade.

## Motor técnico

Critérios nucleares: base ≥4 meses, compressão ATR/semanal, recuperação recente da SMA150, volume ≥3x, preço não esticado, liquidez e market cap.

Acrescenta: força relativa a 20/60 sessões (benchmark IWM, não QQQ — o universo é micro cap); regime de mercado via QQQ+IWM; CLV do impulso; secagem de volume pré-breakout; resistência real da base e estado BREAKOUT/RETEST; persistência acima da SMA150; contagem de falsos breakouts; relação potencial/risco mínima; score multifatorial 0–100.

## Novo em 1.3.0

- **Correção crítica**: `swing_slopes()` deixou de depender de `np.array_split` sobre um DataFrame — comportamento que muda entre versões de pandas/numpy e que, ao falhar, era engolido pelo `try/except` geral e transformava um varrimento partido num falso "nenhum setup encontrado".
- **Alerta de varrimento inválido**: se >25% das candidatas falharem com erro técnico, é enviado um alerta explícito em vez de uma lista vazia silenciosa.
- **Universo limpo**: exclusão de warrants, rights, units e preferred (5.ª letra W/R/U + regex de nome). Abaixo de $1 são abundantes e distorcem qualquer estatística.
- **Diário de sinais estatisticamente válido**:
  - entrada medida sobre a série **ajustada** no dia do sinal, não sobre o preço nominal guardado — os reverse splits são endémicos abaixo de $1 e reescalam retroativamente o histórico, fabricando ganhos fantasma;
  - sinais cuja série de dados morre são marcados `data_missing` em vez de desaparecerem — omiti-los é survivorship bias que inflaciona qualquer taxa de acerto;
  - janelas de máximo/mínimo passaram a ser **estritamente pós-entrada**;
  - **grupo de controlo**: candidatas rejeitadas nas últimas barreiras são registadas sem publicação, para medir se as barreiras acrescentam algo.
- **Calibração com intervalo de Wilson**: taxas de acerto reportadas com IC95 correto para amostras pequenas. Abaixo de 20 sinais resolvidos, recusa reportar.
- **Cache stale como último recurso**: se Yahoo e Stooq falharem, usa o CSV antigo em vez de descartar o ticker.
- **`backtest.py`**: validação walk-forward ponto-no-tempo. Reexecuta toda a pilha de deteção sobre `df[:t]` e mede retornos futuros por bucket de score.

## Utilização

```bash
python scanner.py                      # varrimento diário
python backtest.py --step 5            # calibração histórica
```

## Telegram

Secrets: `TG_BOT_TOKEN` e `TG_CHAT_ID`.

## Limitações que não desaparecem com código

1. **Survivorship bias** — o universo é a lista NASDAQ sub-$1 de hoje. Empresas deslistadas, adquiridas ou reverse-split para fora do intervalo não existem no histórico. Qualquer estatística do backtest é um **limite superior otimista**.
2. **Custos de transação** — spreads de 2–5% são normais neste segmento. Nenhum resultado modelado inclui spread, slippage ou halts.
3. **Amostras sobrepostas** — sinais consecutivos no mesmo ticker são correlacionados; o N efetivo é muito inferior ao N nominal.
4. **Multiplicidade** — os limiares (68, 2.0x, 0.60, ...) foram escolhidos à mão. Cada limiar afinado sobre os mesmos dados é uma comparação múltipla não corrigida.
5. **Diluição e SEC continuam manuais.** Neste segmento, o ATM offering é o risco dominante e é invisível no gráfico.

## Teste de aceitação

O score só é utilizável como probabilidade quando os buckets do `backtest_report.json` forem **monótonos**: 90+ > 80-89 > 70-79 > <70. Se não forem, o score não discrimina e o número não deve ser lido como probabilidade.
