# Heartbeat Stage 2 — Discovery Engine v1.7.1

Scanner diário de ações NASDAQ, micro a mid cap (preço $0.08–$500, capitalização até $10 mil milhões, configurável). Por omissão em **modo descoberta**: sinaliza candidatas plausíveis para avaliação manual em vez de eliminar tudo o que não é perfeito — ver secção abaixo.

## O que este scanner é — e o que não é

É um **filtro de assimetria**: procura setups onde a relação potencial/risco é favorável e regista tudo para calibração empírica posterior.

**Não** entrega "empresas com grande probabilidade de subir". Essa afirmação exige uma probabilidade medida, e uma probabilidade medida exige amostra fora de treino. Enquanto `cache/backtest_report.json` e `cache/signal_journal.json` não tiverem amostra suficiente, o score de 0–100 é um **ranking ordinal**, não uma probabilidade.

## Corrigido em 1.7.1 — bug crítico: cache de OHLCV nunca atualizava

Sintoma reportado: as mesmas 10 ações, com o mesmo score ao décimo, todos os dias desde 13 de agosto.

Causa: o workflow faz `git add cache/ohlcv` e commit a cada execução (para não recomeçar do zero todos os dias). Em CI, um `actions/checkout` novo repõe o mtime de TODOS os ficheiros para "agora" a cada execução — a verificação de frescura antiga (`_fresh()`) olhava só para o mtime do ficheiro em disco, nunca para a data dentro dos dados. Resultado: o cache era sempre visto como "fresco" mesmo com conteúdo de dias antes, e o scanner nunca voltava a contactar a Yahoo/Stooq. Confirmado no repositório real: `cache/ohlcv/OCFC.csv` foi escrito uma única vez (13 ago, dados até 12 ago) e nunca mais atualizado em 5 execuções diárias seguintes.

Corrigido: nova `_cache_has_recent_data()` verifica a **data dentro do CSV**, não a idade do ficheiro — só aceita o cache se a última sessão registada estiver a ≤3 dias de hoje (tolera fim de semana sem forçar obtenções inúteis). `load_ohlcv()` passa a exigir as duas condições (mtime E conteúdo) antes de servir cache; falhando qualquer uma, tenta sempre obter dados novos primeiro. 2 novos testes reproduzem o bug exato (ficheiro com mtime "agora" mas conteúdo de 2020) e confirmam a correção.

## Modo descoberta (padrão desde 1.7.0)

O objetivo deixou de ser "só os melhores" e passou a ser "candidatas plausíveis para eu avaliar manualmente". Isso muda o desenho: em vez de ~10 critérios obrigatórios em AND (onde reprovar em qualquer um apaga a candidata por completo), os critérios de **qualidade** passam a **penalizar o score** em vez de eliminar:

- ETF do setor sem SMA50 a curvar
- SMA50 da própria ação sem curvar
- CLV fraco (fecho longe do topo do impulso)
- Relação potencial/risco abaixo do ideal
- Falsos breakouts em excesso

A candidata continua visível, com score mais baixo e uma linha `⚠` no Telegram a dizer exatamente o que está fraco — quem decide é quem lê, não o filtro.

**Continuam a eliminar** (estruturais, não de qualidade): histórico insuficiente, preço fora do intervalo, liquidez insuficiente, ausência de um breakout da SMA150 na janela, e ausência de uma base de compressão válida — sem estes não há sequer um setup para descrever (preço de entrada, suporte, invalidação dependem de existir uma base).

O piso final de score também desce: `DISCOVERY_MIN_SCORE` (padrão 35) em vez de `MIN_QUALITY_SCORE` (68) — continua a existir para não mostrar autêntico ruído, mas deixou de ser o corte que apagava candidatas moderadas.

Para voltar ao comportamento antigo (só recomendações de alta convicção, tudo o resto eliminado): `DISCOVERY_MODE=0`. Nesse modo os critérios acima voltam a ser cortes rígidos e `MIN_QUALITY_SCORE` volta a mandar.

## Pré-filtro: ETFs de setor primeiro

Antes de analisar qualquer ação, o scanner corre um gate de setor: para ~20 setores (Semicondutores→SMH, Biotecnologia→XBI, Energia→XLE, Petróleo e Gás→XOP, Água→PHO, Saúde→XLV, Software/Tecnologia→IGV, Defesa→ITA, etc. — lista completa em `SECTOR_ETF_MAP` no `scanner.py`), verifica se a SMA50 diária do ETF representativo está a curvar para cima. **Só se procuram ações nos setores cujo ETF passa este teste.** Uma ação cujo setor/indústria não corresponde a nenhum ETF mapeado é rejeitada por defeito (`sector_unmapped`) — não há forma de verificar o filtro sem uma referência.

Configurável via `REQUIRE_SECTOR_ETF_CURL` (padrão `1`, ligado — desligar remove o pré-filtro por completo). Os resultados no Telegram e no CSV/JSON vêm agrupados por setor, com o ETF e a sua inclinação no cabeçalho de cada grupo.

## Motor técnico

Critérios nucleares: base ≥4 meses, compressão ATR/semanal, recuperação recente da SMA150 com volume ≥2x, SMA50 diária a curvar para cima (inclinação positiva sustentada; aceleração/inflexão exata é opcional, ver `REQUIRE_SMA50_CURL_ACCELERATING`), preço não esticado, liquidez e market cap.

Acrescenta: força relativa a 20/60 sessões (benchmark dependente do tamanho — IWM para small/micro cap, MDY/S&P MidCap 400 a partir de $2 mil milhões); regime de mercado via QQQ+IWM+MDY; CLV do impulso; secagem de volume pré-breakout; resistência real da base e estado BREAKOUT/RETEST; persistência acima da SMA150; contagem de falsos breakouts; relação potencial/risco mínima; score multifatorial 0–100.

## Novo em 1.6.1 — auditoria orientada a dados (0 qualificadas há 9 dias)

Diagnóstico a partir do funil real de 9 execuções consecutivas (5–12 ago), não de intuição:

- **`.github/workflows/daily.yml` nunca foi atualizado.** As alterações de 1.4.2 (volume) e 1.5.0 (mid cap) só existiam nos defaults do `scanner.py` — o workflow continuava com `MAX_PRICE=1.00`, `MAX_MARKET_CAP=150000000`, `BREAKOUT_VOLUME_MULT=3.0`, `BREAKOUT_VOLUME_WINDOW=3`. O universo diário ficou preso em ~280 empresas sub-$1 durante todo este tempo — nenhuma das duas melhorias anteriores chegou a produção. Corrigido: os 4 valores agora sincronizados com o código.
- **`no_sma150_cross` era, de longe, a maior perda do funil** (70–85% dos sobreviventes da liquidez, todos os 9 dias) — maior que o filtro de setor, liquidez e tudo o resto juntos. `BREAKOUT_MAX_AGE` (40 sessões, ~8 semanas) exigia que o cruzamento da SMA150 tivesse acontecido numa janela demasiado estreita. Alargado para 90 sessões (~4,5 meses); `BREAKOUT_MIN_AGE` mantido em 10 (evita comprar o próprio dia do cruzamento). Continua protegido por `MAX_GAIN_SINCE_BREAKOUT` e `MAX_DISTANCE_SMA150`, que rejeitam independentemente qualquer cruzamento antigo já esticado — medido no universo real em cache: cruzamentos encontrados sobem de 7 para 12 (quase duplica) sem qualquer outra alteração.
- **O filtro de ETF de setor (1.6.0) NÃO é o gargalo principal** — custa ~35% (126/196 nos 9 dias), secundário face ao ponto acima. Mantido como está, sem alterações; o `breakout_ok` já estava preso em 1–2/dia mesmo nos 3 dias anteriores à existência do filtro de setor.
- Uma única alteração de cada vez, por desenho: sector gate (1.6.0), workflow+janela (1.6.1) e mid cap (1.5.0) são medidos separadamente para se poder atribuir qualquer melhoria (ou piora) à causa certa, não a um pacote de mudanças simultâneas.

## Novo em 1.6.0

- **Pré-filtro de ETF de setor** (o pedido desta versão): ETFs correm primeiro; só se procuram ações nos setores aprovados. Ver secção acima.
- `classify_sector()` mapeia sector+industry (texto NASDAQ) → (rótulo, ETF) por palavra-chave, ordem específica→genérica para não deixar categorias largas engolirem as específicas (ex.: Biotecnologia antes de Saúde).
- Novos campos no resultado: `sector_label`, `sector_etf`, `sector_etf_slope_pct`.
- Telegram reestruturado: secção "ETFs DE SETOR" (aprovados/reprovados com inclinação) antes dos resultados; resultados agrupados por setor, ordenados pela força do próprio ETF.
- `scanner.py` persiste `cache/universe_sectors.json` (ticker→setor/indústria) a cada execução, para o `backtest.py` poder replicar o gate ponto-no-tempo (recalcula a curvatura do ETF em cada data de corte, não usa o estado de hoje). Sem esse ficheiro (nenhuma execução do scanner ainda feita), o `backtest.py` desliga o gate e avisa — não falha silenciosamente.
- 4 novos testes: cobertura de `classify_sector` contra texto realista, bloqueio do gate ao vivo, ordem no código-fonte (setor antes do breakout), e o gate ponto-no-tempo do backtest (zero sinais antes do ETF curvar, sinais depois).

## Novo em 1.5.0

- **Universo alargado a mid caps**: `MAX_PRICE` $1.00 → $500; `MAX_MARKET_CAP` $150M → $10 mil milhões. Deixa de ser um scanner exclusivamente sub-$1.
- **Benchmark de força relativa dependente do tamanho**: candidatas ≥$2 mil milhões passam a comparar-se contra o MDY (S&P MidCap 400) em vez do IWM (Russell 2000, small/micro cap) — usar o IWM para uma empresa de $8 mil milhões era a régua errada.
- **Bónus de market cap no score corrigido**: estava fixo em $50M (calibrado para o teto antigo de $150M). Com o teto novo em $10 mil milhões isso penalizava sistematicamente qualquer mid cap que entrasse no universo — o alargamento seria inútil na prática, porque essas candidatas nunca ganhariam o ranking. Passa a ser proporcional ao teto configurado (1/3 do `MAX_MARKET_CAP`).
- Texto do log e do Telegram deixou de assumir "sub-$1" fixo.

## Novo em 1.4.2

- **Volume de confirmação relaxado**: `BREAKOUT_VOLUME_MULT` 3.0x → 2.0x; `BREAKOUT_VOLUME_WINDOW` ±3 → ±5 sessões à volta do cruzamento da SMA150. O funil estava perto de vazio quase todos os dias — a interseção de breakout confirmado + base + SMA50 a curvar era demasiado rara com o limiar antigo.

## Novo em 1.4.0

- **Critério adicional obrigatório — SMA50 a curvar para cima.** A par da recuperação da SMA150, a SMA50 diária tem agora de mostrar inclinação positiva nas últimas `SMA50_CURL_LOOKBACK` sessões (padrão 10). Ambos os critérios (SMA150 e SMA50) têm de se verificar; nenhum substitui o outro. Configurável via `SMA50_CURL_LOOKBACK`, `MIN_SMA50_SLOPE_PCT` e `REQUIRE_SMA50_CURL_ACCELERATING` (env vars).
  - *Correção 1.4.1*: por omissão já não exige aceleração (inflexão exata) — só inclinação positiva sustentada. Exigir a par o breakout confirmado da SMA150 **e** o instante exato da inflexão da SMA50 no mesmo dia esvaziava o funil quase sempre: são dois eventos raros que raramente coincidem no calendário. Quem quiser a versão mais estrita (inflexão exata) liga `REQUIRE_SMA50_CURL_ACCELERATING=1`.
- Novo campo `sma50_slope_pct` no resultado, no CSV/JSON, na mensagem do Telegram e nas "quase aprovadas".
- `backtest.py` replica o gate exatamente na mesma ordem (SMA150 → SMA50 → base) para a calibração não divergir da produção.
- Rejeitadas por `sma50_not_curling_up` entram no grupo de controlo do diário de sinais, para medir se este critério acrescenta algo de facto.

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

1. **Survivorship bias** — o universo é a lista NASDAQ elegível de hoje (preço e market cap). Empresas deslistadas, adquiridas ou que saíram do intervalo não existem no histórico. Qualquer estatística do backtest é um **limite superior otimista**.
2. **Custos de transação heterogéneos** — spreads de 2–5% são normais na ponta sub-$1/nano cap; em mid caps líquidas o spread é tipicamente uma fração disso. Nenhum resultado modelado inclui spread, slippage ou halts, mas o efeito não é uniforme ao longo do universo alargado — tratar o backtest como uma única estimativa mistura dois regimes de liquidez distintos.
3. **Amostras sobrepostas** — sinais consecutivos no mesmo ticker são correlacionados; o N efetivo é muito inferior ao N nominal.
4. **Multiplicidade** — os limiares (68, 2.0x, 0.60, $2 mil milhões para o corte IWM/MDY, ...) foram escolhidos à mão. Cada limiar afinado sobre os mesmos dados é uma comparação múltipla não corrigida.
5. **Diluição e SEC continuam manuais.** Em nano/micro cap, o ATM offering é o risco dominante e é invisível no gráfico; em mid cap esse risco específico é menos comum, mas outros (guidance, cobertura de analistas, opções) passam a ser relevantes e continuam fora do âmbito do scanner.

## Teste de aceitação

O score só é utilizável como probabilidade quando os buckets do `backtest_report.json` forem **monótonos**: 90+ > 80-89 > 70-79 > <70. Se não forem, o score não discrimina e o número não deve ser lido como probabilidade.
