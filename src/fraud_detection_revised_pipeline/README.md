# Pipeline experimental revisado

Reimplementação do pipeline de detecção de fraudes atendendo ao *Relatório de
Revisão* da primeira versão da dissertação (01/10/2026). A pasta é
autocontida: não importa nada de `linear_regression_naive-bayes_knn/`, e o
pipeline antigo continua intacto para comparação.

Os itens do relatório de revisão são citados pelo código original (G1–G23,
F1–F11, C*.*) ao longo deste documento e nos comentários do código.

---

## 1. O que mudou, em uma tela

| Decisão | Antes | Agora | Item |
|---|---|---|---|
| Bases | 8 configurações de 4 fontes, com `eqm-ref-21` e `ulb-pca` duplicando a ULB | 7 configurações de 4 fontes; `eqm-ref-21` eliminada | G9a |
| PCA | Eixo implícito da grade, confundido com a base | Propriedade fixa de cada base: roda em todas, **menos na ULB** | C5.31 |
| Reserva (`test_data_size`) | Eixo da grade, inerte em 2 das 4 políticas | **Eliminada** da identidade, do manifesto e dos resultados | G9c, G9d |
| Protocolos | 4 políticas, uma delas descartando 20–40% dos dados | 3 protocolos explícitos, nenhum descartando dados | G9 |
| Particionamento | Por linha; cliente/cartão atravessava treino e teste | Por grupo (`GroupShuffleSplit`/`StratifiedGroupKFold`) em todas as bases que têm titular | G7, G8 |
| Protocolo temporal | Inexistente (a "divisão aleatória" era temporal por acidente) | `temporal-holdout` explícito nas bases com tempo | G8 |
| Identificadores | `cc_num`, `trans_num`, `TransactionID`, `Time` entravam como atributos | Removidos de todas as bases, antes do ajuste | G14b, G14d |
| Padronização | Só nas configurações com PCA | Em **todas**, ajustada no treino | G14a |
| Linha de base sem reamostragem | Não existia | Braço `none` nos blocos clássico e de boosting | G10a |
| Estratégia comum entre famílias | Não havia | `random-undersampling` 1:1 nos dois blocos | G10f |
| Remoção de registros incompletos | Removia linhas do **teste** também | Remove **só do treino**; teste mantém a população | G16 |
| Colunas quase vazias no `drop-incomplete-rows` | `dropna()` em todas as colunas esvaziava o treino | Antes da remoção saem as colunas acima do limiar escolhido no treino: o **maior** entre 0–100% (passo 5) que preserva ≥ 75% das fraudes; limiar e colunas vão para o manifesto | G15c |
| Entradas inválidas na PCA e nos executores clássicos | Categóricas descartadas e faltantes completados pela mediana, em silêncio | **Falha** com o nome das colunas | G15 |
| Verificações de sanidade | Não eram gravadas | Relatório por dobra (`<configuration_md5>.fold<k>.preparation.json`, blocos clássico e nativo): faltantes, linhas e fraudes por etapa, componentes da PCA | G15 |
| Métricas | Só AUC-ROC entre as independentes de limiar | **AUC-PR acrescentada**, gravada por todos os blocos | G11-iii |
| Limiar de decisão | Fixo em 0,5 sobre probabilidades deslocadas | **Escolhido na validação** (máximo F2) e aplicado ao teste, em todos os blocos; no `stratified-k-fold`, vem da validação das outras dobras | G11 |
| Pesos de classe | Não existiam | Braços `sqrt-balanced` (√R) e `balanced` (R) no eixo de balanceamento, só sobre configurações sem reamostragem. Regressão logística, CatBoost e XGBoost | G10, C3.14 |
| Regressão linear | Avaliada como classificador | **Removida da grade**: saída não calibrada, limiar sem interpretação | C3.4 |
| `fold_index` | 0-based, vazio quando não havia dobras | **1-based**, sempre preenchido | pedido |
| MD5 dos arquivos | Só identidade da configuração | Identidade **+ MD5 de conteúdo** de treino/validação/teste na linha de resultado | pedido |
| Semente | Única, implícita | `random_seed` na identidade; `SEED_LIST` repete a grade | G13 |
| Cabeça dos Transformers | `linear`, `mlp`, `lstm` | **`linear` removida**; ficam `mlp` e `lstm` | pedido |
| Titular na IEEE-CIS | `card1`, que é informação do cartão | `card1` + `addr1` | G8-iii |
| Registros duplicados | 1.081 linhas idênticas da ULB espalhadas entre treino e teste | Removidas na carga | achado novo |

---

## 2. O que será executado em cada base

### 2.1 Catálogo

`dataset_id` é a identidade persistida e usada em filtros; `dataset_name` é o
rótulo de apresentação. `database_type` continua temporariamente nos novos
manifestos e resultados, sempre como espelho de `dataset_id`. Os nomes físicos
dos artefatos continuam derivados de hashes e não incorporam nenhum desses
nomes.

| `dataset_id` | `dataset_name` | Alias histórico aceito | Fonte | Alvo | Linhas | PCA | Agrupamento | Ordem temporal | Protocolos |
|---|---|---|---|---|---:|:---:|---|---|---|
| `ulb` | ULB | `eqm-ref-18` | ULB `creditcard.csv` | `Class` | 283.726 | **não** | — | `Time` | holdout, k-fold, temporal |
| `home-credit-completa` | Home-Credit-completa | `eqm-ref-19` | Home Credit (junção à esquerda) | `TARGET` | 1.430.155 | sim | `SK_ID_CURR` | — | holdout, k-fold |
| `home-credit-somente-aplicacao` | Home-Credit-somente-aplicação | `eqm-ref-19-application` | Home Credit (só aplicações) | `TARGET` | 307.511 | sim | `SK_ID_CURR` | — | holdout, k-fold |
| `sparkov` | Sparkov | `eqm-ref-20` | Sparkov (2 arquivos) | `is_fraud` | 1.852.394 | sim | `cc_num` | `trans_date_trans_time` | holdout, k-fold, temporal |
| `ieee-cis-completa` | IEEE-CIS-completa | `eqm-ref-22` | IEEE-CIS com identidade | `isFraud` | 144.233 | sim | titular aproximado | `TransactionDT` | holdout, k-fold, temporal |
| `ieee-cis-sem-identidade` | IEEE-CIS-sem-identidade | `eqm-ref-22-noidentity` | variante de `ieee-cis-completa` sem as colunas de identidade | `isFraud` | 144.233 | sim | titular aproximado | `TransactionDT` | holdout, k-fold, temporal |
| `ieee-cis-transacional` | IEEE-CIS-transacional | `eqm-ref-22-transaction` | IEEE-CIS, todas as transações | `isFraud` | 590.540 | sim | titular aproximado | `TransactionDT` | holdout, k-fold, temporal |

São **7 configurações de 4 fontes**, mais **2 bases de ablação** (item G14), que
herdam carregador, linhas e partições da base de origem e removem atributos
declarados em `ablated_columns`:

| `dataset_id` | Origem | Remove | Blocos |
|---|---|---|---|
| `sparkov-sem-titular` | `sparkov` | `zip`, `lat`, `long`, `city`, `city_pop`, `job` (um valor por titular); `cc_num` continua só como agrupamento e sequência | todos |
| `ieee-cis-completa-sem-alta-cardinalidade` | `ieee-cis-completa` | `DeviceInfo`, `id_31`, `id_33`, `card1` | **só boosting** |

`eqm-ref-21` não aparece em lugar nenhum:
era uma republicação da ULB que produzia resultados idênticos aos de
`ulb` e dava peso dobrado a ela em toda média agregada (G9a).

A PCA é aplicada **apenas para os algoritmos clássicos** — boosting e
Transformers nunca a recebem, porque ela eliminaria a semântica individual dos
atributos, impediria o tratamento nativo das categóricas e alteraria a
representação por campo dos Transformers; além disso, por ser não
supervisionada, preserva variância dos preditores, não necessariamente
informação relevante para fraude. Entre as bases, roda em todas **menos na
ULB**: `V1`–`V28` já são componentes principais publicados (C5.31).

### 2.2 Tratamento específico por base

**`ulb` (ULB).** `Time` sai dos atributos e vira `hour_of_day`, uma
marca relativa com significado operacional; como marca absoluta, só localizava
a transação dentro do período coletado (G14d). Não tem atributo categórico nem
valor faltante, então os eixos de codificação e imputação **não variam** nela:
antes, 2 codificações × 4 imputações geravam 8 cópias idênticas de cada
experimento (G9b). **Achado novo:** a base traz 1.081 linhas idênticas em todas
as 31 colunas; removidas na carga, porque uma partição aleatória espalhava
cópias do mesmo registro entre treino e teste e o KNN acertava por memorização.

**`home-credit-completa` (Home Credit).** As sete colunas de `previous_application`
homônimas das de `application_data` deixaram de ser descartadas como
"redundantes" e entram com prefixo `PREV_`: descrevem o pedido **anterior**, e
não o atual (C3.26). `SK_ID_CURR` e `SK_ID_PREV` saem dos atributos e servem
só para a junção e o agrupamento. Toda partição é por cliente, porque a junção
replica cada um em ~4,9 linhas com o mesmo `TARGET` (G7). Não tem protocolo
temporal: a unidade é o cliente, não a transação.

**`sparkov` (Sparkov).** Saem dos atributos: `cc_num`, `trans_num`, o índice
sem nome, `first`, `last`, `street`, `dob`, `unix_time` e a data bruta. Entram:
hora, dia da semana, fim de semana, mês, dia, idade na data da transação e
distância cliente–estabelecimento. As quatro coordenadas (`lat`, `long`,
`merch_lat`, `merch_long`) **permanecem**: a localização de uma transação é
informação legítima para identificar fraude. O `trans_num`, único por
transação, é a explicação mais provável para o XGBoost sinalizar ~80% das
transações nessa base (G14c). O protocolo temporal usa a separação oficial
`fraudTrain`/`fraudTest`.

**`ieee-cis-completa*` (IEEE-CIS).** `TransactionID` e `TransactionDT` saem dos
atributos; de `TransactionDT` ficam `hour_of_day` e `day_of_week`. O
agrupamento usa um **identificador aproximado de titular**, combinando `card1`
+ `addr1`, porque `card1` sozinho é informação do cartão — tipo, categoria,
banco emissor —, e não do portador (G8-iii). Os arquivos `test_transaction.csv`/`test_identity.csv`, sem rótulo,
nunca são lidos.

### 2.3 Protocolos de avaliação

Três, todos sem parâmetro de tamanho:

- **`stratified-holdout`** — partição única 70/15/15, estratificada, respeitando
  o agrupamento. 1 dobra.
- **`stratified-k-fold`** — 5 dobras estratificadas por grupo; cada dobra dá
  treino e validação. É daqui que sai a estimativa de dispersão entre
  partições que faltava (G13). 5 dobras.
- **`temporal-holdout`** — corte cronológico 70/15/15; na Sparkov, respeita os
  arquivos oficiais. É o único que estima o desempenho esperado em operação
  (G8). 1 dobra.

O TabBERT usa os **mesmos protocolos** e a mesma semente dos demais blocos:
`temporal-holdout` onde a base tem ordem temporal, e `stratified-holdout`
agrupado por cliente na Home Credit. As janelas causais usam como histórico os
registros anteriores do titular nos splits anteriores. O antigo
`sequence-chronological` (corte dentro de cada titular) saiu da grade por não
coincidir com partição alguma dos outros blocos.

**Identidade da partição.** Cada split recebe um `partition_md5`, o MD5 do
*conjunto* de linhas brutas que ele contém, calculado antes de qualquer
transformação. Ele vai para o manifesto (coluna 23), e a linha de resultado
recebe o `partition_id` da dobra. Blocos com o mesmo `partition_id` avaliaram
exatamente as mesmas linhas, e a validação P9 recusa divergências.

**Pareamento (G16).** A linha de resultado traz quatro identidades:

| Coluna | O que identifica |
|---|---|
| `partition_id` | linhas brutas de treino, validação e teste, antes de qualquer transformação |
| `train_population_id` | linhas brutas do treino efetivo, depois da remoção de incompletos e antes da reamostragem; sem remoção, é o treino da partição |
| `unit_partition_id` | clientes de cada split, nas bases com unidade de avaliação |
| `train_unit_population_id` | clientes do treino efetivo, nas bases com unidade de avaliação; sem remoção, são os clientes do treino da partição |

| Comparação | Mesmo `partition_id` | Mesmo `train_population_id` |
|---|---|---|
| Estratégias de preenchimento da mesma base | sim | só entre as que não removem linhas |
| Famílias (clássico × nativo) | sim | não exigido (o nativo não remove) |
| Base × variante de ablação, mesma estratégia | sim | sim |
| `home-credit-completa` × `home-credit-somente-aplicacao` | mesmo `unit_partition_id` | mesmo `train_unit_population_id` |

- **Variante de colunas.** A `ieee-cis-sem-identidade` carrega a `ieee-cis-completa` e remove a identidade depois do corte, como as demais variantes de ablação. Por isso tem o mesmo `partition_md5`.
- **Remoção decidida na base de origem.** No `drop-incomplete-rows`, o limiar e as linhas removidas são decididos com as colunas da base de origem (`missing_screening.decide_fold_removal`), e a variante treina com as mesmas linhas.
- **Par da Home Credit.** As duas bases têm linhas diferentes, então o pareamento é por cliente. A completa usa junção à esquerda, para ter os mesmos 307.511 clientes. As duas bases agrupam por `SK_ID_CURR`, e holdout e k-fold sorteiam a tabela de clientes. Assim, cada cliente cai no mesmo split nas duas.
- **Remoção no par da Home Credit.** No `drop-incomplete-rows`, a decisão é tomada na completa: limiar, colunas triadas e linhas que sobrevivem. Um cliente fica no treino das duas bases se ao menos uma das suas linhas sobrevive na completa, e a somente-aplicação remove as colunas de aplicação que a triagem da completa removeu (`missing_screening.adopt_paired_removal`). As duas treinam com os mesmos clientes, e a diferença entre elas fica só nos atributos da junção. O custo é que a somente-aplicação perde clientes por faltantes em colunas que ela não tem. Na semente 42, holdout estratificado: limiar de 20%; a completa treina com 814.628 linhas e a somente-aplicação com 162.437 clientes, os mesmos da completa. Decidindo sozinha, a somente-aplicação escolheria 30% e manteria 170.923 clientes.

Saíram: `random-split` (nome enganoso — era estratificado em quase todas as
bases e temporal por acidente na Sparkov) e `stratified-train-test-split` (que
descartava de 20% a 40% dos dados sem justificativa e, na Sparkov, jogava fora
o `fraudTest.csv` inteiro: 30,0% das transações e 22,2% das fraudes, G9d).

### 2.4 Grade por bloco

Gerada por `experiment_grid.py`, que consulta o catálogo e **só varia um eixo
onde o fenômeno existe**:

```
python3 experiment_grid.py --block classic        --format summary
python3 experiment_grid.py --block boosting       --format summary
python3 experiment_grid.py --block ft-transformer --format summary
python3 experiment_grid.py --block tabbert        --format summary
```

| Bloco | Configurações | Dobras | Algoritmos |
|---|---:|---:|---|
| clássico | 892 | 2.252 | regressão logística (`none` em todas; `sqrt-balanced` e `balanced` nas 223 sem reamostragem), Naive Bayes, KNN |
| boosting nativo | 116 | 292 | CatBoost, XGBoost (`none` em todas; `sqrt-balanced` e `balanced` nas 58 sem reamostragem) |
| FT-Transformer | 96 | 96 | `ft-transformer-mlp`, `ft-transformer-lstm` |
| TabBERT | 12 | 12 | `tabbert-mlp`, `tabbert-lstm`, cada um nos três regimes |

O TabBERT roda só nas cinco bases que têm titular e ordem; a ULB fica de fora
porque suas linhas são transações anônimas, sem portador. Ele também não varia
reamostragem: RUS e SMOTE quebrariam as janelas cronológicas, retirando ou
inventando posições no meio do histórico de um titular. Cada configuração roda
nos três regimes de treino do item C3.19, com o mesmo total de épocas
supervisionadas: `supervised-only` (S, sem pré-treino), `mlm-before-finetune`
(PM→S) e `warmup-mlm-finetune` (S→PM→S: um terço das épocas de aquecimento,
o pré-treino e o restante de ajuste). Cada base roda as duas estratégias de
cardinalidade (`minimum-frequency` e `hashing`) e as duas cabeças (`mlp` e
`lstm`): são 12 configurações e 72 execuções, cerca de 17,5 horas de GPU pela
medição de 31/10/2026 no Acer.

Configurações por base (bloco clássico): ULB 12, Home Credit 64, Sparkov 24,
e 96 em cada uma das três visões IEEE-CIS. A ULB tem poucas porque os eixos de
codificação e imputação não se aplicam a ela.

**Eixos do bloco clássico:** codificação categórica, um fatorial 2×2
(`smoothed-mean:k-fold`, `smoothed-mean:permutation-prefix`,
`log-odds:k-fold`, `log-odds:permutation-prefix`, ou `none` onde não há
categóricas) × imputação (`mean`, `median`, `mode`,
`drop-incomplete-rows`, ou `none` onde não há faltantes; no bloco nativo,
`native` no lugar da remoção, para o estimador tratar a ausência) × balanceamento
(`none`, `random-undersampling`, `RUS+SMOTE`, `tomek-links`) × protocolo ×
semente.

**Pesos de classe.** São dois braços do eixo de balanceamento, ao lado de
`none`, `random-undersampling`, `RUS+SMOTE` e `tomek-links` (G10, C3.14):
`sqrt-balanced` (√R) e `balanced` (R), com R = negativos/positivos medido no
treino de cada dobra. Rodam **só sobre as configurações sem reamostragem**: num
treino já 1:1, R ≈ 1 e os pesos repetiriam o `none`, e com Tomek Links a
ponderação corrigiria o desbalanceamento duas vezes. O código recusa a
combinação (`validate_class_weight_arm`). Valem para regressão logística,
CatBoost e XGBoost; Naive Bayes e KNN não têm noção de peso por classe.

**Limiar de decisão.** Depois de reamostrar ou ponderar, as probabilidades
deixam de corresponder à prevalência real (0,17% a 8,7%) e o limiar de 0,5 vira
arbitrário (G11). Em todos os blocos e algoritmos o limiar é o que **maximiza
F2 sobre as previsões de validação**, aplicado ao teste
(`decision_threshold.py`):

| Protocolo | De onde vem o limiar | `threshold_strategy` |
|---|---|---|
| `stratified-holdout`, `temporal-holdout` | validação da própria dobra | `validation-f2` |
| `stratified-k-fold` | validação das **outras** dobras (aqui a validação é o conjunto avaliado) | `cross-fold-validation-f2` |
| validação sem as duas classes | reserva de 0,5 | `fixed` |

Limiar e estratégia são gravados em cada linha. `THRESHOLD_STRATEGY=fixed`
volta ao 0,5. Medição antiga na ULB, holdout, RUS+SMOTE, ainda com limiares
fixos, que mostra por que o corte precisa ser escolhido:

| Limiar | Precisão | Revocação | F1 | FPR | AUC-ROC |
|---:|---:|---:|---:|---:|---:|
| 0,5 | 0,055 | 0,859 | 0,103 | 0,0247 | 0,970 |
| 0,7 | 0,092 | 0,859 | 0,166 | 0,0142 | 0,970 |
| 0,9 | 0,173 | 0,789 | 0,284 | 0,0063 | 0,970 |

A AUC-ROC não muda: o balanceamento desloca o **ponto de operação**, não a
capacidade de ordenação — exatamente o argumento de C5.9/G10d.

### 2.5 Pré-processamento do item G14

- **`log1p` nos valores monetários** (`Amount`, `amt`, `TransactionAmt`, declarados em
  `monetary_columns`), antes de imputar, padronizar e aplicar a PCA: bloco
  clássico, Isolation Forest e DevNet. Os arquivos **nativos** (boosting) guardam
  o valor original. O Isolation Forest **não** é invariante à escala: sorteia os
  cortes entre o mínimo e o máximo de cada atributo.
- **Códigos categóricos** (`categorical_code_columns`; IEEE-CIS: `card1`, `card2`,
  `card3`, `card5`, `addr1`, `addr2`) viram texto na carga. Como um CSV não guarda
  tipo, a leitura dos arquivos **nativos** os devolve como texto.
- **Colunas constantes** (valor único na base inteira; o ausente conta como valor)
  saem na carga. Na IEEE-CIS-completa são 23: 21 totalmente ausentes e `C5`, `C9`.
- **Registro de colunas**: cada `<dataset_md5>.csv` ganha `<dataset_md5>.columns.json`,
  e o manifesto ganha `feature_schema_md5` (coluna 26). A identidade da
  configuração inclui `PREPROCESSING_REVISION = "g14"`, então **toda configuração
  preparada antes precisa ser preparada de novo** (mova o manifesto antigo para
  o lado).

---

## 3. Estratégia de validação contra vazamento

Dois scripts, dois ângulos. Nenhum confia no código: ambos olham para os dados.

```bash
python3 validate_prepared_datasets.py --mode prepared
python3 validate_prepared_datasets.py --mode partition --dataset sparkov
python3 validate_prepared_datasets.py --mode partition   # todas as bases
```

### 3.1 Modo `prepared` — auditoria dos arquivos gravados

| ID | Verificação | Protege contra |
|---|---|---|
| **P1** | MD5 do conteúdo de cada arquivo confere com o do manifesto | métrica atribuída a dados que não são os daquela configuração |
| **P2** | Nenhuma linha aparece em duas partições da mesma dobra | reuso de linha entre treino e teste |
| **P3** | Nenhum identificador, agrupamento ou marca de tempo sobrou nos atributos | memorização via `trans_num`, `cc_num`, `TransactionID` |
| **P4** | Validação e teste mantêm a prevalência da base | reamostragem vazando para fora do treino |
| **P5** | `fold_index` começa em 1 e é contíguo | dobras perdidas ou renumeradas |
| **P6** | Duas configurações distintas nunca produzem o mesmo conjunto de arquivos | as duplicatas de `eqm-ref-21` e as 8 cópias por base sem categóricas |
| **P7** | `eqm-ref-21`/`ulb-pca` não aparecem no manifesto | reintrodução acidental das bases banidas |
| **P8** | Nos modos de Transformer, o vocabulário da dobra não lista nenhum identificador como campo do modelo | identificador virando atributo no TabBERT, onde a coluna precisa sobreviver no arquivo |
| **P9** | Mesma base de origem, protocolo, semente, dobra e split têm o mesmo `partition_md5` em todos os blocos e variantes; nos pares por unidade, o mesmo `unit_partition_md5` | blocos preparados em momentos diferentes que deixaram de avaliar as mesmas linhas (carga, código ou biblioteca mudou); variante de ablação com outra partição; par da Home Credit com outros clientes |
| **P16** | No `drop-incomplete-rows`, base e variante de ablação treinam com as mesmas linhas (`train_population_md5`) e o par da Home Credit com os mesmos clientes (`train_unit_population_md5`); sem remoção, o treino efetivo é o da partição | remoção que tira linhas ou clientes diferentes das bases comparadas |

**P8 encontrou um vazamento real durante o desenvolvimento.** No modo
sequencial as colunas de titular e de ordem precisam sobreviver no arquivo,
porque o índice de janelas causais as referencia — e, sem essa verificação,
`TransactionID`, `TransactionDT`, `approximate_holder_id` e, na Sparkov,
`cc_num` e `trans_num` entrariam como campos do modelo. A lista de exclusão
passou a ser derivada do catálogo, com uma única fonte de verdade.

P2 distingue duas causas. Se a mesma partição já contém repetições internas, a
coincidência entre partições vem do **dado** (transações distintas com
atributos idênticos) e é registrada como observação; sem repetição interna,
seria reuso de linha e falha. A prova forte de disjunção é D4.

### 3.2 Modo `partition` — reconstrução a partir da base bruta

Verifica o que os arquivos preparados já não conseguem mostrar, porque as
colunas foram removidas. Refaz a partição com a mesma semente, **sem** o
descarte, e inspeciona.

| ID | Verificação | Protege contra |
|---|---|---|
| **D1** | Nenhum cliente/cartão/titular em duas partições | G7 (Home Credit), G8 (Sparkov, IEEE-CIS) |
| **D2** | No protocolo temporal, todo treino é anterior a validação e teste | G8 |
| **D3** | O treino tem as duas classes | partição degenerada |
| **D4** | Os índices de origem das partições são disjuntos | reuso de linha, prova independente dos atributos |

**D1 sob o protocolo temporal é deliberadamente informativo, não falha.** Na
Sparkov, 908 dos 999 cartões aparecem dos dois lados do corte temporal — e
devem aparecer: o que separa as partições é o **período**, não o titular, e
esse é o cenário de operação, em que o sistema conhece o histórico do cartão e
decide sobre a próxima transação dele. Exigir grupos disjuntos ali mediria
outra coisa (generalização para titulares nunca vistos). A garantia desse
protocolo é D2.

### 3.3 Resultados das validações já executadas

| Base | Modo | Verificações | Falhas |
|---|---|---:|---:|
| `ulb` | partition | 15 | 0 |
| `ulb` | prepared (4 configurações) | 34 | 0 |
| `sparkov` | partition | 22 | 0 |
| `ieee-cis-transacional` | partition | 22 | 0 |
| `ieee-cis-completa` (TabBERT, modo sequencial) | prepared | 8 | 0 |

Exemplos do que foi provado com dados:

```
[OK] D4 sparkov/stratified-holdout/dobra-1: 1852394 linhas distribuidas em
     particoes disjuntas por indice de origem (train=1280919, validation=301482,
     test=269993)
[OK] D1 sparkov/stratified-holdout/dobra-1: 999 grupos de 'cc_num'
     totalmente separados entre as particoes
[OK] D2 sparkov/temporal-holdout/dobra-1: treino inteiramente anterior a
     validacao e ao teste: train=[2019-01-01 00:00:18, 2020-03-19 07:30:44],
     validation=[2020-03-19 07:30:50, 2020-06-21 12:13:37],
     test=[2020-06-21 12:14:25, 2020-12-31 23:59:34]
[OK] D1 ieee-cis-transacional/stratified-holdout/dobra-1: 243759 grupos de
     'approximate_holder_id' totalmente separados entre as particoes
[OK] P4 .../dobra-1: prevalencia validacao/teste ['0.0017','0.0017']
     (treino 0.5)
```

A última linha é a demonstração direta de que a reamostragem toca **só** o
treino: ele vai a 1:1 (0,5) enquanto validação e teste permanecem em 0,17%.

### 3.4 Rastreabilidade das métricas

O contrato é de **uma linha por (dobra, split)**, não uma por dobra. Cada
linha carrega `fold_index` (1-based), `split`, `random_seed`,
`classification_threshold` (o limiar aplicado naquela dobra),
`threshold_strategy` e os **MD5 de conteúdo** de treino, validação e
teste. Dada uma métrica, dá para voltar aos bytes exatos que a produziram; e
`load_prepared_data_splits` recusa a execução se o arquivo no disco divergir do
manifesto.

A seleção da melhor configuração usa as linhas `split=validation`;
`split=test` fica reservado para o relato final e `split=train` serve para
diagnosticar sobreajuste (item G12). A versão anterior gravava um único split,
com prioridade `test > validation > train`, e como os três protocolos sempre
produzem teste as métricas de validação eram calculadas e descartadas na
escrita. Nos Transformers o treino não é avaliado: ali exigiria uma passada de
inferência inteira sobre o maior split para uma medida que não entra em seleção
nem em relato. O dashboard pré-seleciona `split=test` ao carregar o arquivo,
para que a leitura padrão não some os splits na mesma média.

### 3.5 Testes automatizados

`tests/unit/test_revised_pipeline.py` — 61 testes, cada um nomeando o item da
revisão que protege:

```bash
python3 -m pytest tests/unit/test_revised_pipeline.py -q
```

Cobrem: bases banidas, política de PCA, descarte de identificadores,
deduplicação, identidade sem `test_data_size` e com semente, disjunção de
grupos nos três protocolos, ordem temporal, padronização ajustada só no treino,
remoção de incompletos só no treino, `fold_index` 1-based, MD5 por arquivo,
estratégia de limiar na chave de execução, limiar escolhido na validação (e,
no k-fold, nunca na dobra avaliada), pesos de classe só sem reamostragem,
ausência de configurações duplicadas na grade, braço sem reamostragem, remoção da cabeça `linear`, o fatorial 2×2 da
codificação categórica (estatística × controle de vazamento, com o teste de
categoria única que detecta vazamento nos dois mecanismos) e a gravação de uma
linha por split.

---

## 4. Comparações possíveis

Toda comparação é pareada pela chave
`configuration_md5 + database_type + algorithm + fold_index + split + regime`,
e agora também por `random_seed`, `class_weight_intensity` e `threshold_strategy`.

### 4.1 Entre algoritmos, sobre a mesma partição

Os três blocos partem do **mesmo catálogo, dos mesmos protocolos e da mesma
semente**. Para um dado protocolo e dobra, os conjuntos de treino/validação/
teste são os mesmos, o que permite comparação pareada direta:

- **dentro da família clássica** — regressão linear × logística × Naive Bayes ×
  KNN;
- **dentro do boosting** — CatBoost × XGBoost;
- **entre famílias** — clássicos × boosting, agora com uma estratégia de
  balanceamento comum (`random-undersampling` 1:1 nos dois blocos). Antes isso
  era impossível: o bloco clássico só tinha RUS+SMOTE e Tomek Links, o nativo
  só RUS, e a opção `none` removia linhas em um e as mantinha no outro, de modo
  que as partições sequer coincidiam (G10f, G16);
- **clássicos/boosting × FT-Transformer** — nos dois protocolos de holdout;
- **TabBERT × demais famílias** — o TabBERT usa os mesmos protocolos e a mesma
  semente (`temporal-holdout` onde há ordem temporal, `stratified-holdout`
  agrupado por cliente na Home Credit), então o par é pela mesma partição. Ele
  não roda na ULB (sem titular) e não tem braço de reamostragem: os pares com
  ele usam só o braço `none`.

### 4.2 Efeito de cada decisão de preparação

Pares que diferem em **um** eixo, com todo o resto fixo:

| Comparação | Eixo isolado | Bases |
|---|---|---|
| `smoothed-mean` × `log-odds`, mecanismo fixo | **estatística** da codificação categórica | 19, 20, 22, 22-noid, 22-trans |
| `k-fold` × `permutation-prefix`, estatística fixa | **mecanismo de controle de vazamento** da codificação | 19, 20, 22, 22-noid, 22-trans |
| `mean` × `median` × `mode` × `drop-incomplete-rows` | imputação | 19, 22, 22-noid, 22-trans |
| `none` × `random-undersampling` × `RUS+SMOTE` × `tomek-links` × `sqrt-balanced` × `balanced` | balanceamento (os dois últimos são pesos de classe, sem reamostragem) | todas |
| `quantile` × `standard` | normalização numérica (FT-Transformer) | todas |
| `minimum-frequency` × `hashing` | alta cardinalidade (FT-Transformer) | todas |
| `mlp` × `lstm` | cabeça de classificação | todas |
| `supervised-only` × `mlm-before-finetune` × `warmup-mlm-finetune` | regime de treino, mesmo orçamento supervisionado (TabBERT, C3.19) | bases do TabBERT |

O braço `none` do balanceamento é o que torna a pergunta "balancear ajuda?"
respondível pela primeira vez (G10a).

### 4.3 Efeito do protocolo de avaliação

`stratified-holdout` × `stratified-k-fold` × `temporal-holdout` sobre a mesma
configuração de preparação. A diferença entre os holdouts embaralhado e
temporal é a medida direta do quanto as partições embaralhadas superestimam o
desempenho esperado em operação (G8) — exatamente a diferença que, na versão
anterior, aparecia como revocação de 0,697 contra 0,996–0,999 e era atribuída a
uma "tarefa mais exigente".

O `stratified-k-fold` fornece a dispersão entre partições (5 dobras), que
permite dizer se uma diferença de 0,007 em F1 entre codificações é sinal ou
ruído — o que o relatório apontou como impossível na versão anterior (G13).

### 4.4 Ablação IEEE-CIS

Três comparações independentes, cada uma isolando um fator:

- **`ieee-cis-completa` × `ieee-cis-completa-sem-alta-cardinalidade`** (só
  CatBoost e XGBoost; item G14) — mesmas linhas e mesmas partições
  (`partition_md5` idêntico), sem `DeviceInfo`, `id_31`, `id_33` e `card1`: isola
  o efeito dos atributos de alta cardinalidade.

- **`ieee-cis-completa` × `ieee-cis-sem-identidade`** — mesmas 144.233 transações, com e
  sem as colunas de identidade. Conjunto de linhas fixo, atributos variando:
  isola o **valor da identidade**.
- **`ieee-cis-sem-identidade` × `ieee-cis-transacional`** — mesmos atributos
  transacionais, 144.233 contra 590.540 linhas: isola o **efeito do volume**.

Com `drop-incomplete-rows`, cada visão escolhia seu próprio limiar, porque a
com identidade tem 40 colunas a mais, muitas quase vazias (G16-i).

Medido sobre os dados brutos, com semente 42:
- no holdout estratificado, a sem-identidade treinava com 1.987 linhas a mais;
- no temporal, 21.632 linhas ficavam só na completa e 1.852 só na sem-identidade.

Agora a `ieee-cis-sem-identidade` é variante de colunas, e a decisão de remoção
é tomada com as colunas da base completa. As duas visões têm o **mesmo teste e
o mesmo treino efetivo**, verificados pela P16 e pela R5, e diferem só nos
atributos.

### 4.5 Entre as bases

Esta é a comparação mais delicada, porque a prevalência varia por um fator de
50 entre as bases:

| Base | Nome | Prevalência | Domínio |
|---|---|---:|---|
| `ulb` | ULB | 0,173% | Fraude em cartão |
| `home-credit-completa` | Home Credit | 8,7% | Risco de crédito (controle) |
| `sparkov` | Sparkov | 0,579% (treino) / 0,386% (teste) | Fraude em cartão (sintética) |
| `ieee-cis-completa` | IEEE-CIS com identidade | 7,85% | Fraude em cartão |
| `ieee-cis-sem-identidade` | IEEE-CIS sem identidade | 7,85% | Fraude em cartão |
| `ieee-cis-transacional` | IEEE-CIS transações | 3,50% | Fraude em cartão |

**F1, F2 e precisão não são comparáveis entre bases.** As três dependem da
prevalência do conjunto avaliado: um F1 de 0,30 na ULB (0,173%) e um F1 de 0,30
na IEEE-CIS (7,85%) descrevem desempenhos muito diferentes. Para comparar entre
bases, use **AUC-ROC**, independente da prevalência, ou o **ranking médio** dos
algoritmos dentro de cada base.

O que se pode comparar entre bases:

| Comparação | O que mede | Métrica adequada |
|---|---|---|
| ULB × Sparkov × IEEE-CIS | Consistência do desempenho relativo entre três fontes de fraude em cartão com características muito diferentes: componentes principais anonimizados, dados sintéticos e dados reais com centenas de atributos | AUC-ROC; ranking médio por base |
| Dimensionalidade: Sparkov (23 colunas) × IEEE-CIS (394–434) | Se a largura da base explica o desempenho. A versão anterior afirmava que sim, mas invertia os números: a Sparkov é a base com *menos* colunas | AUC-ROC pareada por algoritmo |
| Real × sintético: ULB/IEEE-CIS × Sparkov | Se conclusões obtidas em base sintética se transferem. A Sparkov é gerada pela ferramenta *Sparkov Data Generation* | AUC-ROC; ordenação dos algoritmos |
| Prevalência: ULB (0,17%) × IEEE-CIS (7,85%) | Como o desbalanceamento afeta o ganho de cada braço de balanceamento e o limiar escolhido na validação. É aqui que os braços `none` e de pesos de classe se tornam informativos | Diferença de F2 entre braços de balanceamento, dentro de cada base |
| Fraude × inadimplência: bases de cartão × Home Credit | Controle fora do domínio: alvo de inadimplência e unidade de cliente | Relatar **separadamente**, nunca na média das bases de fraude (G7-i) |
| Disponibilidade de protocolo temporal | ULB, Sparkov e IEEE-CIS têm ordem temporal; a Home Credit não. A diferença entre holdout embaralhado e temporal só é mensurável nas três primeiras | Diferença de revocação e AUC-ROC entre protocolos |
| Elegibilidade ao TabBERT | Cinco bases têm titular e ordem; a ULB não. A comparação FT-Transformer × TabBERT só existe nas cinco | AUC-ROC, sobre a mesma base |

**Conte fontes, não configurações.** Ao generalizar, são **4 fontes**, não 6
bases: `ieee-cis-completa`, `-noidentity` e `-transaction` são três visões da mesma
IEEE-CIS. Frases como "vence nas cinco bases" superestimavam a amplitude da
conclusão (G9h).

### 4.6 O que continua não comparável

- **F1, F2, precisão e AUC-PR entre bases com prevalências diferentes** (0,17%
  a 8,7%). A AUC-PR tem como linha de base a própria prevalência, não 0,5. Para
  comparar entre bases, use AUC-ROC ou o ranking médio dentro de cada base.
- **Resultados deste pipeline com os do anterior.** As partições, os atributos e
  a população mudaram. A comparação honesta é conceitual, não numérica.
- **Números de Itoo et al. [24]**, artigo retratado em 30/07/2026 (G1).
- **TabBERT na ULB e com reamostragem.** A ULB não tem titular, e RUS e SMOTE
  quebrariam as janelas cronológicas. Nas demais bases e no braço `none`, o
  TabBERT é comparável às outras famílias pela mesma partição.
- **Um modelo entre bases.** Cada modelo é treinado e avaliado dentro da
  própria base, e as bases diferem em atributos, alvo, prevalência e unidade:
  não há transferência a medir. A consistência entre bases é *robustez entre
  fontes* (ranking conjunto, `analise_hipoteses.py`), e não generalização; o
  termo fica para o `temporal-holdout` (G20).

---

## 5. Estimativa de tempo de execução

```bash
python3 estimate_execution_time.py
```

O script é auditável: todas as premissas de extrapolação são constantes
nomeadas no topo do arquivo.

**Base de medição.** (1) Medições próprias neste pipeline, na ULB, holdout sem
reamostragem — o cenário mais caro dos clássicos, porque o treino inteiro entra
no ajuste. (2) Medianas das execuções históricas por algoritmo e base,
extraídas de `resultados_originais_*.tar.gz` — cobrem CatBoost e XGBoost, que
não estão instalados neste ambiente. (3) Medições de GPU do FT-Transformer da
execução anterior (RTX 4050).

Medido aqui (ULB, 198.608 linhas de treino, 42.559 de teste, 30 atributos):

| Algoritmo | Segundos por dobra |
|---|---:|
| Naive Bayes | 2,1 |
| regressão linear | 3,6 |
| regressão logística | 3,8 |
| KNN | 39,2 |

**Premissas.** Regressões e Naive Bayes crescem linearmente com as linhas;
o KNN cresce com `n^1.5` (entre linear e quadrático, por causa da indexação em
árvore); boosting usa as medianas históricas × 2,5, fator que corrige a
introdução do braço sem reamostragem. A PCA na Sparkov e o descarte dos
identificadores **reduzem** o custo e não entram na conta: a estimativa é
conservadora nesse ponto.

### Resultado

| Algoritmo | Dobras | Estimativa | Faixa |
|---|---:|---:|---|
| XGBoost `[none]` | 292 | **10,2 d** | 6,1 d – 18,4 d |
| CatBoost `[none]` | 292 | **9,9 d** | 5,9 d – 17,8 d |
| XGBoost `[sqrt-balanced]`, `[balanced]` | 146 cada | **5,1 d** cada | 3,1 d – 9,2 d |
| CatBoost `[sqrt-balanced]`, `[balanced]` | 146 cada | **4,9 d** cada | 3,0 d – 8,9 d |
| TabBERT (2 cabeças × 2 regimes) | 10 cada | **3,6–4,4 d** cada | 2,2 d – 7,9 d |
| KNN | 2.252 | **3,8 d** | 2,3 d – 6,8 d |
| preparação | 1.114 | **3,2 d** | 45,6 h – 5,7 d |
| FT-Transformer MLP, LSTM | 96 cada | **6,8 h** cada | 4,1 h – 12,2 h |
| regressão logística `[none]` | 2.252 | **4,7 h** | 2,8 h – 8,5 h |
| Naive Bayes | 2.252 | **2,6 h** | 1,6 h – 4,7 h |
| regressão logística `[sqrt-balanced]`, `[balanced]` | 563 cada | **71 min** cada | 43 min – 2,1 h |

| Bloco | Estimativa |
|---|---:|
| clássico | 4,2 d |
| boosting nativo | 40,2 d |
| FT-Transformer | 13,5 h |
| TabBERT | 16,0 d |
| preparação | 3,2 d |
| **TOTAL sequencial** | **64,1 d** (faixa 38,5 d – 115,4 d) |

Restringir os pesos de classe às configurações sem reamostragem cortou o
boosting de ~60 para ~40 dias: cada braço ponderado cobre metade das
configurações.

### O gargalo

As cinco combinações mais caras concentram ~17,5 dos 64,1 dias:

| Algoritmo | Base | Dobras | s/dobra | Total |
|---|---|---:|---:|---:|
| XGBoost | `home-credit-completa` | 48 | 8.378 | 4,7 d |
| CatBoost | `home-credit-completa` | 48 | 8.211 | 4,6 d |
| XGBoost | `ieee-cis-transacional` | 56 | 4.618 | 3,0 d |
| CatBoost | `ieee-cis-transacional` | 56 | 4.411 | 2,9 d |
| XGBoost `[sqrt-balanced]` | `home-credit-completa` | 24 | 8.378 | 2,3 d |

O boosting na Home Credit domina tudo: 1,41 milhão de linhas, 29 atributos
categóricos de alta cardinalidade e 1.000 iterações no padrão da biblioteca.

**Reduções possíveis, se o prazo apertar** (nenhuma aplicada; todas mudam o
escopo e precisam de decisão sua):

1. Limitar `home-credit-completa` ao protocolo `stratified-holdout` no boosting (as 5
   dobras do k-fold são 4/5 do custo dessa base) → economiza ~7 dias.
2. Reduzir as iterações de CatBoost/XGBoost com parada antecipada na validação,
   que o relatório pede de qualquer forma (G12) → tipicamente 3–5× mais rápido.
3. Rodar os blocos em paralelo: são processos independentes, e o clássico e o
   boosting usam CPU enquanto o FT-Transformer usa GPU.

---

## 6. Como executar

Todos os scripts recebem como **primeiro argumento** o diretório dos arquivos
preparados — onde a preparação grava e de onde a execução lê. Os valores
aceitos são `/media/marcospiaia/dayasets` e `/media/marcospiaia/new-datasets`
(lista em `data_directory_common.sh`); o HD precisa estar montado ali.

```bash
DADOS=/media/marcospiaia/new-datasets   # ou /media/marcospiaia/dayasets

# Preparação (grava no HD externo; honra FORCE_RECREATE_PREPARED_DATASETS)
./prepare_lr_nb_knn_specific_experiment_datasets.sh "$DADOS"
./prepare_catboost_xgboost_native_experiment_datasets.sh "$DADOS"
./prepare_ft_transformer_experiment_datasets.sh "$DADOS"
./prepare_tabbert_experiment_datasets.sh "$DADOS"

# Validação — rode ANTES de executar os experimentos
python3 validate_prepared_datasets.py --mode prepared --prepared-data-directory "$DADOS"
python3 validate_prepared_datasets.py --mode partition --prepared-data-directory "$DADOS"

# Execução (honra FORCE_REPROCESS e EXECUTION_DEVICE)
./run_lr_nb_knn_specific_prepared_experiments.sh "$DADOS" [all|knn naive-bayes ...]
./run_catboost_xgboost_native_prepared_experiments.sh "$DADOS" [all|catboost|xgboost]
./run_ft_transformer_prepared_experiments.sh "$DADOS" [all|mlp|lstm]
./run_tabbert_prepared_experiments.sh "$DADOS" [all|mlp|lstm]
./run_anomaly_detection_prepared_experiments.sh "$DADOS" [all|isolation-forest|devnet]

# Uma única linha do manifesto, para depurar
./run_lr_nb_knn_specific_prepared_experiment_line.sh "$DADOS" <linha>=2.. [algoritmos]

# Contagem de execuções previstas x realizadas (G17) e tabelas do apêndice
python3 validate_prepared_datasets.py --mode results --prepared-data-directory "$DADOS" \
    --manifest <manifesto.tsv> --results <resultados>/resultados_*.csv \
    --execution-log <resultados>/registro_execucoes.csv --appendix-output apendice/

# Figuras e números do Capítulo 5 da dissertação, sobre os resultados gravados
python3 gerar_discussao_resultados.py --results-directory <resultados> \
    --figures-directory <eqm_paper>/figuras --summary-output discussao_resultados_resumo.json

# Hipóteses H1–H3 e questões Q1–Q3 da dissertação (G20): seleção na validação,
# ranking conjunto, Friedman/Nemenyi e desfechos em resumo_hipoteses.json
python3 analise_hipoteses.py --results-directory <resultados> \
    --output-directory analise_hipoteses

# Ablação dos atributos de identidade da IEEE-CIS (H4), com as margens do Capítulo 3
python3 analise_ablacao_ieee.py --results-directory <resultados> \
    --output-directory analise_ablacao_ieee

# Figuras do Capítulo 5 sobre a grade revisada (todas as famílias), no padrão
# da dissertação: vírgula decimal, nomes descritivos, legenda fora dos painéis
python3 figuras_discussao.py --results-directory <resultados> \
    --figures-directory <eqm_paper>/figuras --summary-output figuras_discussao_resumo.json

# Tabelas LaTeX (booktabs, legenda acima, vírgula decimal, siglas via acro)
python3 tabelas_latex.py --results-directory <resultados> \
    --hypotheses-directory analise_hipoteses --ablation-directory analise_ablacao_ieee \
    --output-directory <eqm_paper>/tabelas

# Roteiro da reescrita: cada \todo dos Capítulos 4–6 e a saída que o responde
python3 roteiro_capitulos.py --thesis-directory <eqm_paper> \
    --output-directories <eqm_paper>/figuras <eqm_paper>/tabelas analise_hipoteses analise_ablacao_ieee . \
    --output roteiro_capitulos.md
```

**Figuras e tabelas da dissertação (Seção 4 do relatório de revisão, F6 e F7).**
`gerar_discussao_resultados.py` continua cobrindo a grade de 22/09 (só
clássicos e boosting, com os eixos antigos); para a grade revisada, as figuras
vêm de `figuras_discussao.py`, que cobre também a detecção de anomalias e os
Transformers. Os dois usam `estilo_figuras.py`: rótulos em português, AUC-ROC,
AUC-PR e F2, `dataset_name` no lugar de `eqm-ref-NN`, vírgula decimal sem zero
negativo, fonte de 8 pt na largura impressa de 16 cm, legenda fora dos painéis,
marcadores e hachuras além da cor e paleta `colorblind`. Os scripts de análise
só leem resultados: nenhum deles é importado pelos executores.

**Hipóteses (G20).** `analise_hipoteses.py` só lê os resultados e aplica as
regras fixadas no Capítulo 3 da dissertação. A execução representativa de cada
algoritmo e base é a de maior $F_2$ na **validação**. As fontes de fraude são
ULB, Sparkov e IEEE-CIS-completa, no `temporal-holdout`, e a Home Credit é
ordenada à parte. O IC da diferença de $F_2$ usa, de preferência, a unidade
`transaction`: *bootstrap* pareado e estratificado por classe sobre as linhas do
teste, a partir dos escores gravados por linha. Os blocos clássico, de boosting
e de anomalia gravam esses escores em `resultados_<algoritmo>_test_scores.jsonl`
(um registro por dobra, escores em float32 e rótulos em bits, em base64), e os
Transformers, no `transaction_predictions` do `_details.jsonl`. O par só é
pareado se as duas execuções avaliaram as mesmas linhas (mesmas sementes e
dobras, mesma sequência de rótulos e, quando gravado, o mesmo
`test_partition_md5`). Sem isso, usa a unidade `seed` (pareada) com duas ou
mais sementes. Com semente única, usa `confusion`: um *bootstrap* da
matriz de confusão reconstruída de precisão, revocação, FPR e tamanho do teste,
não pareado e, por isso, conservador. Sem essas informações, a unidade é
`none` e o desfecho que depende do IC fica inconclusivo. A unidade usada vai
para o resumo. Na H1, a margem de não inferioridade é 0,010 em $F_2$, a mesma
da ablação IEEE-CIS; `h1_sensibilidade_margem.csv` repete o desfecho de cada
par e o global com 0,005 e 0,020, e o resumo indica se ele muda com a margem.
Q1–Q3 são exploratórias e não têm desfecho.

**Contagem de execuções e motivo das falhas (G17).** Toda linha
`execution_status=failure` grava `failure_category` (`memoria`,
`convergencia`, `particao-vazia`, `armazenamento` ou `outra`) e
`failure_message`. Um treino sem fraudes ou um split vazio interrompe a
execução antes do ajuste, como `particao-vazia`. Cada tentativa de comando dos
executores acrescenta uma linha a `registro_execucoes.csv` (no diretório de
resultados) com o código de saída: um processo morto pelo OOM killer (137) não
grava resultado, e é esse registro que explica a ausência. O R4 cruza a grade
prevista de todos os blocos — multiplicada por `SEED_LIST` e, no DevNet, por
`LABEL_FRACTIONS` e `LABEL_DRAW_SEEDS` — com os resultados e o registro e grava
`appendix_run_counts.csv` (previstas, sucesso, falha, incompleta e ausente, em
execuções e em linhas de teste, por bloco, base, algoritmo e eixos) e
`appendix_run_failures.csv` (uma linha por execução sem sucesso, com o
motivo). A tabela mostra o estado final de cada execução e o número de
tentativas.

**Pareamento estrito (G17).** Toda comparação do
`gerar_discussao_resultados.py` passa por `paired_comparison`: um par são duas
execuções iguais em todos os eixos, exceto o comparado, com sucesso em todas as
dobras previstas e avaliadas nas mesmas partições. O par incompleto é
descartado, nunca diluído numa média. `pares_por_comparacao.csv`, gravado ao
lado do resumo, traz por comparação, base e algoritmo os pares previstos (da
grade, não dos dados), realizados e descartados por motivo. As células das
figuras de codificação e de reamostragem por base usam só as execuções que
formam par e mostram o *n*. Só entram linhas de teste avaliadas por registro.

`gerar_discussao_resultados.py` foi escrito sobre o esquema da grade de
22/09/2026 (reserva em `test_data_size`, políticas antigas,
`ordered-target`/`woe`). Antes de gerar figuras, ele compara as colunas e os
valores de cada eixo com os resultados carregados e imprime cada divergência
como `[revisar]`; análises sem as colunas ou os valores que supõem são puladas
com aviso. Sempre que `RESULT_CSV_COLUMNS` ou os valores de um eixo da grade
mudarem, revisar essas análises antes de levar uma figura ou um número para o
texto.

Pelo item G19, o script não gera mais as figuras da comparação com Itoo et al.
(artigo retratado: atributos e classes, métricas, particionamento,
balanceamento, limiar 0,5 e distribuições da ULB) nem as da PCA na ULB
(`pca_efeito_eqm_ref18.png` e `comparacao_bases_pca.png`). As linhas das bases
que saíram do catálogo (`eqm-ref-21` e `eqm-ref-18-pca`) são descartadas na
carga, de modo que os CSVs de 22/09 ainda podem ser lidos.

Variáveis de ambiente: `PYTHON_COMMAND`, `CONFIGURATION_MANIFEST`,
`RESULTS_DIRECTORY`, `LOCAL_STAGING_DIRECTORY`, `EXECUTION_DEVICE`,
`DATASET_FILTER`, `SEED_LIST`, `MAX_PREPARED_CONFIGURATIONS`,
`FORCE_RECREATE_PREPARED_DATASETS`, `FORCE_REPROCESS`,
`THRESHOLD_STRATEGY`, `STORAGE_DEVICE_UUID`, `STORAGE_RETRY_SECONDS`,
`STORAGE_RETRY_ATTEMPTS`, `GRID_PLAN`, `GRID_LANE`, `LABEL_FRACTIONS`,
`LABEL_DRAW_SEEDS`, `EXECUTION_LOG_FILE` (padrão:
`$RESULTS_DIRECTORY/registro_execucoes.csv`).

O contrato do **código de saída 75** foi preservado: quando o armazenamento
preparado fica ilegível, o Python levanta `PreparedDatasetStorageError` e sai
com 75. O script shell então espera o armazenamento voltar e repete o mesmo
comando; com um UUID configurado, também remonta o HD pelo `udisksctl` antes
de cada espera. A espera (`wait_for_storage`) confere o diretório a cada
`STORAGE_RETRY_SECONDS` (2 minutos), sem limite (`STORAGE_RETRY_ATTEMPTS=0`),
imprime cada tentativa que falha e avisa quando o acesso volta; o passo
interrompido (o comando do runner ou a preparação de uma configuração) é
refeito desde o início, sem linha de falha. Um 75 com o diretório acessível
(arquivo lateral ausente, relatório divergente) é repetido uma vez e depois
vira falha comum.

### Proveniência, tempos e execução em várias máquinas (item G18)

- **Proveniência:** cada linha de resultado grava o commit, o hash do ambiente de referência (`requirements.lock.txt`) e a máquina. Divergências de commit, árvore Git, pacotes ou versão do Python são informativas e não bloqueiam os scripts.
- **Tempos:** as dobras gravam `fit_time` e `scoring_time`, e a preparação grava o tempo de cada etapa em `<configuration_md5>.preparation_timing.json`.
- **Inferência:** `run_inference_benchmark.sh` mede a inferência numa máquina só.
- **Tabelas:** `build_reproducibility_tables.py` gera as tabelas de hiperparâmetros e de tempos da dissertação.
- **Várias máquinas:** a reexecução se divide entre o Acer, o E470 e o Dell, cada um com vários processos (`calibrate_machine.py --cpu-lanes`, `grid_plan.py`, `GRID_PLAN` e `GRID_LANE` nos runners). Os dados preparados ficam nos HDs da rede; `stage_machine_configurations.sh ... local` copia as configurações do Lenovo para o HD dele, `backup_lane_results.sh` leva os resultados de cada processo para a rede, e `merge_machine_results.py` os une.
- **Estratos de histórico (C3.19):** o CatBoost, o XGBoost, o FT-Transformer e o TabBERT gravam métricas, transações e prevalência por estrato de histórico; a checagem R6 confere as contagens, e `history_strata_analysis.py` gera a tabela.
- **Diretório de dados:** cada máquina declara o próprio diretório em `machine_storage.conf`.

O passo a passo está em [`../../REPRODUCAO.md`](../../REPRODUCAO.md).

---

## 7. Arquivos

| Arquivo | Papel |
|---|---|
| `experiment_datasets.py` | catálogo das bases: carga, identificadores, PCA, agrupamento, tempo |
| `data_partitioning.py` | os três protocolos de avaliação, todos group-aware |
| `experiment_grid.py` | geração da grade sem configurações duplicadas |
| `prepared_dataset_storage.py` | identidade MD5, gravação, MD5 de conteúdo, carga com verificação |
| `missing_screening.py` | limiar de faltantes por dobra antes do `drop-incomplete-rows` (G15c) |
| `preparation_report.py` | relatório de preparação por dobra e verificação do esquema comum (G15) |
| `prepare_lr_nb_knn_datasets.py` | preparação de uma configuração |
| `data_directory_common.sh` | diretórios de dados aceitos e validação do argumento |
| `prepare_grid_common.sh` | laço de preparação compartilhado pelos três blocos |
| `prepare_*_experiment_datasets.sh` | entradas de preparação por bloco |
| `decision_threshold.py` | escolha do limiar na validação (máximo F2; no k-fold, pelas outras dobras) |
| `lr_nb_knn_specific.py` | algoritmos clássicos, métricas e gravação de resultados |
| `catboost_xgboost_specific.py` | CatBoost/XGBoost com categorias nativas |
| `run_transformer_prepared_experiments.py` | FT-Transformer (cabeças `mlp` e `lstm`) |
| `run_storage_common.sh` | recuperação do armazenamento nos executores: espera, repetição do passo e processos (`GRID_LANE`) |
| `data_directory_common.sh` | escolha do diretório de dados e `wait_for_storage` (uma tentativa a cada 2 minutos, sem limite) |
| `history_strata.py` | métricas, transações e prevalência por estrato de histórico; checagem R6 (C3.19) |
| `history_strata_analysis.py` | tabela de métricas por estrato de histórico para a dissertação (C3.19) |
| `backup_lane_results.sh` | cópia periódica dos resultados de cada processo para um HD da rede |
| `execution_log_common.sh` | registro de cada tentativa de comando em `registro_execucoes.csv` (G17) |
| `execution_failures.py` | categoria e mensagem das falhas; verificação de partição vazia antes do ajuste (G17) |
| `run_*_prepared_experiments.sh` | executores por bloco |
| `review_items.py` | como cada observação da revisão foi endereçada |
| `build_public_report.py` | gera o relatório público em HTML |
| `DEPLOY.md` | como publicar o relatório no Firebase Hosting |
| `validate_prepared_datasets.py` | validação contra vazamento e inconsistência; ponto de entrada também de P10–P16 e do modo `results` (R1–R6) |
| `preparation_sanity_checks.py` | verificações de sanidade do G15/G17: cada eixo muda arquivos e resultados (P11, P12, R2), triagem, PCA, identidade, contagens, pareamento (P16, R5) e tabelas do apêndice, inclusive a de pareamento (`appendix_pairing.csv`) |
| `estimate_execution_time.py` | estimativa de tempo auditável |
| `gerar_discussao_resultados.py` | figuras e resumo numérico do Cap. 5 da dissertação; pareamento estrito e `pares_por_comparacao.csv` (G17); sem as figuras retiradas no G19; aponta com `[revisar]` o que diverge do esquema de 22/09 |
| `analise_hipoteses.py` | evidência de H1–H3 e Q1–Q3 da dissertação: seleção na validação, ranking conjunto, Friedman/Nemenyi, IC pareado por transação, por semente ou por matriz de confusão (G20) |
| `tests/unit/test_revised_pipeline.py` | 22 testes das garantias da revisão |
| `tests/unit/test_g15_screening_and_report.py` | triagem, relatório de preparação e proteções de entrada (G15) |
| `tests/unit/test_g15_verifier.py` | verificador P10–P15 e R1–R4, um caso que passa e um que falha por verificação |
| `tests/unit/test_g17_failure_reasons.py` | motivo das falhas e partição vazia (G17) |
| `tests/unit/test_g17_execution_log.py` | registro de execuções dos executores shell (G17) |
| `tests/unit/test_g17_run_counts.py` | grade prevista de todos os blocos e R4 por estado, motivo e linhas (G17) |
| `tests/unit/test_g17_strict_pairing.py` | pareamento estrito, descarte e contagem de pares (G17) |
| `tests/unit/test_g19_discussion_figures.py` | figuras e chaves do resumo retiradas da discussão e descarte das bases removidas (G19) |
| `tests/unit/test_tabbert_scope_and_regimes.py` | regimes S e PM→S com o mesmo orçamento e grade reduzida do TabBERT (C3.19) |
| `tests/unit/test_history_strata.py` | estrato igual ao do TabBERT, arquivo lateral, métricas por estrato, R6 e tabela (C3.19) |
| `tests/unit/test_storage_wait.py` | espera pelos HDs da rede e repetição do passo interrompido |
| `tests/unit/test_grid_lanes.py` | processos por máquina, memória simultânea, leitura pela rede, posicionamento, backup e junção |
| `tests/unit/test_g16_pairing.py` | pareamento do G16: variante de identidade, remoção decidida na base de origem, par da Home Credit por cliente, rótulo `native`, P9, P16 e R5 |
| `tests/unit/test_g20_hypotheses_analysis.py` | seleção pela validação, desfechos de H1–H3, unidades de incerteza, Home Credit à parte e questões exploratórias (G20) |
| `tests/unit/test_g20_test_scores.py` | gravação dos escores do teste por linha, leitura pelo script de análise e *bootstrap* pareado sobre as transações (G20) |

---

## 8. Itens da revisão que este pipeline **não** resolve

São de texto, de análise ou de escopo, e ficam para a dissertação:

- **G1** — retratação de Itoo et al.; **G3** — enquadramento (anomalias ×
  classificação supervisionada); **G5** — Resumo e *Abstract*; **G6** —
  objetivos e hipóteses; **G19** — organização dos capítulos (o pipeline só
  retira as figuras redundantes); **G21** — revisão bibliográfica;
  **F1**–**F11** — formatação.
- **G20** — "generalização": o texto é da dissertação. A evidência das
  hipóteses reformuladas vem de `analise_hipoteses.py`. Os escores do teste
  por linha vão para `resultados_<algoritmo>_test_scores.jsonl`
  (`record_algorithm_result_to_csv`), coletado e mesclado entre máquinas como
  os demais companheiros, e dão o *bootstrap* pareado sobre as transações.
- **G12** — seleção na validação e busca de hiperparâmetros. O pipeline agora
  **produz** o conjunto de validação em todos os protocolos e grava suas
  métricas. A seleção é feita na análise: `analise_hipoteses.py` escolhe pela
  validação; as figuras de `gerar_discussao_resultados.py` ainda mostram o
  teto de cada métrica no teste.
- **G13** — testes estatísticos (Wilcoxon/Friedman). O pipeline fornece a
  dispersão entre dobras e o `SEED_LIST`; os testes são da fase de análise.
  O Friedman e o Nemenyi das hipóteses estão em `analise_hipoteses.py`.
- **Dashboard e `test_data_size`.** A AUC-PR já entrou como métrica opcional em
  `experiment_results_dashboard/app/constants.py`, de forma retrocompatível.
  Mas o dashboard ainda exige `test_data_size` em `FILTER_COLUMNS` e
  `REQUIRED_COLUMNS`, coluna que este pipeline não grava mais — os resultados
  novos seriam rejeitados na carga. Resolver isso é uma decisão sua sobre o
  dashboard.
