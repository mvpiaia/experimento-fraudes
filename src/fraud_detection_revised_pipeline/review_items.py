"""Como cada observacao do relatorio de revisao foi endereçada.

Os itens seguem a numeracao do *Relatorio de Revisao* da primeira versao da
dissertacao (01/10/2026): G1-G23 (problemas estruturais e metodologicos),
F1-F11 (formatacao, LaTeX, figuras, tabelas e referencias) e os itens C*.*
pontuais que dependem de codigo.

Cada item tem um ``status``:

``resolvido``
    O pipeline implementa a correcao e ha teste ou validacao que a protege.
``parcial``
    O pipeline faz a parte que lhe cabe; o restante e da fase de analise ou
    da redacao.
``dissertacao``
    Nao depende de codigo: e texto, estrutura, referencia ou formatacao.
``decisao``
    Depende de uma decisao do aluno ou do orientador antes de implementar.
"""

from __future__ import annotations

from dataclasses import dataclass


STATUS_RESOLVED = "resolvido"
STATUS_PARTIAL = "parcial"
STATUS_THESIS = "dissertacao"
STATUS_DECISION = "decisao"

SEVERITY_CRITICAL = "CRITICO"
SEVERITY_IMPORTANT = "IMPORTANTE"
SEVERITY_MINOR = "MENOR"


@dataclass(frozen=True)
class ReviewItem:
    identifier: str
    title: str
    severity: str
    status: str
    # O que a revisao apontou, em uma ou duas frases.
    problem: str
    # O que foi feito. Vazio quando o item nao depende de codigo.
    response: str
    # Onde conferir: arquivo, funcao, teste ou verificacao.
    evidence: tuple[str, ...] = ()


REVIEW_ITEMS: tuple[ReviewItem, ...] = (
    # ---------------------------------------------------------------- G1-G6
    ReviewItem(
        "G1",
        "Estudo-base retratado",
        SEVERITY_CRITICAL,
        STATUS_THESIS,
        "Itoo et al. [24], usado como ponto de partida do desenho experimental "
        "e como referencia numerica, foi retratado pela revista em 30/07/2026.",
        "Nao depende de codigo. O desenho experimental deste pipeline nao "
        "reproduz mais o arranjo daquele artigo: os algoritmos classicos "
        "entram como linhas de base por serem classicos, e nenhuma comparacao "
        "numerica com ele e feita pelo pipeline.",
    ),
    ReviewItem(
        "G2",
        "Versao incompleta",
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "Faltavam resultados dos Transformers, a ablacao IEEE-CIS, tres "
        "configuracoes nunca executadas, a FPR nos blocos classico e nativo e "
        "o identificador da dobra em cada linha de resultado.",
        "O pipeline cobre as partes que lhe cabem: FT-Transformer e TabBERT "
        "estao na grade; as tres visoes da IEEE-CIS entram e a ablacao fica "
        "pareada; `false_positive_rate` e gravada por todos os blocos; e "
        "`fold_index` (1-based) e `split` identificam cada linha. O capitulo "
        "de conclusoes continua sendo da dissertacao.",
        (
            "lr_nb_knn_specific.py: RESULT_CSV_COLUMNS",
            "experiment_grid.py: build_ft_transformer_grid, build_tabbert_grid",
            "teste: test_result_schema_records_seed_threshold_and_split_md5",
        ),
    ),
    ReviewItem(
        "G3",
        "Enquadramento incoerente com os experimentos",
        SEVERITY_CRITICAL,
        STATUS_PARTIAL,
        "Titulo e Resumo prometem deteccao de anomalias e metodos nao "
        "supervisionados, mas todos os experimentos sao de classificacao "
        "binaria supervisionada.",
        "O escopo foi completado, e nao reduzido. O bloco de deteccao de "
        "anomalias compara, sobre as mesmas particoes nativas do boosting e "
        "sem reamostragem, o Isolation Forest (nao supervisionado, "
        "hiperparametros de Liu et al.) e o DevNet (semissupervisionado, com "
        "1%, 5% e 10% das fraudes do treino rotuladas e cinco sorteios por "
        "fracao) aos "
        "classificadores supervisionados. As categorias sao codificadas pela "
        "frequencia no treino, sem rotulo, e os valores monetarios recebem "
        "log1p (G14). A Secao 3.1.1.4 descreve o bloco, e o Resumo e o "
        "Abstract passam a anunciar a comparacao entre os tres regimes de "
        "supervisao. As fracoes sao as proporcoes sugeridas pela revisao, e "
        "nao contagens absolutas: cada dobra rotula ceil(fracao x fraudes do "
        "treino), o que torna o eixo comparavel entre bases de tamanhos "
        "diferentes. A Figura 3.1 passa a incluir o CatBoost. Pendente: a "
        "execucao do bloco.",
        (
            "anomaly_detection_specific.py: execute_isolation_forest_fold, "
            "execute_devnet_fold",
            "devnet_model.py: DevNetSettings, train_devnet",
            "label_free_encoding.py: LabelFreeEncoder",
            "devnet_model.py: labeled_anomaly_count",
            "run_anomaly_detection_prepared_experiments.sh: LABEL_FRACTIONS, "
            "LABEL_DRAW_SEEDS",
            "testes: test_anomaly_detection.py, test_devnet.py, "
            "test_g14_monetary_log_transform.py",
            "dissertacao: Secao 3.1.1.4, Resumo e Abstract",
        ),
    ),
    ReviewItem(
        "G4",
        "Texto herdado do projeto de qualificacao",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "Varias passagens descrevem no futuro o que ja foi feito, e o futuro "
        "esconde decisoes que o leitor precisa conhecer.",
        "Nao depende de codigo. As decisoes que estavam em aberto na Secao "
        "3.1.1.3 (normalizacao, limiar de frequencia, hashing, numero de "
        "faixas) agora sao parametros explicitos da grade, entao o texto tem "
        "o que registrar.",
        ("experiment_grid.py: TRANSFORMER_NORMALIZATIONS, TRANSFORMER_CARDINALITY",),
    ),
    ReviewItem(
        "G5",
        "Resumo e Abstract sem resultados",
        SEVERITY_CRITICAL,
        STATUS_THESIS,
        "Os dois textos terminam com expectativas, sem nenhum resultado ou "
        "conclusao.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "G6",
        "Objetivos, contribuicoes e hipoteses vagos",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "Nao ha objetivo geral; as hipoteses sao ambiguas, sobrepostas e nunca "
        "retomadas no Capitulo 5.",
        "Nao depende de codigo. As comparacoes que o pipeline torna possiveis "
        "estao listadas na secao de comparacoes deste relatorio e podem "
        "sustentar questoes de pesquisa verificaveis.",
    ),
    # ---------------------------------------------------------------- G7-G11
    ReviewItem(
        "G7",
        "Home Credit: tarefa fora do dominio e vazamento por clientes replicados",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "A juncao replica cada cliente em ~4,9 linhas, com os mesmos atributos "
        "e o mesmo TARGET. Como as particoes sorteavam linhas, o mesmo cliente "
        "aparecia em treino e teste, e nao se dizia se SK_ID_CURR/SK_ID_PREV "
        "entravam como atributos.",
        "Toda particao da home-credit-completa e feita por SK_ID_CURR, com "
        "GroupShuffleSplit e StratifiedGroupKFold; SK_ID_CURR e SK_ID_PREV "
        "saem dos atributos apos a particao, servindo so para a juncao e o "
        "agrupamento. A base e declarada controle fora do dominio de fraude e "
        "nao deve ser misturada as medias das bases de fraude.",
        (
            "experiment_datasets.py: home-credit-completa (group_column, identifier_columns_to_drop)",
            "data_partitioning.py: _split_positions, _stratified_k_fold",
            "validacao D1 e D4 (--mode partition)",
            "teste: test_group_never_appears_in_two_partitions",
        ),
    ),
    ReviewItem(
        "G8",
        "Avaliacao que ignora o tempo e a identidade do titular",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "Sparkov e IEEE-CIS sao temporais, mas as particoes eram embaralhadas; "
        "a unica avaliacao temporal era acidental. No TabBERT, agrupar a "
        "IEEE-CIS por card1 nao produz sequencias de um titular, porque "
        "card1-card6 sao informacoes do cartao.",
        "Entrou o protocolo `temporal-holdout` explicito, que na Sparkov "
        "respeita a separacao oficial fraudTrain/fraudTest. Os protocolos "
        "embaralhados tambem respeitam o titular (cc_num na Sparkov, titular "
        "aproximado na IEEE-CIS). O titular aproximado da IEEE-CIS combina "
        "card1 + addr1, e nao card1 sozinho. O TabBERT passou a usar o mesmo "
        "temporal-holdout dos outros blocos, inclusive o teste oficial da "
        "Sparkov, em vez de um corte proprio dentro de cada titular.",
        (
            "data_partitioning.py: _temporal_holdout",
            "experiment_datasets.py: _build_approximate_holder_id",
            "transformer_sequence_windows.py: SEQUENCE_KEYS",
            "validacao D1 e D2 (--mode partition)",
            "teste: test_temporal_holdout_keeps_training_in_the_past",
        ),
    ),
    ReviewItem(
        "G9",
        "Grade com fatores inocuos, configuracoes duplicadas e nomes enganosos",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "ulb e eqm-ref-21 eram o mesmo arquivo; na ULB, 2 codificacoes "
        "x 4 imputacoes geravam 8 copias identicas; a reserva nao tinha efeito "
        "em duas politicas; stratified-train-test-split descartava de 20% a "
        "40% dos dados; e politica/reserva eram tratadas como preparacao.",
        "eqm-ref-21 e ulb-pca foram eliminadas. A grade e gerada por "
        "experiment_grid.py, que so varia um eixo onde o fenomeno existe. O "
        "eixo test_data_size nao existe mais. Restaram tres protocolos "
        "explicitos, nenhum descartando dados, e eles sao tratados como "
        "protocolo de avaliacao, nao como tecnica de preparacao.",
        (
            "experiment_grid.py: _encodings_for, _classic_fill_na_for",
            "prepared_dataset_storage.py: build_prepared_dataset_md5",
            "validacao P6 e P7 (--mode prepared)",
            "testes: test_grid_does_not_repeat_identical_configurations, "
            "test_eqm_ref_21_and_duplicated_views_are_gone",
        ),
    ),
    ReviewItem(
        "G10",
        "Balanceamento sem linha de base e com razoes diferentes entre as familias",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "Nao havia configuracao sem reamostragem, o que impedia medir o efeito "
        "de balancear; os classicos usavam RUS 1:2 + SMOTE e o boosting so RUS "
        "1:1, de modo que nenhuma estrategia era comum as duas familias.",
        "Entrou o braco `none` (sem reamostragem) nos blocos classico e de "
        "boosting, e `random-undersampling` 1:1 passou a existir nos dois, "
        "dando as familias uma estrategia comum. Entraram tambem os bracos de "
        "pesos de classe pedidos no item: `class_weight` na regressao "
        "logistica, `scale_pos_weight` no XGBoost e `auto_class_weights` no "
        "CatBoost, em duas intensidades -- sqrt-balanced (sqrt(R)) e balanced "
        "(R) --, com R = negativos/positivos medido no treino de cada dobra. "
        "Eles sao bracos do eixo de balanceamento, ao lado de none, RUS, "
        "RUS+SMOTE e Tomek Links, e por isso so rodam sobre as configuracoes "
        "sem reamostragem, onde R e a razao natural: num treino ja 1:1, R ~ 1 e "
        "o peso so repetiria o braco none. Medicao na ULB (R = 599): a "
        "ponderacao moderada vence em F1 (0,726 contra 0,689 e 0,100) e em F2 "
        "(0,770 contra 0,627 e 0,213); a completa leva a revocacao a 0,859 mas "
        "derruba a precisao de 0,824 para 0,053 -- medicao feita ainda com "
        "limiar fixo em 0,5. A reamostragem continua tocando apenas o treino.",
        (
            "experiment_grid.py: CLASSIC_BALANCING, BOOSTING_BALANCING, "
            "run_entries_for",
            "prepare_lr_nb_knn_datasets.py: reamostragem so em train_data",
            "lr_nb_knn_specific.py: CLASS_WEIGHT_INTENSITIES, "
            "validate_class_weight_arm, resolve_logistic_class_weight",
            "catboost_xgboost_specific.py: resolve_scale_pos_weight, "
            "resolve_auto_class_weights",
            "validacao P4 (--mode prepared)",
            "teste: test_grid_includes_a_no_resampling_baseline",
            "teste: test_weighted_arms_only_cover_configurations_without_resampling",
            "teste: test_weighting_is_rejected_on_resampled_configurations",
        ),
    ),
    ReviewItem(
        "G11",
        "Limiar de decisao fixo em 0,5 apos a reamostragem",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "Depois de reamostrar o treino para 1:1, as probabilidades deixam de "
        "corresponder a prevalencia real (0,17% a 8,7%) e o limiar de 0,5 vira "
        "uma escolha arbitraria que produz pontos de operacao extremos.",
        "Adotada a correcao (i): em todos os blocos e algoritmos o limiar e o "
        "que maximiza F2 sobre as previsoes de validacao, e e esse limiar que "
        "se aplica ao teste. Nos protocolos com teste (os dois holdouts e o "
        "corte cronologico do TabBERT) ele vem da validacao da propria dobra. "
        "No stratified-k-fold, em que a validacao e o proprio conjunto "
        "avaliado, o limiar de cada dobra vem da validacao das OUTRAS dobras, "
        "para que nenhuma metrica seja calculada com um limiar escolhido nos "
        "mesmos dados. O limiar e a forma como foi obtido sao gravados em cada "
        "linha (classification_threshold, threshold_strategy). Os pesos de "
        "classe nao resolvem este item: eles deslocam a probabilidade a priori "
        "como a reamostragem, e por isso tambem passam pelo limiar escolhido; "
        "sao bracos de balanceamento, registrados em G10. A parte (iii) esta "
        "em G11c, e a (iv), descrever o limiar na Secao 3.3, e da dissertacao.",
        (
            "decision_threshold.py: select_f2_threshold, select_fold_thresholds",
            "lr_nb_knn_specific.py: apply_decision_thresholds, "
            "RESULT_CSV_COLUMNS (threshold_strategy)",
            "catboost_xgboost_specific.py: execute_boosting_algorithm",
            "run_transformer_prepared_experiments.py: choose_fold_threshold",
            "teste: test_k_fold_threshold_never_comes_from_the_evaluated_fold",
            "teste: test_classic_execution_applies_the_selected_threshold",
            "teste: test_fixed_threshold_results_do_not_satisfy_a_validation_run",
        ),
    ),
    # --------------------------------------------------------------- G12-G18
    ReviewItem(
        "G12",
        "Melhor configuracao escolhida no teste; validacao sem uso",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "A melhor configuracao era escolhida pela metrica calculada na mesma "
        "particao de teste em que era relatada, e o conjunto de validacao "
        "nunca era usado para selecao nem para parada antecipada.",
        "Os tres protocolos produzem conjunto de validacao, e o gravador passou "
        "a emitir uma linha por (dobra, split) em vez de uma linha por dobra. "
        "Antes ele escolhia um unico split, com prioridade test > validation > "
        "train: como os tres protocolos sempre produzem teste, nenhuma linha "
        "`split=validation` chegava ao CSV e as metricas de validacao eram "
        "calculadas e descartadas na escrita. Nos blocos classico e de boosting "
        "isso nao custa calculo algum, porque as metricas dos tres splits ja "
        "estavam em `split_results`; nos Transformers a validacao passou a ser "
        "efetivamente avaliada, e nao apenas o teste. O dashboard pre-seleciona "
        "`split=test` ao carregar o arquivo, para que a leitura padrao nao some "
        "os splits na mesma media. A selecao em si, e a busca de "
        "hiperparametros, continuam sendo da fase de analise, fora do pipeline.",
        (
            "data_partitioning.py: _stratified_holdout, _stratified_k_fold, "
            "_temporal_holdout",
            "lr_nb_knn_specific.py: _iterate_execution_metrics",
            "run_transformer_prepared_experiments.py: EVALUATED_SPLITS",
            "tests/unit/test_revised_pipeline.py: "
            "test_every_evaluated_split_becomes_its_own_result_row",
        ),
    ),
    ReviewItem(
        "G13",
        "Ausencia de estimativas de incerteza e de testes estatisticos",
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "Cada configuracao tinha uma unica semente e uma unica particao, e os "
        "intervalos bootstrap mediam a heterogeneidade da grade, nao a "
        "incerteza de estimacao.",
        "`random_seed` entrou na identidade da configuracao e na linha de "
        "resultado, e `SEED_LIST` repete a grade inteira com varias sementes. "
        "O protocolo `stratified-k-fold` fornece cinco dobras da mesma "
        "configuracao, que e a dispersao entre particoes que faltava. A "
        "semente chega tambem a particao por cliente do TabBERT na Home "
        "Credit, que antes fixava random_state=42 e repetia os mesmos clientes "
        "a cada semente. Os protocolos temporais sao deterministicos: ali, "
        "sementes diferentes variam so o modelo. Os testes estatisticos "
        "(Wilcoxon, Friedman) sao da fase de analise.",
        (
            "prepared_dataset_storage.py: build_prepared_dataset_md5 (random_seed)",
            "experiment_grid.py: _configured_seeds",
            "transformer_sequence_windows.py: _split_chronologically_group_exclusive",
            "teste: test_configuration_identity_has_no_test_data_size_and_has_the_seed",
            "teste: test_tabbert_group_exclusive_split_follows_the_seed",
        ),
    ),
    ReviewItem(
        "G14",
        "Pre-processamento: escalas, atributos identificadores e resultados degenerados",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "A padronizacao so era aplicada antes da PCA, deixando KNN, SMOTE, "
        "Tomek Links, regressao logistica e Naive Bayes sem escalonamento. "
        "Identificadores como cc_num e trans_num entravam como atributos, e as "
        "marcas de tempo absolutas permitiam explorar a proximidade temporal "
        "entre treino e teste.",
        "A padronizacao passou a valer para TODA configuracao classica, "
        "ajustada no treino de cada particao. Identificadores e dados pessoais "
        "saem dos atributos em todas as bases. As marcas de tempo absolutas "
        "deixam de ser atributos e viram atributos relativos (hora do dia, dia "
        "da semana, idade, distancia cliente-estabelecimento).",
        (
            "lr_nb_knn_specific.py: standardize_data_splits",
            "experiment_datasets.py: identifier_columns_to_drop, "
            "_add_haversine_distance",
            "validacao P3 (--mode prepared)",
            "testes: test_standardization_is_fitted_on_training_only, "
            "test_identifier_columns_leave_the_attributes",
        ),
    ),
    ReviewItem(
        "G15",
        "Padroes de resultado que sugerem erro no pipeline",
        SEVERITY_CRITICAL,
        STATUS_PARTIAL,
        "Imputacoes por media, mediana e moda produziam resultados identicos "
        "ate a terceira casa decimal, e OTE e WoE coincidiam na ieee-cis-completa. "
        "Os sintomas apontavam para arquivos reaproveitados ou para um eixo "
        "sem efeito.",
        "Causa confirmada nos resultados de 22/09 (pares identicos bit a bit em "
        "100% dos casos na eqm-ref-19 e na eqm-ref-22): uma PCA dentro de "
        "load_data, antes da divisao, descartava as categoricas e completava "
        "os faltantes com a mediana global. A carga ficou estrutural e a PCA "
        "passou a ser por particao. A PCA e os executores classicos agora "
        "FALHAM diante de colunas nao numericas ou faltantes, em vez de "
        "corrigi-los em silencio. O drop-incomplete-rows remove antes as "
        "colunas acima do limiar escolhido no treino (maior limiar que preserva "
        "75% das fraudes), registrado no manifesto. Cada dobra grava um "
        "relatorio de preparacao com faltantes, linhas, fraudes e componentes "
        "da PCA. O verificador testa se cada eixo muda arquivos e resultados "
        "(P11, P12, R2) e gera as tabelas do apendice; rodado sobre os CSVs de "
        "22/09, reproduz o defeito. Pendente: reexecucao e apendice da "
        "dissertacao.",
        (
            "lr_nb_knn_specific.py: require_numeric_complete_features, _apply_pca",
            "missing_screening.py: select_missing_screening",
            "preparation_report.py: relatorio por dobra",
            "testes: test_g15_screening_and_report.py",
            "preparation_sanity_checks.py: P10-P15 e R1-R4 "
            "(validate_prepared_datasets.py --mode prepared|results)",
            "testes: test_g15_verifier.py",
        ),
    ),
    ReviewItem(
        "G16",
        'Remocao de registros como estrategia de "preenchimento"',
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "A remocao de casos incompletos era aplicada a TODAS as particoes, "
        "inclusive ao teste, mudando a populacao avaliada de uma estrategia "
        "para outra. Alem disso, `none` significava remocao no bloco classico "
        "e tratamento nativo no bloco de boosting.",
        "A estrategia passou a se chamar `drop-incomplete-rows` e remove "
        "linhas apenas do TREINO; validacao e teste mantem a populacao "
        "completa e recebem a estatistica calculada no treino. Antes da "
        "remocao, as colunas com faltantes acima do limiar escolhido no treino "
        "saem de todos os splits (missing_screening.py, item G15c). Para que as "
        "particoes sejam de fato as mesmas em todos os blocos, cada split "
        "recebe um partition_md5 -- o MD5 do conjunto de linhas brutas que ele "
        "contem, calculado antes de qualquer transformacao --, gravado no "
        "manifesto, e cada linha de resultado recebe o partition_id da dobra. "
        "A validacao P9 recusa base, protocolo, semente e dobra com particoes "
        "diferentes entre blocos. O TabBERT passou a usar os mesmos protocolos "
        "e a mesma semente dos demais, e entra no pareamento nos holdouts. "
        "Faltava o pareamento das ablacoes. Medido nos dados brutos (semente "
        "42), a ieee-cis-sem-identidade tinha as mesmas linhas da completa, mas "
        "outro partition_md5. Na remocao, escolhia outro limiar (45% contra 10% "
        "no holdout estratificado e 35% contra 5% no temporal) e treinava com "
        "outras linhas: 1.987 a mais no estratificado; no temporal, 21.632 so na "
        "completa e 1.852 so na sem-identidade. A sem-identidade passou a "
        "ser variante de colunas da completa, e a remocao e decidida com as "
        "colunas da base de origem (missing_screening.decide_fold_removal); o "
        "treino efetivo ganhou identidade propria (train_population_md5, no "
        "relatorio e no manifesto; train_population_id no resultado). O par "
        "da Home Credit tambem comparava clientes diferentes: juncao interna "
        "(16.454 clientes a menos, com inadimplencia de 5,96% contra 8,19%) e "
        "particao por linha na somente-aplicacao. A completa usa juncao a "
        "esquerda, as duas agrupam por SK_ID_CURR, o k-fold sorteia a tabela de "
        "clientes, e cada split ganhou unit_partition_md5 (unit_partition_id no "
        "resultado). No braco de remocao, a somente-aplicacao adota a decisao "
        "da completa e treina com os mesmos clientes (train_unit_population_md5; "
        "train_unit_population_id no resultado): na semente 42, holdout "
        "estratificado, 162.437 clientes nas duas, contra 170.923 se decidisse "
        "sozinha. No bloco nativo, o tratamento interno passou a se chamar "
        "`native`; `none` ficou so para as bases sem faltantes. P9 agrupa pela "
        "base de origem e pelos pares por unidade, P16 confere o treino efetivo "
        "e R5 o pareamento nos resultados, inclusive os clientes do treino no "
        "par da Home Credit; o apendice ganhou a tabela de pareamento "
        "(appendix_pairing.csv). Pendente: reexecucao com os dados "
        "regerados.",
        (
            "lr_nb_knn_specific.py: DROP_INCOMPLETE_ROWS_STRATEGY, "
            "_impute_data_splits",
            "data_partitioning.py: row_fingerprints, partition_md5, "
            "fold_partition_id",
            "validate_prepared_datasets.py: validate_partition_pairing (P9)",
            "teste: test_dropping_incomplete_rows_touches_training_only",
            "teste: test_threshold_is_chosen_on_training_data_only",
            "teste: test_same_protocol_and_seed_give_the_same_partition_id",
            "teste: test_tabbert_uses_a_protocol_shared_with_the_other_blocks",
            "experiment_datasets.py: ieee-cis-sem-identidade como variante, "
            "juncao a esquerda na home-credit-completa, paired_dataset_id",
            "missing_screening.py: decide_fold_removal, adopt_paired_removal",
            "preparation_report.py: train_population_md5, "
            "train_unit_population_md5",
            "data_partitioning.py: unit_partition_md5, k-fold pela tabela de grupos",
            "preparation_sanity_checks.py: check_training_population (P16), "
            "check_result_pairing (R5), pairing_table (apendice)",
            "teste: tests/unit/test_g16_pairing.py",
        ),
    ),
    ReviewItem(
        "G17",
        "Contagens de execucoes e de pares inconsistentes com o desenho",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "O desenho previa 3.840 unidades e o texto informava 3.824, sem "
        "explicar as 40 falhas nem as 96 linhas que faltavam.",
        "A grade e deterministica e contavel antes de rodar: "
        "`experiment_grid.py --format summary` imprime configuracoes e "
        "execucoes por base e por algoritmo, em todos os blocos. Cada falha "
        "grava `failure_category` e `failure_message`, e cada comando dos "
        "executores vira uma linha de `registro_execucoes.csv`, com o codigo de "
        "saida, o que explica tambem as execucoes que nao gravaram resultado. O "
        "R4 conta previstas, sucesso, falha, incompleta e ausente, em execucoes "
        "e em linhas de teste, por bloco, base, algoritmo e eixo. As "
        "comparacoes pareadas descartam e contam os pares incompletos, em vez "
        "de tirar a media de configuracoes distintas.",
        (
            "experiment_grid.py: algorithm_runs_for, result_rows_for, anomaly_arms",
            "execution_failures.py: classify_failure, check_fold_partitions",
            "execution_log_common.sh: record_execution_attempt",
            "preparation_sanity_checks.py: expected_runs, check_run_counts (R4, "
            "previstas x sucesso x falha x incompleta x ausente, com motivo)",
            "gerar_discussao_resultados.py: paired_comparison, "
            "pares_por_comparacao.csv",
        ),
    ),
    ReviewItem(
        "G18",
        "Reprodutibilidade e recursos computacionais",
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "Hiperparametros so apareciam na discussao de uma base; versoes de "
        "biblioteca, repositorio e tempos nao eram informados.",
        "Cada linha de resultado grava o commit, o hash do ambiente de "
        "referencia (requirements.lock.txt, conferido sem bloquear) e a maquina "
        "(CPU, RAM, GPU, sistema). As dobras gravam fit_time e scoring_time, "
        "a preparacao grava o tempo de cada etapa, e um benchmark rodado no "
        "Acer mede ms por 1.000 transacoes e a latencia unitaria (p50, p95 e "
        "IC do p95, amostra de 1.000 justificada pela precisao do p95). As "
        "tabelas de hiperparametros efetivos e de tempos sao geradas dos "
        "resultados. A grade e dividida entre Acer, E470 e Dell por "
        "calibracao e plano por (algoritmo, base), com a GPU no Acer, copia "
        "conferida por MD5 e juncao estrita. As bases e o codigo da tag vao "
        "para o repositorio publico experimento-fraudes. Pendente: "
        "adaptador dos Transformers no benchmark, reexecucao, tag e texto.",
        (
            "execution_provenance.py: collect_provenance, enforce_locked_environment",
            "requirements.lock.txt",
            "preparation_timing.py",
            "benchmark_inference.py, run_inference_benchmark.sh",
            "build_reproducibility_tables.py",
            "calibrate_machine.py, grid_plan.py",
            "stage_machine_configurations.sh, collect_machine_results.sh, "
            "merge_machine_results.py",
            "verify_datasets.py, ../datasets/CHECKSUMS.md5",
            "../../scripts/export_public_repository.sh, ../../REPRODUCAO.md",
            "testes: test_g18_*.py",
        ),
    ),
    # --------------------------------------------------------------- G19-G23
    ReviewItem(
        "G19",
        "Organizacao dos Capitulos 3 a 5 e redundancias",
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "O Capitulo 4 quase nao traz resultados e descreve engenharia de "
        "software que pertence a metodologia ou a um apendice; o Capitulo 5 "
        "tem figuras redundantes.",
        "O script da discussao deixou de gerar as figuras da comparacao com "
        "Itoo et al. (artigo retratado) e as da PCA na ULB, base que saiu do "
        "catalogo. A reorganizacao dos capitulos e da dissertacao.",
        ("gerar_discussao_resultados.py", "tests/unit/test_g19_discussion_figures.py"),
    ),
    ReviewItem(
        "G20",
        'Uso improprio de "generalizacao"',
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        'O termo era usado no sentido de "vence em varias bases", mas cada '
        "modelo e treinado e avaliado dentro da mesma base.",
        "O texto troca a generalizacao entre bases por robustez entre fontes "
        "(H2) e reserva o termo ao protocolo temporal (Q1). "
        "analise_hipoteses.py produz a evidencia: selecao na validacao, "
        "ranking conjunto nas tres fontes de fraude, Friedman e Nemenyi, "
        "desfechos de H1-H3 e a diferenca temporal - estratificado. Os "
        "executores gravam os escores do teste por linha "
        "(resultados_*_test_scores.jsonl), e o IC e um bootstrap pareado sobre "
        "as transacoes; sem eles, e pareado por semente ou, com semente unica, "
        "conservador. Falta a reexecucao da grade.",
        (
            "analise_hipoteses.py",
            "data_partitioning.py: _temporal_holdout",
            "lr_nb_knn_specific.py: record_algorithm_result_to_csv, "
            "encode_test_scores",
            "tests/unit/test_g20_hypotheses_analysis.py",
            "tests/unit/test_g20_test_scores.py",
        ),
    ),
    ReviewItem(
        "G21",
        "Revisao bibliografica: foco, lacunas e imprecisoes",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "A taxonomia de Pang et al. e descrita incorretamente e faltam "
        "referencias classicas de fraude, desbalanceamento e vazamento.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "G22",
        "Terminologia inconsistente",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "O mesmo conceito aparece com nomes diferentes ao longo do texto.",
        "Nao depende de codigo. Os nomes usados pelo pipeline foram "
        "padronizados e podem servir de referencia ao texto: `roc_auc`, "
        "`f_beta_2`, `false_positive_rate`, `random_seed`, "
        "`classification_threshold`.",
        ("lr_nb_knn_specific.py: RESULT_CSV_COLUMNS",),
    ),
    ReviewItem(
        "G23",
        "Padroes recorrentes de redacao",
        SEVERITY_MINOR,
        STATUS_THESIS,
        'Primeira pessoa, "onde" sem sentido de lugar, enclise indevida, '
        "gerundismo e decalques do ingles.",
        "Nao depende de codigo.",
    ),
    # ----------------------------------------------------------------- F1-F11
    ReviewItem(
        "F1",
        "Efeitos colaterais do destaque azul",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "A caixa colorida corta texto na margem e desalinha marcadores.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F2",
        "Formato Unicamp/IC e numeracao das paginas",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "Papel Carta em vez de A4 e numeracao romana reiniciada.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F3",
        "Sumario: titulo e pagina da bibliografia",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "Entradas e paginas erradas no sumario.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F4",
        "Caixas coloridas dos hyperlinks",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "As bordas do hyperref aparecem na impressao.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F5",
        "Equacoes: simbolos errados e notacao inconsistente",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "Funcao indicadora errada na Equacao 3.1, algarismos em estilo antigo "
        "e operadores em italico.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F6",
        "Tabelas",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "Legendas abaixo, separador decimal inconsistente e zero negativo.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F7",
        "Figuras",
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "Texto ilegivel, rotulos em duas linguas e fluxogramas que nao "
        "correspondem a logica descrita: a Figura 4.1 verificava a "
        "configuracao so pelo configuration_md5, sem o algoritmo, e a Figura "
        "3.1 mostrava uma unica preparacao, anterior ao particionamento.",
        "O fluxo do pipeline mudou e as figuras precisam ser refeitas a partir "
        "dele: dados -> particionamento -> preparacao ajustada so no treino -> "
        "algoritmos -> metricas. A chave de 'ja executado' agora inclui "
        "algoritmo e limiar, nao so o configuration_md5.",
        (
            "lr_nb_knn_specific.py: is_experiment_already_recorded",
            "prepare_lr_nb_knn_datasets.py: ordem das operacoes",
        ),
    ),
    ReviewItem(
        "F8",
        "Codigo e identificadores no texto",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "Aspas tipograficas dentro de \\texttt e nomes de scripts quebrados.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F9",
        "Pre-textuais, titulos das secoes e legendas longas",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "Titulos sem padrao de maiusculas e duas secoes com o mesmo titulo.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F10",
        "Referencias bibliograficas",
        SEVERITY_IMPORTANT,
        STATUS_THESIS,
        "Autor institucional lido como pessoa, versoes arXiv de trabalhos "
        "publicados, DOIs ausentes e metodos descritos sem a referencia "
        "original.",
        "Nao depende de codigo.",
    ),
    ReviewItem(
        "F11",
        "Siglas e simbolos",
        SEVERITY_MINOR,
        STATUS_THESIS,
        "Siglas nunca expandidas e um mesmo simbolo designando objetos "
        "diferentes.",
        "Nao depende de codigo.",
    ),
    # --------------------------------------------------- itens C*.* de codigo
    ReviewItem(
        "C3.6",
        "Codificacao da validacao e do teste (OTE x WoE)",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "A comparacao OTE x WoE misturava dois fatores: a estatistica (media "
        "suavizada x log-razao) e o controle de vazamento (prefixo de uma "
        "permutacao x cinco dobras).",
        "O eixo unico foi substituido por um desenho fatorial 2x2: a "
        "estatistica (media suavizada x log-razao) e o mecanismo de controle "
        "de vazamento (prefixo de permutacao x cross-fitting de cinco dobras) "
        "passaram a ser coordenadas independentes, e as quatro celulas estao "
        "na grade. O mecanismo foi reimplementado uma unica vez, por fora do "
        "codificador, de modo que seja literalmente o mesmo codigo nas duas "
        "estatisticas. A medicao justifica o desenho: o mecanismo de prefixo "
        "entrega ao modelo codificacoes de treino com ruido/sinal de 0,32 e "
        "0,30 (media suavizada e log-razao), contra 0,05 e 0,06 do "
        "cross-fitting -- um efeito cinco vezes maior que o da estatistica, "
        "que antes seria lido como diferenca entre OTE e WoE.",
        (
            "lr_nb_knn_specific.py: _encode_training_rows_with_leakage_control",
            "lr_nb_knn_specific.py: FACTORIAL_CATEGORICAL_ENCODINGS",
            "experiment_grid.py: CLASSIC_ENCODINGS",
            "tests/unit/test_revised_pipeline.py: "
            "test_encoding_axis_is_a_complete_two_by_two_factorial",
        ),
    ),
    ReviewItem(
        "C3.10",
        "Padronizacao e PCA",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "A padronizacao so era aplicada nas configuracoes que recebiam PCA, e "
        "a presenca de PCA estava confundida com a base.",
        "A padronizacao e independente da PCA e vale para toda configuracao "
        "classica. A PCA virou propriedade fixa de cada base -- roda em todas "
        "menos na ULB, cujos V1-V28 ja sao componentes principais publicados "
        "-- e e aplicada EXCLUSIVAMENTE no bloco classico, ajustada no treino de "
        "cada particao. Boosting e Transformers nunca a recebem: ela eliminaria "
        "a semantica individual dos atributos, impediria o tratamento nativo das "
        "categoricas e alteraria a representacao por campo dos Transformers.",
        (
            "lr_nb_knn_specific.py: standardize_data_splits",
            "experiment_datasets.py: applies_pca_in_classic_block",
            "teste: test_pca_runs_everywhere_except_the_ulb",
        ),
    ),
    ReviewItem(
        "C3.14",
        "Grade nativa: razao do RUS, pesos de classe e hiperparametros",
        SEVERITY_IMPORTANT,
        STATUS_PARTIAL,
        "O RUS nativo usava 1:1 e o classico 0,5; faltava a ponderacao de "
        "classes usual em boosting (scale_pos_weight, auto_class_weights); os "
        "hiperparametros eram os padroes da biblioteca e nao estavam listados.",
        "A razao do RUS e a mesma nas duas familias (G10). CatBoost e XGBoost "
        "ganharam os bracos sem reamostragem e de pesos de classe "
        "(sqrt-balanced e balanced), estes so sobre as configuracoes sem "
        "reamostragem. O rotulo none do eixo de faltantes e o eixo que nao se "
        "aplica a ULB estao em G16 e G9b. Falta listar os hiperparametros "
        "efetivos e decidir sobre parada antecipada (G12, G18).",
        (
            "experiment_grid.py: BOOSTING_BALANCING, run_entries_for",
            "catboost_xgboost_specific.py: create_catboost_classifier, "
            "create_xgboost_classifier",
            "run_catboost_xgboost_native_prepared_experiments.sh",
        ),
    ),
    ReviewItem(
        "C3.26",
        "Home Credit: as sete colunas 'redundantes'",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "Sete colunas de previous_application homonimas das de "
        "application_data eram descartadas como redundantes, mas descrevem o "
        "pedido anterior, e nao o atual.",
        "Elas deixaram de ser descartadas e entram com prefixo PREV_.",
        (
            "experiment_datasets.py: PREVIOUS_APPLICATION_OVERLAPPING_COLUMNS, "
            "_load_home_credit",
        ),
    ),
    ReviewItem(
        "C3.27",
        "Sparkov: base sintetica e colunas identificadoras",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "Nao se dizia quais colunas foram removidas; cc_num, trans_num, nomes "
        "e endereco entravam como atributos.",
        "Saem dos atributos: indice sem nome, cc_num, trans_num, first, last, "
        "street, dob, unix_time e a data bruta. Entram hora, dia da semana, fim "
        "de semana, mes, dia, idade na data da transacao e distancia "
        "cliente-estabelecimento. As quatro coordenadas PERMANECEM: a "
        "localizacao de uma transacao e informacao legitima para identificar "
        "fraude. O carater sintetico da base esta registrado no catalogo.",
        (
            "experiment_datasets.py: _load_sparkov, sparkov",
            "validacao P3 (--mode prepared)",
        ),
    ),
    ReviewItem(
        "C3.28",
        "IEEE-CIS: TransactionID e ordem temporal",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "TransactionID deveria ser excluido dos atributos e TransactionDT "
        "considerado na particao.",
        "Os dois saem dos atributos; de TransactionDT ficam hora do dia e dia "
        "da semana, e ele passa a ordenar o protocolo temporal.",
        ("experiment_datasets.py: ieee-cis-completa*, _derive_temporal_features",),
    ),
    ReviewItem(
        "C3.22",
        "Ablacao IEEE-CIS: hipotese e margens contraditorias",
        SEVERITY_CRITICAL,
        STATUS_PARTIAL,
        "A hipotese tinha direcoes opostas em duas secoes, as margens eram "
        "diferentes e a regra de decisao mudava. Alem disso, com remocao de "
        "registros o par com/sem identidade deixava de comparar a mesma "
        "populacao.",
        "O pipeline garante o pareamento: as tres visoes existem, e como a "
        "remocao de incompletos agora vale so para o treino, o conjunto de "
        "teste permanece identico entre as visoes com e sem identidade. A "
        "hipotese unica e as margens sao da dissertacao.",
        (
            "experiment_datasets.py: ieee-cis-completa, -noidentity, -transaction",
            "lr_nb_knn_specific.py: DROP_INCOMPLETE_ROWS_STRATEGY",
        ),
    ),
    ReviewItem(
        "C4.6",
        "Execucao dos Transformers: validacao usada como teste",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "O executor avaliava na particao de teste 'ou na de validacao quando a "
        "politica nao produz conjunto de teste', misturando os protocolos.",
        "Os tres protocolos produzem validacao, e os dois de holdout produzem "
        "tambem teste. O FT-Transformer usa os mesmos protocolos das demais "
        "familias, de modo que suas metricas sao pareaveis sobre as mesmas "
        "particoes.",
        ("prepare_lr_nb_knn_datasets.py: _build_transformer_partition",),
    ),
    ReviewItem(
        "C4.7",
        "Limitacoes do registro: FPR, dobra e acuracia",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "A FPR nao era gravada nos blocos classico e nativo, a dobra nao era "
        "identificada e o bloco dos Transformers nao gravava acuracia.",
        "O esquema canonico de resultado grava `false_positive_rate`, "
        "`accuracy`, `fold_index` (1-based), `split`, `random_seed`, "
        "`classification_threshold` e os MD5 de conteudo de treino, validacao "
        "e teste, para todos os blocos.",
        (
            "lr_nb_knn_specific.py: RESULT_CSV_COLUMNS",
            "teste: test_fold_index_starts_at_one",
        ),
    ),
    ReviewItem(
        "C5.26",
        "Descarte da reserva amostral",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "stratified-train-test-split descartava de 20% a 40% dos dados sem "
        "justificativa e, na Sparkov, o fraudTest.csv inteiro.",
        "A politica foi eliminada, junto com o eixo test_data_size. Nenhum dos "
        "tres protocolos descarta dados.",
        (
            "data_partitioning.py: docstring do modulo",
            "teste: test_configuration_identity_has_no_test_data_size_and_has_the_seed",
        ),
    ),
    ReviewItem(
        "C5.31",
        "PCA confundida com a padronizacao e com a base",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "O contraste ulb x ulb-pca nao isolava a PCA, porque a "
        "padronizacao so era aplicada nas configuracoes com PCA.",
        "O contraste deixou de existir junto com ulb-pca. A "
        "padronizacao e universal e a PCA e propriedade da base, de modo que "
        "nenhuma comparacao mistura os dois efeitos.",
        ("experiment_datasets.py: applies_pca_in_classic_block", "lr_nb_knn_specific.py: standardize_data_splits"),
    ),
    # --------------------------------------------------- achados deste trabalho
    ReviewItem(
        "C3.4",
        "Regressao linear como classificador",
        SEVERITY_MINOR,
        STATUS_RESOLVED,
        "A regressao linear entrava como classificador sem que se dissessem "
        "suas limitacoes: e um modelo de probabilidade linear, cuja saida nao e "
        "uma probabilidade calibrada e pode cair fora de [0, 1].",
        "Removida da grade. O limiar de 0,5 nao tem interpretacao sobre uma "
        "saida nao calibrada, e mantida ela so acrescentava uma linha de base "
        "sem significado decisorio.",
        (
            "experiment_grid.py: CLASSIC_ALGORITHMS",
            "teste: test_only_weightable_algorithms_sweep_class_weight_intensities",
        ),
    ),
    ReviewItem(
        "G7b",
        "Home Credit sem a juncao, como par de controle",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "A juncao com previous_application replicava cada cliente em ~4,9 "
        "linhas e descartava os 16.454 clientes sem pedidos anteriores. Nao "
        "havia como separar o que o historico acrescentava do que a replicacao "
        "inflava.",
        "Entrou home-credit-somente-aplicacao, a mesma base lida apenas de "
        "application_data.csv: 307.511 pedidos, um por cliente. Pareada com "
        "home-credit-completa, isola o efeito da juncao. Como nenhum cliente se repete, a "
        "particao aleatoria ja e segura por construcao.",
        (
            "experiment_datasets.py: _load_home_credit_application_only",
            "teste: test_application_only_home_credit_has_one_row_per_client",
        ),
    ),
    ReviewItem(
        "G11c",
        "AUC-PR entre as metricas",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "O item G11-iii pede metricas independentes de limiar e adequadas ao "
        "desbalanceamento. So a AUC-ROC era gravada, e ela e otimista sob forte "
        "desbalanceamento: sua taxa de falsos positivos tem como denominador o "
        "numero de negativos, enorme nestas bases, de modo que um aumento "
        "grande em termos absolutos de falsos alarmes mal desloca a curva.",
        "A AUC-PR (precisao media) e gravada por todos os blocos, ao lado da "
        "AUC-ROC. Sem exemplos positivos ela devolve NaN, e nao o 0,0 do "
        "scikit-learn, que se confundiria com desempenho pessimo. A diferenca "
        "entre as duas ja aparece na primeira medicao: na ULB, a AUC-ROC diz "
        "que a ponderacao moderada e a melhor (0,971 contra 0,961), enquanto a "
        "AUC-PR diz que e a ausencia de ponderacao (0,686 contra 0,675). "
        "ATENCAO: a AUC-PR depende da prevalencia -- sua linha de base e a "
        "propria prevalencia, e nao 0,5 --, entao ela compara algoritmos "
        "DENTRO de uma base, nunca entre bases.",
        (
            "lr_nb_knn_specific.py: _average_precision, RESULT_CSV_COLUMNS",
            "run_transformer_prepared_experiments.py: metrics",
            "experiment_results_dashboard/app/constants.py: OPTIONAL_METRICS",
            "teste: test_average_precision_matches_the_prevalence_baseline",
            "teste: test_average_precision_is_undefined_without_positives",
        ),
    ),
    ReviewItem(
        "N1",
        "Registros exatamente duplicados na ULB (achado novo)",
        SEVERITY_CRITICAL,
        STATUS_RESOLVED,
        "Nao apontado pela revisao. A base da ULB traz 1.081 linhas identicas "
        "em todas as 31 colunas. Com particao aleatoria, copias do mesmo "
        "registro caiam em treino e teste, e o modelo acertava por "
        "memorizacao; o KNN e o caso extremo, porque o vizinho mais proximo de "
        "uma linha de teste passa a ser a sua propria copia no treino.",
        "Registros exatamente repetidos sao removidos na carga, antes de "
        "qualquer particao. Nenhuma transacao distinta e descartada: linhas "
        "que ficam identicas apenas depois do descarte da marca de tempo sao "
        "transacoes diferentes e permanecem na base.",
        (
            "experiment_datasets.py: drop_exact_duplicate_records",
            "validacao P2 (--mode prepared)",
            "teste: test_exact_duplicate_records_are_removed_before_partitioning",
        ),
    ),
    ReviewItem(
        "N2",
        "Integridade dos arquivos preparados (achado novo)",
        SEVERITY_IMPORTANT,
        STATUS_RESOLVED,
        "Nao apontado pela revisao. Nada garantia que o arquivo lido na "
        "execucao fosse o mesmo que a preparacao gravou: uma reescrita "
        "parcial, uma falha do HD externo ou uma preparacao interrompida "
        "produziriam metricas atribuidas a dados errados.",
        "Cada arquivo carrega um MD5 de conteudo no manifesto, e a carga "
        "recusa a execucao se os bytes divergirem. Os MD5 de treino, "
        "validacao e teste vao para a linha de resultado, de modo que uma "
        "metrica sempre aponta para os dados exatos que a produziram.",
        (
            "prepared_dataset_storage.py: compute_file_content_md5, "
            "load_prepared_data_splits",
            "validacao P1 (--mode prepared)",
            "teste: test_each_saved_file_carries_its_content_md5",
        ),
    ),
)


def items_by_status() -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = {}
    for item in REVIEW_ITEMS:
        grouped.setdefault(item.status, []).append(item)
    return grouped


def status_counts() -> dict[str, int]:
    return {status: len(items) for status, items in items_by_status().items()}
