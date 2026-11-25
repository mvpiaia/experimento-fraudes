# Detecção de fraudes em transações financeiras: código e bases

Código e bases do pipeline experimental da dissertação de mestrado *Detecção de Fraudes em Transações Financeiras: Avaliação Comparativa de Métodos com Diferentes Níveis de Supervisão em Múltiplas Bases de Dados*, de Marcos Vinícius Piaia (Instituto de Computação, Unicamp). A versão exata usada no texto está no arquivo `VERSION`: a tag e o commit do repositório de origem. Cada linha de resultado também grava o commit (`code_commit`), o ambiente (`environment_md5`) e a máquina que a produziu.

## Bases

As bases originais estão em `src/datasets/`, nos arquivos `.zip` como baixados do Kaggle, versionados em Git LFS. Depois de clonar, rode `git lfs pull` e extraia cada `.zip` na própria pasta. O MD5 de cada arquivo usado está em `src/datasets/CHECKSUMS.md5`, e a preparação confere esses valores antes de começar.

| Base | Arquivo compactado | Arquivos usados | Fonte | Referência na dissertação |
|---|---|---|---|---|
| ULB | `ref-18/fraud-detection-ref-18.zip` | `creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `kaggle-eur-credita-cartd-archive` |
| Home Credit | `ref-19/credit-card-fraud-detection-ref-19.zip` | `application_data.csv`, `previous_application.csv` | https://www.kaggle.com/datasets/mishra5001/credit-card (dados da competição https://www.kaggle.com/competitions/home-credit-default-risk) | `mishra-credita-card`, `home-credit-default-risk` |
| Sparkov | `ref-20/credit-card-transaction-ref-20.zip` | `fraudTrain.csv`, `fraudTest.csv` | https://www.kaggle.com/datasets/kartik2112/fraud-detection | `kaggle-credit-card-2019-2020-archive` |
| IEEE-CIS | `ref-22/ieee-fraud-detection-ref-22.zip` | `train_transaction.csv`, `train_identity.csv` | https://www.kaggle.com/competitions/ieee-fraud-detection/data | `ieee-fraud-detection` |

```bash
git lfs pull
cd src/datasets
for zip in */*.zip; do unzip -n "$zip" -d "$(dirname "$zip")"; done
cd ../..
python3 src/fraud_detection_revised_pipeline/verify_datasets.py
```

Os `.zip` trazem também arquivos que a grade não usa (por exemplo, o conjunto de teste sem rótulos da IEEE-CIS); eles ficam como na distribuição original.

## Ambiente

O ambiente de referência usa as versões de `src/fraud_detection_revised_pipeline/requirements.lock.txt`. Diferenças de pacotes ou da versão do Python não bloqueiam os scripts; pacotes divergentes fazem as linhas gravarem `environment_locked=false`. `ALLOW_UNLOCKED_ENVIRONMENT` é mantida apenas por compatibilidade e seu comportamento efetivo é sempre permissivo.

```bash
python3 -m venv .venv
.venv/bin/pip install -r src/fraud_detection_revised_pipeline/requirements.lock.txt
.venv/bin/python src/fraud_detection_revised_pipeline/execution_provenance.py --check
.venv/bin/python src/fraud_detection_revised_pipeline/execution_provenance.py --show
```

A roda padrão do PyTorch no Linux já inclui CUDA. Os Transformers precisam de GPU; os demais blocos rodam em CPU.

## Diretório de dados preparados

Copie `src/fraud_detection_revised_pipeline/machine_storage.conf.example` para `machine_storage.conf` e defina `MACHINE_PREPARED_DATA_DIRECTORY`. O exemplo traz a configuração de cada máquina da dissertação.

Quando perdem o acesso ao diretório, a preparação e os runners não desistem:
- verificam o diretório a cada 2 minutos (`STORAGE_RETRY_SECONDS`), sem limite de tentativas (`STORAGE_RETRY_ATTEMPTS=0`);
- imprimem no terminal cada tentativa que falha;
- quando o acesso volta, avisam e executam de novo, desde o início, o passo que foi interrompido.

Um passo interrompido assim não grava linha de falha. Já um arquivo preparado ausente, com o diretório acessível, é repetido uma vez e depois registrado como falha.

Diretórios no sistema de arquivos raiz são recusados, para que um HD desmontado não faça a gravação cair no disco do sistema. Quem reproduz o trabalho sem HD libera isso com `MACHINE_STORAGE_ALLOW_ROOT_FILESYSTEM=true`.

Nos comandos abaixo, `$DADOS` é esse diretório.

## Preparação

```bash
cd src/fraud_detection_revised_pipeline
./prepare_lr_nb_knn_specific_experiment_datasets.sh "$DADOS"       # bloco clássico
./prepare_catboost_xgboost_native_experiment_datasets.sh "$DADOS"  # boosting e detecção de anomalias
./prepare_ft_transformer_experiment_datasets.sh "$DADOS"
./prepare_tabbert_experiment_datasets.sh "$DADOS"
python3 validate_prepared_datasets.py --mode prepared --prepared-data-directory "$DADOS"
```

Para encurtar a preparação, ela pode ser dividida em partes que rodam em paralelo na mesma máquina, cada uma com um manifesto próprio. As partes depois são juntadas no manifesto principal:

```bash
./prepare_partition.sh "$DADOS" "$DADOS/parte1.tsv" classic home-credit-completa &
./prepare_partition.sh "$DADOS" "$DADOS/parte2.tsv" boosting ulb sparkov &
wait
./merge_partition_manifests.sh "$DADOS" "$DADOS/parte1.tsv" "$DADOS/parte2.tsv"
```

Cada parte começa com o que o manifesto principal já tem, de modo que as configurações já preparadas são reaproveitadas. Rodar de novo uma parte retoma o que faltou.

Cada configuração grava, ao lado dos CSVs, o relatório de preparação de cada dobra e o tempo de cada etapa (`<configuration_md5>.preparation_timing.json`).

## Execução numa máquina

```bash
./run_lr_nb_knn_specific_prepared_experiments.sh "$DADOS"
./run_catboost_xgboost_native_prepared_experiments.sh "$DADOS"
./run_anomaly_detection_prepared_experiments.sh "$DADOS"
./run_ft_transformer_prepared_experiments.sh "$DADOS"
./run_tabbert_prepared_experiments.sh "$DADOS"
python3 validate_prepared_datasets.py --mode results --manifest "$DADOS/prepared_dataset_configurations.tsv"
```

## Execução dividida entre máquinas

A dissertação dividiu a grade entre três máquinas. Os dados preparados ficaram em dois HDs compartilhados pela rede, montados nas três máquinas nos mesmos caminhos:

| HD | Local para | Uso |
|---|---|---|
| `/mnt/network-black-hd` | Acer | dados preparados; o Dell lê daqui pela rede |
| `/mnt/network-transparent-hd` | Lenovo | cópia das configurações do Lenovo |

Princípios da divisão:
- A preparação é feita uma única vez, no Acer.
- Cada par (algoritmo, base) roda inteiro numa única máquina.
- Cada máquina roda vários **processos** ao mesmo tempo. A GPU do Acer é um processo à parte dos processos de CPU dele.

O fluxo é:

1. **Calibração**, em cada máquina, com o número de processos que ela vai rodar:
   ```bash
   python3 calibrate_machine.py --prepared-data-directory "$DADOS" --cpu-lanes <n> [--gpu-lanes <n>]
   ```
   Com mais de um processo, o tempo usado no fator de velocidade é medido com essa quantidade de cópias da carga rodando juntas. A calibração também registra se o diretório é lido pela rede. Copie os `calibration/calibration_<host>.json` para o Acer.
2. **Plano**, no Acer, gravado no HD, para que as outras máquinas o leiam:
   ```bash
   python3 grid_plan.py allocate --manifest "$DADOS/prepared_dataset_configurations.tsv" \
     --prepared-data-directory "$DADOS" --calibration-directory calibration \
     --reference-host <host-do-acer> --plan-directory "$DADOS/grid_plan"
   python3 estimate_execution_time.py --plan "$DADOS/grid_plan"   # tempo de cada processo e makespan
   ```
   O plano distribui os pares entre os processos pela heurística de maior tarefa primeiro e só manda os pares de GPU para o processo de GPU. Também respeita a memória somada dos processos de cada máquina e cobra das máquinas que leem pela rede a leitura de cada configuração (`--network-bytes-per-second`).
3. **Posicionamento**, no Acer, das configurações do Lenovo no HD local dele:
   ```bash
   ./stage_machine_configurations.sh "$DADOS/grid_plan" <host-do-lenovo> local /mnt/network-transparent-hd/prepared
   ```
   Depois de conferir o MD5 de cada arquivo, a cópia leva o plano, com os processos do Lenovo, para `/mnt/network-transparent-hd/prepared/grid_plan`.
4. **Execução**: em cada máquina, um terminal por processo. O `run_lane.sh <dados> <plano> <processo>` roda todos os runners de um processo: os Transformers no processo de GPU e os demais blocos nos de CPU. Equivale a chamar cada runner com `GRID_PLAN` e `GRID_LANE`:
   ```bash
   GRID_PLAN="$DADOS/grid_plan" GRID_LANE=<host>-cpu1 ./run_catboost_xgboost_native_prepared_experiments.sh "$DADOS"
   ```
   O runner só executa os pares daquele processo. Ele usa o manifesto do processo e grava resultados e *staging* em `lanes/<processo>/`, no disco local. Os resultados nunca são gravados ao vivo num HD da rede.
5. **Backup**, pelo cron de cada máquina, para o HD que não é local dela:
   ```bash
   */30 * * * * cd <pipeline> && ./backup_lane_results.sh /mnt/network-transparent-hd/resultados   # no Acer
   */30 * * * * cd <pipeline> && ./backup_lane_results.sh /mnt/network-black-hd/resultados         # no Lenovo e no Dell
   ```
   A cópia é segura com o processo gravando: uma linha ainda incompleta fica para o próximo backup.
6. **Junção**, no Acer, sobre as duas árvores de backup reunidas num diretório de coleta (um subdiretório por processo):
   ```bash
   python3 merge_machine_results.py --plan-directory "$DADOS/grid_plan" --collection-directory coleta --output-directory resultados
   ```
   A junção para quando encontra linhas duplicadas e lista as lacunas do plano em `resultados/merge_report.json`. Ela também concatena os arquivos companheiros: os `.jsonl` de detalhes e os de estratos de histórico.

## Benchmark de inferência

Rode o benchmark numa máquina só (na dissertação, o Acer), sem outros processos da grade rodando ao mesmo tempo. Ele mede, por algoritmo e base, os milissegundos por 1.000 transações e a latência unitária (p50, p95 e intervalo de confiança do p95, numa amostra de 1.000 transações):

```bash
./run_inference_benchmark.sh "$DADOS"
```

## Tabelas da dissertação

```bash
python3 build_reproducibility_tables.py --results-directory resultados \
  --prepared-data-directory "$DADOS" --prepared-dataset-manifest "$DADOS/prepared_dataset_configurations.tsv" \
  --benchmark-file resultados_benchmark_inferencia.csv \
  --latex-directory tabelas --markdown-output tabelas/reprodutibilidade.md
```

## Estratos de histórico

Nas bases do TabBERT e no protocolo dele, a preparação nativa e a do FT-Transformer gravam, ao lado da validação e do teste, o estrato de histórico de cada transação (`<dataset_md5>.history.csv`). O valor é calculado como no índice de janelas do TabBERT.

O CatBoost, o XGBoost, o FT-Transformer e o TabBERT gravam métricas, transações, fraudes e prevalência por estrato:
- no boosting, em `resultados_<algoritmo>_history_strata.jsonl`;
- nos Transformers, no arquivo de detalhes.

A checagem R6 do `validate_prepared_datasets.py --mode results` confere que todos os algoritmos têm as mesmas contagens em cada estrato. A tabela da dissertação sai de:

```bash
python3 history_strata_analysis.py --results-directory resultados --output-directory tabelas
```

## Testes

```bash
python3 -m pytest src/fraud_detection_revised_pipeline/tests/unit -q
```
