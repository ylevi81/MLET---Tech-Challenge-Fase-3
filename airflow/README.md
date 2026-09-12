# Orquestração do treinamento com Airflow

A DAG `train_medical_abstracts_classifier` executa duas tarefas em ordem:

1. baixa, durante a execução da tarefa, o dataset público
   `saharalaa/medical-abstracts-tc-corpus` com `kagglehub`;
2. usa `medical_triage.model.train_and_save` para agrupar os rótulos por resumo,
   treinar as cinco classes como um problema multilabel e promover os artefatos.

O parse da DAG não baixa dados nem inicia treinamento. Isso mantém o scheduler
rápido e permite que as tentativas e falhas sejam administradas pelo Airflow.

## Dependências e configuração

Instale as dependências a partir da raiz do repositório e disponibilize o pacote
do projeto no mesmo ambiente dos workers:

Use Python 3.11 ou 3.12 para esta configuração com Airflow 2.x.

```bash
python -m pip install -r airflow/requirements.txt
python -m pip install -e .
```

Configure a pasta desta DAG no Airflow ou copie `airflow/dags/train_model_dag.py`
para a pasta de DAGs da instalação. O diretório de saída padrão é `artifacts/` no
diretório de trabalho do worker. Para usar um volume persistente, defina:

```bash
export MEDICAL_TRIAGE_ARTIFACT_DIR=/opt/airflow/artifacts
```

O disparo é manual por padrão. Opcionalmente, defina um cron preset ou uma
expressão cron antes de iniciar o scheduler, por exemplo:

```bash
export AIRFLOW_TRAIN_SCHEDULE='@weekly'
```

O download público normalmente não exige credenciais. Se o Kaggle solicitar
autenticação no ambiente de execução, configure as credenciais suportadas pelo
`kagglehub` como segredo do worker, e nunca no arquivo da DAG.

As duas tasks trocam o caminho local do dataset. Use `LocalExecutor` para a
execução local ou monte o mesmo cache/volume do Kaggle em todos os workers de um
executor distribuído. O diretório de artefatos também deve apontar para um volume
compartilhado e persistente nesse cenário.

## Execução

Sem `AIRFLOW_TRAIN_SCHEDULE`, a DAG deve ser disparada quando um novo
treinamento for necessário:

```bash
airflow dags trigger train_medical_abstracts_classifier
```

Para uma verificação local de uma execução lógica:

```bash
airflow dags test train_medical_abstracts_classifier 2026-09-12
```

Ao final, o diretório configurado contém `model.joblib`, `metrics.json` e
`promotion.json`. O modelo é um bundle que inclui o pipeline multilabel e o
binarizador de rótulos; o manifesto é publicado por último e funciona como o
marcador de uma promoção completa.
