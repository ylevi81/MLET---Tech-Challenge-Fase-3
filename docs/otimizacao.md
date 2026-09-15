# Etapa 4 — Otimização e serving ONNX

## Escopo

A otimização usa o mesmo **Medical Abstracts TC Corpus**, TF-IDF e regressões
logísticas one-vs-rest das etapas anteriores. As cinco categorias, o problema
multilabel e a deduplicação são preservados. O modelo treinado é exportado para
ONNX FP32 e quantizado em INT8; o README registra os obstáculos de conversão e
as medições locais anteriores.

O runtime é selecionado por `MEDICAL_TRIAGE_BACKEND`:

| Valor | Inferência | Arquivo padrão ao lado do Joblib |
|---|---|---|
| `sklearn` | Pipeline scikit-learn | `model.joblib` |
| `onnx_fp32` | ONNX Runtime, pesos FP32 | `model.onnx` |
| `onnx_int8` | ONNX Runtime, pesos quantizados | `model_int8.onnx` |

Localmente, o padrão é `sklearn`; no Compose é `onnx_int8`. O pacote instalado
com `pip install -e ".[onnx]"` inclui as dependências de exportação e serving.
A imagem Docker já instala essas dependências via `requirements.txt`.
Ela também instala o locale `en_US.UTF-8`, necessário para o
[StringNormalizer do ONNX](https://onnx.ai/onnx/operators/onnx__StringNormalizer.html)
no Linux. O entrypoint tem finais de linha normalizados para funcionar em
checkouts feitos no Windows.

## Preparar e servir

Depois de treinar com o código atual:

```powershell
python -m medical_triage.serving --model-path artifacts/model.joblib --output-dir artifacts
$env:MEDICAL_TRIAGE_BACKEND = "onnx_int8"
$env:MEDICAL_TRIAGE_MODEL_PATH = "artifacts/model.joblib"
uvicorn medical_triage.api:app --host 0.0.0.0 --port 8000
```

`MEDICAL_TRIAGE_ONNX_PATH` permite um caminho explícito na execução local.
No Compose, o entrypoint prepara `model.onnx` e `model_int8.onnx` ao lado do
Joblib antes de iniciar Uvicorn. Modelos ONNX válidos e correspondentes são
reutilizados. Um novo `artifact_id` provoca nova exportação no próximo início.

O grafo contém metadados com identificação do treinamento, ordem dos IDs das
classes e variante. A API rejeita arquivos ausentes ou incompatíveis: `/health`
e `/predict` respondem 503. Não há fallback automático para sklearn. O
`/health` inclui `inference_backend` para tornar a escolha verificável.

O bundle Joblib continua carregado para manter metadados e o binarizador.
Somente `predict_proba` é substituído pelo adaptador ONNX: normalização de
texto, limiar, seleção de labels, fallback para top-1 e formato JSON permanecem
compartilhados. Isso não reduz necessariamente a RAM total do processo, pois
o bundle original também é carregado. A sessão ONNX é criada uma vez, com uma
thread, e reutilizada nas requisições.

## Migrar artefatos anteriores

O modelo antigo com `strip_accents="unicode"` não é exportável. Preserve uma
cópia dos artefatos antes de retreinar. Para o volume Docker:

```bash
docker compose stop api
docker compose run --rm --entrypoint python api -m medical_triage.train --model-path /app/artifacts/model.joblib --metrics-path /app/artifacts/metrics.json
docker compose up -d api
```

Não é preciso apagar volumes nem baixar novamente se o corpus está no cache.
A API não retreina silenciosamente um artefato existente incompatível. Um
modelo só é treinado automaticamente quando ausente e `TRAIN_IF_MISSING=1`.

## Avaliar paridade e latência

```bash
python -m medical_triage.optimize --dataset-dir data/raw --runs 400
```

Esse comando exige os três CSVs do corpus no diretório informado. Ele gera
`artifacts/optimization.json` com comparação de probabilidades e concordância
por decisão binária, além de latência local para batches 1, 8 e 32.
A amostra usa os primeiros abstracts agrupados do corpus, não exclusivamente
o conjunto de teste; a concordância é uma comparação entre runtimes, não uma
nova estimativa de qualidade preditiva. O relatório de paridade não impõe um
limiar automático de aprovação. A preparação no entrypoint usa apenas duas
frases para verificação básica; não substitui essa avaliação do corpus.

INT8 pode alterar probabilidades e decisões próximas do limiar. Os testes
offline cobrem equivalência FP32, tolerância INT8, contrato HTTP, fallback
top-1, metadados incompatíveis e reutilização dos modelos.

Para medir o efeito real na API, compare os backends no mesmo host, artefato,
texto, warm-up e quantidade de chamadas. Em PowerShell:

```powershell
$env:MEDICAL_TRIAGE_BACKEND = "sklearn"
docker compose up -d --force-recreate api
# Aguarde /health retornar 200.
python scripts/benchmark.py --warmup 10 --requests 500 --output artifacts/http-sklearn.json

$env:MEDICAL_TRIAGE_BACKEND = "onnx_int8"
docker compose up -d --force-recreate api
# Aguarde /health retornar 200 e inference_backend=onnx_int8.
python scripts/benchmark.py --warmup 10 --requests 500 --output artifacts/http-int8.json
```

Os números locais do README não são uma promessa de aceleração HTTP. Rede,
serialização, validação e instrumentação também compõem a latência do serviço.
As métricas do Grafana incluem todas essas fases dentro do servidor.

## Promoção pelo Airflow

A DAG continua promovendo o bundle Joblib. Após a promoção, reinicie a API
para o entrypoint exportar grafos correspondentes ao novo treinamento.
Não existe troca do modelo em memória sem reinício. Em produção, prepare e
valide os ONNX antes do deploy, distribuindo Joblib e grafos da mesma versão.
