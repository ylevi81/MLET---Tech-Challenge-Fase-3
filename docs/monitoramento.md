# Etapa 3 — Monitoramento e observabilidade

## Iniciar a stack

Requisitos: Docker Desktop com engine Linux ativo e Docker Compose. Na raiz:

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail 50 api
```

O arquivo `compose.yaml` reúne API, Prometheus e Grafana. Na primeira execução,
a API baixa o corpus real, treina e prepara ONNX/INT8. Esse início pode levar
alguns minutos. Com um volume antigo, siga a migração em [etapa 4](otimizacao.md).
Os serviços são publicados apenas em `127.0.0.1`.

| Serviço | Endereço | Uso |
|---|---|---|
| API | http://localhost:8000/docs | Executar `POST /predict` |
| Readiness | http://localhost:8000/health | Modelo carregado e runtime ativo |
| Métricas | http://localhost:8000/metrics | Exposição Prometheus |
| Prometheus | http://localhost:9090/targets | Confirmar `medical-api` como UP |
| Grafana | http://localhost:3000/d/medical-api | Dashboard provisionado |

Login local inicial do Grafana: `admin` / `admin`; o primeiro acesso pode pedir
troca de senha. Para configurar antes do primeiro início, copie `.env.example`
para `.env` e ajuste `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`.
Depois que o banco do Grafana existe no volume, altere a senha na interface.

## Gerar tráfego

Execute o benchmark HTTP existente (com o ambiente Python do projeto ativo):

```bash
python scripts/benchmark.py --warmup 10 --requests 500 --output artifacts/benchmark.json
```

Também é possível usar **Try it out** em `/docs`, com:

```json
{"text": "A patient with persistent chest discomfort underwent cardiac evaluation."}
```

Para ver erros de validação no gráfico de status, envie `{"text": ""}`: a API
deve responder 422. Aguarde duas coletas (cerca de 10 segundos) e use a janela
dos últimos 15 minutos no Grafana. Sem tráfego, a latência pode aparecer como
"No data": não se inventa um tempo zero quando não houve requisições.

## Métricas e interpretação

| Métrica | Tipo | Significado |
|---|---|---|
| `http_requests_total` | Counter | Chamadas concluídas por método, rota e status |
| `http_request_duration_seconds` | Histogram | Duração de cada chamada HTTP, em segundos |
| `up{job="medical-api"}` | Gerada pelo Prometheus | Sucesso da coleta de `/metrics` |

O histograma inclui séries `_bucket`, `_sum` e `_count`. O dashboard usa
`rate` para chamadas/s e latência média, e `histogram_quantile` para estimar
p95. Essa estimativa usa os intervalos dos buckets; não precisa coincidir
exatamente com o p95 calculado pelo cliente HTTP.

Labels: `method`, `route` e `status_code`. Rotas usam templates do FastAPI;
URLs sem rota são agrupadas como `unmatched`. `/metrics` é excluído para a
coleta não inflar o tráfego. São contados sucessos, erros de validação,
indisponibilidade e exceções. Textos recebidos não são labels.

`up=1` significa que o endpoint de métricas respondeu, não que o modelo esteja
pronto: `/metrics` continua disponível quando `/health` e `/predict` retornam
503. O healthcheck Docker usa `/health` para verificar readiness.

A configuração executa **um processo Uvicorn por contêiner**. As métricas ficam
em memória e os contadores reiniciam quando o processo reinicia. Prometheus
mantém histórico no volume, com retenção de 15 dias. Antes de usar múltiplos
workers, configure a coleta multiprocess do `prometheus_client`.

## Dashboard e entregável

O [JSON versionável](../monitoring/grafana/dashboards/medical-api.json) é o
entregável do dashboard. O Grafana o lê automaticamente, junto com a fonte
Prometheus, pelos arquivos de provisioning em `monitoring/grafana/provisioning`.
Não é necessário importar manualmente. O dashboard tem seis painéis:

1. Sucesso da coleta da API.
2. Total de chamadas `/predict` desde o início do processo.
3. Requisições por segundo por rota.
4. Latência HTTP média de `/predict`.
5. Latência HTTP p95 de `/predict`.
6. Taxa de chamadas de `/predict` por status HTTP.

Edite o JSON para persistir mudanças; salvar pela UI está desabilitado para
manter o dashboard reproduzível. Um print pode ser capturado da página acima
depois de gerar tráfego. O JSON atende à alternativa "print/JSON" do enunciado.

## Validar e parar

```bash
docker compose config --quiet
docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
python -m pytest
python -m ruff check .
python scripts/check_stack.py
docker compose down
```

`check_stack.py` requer a stack já pronta. Ele gera tráfego por cerca de
20 segundos, verifica um erro 422, consulta as métricas pelo Prometheus e
confirma dashboard e datasource pelo Grafana. O resultado real é salvo em
`artifacts/stack-validation.json`. Se a senha do Grafana foi alterada, defina
`GRAFANA_ADMIN_USER` e `GRAFANA_ADMIN_PASSWORD` no ambiente do terminal.

`down` preserva os volumes. Alertas e monitoramento de drift não fazem parte
desta implementação.

Referências oficiais: [cliente Python](https://prometheus.github.io/client_python/exporting/http/asgi/),
[configuração Prometheus](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
e [provisioning do Grafana](https://grafana.com/docs/grafana/latest/administration/provisioning/).
