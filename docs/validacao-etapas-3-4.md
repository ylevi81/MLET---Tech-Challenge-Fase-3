# Validação das etapas 3 e 4

Execução em 14/09/2026 (America/Sao_Paulo), com Docker Desktop no Windows e
contêineres Linux. O treinamento usou o corpus real no cache Kaggle; fixtures
foram usadas apenas na suíte offline.

## Resultado

- 26 testes passaram, incluindo métricas de sucesso/erro, contrato dos dois
  runtimes ONNX, rejeição de metadados incompatíveis e reutilização de grafos.
- Ruff e `git diff --check` passaram.
- A imagem Docker foi construída e a API iniciou com `onnx_int8`.
- `promtool check config` validou a configuração do Prometheus.
- A coleta registrou respostas 200 e 422 e produziu latências média/p95.
- A API do Grafana confirmou seis painéis provisionados e datasource saudável.

O [registro HTTP da stack](evidencias/stack-validation.json) foi produzido por
`python scripts/check_stack.py`. As contagens são acumuladas desde o início do
processo; incluem duas execuções do verificador. A latência registrada é do
servidor durante esse tráfego curto, não um benchmark comparativo ou SLA.

## Paridade do modelo servido

O [relatório de paridade](evidencias/serving-parity.json) compara o Joblib com
os mesmos grafos usados no serving, sobre os primeiros 800 abstracts agrupados
do corpus, usando limiar 0,5:

| Variante | Concordância por decisão binária | Diferença média de probabilidade |
|---|---:|---:|
| ONNX FP32 | 100% | 0,000000302 |
| ONNX INT8 | 99,775% | 0,002401 |

Concordância por decisão binária não significa que 99,775% dos documentos têm
o conjunto inteiro de rótulos idêntico. Essa verificação não mede acurácia
contra anotações nem garante igualdade para qualquer entrada ou limiar.

O novo treinamento reproduziu subset accuracy `0,580142`, micro-F1 `0,791135`
e macro-F1 `0,800715`, conforme o README. Não foi atribuído um ganho HTTP aos
tempos locais da etapa 4; o procedimento de comparação HTTP entre backends
está em [otimização](otimizacao.md).

## Condições de reprodução

Foi necessário configurar `en_US.UTF-8` na imagem Linux para o operador de
texto do ONNX e normalizar os finais de linha do entrypoint. Essas correções
estão no Dockerfile e em `.gitattributes`.

O volume de modelos continha um artefato anterior à etapa 4. Ele foi copiado
para `/app/artifacts/backup-before-onnx-1789435738774388613` no próprio volume
antes do novo treinamento. Os testes não apagaram volumes persistentes.

Os testes emitiram dois avisos de depreciação das dependências Starlette/httpx
e AnyIO, sem falhas. O procedimento de execução está em
[monitoramento](monitoramento.md).
