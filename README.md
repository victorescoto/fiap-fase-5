# Passos Mágicos — Modelo Preditivo de Risco de Defasagem Escolar

## Contexto do Negócio

A **Associação Passos Mágicos** atua na transformação da vida de crianças e jovens
de baixa renda, oferecendo educação de qualidade. O programa coleta diversos
indicadores educacionais dos estudantes (desempenho acadêmico, engajamento,
autoavaliação, indicadores psicossociais e psicopedagógicos) que são utilizados
para identificar alunos em risco de defasagem escolar.

Este projeto constrói um **modelo de Machine Learning** que classifica os alunos
em três níveis de risco — **alto**, **médio** e **baixo** — permitindo à equipe
pedagógica priorizar intervenções de forma proativa.

## Descrição da Solução

O sistema é composto por:

1. **Pipeline de ML** — treinamento, avaliação e serialização do modelo
2. **API REST** — FastAPI servindo predições em tempo real
3. **Monitoramento Contínuo** — logging de predições e detecção de drift
4. **Dashboard** — Streamlit para visualização de drift e métricas
5. **Infraestrutura** — Docker para empacotamento e deploy

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| API | FastAPI + Uvicorn |
| ML | scikit-learn, pandas, numpy |
| Monitoramento | PredictionLogger (custom) + Streamlit |
| Serialização | pickle (.pkl) + JSON (metadados) |
| Testes | pytest + pytest-cov (≥80% cobertura) |
| Empacotamento | Docker |
| Gerenciador de pacotes | uv |
| Logging | python-json-logger (structured JSON) |

## Estrutura do Projeto

```
app/
├── __init__.py
├── main.py                     # Aplicação FastAPI, lifespan, middlewares
├── routes.py                   # Endpoints da API
├── schemas.py                  # Modelos Pydantic (request/response)
├── model_loader.py             # Carregamento do modelo e metadados
├── monitoring.py               # Logging de predições e detecção de drift
├── logging_config.py           # Configuração de logging JSON
└── model/
    ├── model.pkl               # Modelo serializado
    └── model_metadata.json     # Metadados do modelo
dashboard/
└── app.py                      # Dashboard Streamlit (drift monitoring)
notebooks/
└── analise-exploratoria.ipynb  # Análise exploratória dos dados
src/
└── model/
    ├── preprocessing.py        # Limpeza e tipagem de colunas
    ├── feature_engineering.py  # Criação de features e remoção de leakage
    ├── train.py                # Treinamento (LogisticRegression + CV)
    ├── evaluate.py             # Métricas de avaliação
    └── run_training.py         # Orquestrador do pipeline completo
tests/
├── conftest.py                 # Fixtures compartilhadas
├── unit/
│   ├── test_schemas.py
│   ├── test_model_loader.py
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_evaluate.py
│   ├── test_train.py
│   └── test_monitoring.py
└── integration/
    └── test_api.py             # Testes dos endpoints via TestClient
Dockerfile
pyproject.toml
```

## Pipeline de Machine Learning

### 1. Pré-processamento (`src/model/preprocessing.py`)

- Identificação automática de colunas numéricas vs. categóricas
- Imputação (mediana para numéricos, constante para categóricos)
- Padronização (StandardScaler) e codificação (OneHotEncoder)
- Pipeline sklearn para reprodutibilidade

### 2. Engenharia de Features (`src/model/feature_engineering.py`)

- Criação da variável-alvo a partir da coluna `Defas` (alto/medio/baixo)
- Construção de features derivadas (médias, desvios-padrão, contagens)
- **Remoção de data leakage** — exclusão de colunas que codificam diretamente
  a classificação do aluno (Pedra, INDE 22, indicadores derivados de Pedra)

### 3. Treinamento (`src/model/train.py`)

- **Modelo:** LogisticRegression com `class_weight='balanced'`
- **Validação:** StratifiedKFold (5 folds) com F1 Macro
- **Balanceamento:** Pesos de classe automáticos para lidar com desbalanceamento

### 4. Avaliação (`src/model/evaluate.py`)

- Métricas: F1 Macro, F1 Weighted, Accuracy, Precision, Recall
- Matriz de confusão formatada
- Classification report detalhado por classe

### 5. Serialização (`src/model/train.py`)

- Modelo salvo em `app/model/model.pkl` (pickle)
- Metadados salvos em `app/model/model_metadata.json` (JSON)

### 6. Serving (`app/`)

- FastAPI carrega modelo no startup via lifespan
- Predições servidas em `/api/v1/predict`
- Cada predição é registrada pelo `PredictionLogger` para monitoramento

## Justificativa da Métrica: F1 Macro

A métrica principal escolhida para avaliação do modelo é o **F1 Macro** pelos
seguintes motivos:

1. **Desbalanceamento de classes** — A classe "alto risco" representa menos de
   2% dos dados. Accuracy (98.8%) seria enganosa pois um modelo que sempre
   previsse "baixo risco" já teria alta acurácia. O F1 Macro trata todas as
   classes igualmente, independente de seu tamanho.

2. **Equilíbrio entre Precision e Recall** — Em contexto educacional, tanto
   falsos positivos (alarme falso de risco) quanto falsos negativos (não
   identificar um aluno em risco) são prejudiciais. O F1 combina essas duas
   métricas de forma harmônica.

3. **Sensibilidade à classe minoritária** — Com F1 Macro = 0.9494, sabemos que
   mesmo para a classe rara (alto risco) o modelo tem desempenho forte
   (F1 = 0.8571, Recall = 1.0), o que significa que **nenhum aluno de alto risco
   conhecido no teste foi perdido**.

### Métricas Atuais do Modelo

| Métrica | Valor |
|---|---|
| **F1 Macro** | **0.9494** |
| F1 Weighted | 0.9892 |
| Accuracy | 0.9884 |
| Recall Macro | 0.9942 |
| Precision Macro | 0.9167 |
| CV F1 Mean ± Std | 0.9936 ± 0.0128 |

**Desempenho por classe:**

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| baixo | 1.00 | 1.00 | 1.00 |
| medio | 1.00 | 0.98 | 0.99 |
| alto  | 0.75 | 1.00 | 0.86 |

> O modelo identifica **100% dos alunos de alto risco** (Recall = 1.0) com uma
> taxa de falso positivo controlada (Precision = 0.75).

## Setup Local

### Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Instalação e Execução

```bash
# Instalar dependências
uv sync --all-groups

# Rodar API em modo desenvolvimento
uv run uvicorn app.main:app --reload
```

A API estará disponível em http://localhost:8000  
Documentação interativa (Swagger): http://localhost:8000/docs

### Treinar o Modelo

```bash
uv run python -m src.model.run_training
```

O modelo treinado será salvo em `app/model/model.pkl` e os metadados em
`app/model/model_metadata.json`.

### Dashboard de Monitoramento

```bash
uv run streamlit run dashboard/app.py
```

O dashboard estará disponível em http://localhost:8501 e exibe:
- Status do modelo e métricas
- Distribuição de predições (baseline vs. atual)
- Alertas de drift (warning e critical)
- Histórico de predições recentes

## Setup com Docker

```bash
# Build
docker build -t challenge-api .

# Run
docker run -p 8000:8000 challenge-api
```

## Endpoints da API

### `GET /health`

Health check.

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy", "model_loaded": true}
```

### `POST /api/v1/predict`

Predição de risco de defasagem escolar.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Fase": 4,
      "Ano nasc": 2010,
      "ANOS PM": 3,
      "INDE": 7.5,
      "IAA": 8.0,
      "IEG": 7.0,
      "IPS": 6.5,
      "IDA": 7.8,
      "IPP": 6.0,
      "IPV": 7.2,
      "IAN": 8.5,
      "Destaque IEG": 0,
      "Destaque IDA": 0,
      "Destaque IPV": 1
    }
  }'
```

```json
{
  "prediction": "baixo",
  "probability": 0.92,
  "model_version": "1.0.0"
}
```

### `GET /api/v1/model/info`

Metadados do modelo carregado.

```bash
curl http://localhost:8000/api/v1/model/info
```

### `GET /api/v1/monitoring/stats`

Estatísticas de monitoramento (predições recentes, distribuição, drift).

```bash
curl http://localhost:8000/api/v1/monitoring/stats
```

```json
{
  "total_predictions": 150,
  "prediction_distribution": {"baixo": 0.80, "medio": 0.15, "alto": 0.05},
  "avg_confidence": 0.87,
  "drift_status": {"severity": "none", "details": {}},
  "recent_predictions": [...]
}
```

## Monitoramento Contínuo

O sistema inclui monitoramento em tempo de execução:

- **PredictionLogger** (`app/monitoring.py`) — registra cada predição (classe,
  confiança, timestamp) em um buffer thread-safe de tamanho configurável
- **Detecção de Drift** — compara a distribuição atual de predições com o
  baseline de treinamento. Limiares:
  - `< 0.15` → sem drift
  - `0.15 – 0.30` → **warning**
  - `> 0.30` → **critical**
- **Endpoint `/api/v1/monitoring/stats`** — expõe estatísticas para consumo
  do dashboard ou alertas externos
- **Dashboard Streamlit** (`dashboard/app.py`) — visualização interativa com
  auto-refresh a cada 30 segundos

## Testes

```bash
# Rodar testes com cobertura
uv run pytest tests/ -v --cov=app --cov=src --cov-report=term-missing
```

Cobertura atual: **85.4%** (mínimo configurado: 80%).

Os testes cobrem:
- **Unitários:** schemas, model_loader, feature_engineering, preprocessing,
  evaluate, train, monitoring
- **Integração:** endpoints da API via TestClient

## Modo Degradado (Graceful Degradation)

A API sobe mesmo sem o modelo disponível:

| Endpoint | Comportamento |
|---|---|
| `GET /health` | `{"model_loaded": false}` |
| `POST /api/v1/predict` | HTTP 503 |
| `GET /api/v1/model/info` | Metadados padrão |
| `GET /api/v1/monitoring/stats` | Estatísticas zeradas |

Isso permite que a infraestrutura (Docker, K8s) inicie a API antes do modelo
estar pronto, facilitando health checks e rolling deployments.
