# Passos Mágicos — Prediction API

API para previsão de risco de defasagem escolar dos estudantes da Associação Passos Mágicos.

## Stack Tecnológica

- **Linguagem:** Python 3.12
- **API:** FastAPI + Uvicorn
- **ML:** scikit-learn, pandas, numpy
- **Serialização:** joblib / pickle
- **Testes:** pytest + pytest-cov (≥80% cobertura)
- **Empacotamento:** Docker
- **Gerenciador de pacotes:** uv
- **Logging:** python-json-logger (structured JSON)

## Estrutura do Projeto

```
app/
├── __init__.py
├── main.py                     # Aplicação FastAPI, lifespan, middlewares
├── routes.py                   # Endpoints da API
├── schemas.py                  # Modelos Pydantic (request/response)
├── model_loader.py             # Carregamento do modelo e metadados
├── logging_config.py           # Configuração de logging JSON
└── model/
    ├── model.pkl               # Modelo serializado (não versionado)
    └── model_metadata.json     # Metadados do modelo
notebooks/                      # Jupyter Notebooks (exploração, EDA, etc.)
src/                            # Pipeline de ML (preprocessing, training, etc.)
tests/
├── conftest.py                 # Fixtures compartilhadas
├── unit/
│   ├── test_schemas.py         # Testes dos Pydantic models
│   └── test_model_loader.py    # Testes de carregamento do modelo
└── integration/
    └── test_api.py             # Testes dos endpoints via TestClient
Dockerfile                      # Imagem Docker da API
pyproject.toml                  # Dependências e configurações
```

## Setup Local

### Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)

### Instalação

```bash
# Instalar dependências
uv sync --all-groups

# Rodar a API em modo desenvolvimento
uv run uvicorn app.main:app --reload
```

A API estará disponível em http://localhost:8000

### Swagger UI

Documentação interativa automática: http://localhost:8000/docs

## Setup com Docker

```bash
# Build da imagem
docker build -t challenge-api .

# Rodar o container
docker run -p 8000:8000 challenge-api
```

## Endpoints

### `GET /health`

Health check da API.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /api/v1/predict`

Realiza predição de risco de defasagem escolar.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.5, "feature_2": 1.0, "feature_3": 10}}'
```

**Response (200):**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "model_version": "1.0.0"
}
```

**Response (503) — modelo não carregado:**
```json
{
  "detail": "Model not loaded. Service unavailable."
}
```

### `GET /api/v1/model/info`

Retorna metadados do modelo carregado.

```bash
curl http://localhost:8000/api/v1/model/info
```

**Response:**
```json
{
  "version": "1.0.0",
  "metrics": {"accuracy": 0.92, "f1": 0.89},
  "features": ["feature_1", "feature_2", "feature_3"]
}
```

## Testes

```bash
# Rodar testes com cobertura
uv run pytest tests/ -v --cov=app --cov-report=term-missing
```

Cobertura mínima configurada: **80%** (atualmente ~99%).

## Pipeline de Machine Learning

1. **Pré-processamento dos Dados** — limpeza e preparação (`src/preprocessing.py`)
2. **Engenharia de Features** — criação de atributos (`src/feature_engineering.py`)
3. **Treinamento e Validação** — treino do modelo (`src/train.py`)
4. **Avaliação** — métricas de performance (`src/evaluate.py`)
5. **Serialização** — modelo salvo em `app/model/model.pkl` via joblib
6. **Serving** — API FastAPI carrega o modelo no startup e serve predições

## Modo Degradado

A API sobe mesmo sem o modelo disponível. Nesse caso:
- `GET /health` retorna `{"model_loaded": false}`
- `POST /api/v1/predict` retorna HTTP 503
- `GET /api/v1/model/info` retorna metadados padrão

Isso permite que a infraestrutura (Docker, K8s) suba a API antes do modelo estar pronto.
