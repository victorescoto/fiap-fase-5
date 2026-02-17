import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.logging_config import setup_logging
from app.model_loader import load_metadata, load_model
from app.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup and shutdown lifecycle for the application."""
    setup_logging()
    logger.info("Starting application...")

    app.state.model = load_model()
    app.state.metadata = load_metadata()

    if app.state.model is not None:
        logger.info("Model loaded successfully — ready to serve predictions")
    else:
        logger.warning(
            "No model found — API running in degraded mode (503 on /predict)"
        )

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title="Passos Mágicos — Prediction API",
    description="API para previsão de risco de defasagem escolar dos estudantes da Associação Passos Mágicos.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):  # noqa: ANN001
    """Log every request with method, path, status code and latency."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info(
        "%s %s → %s (%.4fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


app.include_router(router)
