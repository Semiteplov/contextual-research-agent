from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from api.config import get_api_settings
from api.lifespan import lifespan
from api.routes import health, query, stats, trace
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_api_settings()

    app = FastAPI(
        title="Multi-Agent Research Assistant API",
        description=(
            "FastAPI backend for the multi-agent research assistant. "
            "Provides query execution through a LangGraph-based agent system "
            "with retrieval, generation, post-hoc verification (Critic), "
            "and synthesis stages."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logger(
        request: Request,
        call_next: Callable[[Request], Awaitable],
    ):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "[%s] %s %s — unhandled exception",
                request_id,
                request.method,
                request.url.path,
            )
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[%s] %s %s — %d (%.0fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("[%s] Unhandled exception", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    app.include_router(health.router)
    app.include_router(stats.router)
    app.include_router(query.router)
    app.include_router(trace.router)

    @app.get("/", tags=["meta"])
    async def root():
        return {
            "name": "Multi-Agent Research Assistant API",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": {
                "query": "POST /api/query",
                "query_stream": "POST /api/query/stream",
                "trace_websocket": "WS /ws/trace",
                "health": "GET /api/health",
                "ready": "GET /api/health/ready",
                "stats": "GET /api/stats",
                "config": "GET /api/config",
            },
        }

    return app
