from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from api.config import get_api_settings
from api.lifespan import get_service
from api.routes.query import _trace_to_response
from api.schemas import CognitiveModeAPI, QueryRequest
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


async def _validate_ws_api_key(websocket: WebSocket, api_key: str | None) -> bool:
    settings = get_api_settings()
    if not settings.auth_enabled:
        return True

    if not api_key or api_key not in settings.api_keys_set:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid API key",
        )
        return False

    return True


@router.websocket("/ws/trace")
async def trace_websocket(
    websocket: WebSocket,
    api_key: str | None = Query(default=None, alias="api_key"),
):
    await websocket.accept()

    if not await _validate_ws_api_key(websocket, api_key):
        return

    logger.info("WebSocket connected: %s", websocket.client)

    try:
        while True:
            try:
                message_text = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected: %s", websocket.client)
                break

            try:
                message = json.loads(message_text)
            except json.JSONDecodeError as e:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"Invalid JSON: {e}",
                    }
                )
                continue

            action = message.get("action", "").lower()

            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if action != "query":
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"Unknown action: {action}. Expected 'query' or 'ping'.",
                    }
                )
                continue

            try:
                mode_str = message.get("mode")
                request = QueryRequest(
                    query=message.get("query", ""),
                    mode=CognitiveModeAPI(mode_str) if mode_str else None,
                    document_ids=message.get("document_ids"),
                    include_trace=message.get("include_trace", True),
                )
            except Exception as e:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"Invalid request: {e}",
                    }
                )
                continue

            query_id = str(uuid.uuid4())
            await websocket.send_json(
                {
                    "type": "ack",
                    "query_id": query_id,
                    "query": request.query,
                }
            )

            await _execute_and_stream(websocket, request, query_id)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during execution")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


async def _execute_and_stream(
    websocket: WebSocket,
    request: QueryRequest,
    query_id: str,
) -> None:
    t_start = time.perf_counter()
    service = get_service()

    try:
        result = await service.query(
            text=request.query,
            mode=request.mode.value if request.mode else None,
            document_ids=request.document_ids,
        )

        cumulative_ms = 0.0
        for event in result.trace.events:
            cumulative_ms += event.get("latency_ms", 0.0)

            event_type = "node_error" if event.get("error") else "node_complete"

            await websocket.send_json(
                {
                    "type": event_type,
                    "query_id": query_id,
                    "node": event.get("node", ""),
                    "status": event.get("status", ""),
                    "latency_ms": event.get("latency_ms", 0.0),
                    "data": event.get("data", {}),
                    "error": event.get("error"),
                    "timestamp_ms": cumulative_ms,
                }
            )
            await asyncio.sleep(0.01)

        final_response = _trace_to_response(
            result.trace,
            query=request.query,
            include_trace=request.include_trace,
        )

        await websocket.send_json(
            {
                "type": "final",
                "query_id": query_id,
                "response": final_response.model_dump(mode="json"),
                "timestamp_ms": (time.perf_counter() - t_start) * 1000,
            }
        )

    except Exception as e:
        logger.exception("WebSocket query execution failed")
        await websocket.send_json(
            {
                "type": "error",
                "query_id": query_id,
                "error": str(e),
                "timestamp_ms": (time.perf_counter() - t_start) * 1000,
            }
        )
