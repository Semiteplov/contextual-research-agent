from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(
        self,
        api_url: str,
        ws_url: str,
        api_key: str = "",
        timeout: float = 600.0,
    ):
        self._api_url = api_url.rstrip("/")
        self._ws_url = ws_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    async def health(self) -> dict[str, Any]:
        """GET /api/health"""
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{self._api_url}/api/health")
            r.raise_for_status()
            return r.json()

    async def readiness(self) -> dict[str, Any]:
        """GET /api/health/ready"""
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self._api_url}/api/health/ready")
            r.raise_for_status()
            return r.json()

    async def stats(self) -> dict[str, Any]:
        """GET /api/stats"""
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self._api_url}/api/stats", headers=self.headers)
            r.raise_for_status()
            return r.json()

    async def config(self) -> dict[str, Any]:
        """GET /api/config"""
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self._api_url}/api/config", headers=self.headers)
            r.raise_for_status()
            return r.json()

    async def sync_query(
        self,
        query: str,
        mode: str | None = None,
        document_ids: list[str] | None = None,
        include_trace: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "mode": mode,
            "document_ids": document_ids,
            "include_trace": include_trace,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(
                f"{self._api_url}/api/query",
                json=payload,
                headers=self.headers,
            )
            r.raise_for_status()
            return r.json()

    async def stream_query(
        self,
        query: str,
        mode: str | None = None,
        document_ids: list[str] | None = None,
        include_trace: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        ws_endpoint = f"{self._ws_url}/ws/trace"
        if self._api_key:
            ws_endpoint = f"{ws_endpoint}?api_key={self._api_key}"

        try:
            async with websockets.connect(
                ws_endpoint,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=10 * 1024 * 1024,
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "action": "query",
                            "query": query,
                            "mode": mode,
                            "document_ids": document_ids,
                            "include_trace": include_trace,
                        }
                    )
                )

                while True:
                    try:
                        msg = await ws.recv()
                    except ConnectionClosed:
                        logger.warning("WebSocket closed unexpectedly")
                        break

                    try:
                        event = json.loads(msg)
                    except json.JSONDecodeError as e:
                        logger.error("Invalid JSON from server: %s", e)
                        continue

                    yield event

                    if event.get("type") in ("final", "error"):
                        break

        except Exception as e:
            logger.exception("WebSocket connection failed")
            yield {
                "type": "error",
                "error": f"Connection failed: {e}",
            }
