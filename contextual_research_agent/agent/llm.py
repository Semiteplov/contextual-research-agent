from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from contextual_research_agent.common import logging

logger = logging.get_logger(__name__)


@dataclass
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult: ...


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

        logger.info(f"OllamaProvider initialized: model={model}, host={host}")

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        start = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self._host}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            elapsed_ms = (time.perf_counter() - start) * 1000

            text = data.get("message", {}).get("content", "")

            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            return GenerationResult(
                text=text,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Ollama generation failed: {e}")
            raise

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


def create_llm_provider(
    provider: str = "ollama",
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    if provider == "ollama":
        return OllamaProvider(
            model=model or "qwen3:8b",
            host=host or "http://localhost:11434",
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
