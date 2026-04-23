from __future__ import annotations

import time

import httpx

from contextual_research_agent.agent.llm import GenerationResult, LLMProvider
from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        model: str = "openai/gpt-5.4-mini",
        api_key: str = "",
        host: str = "https://openrouter.ai/api/v1",
        timeout: float = 120.0,
    ):
        self._model = model
        self._api_key = api_key
        self._host = host.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)
        )
        logger.info(f"OpenRouterProvider initialized: model={model}")

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> GenerationResult:
        start = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self._host}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            elapsed_ms = (time.perf_counter() - start) * 1000

            choice = data["choices"][0]
            text = choice["message"].get("content") or ""

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return GenerationResult(
                text=text,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter HTTP error: {e.response.status_code} {e.response.text[:200]}")
            raise
        except Exception as e:
            logger.exception(f"OpenRouter generation failed: {e}")
            raise

    async def close(self) -> None:
        await self._client.aclose()
