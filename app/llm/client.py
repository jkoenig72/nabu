import asyncio
import json
import logging

import httpx

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMClient:
    """OpenAI-compatible chat completion client."""

    def __init__(self, config):
        llm_cfg = config["llm"]
        self.url = llm_cfg["url"]
        self.model = llm_cfg["model"]
        self.max_tokens = llm_cfg.get("max_tokens", 256)
        self.temperature = llm_cfg.get("temperature", 0.7)
        self.timeout = llm_cfg.get("timeout", 10.0)

        base_url = self.url.rsplit("/v1/", 1)[0]

        self._loop = asyncio.new_event_loop()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
        )
        self._sync_client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=10.0),
        )

    async def complete(self, system_prompt, messages, max_tokens=None, temperature=None):
        """Return assistant message text for the given conversation."""
        import time
        t0 = time.monotonic()
        tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        log.debug("LLM request: %d messages, max_tokens=%d, temp=%.1f", len(messages), tok, temp)
        log.debug("LLM system prompt: '%s'", system_prompt[:120])
        if messages:
            last = messages[-1].get("content", "")
            log.debug("LLM last message [%s]: '%s'", messages[-1].get("role"), last[:120])

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": tok,
            "temperature": temp,
        }

        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            log.error("LLM timeout after %.1fs", time.monotonic() - t0)
            raise LLMError(f"LLM request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot connect to LLM server: {e}") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM server error {e.response.status_code}: {e}") from e

        data = response.json()
        usage = data.get("usage", {})
        elapsed = time.monotonic() - t0

        choices = data.get("choices", [])
        if not choices:
            log.warning("LLM returned empty choices")
            return ""

        content = choices[0].get("message", {}).get("content")
        if content is None:
            log.warning("LLM returned None content")
            return ""

        log.debug("LLM response in %.2fs: %d prompt + %d completion tokens",
                  elapsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        log.debug("LLM answer: '%s'", content.strip()[:150])

        return content.strip()

    def complete_sync(self, system_prompt, messages, max_tokens=None, temperature=None):
        """Synchronous wrapper for complete()."""
        return self._loop.run_until_complete(
            self.complete(system_prompt, messages, max_tokens, temperature)
        )

    def stream_tokens_sync(self, system_prompt, messages, max_tokens=None, temperature=None):
        """Yield tokens as they arrive from the LLM via SSE streaming."""
        tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        log.debug("LLM stream: %d messages, max_tokens=%d", len(messages), tok)

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": tok,
            "temperature": temp,
            "stream": True,
        }

        try:
            with self._sync_client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
        except httpx.TimeoutException as e:
            raise LLMError(f"LLM stream timed out: {e}") from e
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot connect to LLM server: {e}") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM server error {e.response.status_code}: {e}") from e

    async def health_check(self):
        """Return True if LLM server is reachable."""
        try:
            response = await self._client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    def health_check_sync(self):
        return self._loop.run_until_complete(self.health_check())

    async def close(self):
        await self._client.aclose()

    def close_sync(self):
        self._loop.run_until_complete(self.close())
        self._loop.close()
        self._sync_client.close()
