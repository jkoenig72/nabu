import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from app.llm.client import LLMClient, LLMError


@pytest.fixture
def llm_config():
    return {
        "llm": {
            "url": "http://192.168.10.11:8000/v1/chat/completions",
            "model": "google/gemma-3-12b-it",
            "max_tokens": 256,
            "temperature": 0.7,
            "timeout": 10.0,
            "system_prompt": "Du bist Nabu.",
            "max_history_turns": 6,
        }
    }


MOCK_COMPLETION_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hallo! Wie kann ich dir helfen?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

MOCK_MODELS_RESPONSE = {
    "object": "list",
    "data": [{"id": "google/gemma-3-12b-it", "object": "model"}],
}


class TestClientInit:
    def test_sets_config_values(self, llm_config):
        client = LLMClient(llm_config)
        assert client.model == "google/gemma-3-12b-it"
        assert client.max_tokens == 256
        assert client.temperature == 0.7
        assert client.timeout == 10.0
        client.close_sync()


class TestHealthCheck:
    def test_success(self, llm_config):
        client = LLMClient(llm_config)
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            assert client.health_check_sync() is True
        client.close_sync()

    def test_connection_error_returns_false(self, llm_config):
        client = LLMClient(llm_config)

        with patch.object(client._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            assert client.health_check_sync() is False
        client.close_sync()


class TestComplete:
    def test_success(self, llm_config):
        client = LLMClient(llm_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = client.complete_sync(
                system_prompt="Du bist Nabu.",
                messages=[{"role": "user", "content": "Hallo"}],
            )
        assert result == "Hallo! Wie kann ich dir helfen?"
        client.close_sync()

    def test_builds_correct_payload(self, llm_config):
        client = LLMClient(llm_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            client.complete_sync(
                system_prompt="Test system prompt",
                messages=[
                    {"role": "user", "content": "Msg 1"},
                    {"role": "assistant", "content": "Reply 1"},
                    {"role": "user", "content": "Msg 2"},
                ],
                max_tokens=100,
                temperature=0.3,
            )

            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "google/gemma-3-12b-it"
            assert payload["messages"][0] == {"role": "system", "content": "Test system prompt"}
            assert payload["messages"][1] == {"role": "user", "content": "Msg 1"}
            assert payload["messages"][2] == {"role": "assistant", "content": "Reply 1"}
            assert payload["messages"][3] == {"role": "user", "content": "Msg 2"}
            assert payload["max_tokens"] == 100
            assert payload["temperature"] == 0.3
        client.close_sync()

    def test_timeout_raises_llm_error(self, llm_config):
        client = LLMClient(llm_config)

        with patch.object(client._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(LLMError, match="timed out"):
                client.complete_sync("system", [{"role": "user", "content": "hi"}])
        client.close_sync()

    def test_connection_error_raises_llm_error(self, llm_config):
        client = LLMClient(llm_config)

        with patch.object(client._client, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            with pytest.raises(LLMError, match="Cannot connect"):
                client.complete_sync("system", [{"role": "user", "content": "hi"}])
        client.close_sync()

    def test_http_error_raises_llm_error(self, llm_config):
        client = LLMClient(llm_config)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(LLMError, match="server error"):
                client.complete_sync("system", [{"role": "user", "content": "hi"}])
        client.close_sync()

    def test_empty_choices_returns_empty_string(self, llm_config):
        client = LLMClient(llm_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = client.complete_sync("system", [{"role": "user", "content": "hi"}])
        assert result == ""
        client.close_sync()


@pytest.mark.network
class TestLiveServer:
    def test_health_check(self, llm_config):
        client = LLMClient(llm_config)
        assert client.health_check_sync() is True
        client.close_sync()

    def test_completion(self, llm_config):
        client = LLMClient(llm_config)
        result = client.complete_sync(
            system_prompt="Antworte auf Deutsch in einem Satz.",
            messages=[{"role": "user", "content": "Was ist 2 plus 2?"}],
            max_tokens=512,
        )
        assert len(result) > 0
        assert isinstance(result, str)
        client.close_sync()
