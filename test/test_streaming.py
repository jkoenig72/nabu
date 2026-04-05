"""Tests for LLM streaming and TTS sentence streaming integration."""

from unittest.mock import MagicMock, patch, call
import json

import pytest

from app.llm.client import LLMClient, LLMError


@pytest.fixture
def llm_config():
    return {
        "llm": {
            "url": "http://localhost:8000/v1/chat/completions",
            "model": "test-model",
            "max_tokens": 100,
            "temperature": 0.7,
            "timeout": 10.0,
        }
    }


class TestStreamTokensSync:
    def test_yields_tokens(self, llm_config):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hallo"}}]}',
            'data: {"choices":[{"delta":{"content":" Welt"}}]}',
            'data: {"choices":[{"delta":{"content":"."}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("app.llm.client.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = mock_resp
            client = LLMClient(llm_config)
            tokens = list(client.stream_tokens_sync("system", [{"role": "user", "content": "hi"}]))

        assert tokens == ["Hallo", " Welt", "."]

    def test_sends_stream_true(self, llm_config):
        sse_lines = ["data: [DONE]"]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("app.llm.client.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = mock_resp
            client = LLMClient(llm_config)
            list(client.stream_tokens_sync("sys", []))

        call_args = MockClient.return_value.stream.call_args
        payload = call_args.kwargs["json"]
        assert payload["stream"] is True

    def test_skips_empty_deltas(self, llm_config):
        sse_lines = [
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{"content":"Ok"}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("app.llm.client.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = mock_resp
            client = LLMClient(llm_config)
            tokens = list(client.stream_tokens_sync("sys", []))

        assert tokens == ["Ok"]

    def test_skips_non_data_lines(self, llm_config):
        sse_lines = [
            "",
            ": keepalive",
            'data: {"choices":[{"delta":{"content":"Ja"}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("app.llm.client.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = mock_resp
            client = LLMClient(llm_config)
            tokens = list(client.stream_tokens_sync("sys", []))

        assert tokens == ["Ja"]

    def test_connection_error_raises_llm_error(self, llm_config):
        import httpx as real_httpx

        with patch("app.llm.client.httpx.Client") as MockClient:
            MockClient.return_value.stream.side_effect = real_httpx.ConnectError("refused")
            client = LLMClient(llm_config)
            with pytest.raises(LLMError, match="Cannot connect"):
                list(client.stream_tokens_sync("sys", []))


class TestStreamingPipelineIntegration:
    """Test the sentence_splitter + stream_tokens_sync chain."""

    def test_tokens_to_sentences(self, llm_config):
        from app.llm.sentence_splitter import split_sentences

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Satz"}}]}',
            'data: {"choices":[{"delta":{"content":" eins."}}]}',
            'data: {"choices":[{"delta":{"content":" Satz"}}]}',
            'data: {"choices":[{"delta":{"content":" zwei."}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("app.llm.client.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = mock_resp
            client = LLMClient(llm_config)
            tokens = client.stream_tokens_sync("sys", [])
            sentences = list(split_sentences(tokens))

        assert sentences == ["Satz eins.", "Satz zwei."]

    def test_llm_error_propagates_through_splitter(self, llm_config):
        from app.llm.sentence_splitter import split_sentences

        def failing_generator():
            yield "Partial "
            raise LLMError("connection lost")

        with pytest.raises(LLMError, match="connection lost"):
            list(split_sentences(failing_generator()))
