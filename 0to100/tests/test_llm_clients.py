"""Tests for ClaudeClient + DeepSeekClient (mocked – no network)."""
from unittest.mock import MagicMock

from sq_ai.backend.llm_clients import ClaudeClient, DeepSeekClient


def test_claude_client_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = ClaudeClient(api_key=None)
    assert c.available is False
    assert c.generate("hi") is None


def test_claude_client_placeholder_key_is_unavailable():
    c = ClaudeClient(api_key="sk-ant-REPLACE_ME")
    assert c.available is False


def test_claude_client_generate_with_mock():
    c = ClaudeClient(api_key="sk-ant-real")
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text='{"action":"BUY"}')]
    c._client = MagicMock()
    c._client.messages.create.return_value = fake_msg
    out = c.generate("prompt", max_tokens=50, temperature=0.1)
    assert out == '{"action":"BUY"}'
    kwargs = c._client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-3-sonnet-20240229"
    assert kwargs["max_tokens"] == 50
    assert kwargs["temperature"] == 0.1
    assert kwargs["messages"][0]["role"] == "user"


def test_claude_client_system_prompt_passed():
    c = ClaudeClient(api_key="sk-ant-real")
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="ok")]
    c._client = MagicMock()
    c._client.messages.create.return_value = fake_msg
    c.generate("hi", system="be concise")
    assert c._client.messages.create.call_args.kwargs["system"] == "be concise"


def test_deepseek_client_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    d = DeepSeekClient(api_key=None)
    assert d.available is False
    assert d.generate("hi") is None


def test_deepseek_client_generate_with_mock():
    d = DeepSeekClient(api_key="sk-deepseek-real")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ranked_tickers": []}'))]
    d._client = MagicMock()
    d._client.chat.completions.create.return_value = fake_resp
    out = d.generate("hello", max_tokens=120, temperature=0.0, system="sys")
    assert out == '{"ranked_tickers": []}'
    kwargs = d._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["max_tokens"] == 120
    # system + user
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1]["role"] == "user"


def test_deepseek_client_default_base_url():
    d = DeepSeekClient(api_key="sk-real")
    assert d.base_url == "https://api.deepseek.com/v1"
