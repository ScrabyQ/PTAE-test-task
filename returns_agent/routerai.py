"""The single network boundary. Only the Chat Completions function-call envelope is consumed."""

import json
import re
import time
from dataclasses import dataclass

import httpx

from .config import Settings
from .contracts import AgentError, tool_schema
from .privacy import PrivacyGate


@dataclass(frozen=True)
class ToolReply:
    calls: list


class RouterAI:
    def __init__(self, settings: Settings, guard: PrivacyGate, transport=None):
        settings.require_access()
        self.model = settings.model
        self.guard = guard
        self.client = httpx.Client(
            base_url=settings.base_url + "/", headers={"Authorization": "Bearer " + settings.api_key},
            timeout=httpx.Timeout(45, connect=12), transport=transport, follow_redirects=False,
        )
        self.http_requests = 0
        self.mode = "routerai" if transport is None else "test_transport"

    def close(self):
        self.client.close()

    def body(self, messages: list[dict], expected: str) -> dict:
        # Only the current capability is advertised. The host still checks every returned call.
        return {
            "model": self.model, "messages": messages, "tools": [tool_schema(expected)],
            "tool_choice": {"type": "function", "function": {"name": expected}},
            "parallel_tool_calls": False, "temperature": 0, "max_tokens": 1800,
        }

    def next_tool(self, messages: list[dict], expected: str, trace) -> ToolReply:
        body = self.body(messages, expected)
        for attempt in range(2):
            req = self.client.build_request("POST", "chat/completions", json=body)
            # Inspect exactly what httpx serialized, on each attempt, including all tool history.
            self.guard.guard(json.loads(req.content))
            trace("llm_request", model=self.model, expected_tool=expected, attempt=attempt + 1)
            self.http_requests += 1
            try:
                response = self.client.send(req)
            except httpx.ConnectTimeout:
                if attempt:
                    raise AgentError("ROUTERAI_UNAVAILABLE") from None
            except httpx.TimeoutException:
                if attempt:
                    raise AgentError("MODEL_TIMEOUT") from None
            except httpx.RequestError:
                if attempt:
                    raise AgentError("ROUTERAI_UNAVAILABLE") from None
            else:
                status = response.status_code
                if status == 200:
                    return self._reply(response, trace)
                trace("llm_http_error", status=status)
                if status not in {408, 429, 500, 502, 503, 504} or attempt:
                    raise AgentError(f"ROUTERAI_HTTP_{status}")
            time.sleep(0.4)
        raise AgentError("ROUTERAI_UNAVAILABLE")

    @staticmethod
    def _reply(response, trace) -> ToolReply:
        try:
            value = response.json()
            if value.get("error"):
                raise AgentError("ROUTERAI_RESPONSE_ERROR")
            choice = value["choices"][0]
            calls = choice["message"].get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError
            model = value.get("model")
            model = model if isinstance(model, str) and re.fullmatch(r"~?deepseek/[A-Za-z0-9.-]+", model) else None
            usage = value.get("usage") or {}
            tokens = {k: usage[k] for k in ("prompt_tokens", "completion_tokens")
                      if type(usage.get(k)) is int and usage[k] >= 0}
            trace("llm_response", resolved_model=model, tokens=tokens, proposed_calls=len(calls))
            return ToolReply(calls)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            raise AgentError("INVALID_MODEL_RESPONSE") from None
