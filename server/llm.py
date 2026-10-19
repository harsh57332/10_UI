"""LLM adapter: same .call(action, params) shape as the other model services, backed by vLLM's
OpenAI-compatible API, so swapping the model never changes the callers.

call("chat", {"system": str, "user": str, "schema": {json schema} (optional), "max_tokens": int})
  -> {"text": raw text, "json": parsed object or None}
"""
import json
import re

import httpx

from common import config

from .services import ServiceError


class LLM:
    def __init__(self):
        s = config.load()["services"]["llm"]
        self.s = s
        self.url = f"http://{s['host']}:{s['port']}/v1"
        self.http = httpx.Client(timeout=s["timeout_s"])

    def call(self, action, params):
        if action != "chat":
            raise ServiceError(f"llm: unknown action '{action}'")
        body = {
            "model": self.s["served_name"],
            "messages": [{"role": "system", "content": params["system"]},
                         {"role": "user", "content": params["user"]}],
            "max_tokens": params.get("max_tokens", self.s["max_tokens"]),
            "temperature": params.get("temperature", self.s["temperature"]),
        }
        if params.get("schema"):
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": params["schema"]}}
        try:
            r = self.http.post(f"{self.url}/chat/completions", json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise ServiceError(f"llm unreachable or failed: {e}")
        return {"text": text, "json": parse_json(text)}

    def health(self):
        try:
            return self.http.get(f"{self.url}/models", timeout=2).json()
        except (httpx.HTTPError, ValueError):
            return None


def parse_json(text):
    """First JSON object in the text (reasoning models may wrap it in prose or <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return None
