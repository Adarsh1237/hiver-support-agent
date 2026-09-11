"""
Minimal wrapper around the Anthropic Messages API. Kept dependency-free
(uses `requests`, not the `anthropic` SDK) so the repo installs fast.

Every caller in this project is written to work WITHOUT this client too
(rule-based fallback) -- see decision_log.md #2 for why.
"""
import json
import requests
from . import config

API_URL = "https://api.anthropic.com/v1/messages"


def is_available() -> bool:
    return bool(config.ANTHROPIC_API_KEY)


def complete(prompt: str, system: str = None, max_tokens: int = 500) -> str:
    if not is_available():
        raise RuntimeError("ANTHROPIC_API_KEY not set -- LLM client unavailable.")

    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": config.LLM_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    resp = requests.post(API_URL, headers=headers, data=json.dumps(body), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
