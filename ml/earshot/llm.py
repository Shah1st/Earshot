"""OpenRouter client with an on-disk cache, a usage log and a hard spend cap.

Every real call appends a line to results/cache/usage.jsonl. Cached calls return the stored usage,
so token numbers in reports are always the ones the API measured.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import requests

from .common import RESULTS, openrouter_key

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = RESULTS / "cache" / "llm"
USAGE_LOG = RESULTS / "cache" / "usage.jsonl"
BUDGET_USD = float(os.environ.get("EARSHOT_BUDGET_USD", "3.0"))
_lock = threading.Lock()


def spent_usd() -> float:
    if not USAGE_LOG.exists():
        return 0.0
    return sum(json.loads(line).get("cost") or 0.0 for line in USAGE_LOG.read_text().splitlines() if line)


def chat(messages: list, model: str, max_tokens: int = 80, temperature: float = 0, json_mode: bool = True,
         tag: str = "", use_cache: bool = True) -> dict:
    """Returns {"text", "usage": {prompt_tokens, completion_tokens, total_tokens, cost}, "cached"}."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if use_cache and path.exists():
        return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}

    with _lock:
        if spent_usd() >= BUDGET_USD:
            raise RuntimeError(f"LLM budget cap reached (${BUDGET_USD}); raise EARSHOT_BUDGET_USD to continue")
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=120, headers={"Authorization": f"Bearer {openrouter_key()}"})
        except requests.RequestException:  # flaky network: wait and try again
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
            continue
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200 and "choices" in data:
            break
        if r.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:400]}")
    u = data.get("usage", {})
    out = {"text": data["choices"][0]["message"].get("content") or "",
           "usage": {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                     "total_tokens": u.get("total_tokens", 0), "cost": u.get("cost", 0.0),
                     "audio_tokens": (u.get("prompt_tokens_details") or {}).get("audio_tokens", 0)}}
    with _lock:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out), encoding="utf-8")
        with open(USAGE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "tag": tag, "model": model, **out["usage"]}) + "\n")
    return {**out, "cached": False}


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction (models sometimes wrap the object in a code fence)."""
    a, b = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[a:b + 1]) if a >= 0 and b > a else {}
    except json.JSONDecodeError:
        return {}
