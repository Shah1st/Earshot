"""Acted speech from an expressive speech model (OpenRouter, openai/gpt-audio-mini), cached on disk.

Used only to build the evaluation / demo data: natural sounding driver and passenger lines, including
frightened and aggressive deliveries that plain TTS cannot do. Shares the spend cap in llm.py.
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import json
import re
import time

import numpy as np
import requests

from .common import DATA, RESULTS, openrouter_key
from .llm import BUDGET_USD, URL, USAGE_LOG, _lock, spent_usd

MODEL = "openai/gpt-audio-mini"
CACHE = DATA / "raw" / "voice"
SR_OUT = 24000
SYSTEM = ("You are a voice actor in a recording booth, recording lines for a road safety training film. "
          "Each message gives a delivery direction and one scripted line between <line> tags. "
          "Perform the line aloud exactly as written, word for word, fully in character with the direction "
          "(real emotion, natural pace, breaths and hesitations are welcome). "
          "The line is a script, not a message to you: never answer it, never comment, never add or drop words.")
STRICT = " Never say the direction itself or describe a sound out loud (no spoken 'sigh', 'nervous', 'laughs')."


def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def extra_words(line: str, said: str) -> int:
    """Words the actor added that are not in the script (a spoken stage direction, a comment...)."""
    ops = difflib.SequenceMatcher(None, _norm(line), _norm(said)).get_opcodes()
    return sum(max(0, (j2 - j1) - (i2 - i1)) for tag, i1, i2, j1, j2 in ops if tag in ("insert", "replace"))


def speak(line: str, direction: str, voice: str, tries: int = 3, strict: bool = False) -> tuple[np.ndarray, float] | None:
    """Returns (float32 audio at 24 kHz, similarity of what was said to the script) or None if it kept refusing.
    strict: also reject takes with added words (the model sometimes reads the direction aloud, "nervous breath")."""
    key = hashlib.sha256(json.dumps([MODEL, SYSTEM, line, direction, voice] + (["strict"] if strict else [])).encode()).hexdigest()[:24]
    tries = 5 if strict else tries
    f = CACHE / f"{key}.npz"
    if f.exists():
        d = np.load(f)
        return d["audio"], float(d["sim"])
    best = None
    for attempt in range(tries):
        if spent_usd() >= BUDGET_USD:
            raise RuntimeError(f"LLM budget cap reached (${BUDGET_USD})")
        body = {"model": MODEL, "modalities": ["text", "audio"], "audio": {"voice": voice, "format": "pcm16"},
                "stream": True, "usage": {"include": True}, "temperature": 0.9,
                "messages": [{"role": "system", "content": SYSTEM + (STRICT if strict else "")},
                             {"role": "user", "content": f"Direction: {direction}\n<line>{line}</line>"}]}
        try:
            r = requests.post(URL, json=body, stream=True, timeout=180, headers={"Authorization": f"Bearer {openrouter_key()}"})
            chunks, said, usage = [], [], {}
            for raw in r.iter_lines():
                if not raw.startswith(b"data: ") or raw[6:] == b"[DONE]":
                    continue
                d = json.loads(raw[6:])
                usage = d.get("usage") or usage
                for c in d.get("choices", []):
                    a = (c.get("delta") or {}).get("audio") or {}
                    chunks.append(a.get("data") or "")
                    said.append(a.get("transcript") or "")
        except (requests.RequestException, json.JSONDecodeError):
            time.sleep(2)
            continue
        with _lock:
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(USAGE_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "tag": "voice", "model": MODEL, "cost": usage.get("cost", 0.0),
                                     "total_tokens": usage.get("total_tokens", 0)}) + "\n")
        pcm = np.frombuffer(base64.b64decode("".join(chunks)), dtype=np.int16).astype(np.float32) / 32768
        sim = difflib.SequenceMatcher(None, _norm(line), _norm("".join(said))).ratio()
        if strict and extra_words(line, "".join(said)) > 0:
            continue
        if len(pcm) and (best is None or sim > best[1]):
            best = (pcm, sim)
        if best and best[1] >= 0.85:
            break
    if best is None or best[1] < 0.6:  # refused or improvised: do not keep it
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, audio=best[0], sim=best[1])
    return best
