"""LLM judge (plan 4): text only, one small call per ambiguous event, plus operator Q&A."""
from __future__ import annotations

import json

from .llm import chat, parse_json

DECISIONS = ("dismiss", "check_in", "alert")


class Judge:
    def __init__(self, cfg: dict):
        self.c = cfg["judge"]

    def __call__(self, event: dict) -> dict:
        """event: the event JSON (plan 2.3) with "keywords" and "transcript" added."""
        res = chat([{"role": "system", "content": self.c["system_prompt"]},
                    {"role": "user", "content": json.dumps(event, separators=(",", ":"))}],
                   model=self.c["model"], max_tokens=self.c["max_tokens"], temperature=self.c["temperature"], tag="judge")
        out = parse_json(res["text"])
        decision = out.get("decision") if out.get("decision") in DECISIONS else "check_in"  # unreadable reply: ask the rider
        return {"decision": decision, "severity": out.get("severity", 2), "reason": out.get("reason", ""),
                "usage": res["usage"]}

    def ask(self, event_log: dict, question: str) -> dict:
        """Operator question about one event; the model sees that event's log and transcript only."""
        res = chat([{"role": "system", "content": "You help a taxi safety operator. Answer the question in one or two "
                                                  "sentences using only the event log below. The transcript comes from a "
                                                  "small offline recognizer and may be wrong; say so if unsure.\n"
                                                  + json.dumps(event_log, separators=(",", ":"))},
                    {"role": "user", "content": question}],
                   model=self.c["model"], max_tokens=120, temperature=self.c["temperature"], json_mode=False, tag="qa")
        return {"answer": res["text"].strip(), "usage": res["usage"]}


class NoJudge:
    """Offline stand-in: every judged event becomes a check in, 0 tokens. For tuning the detector."""

    def __call__(self, event: dict) -> dict:
        return {"decision": "check_in", "severity": 2, "reason": "judge disabled",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}
