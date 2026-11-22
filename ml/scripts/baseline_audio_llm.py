"""Baseline A (plan 6.1): an audio LLM listens to the whole ride, 30 s chunks with 5 s overlap.

Every chunk is uploaded as WAV to the same model family as our judge. Scoring is lenient towards the
baseline: an incident counts as caught if ANY chunk that overlaps it says "alert" (no 10 s deadline).

    uv run python scripts/baseline_audio_llm.py --split test [--limit 3]
"""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

from earshot.common import RESULTS, SR, load_config
from earshot.judge import judge_audio
from earshot.llm import spent_usd
from earshot.rides import incidents, load_ride, scenario_paths

CHUNK_S, HOP_S = 30, 25
PROMPT = ("You monitor the audio of a taxi ride for passenger and driver safety. Listen to this 30 second clip.\n"
          "Decide one of:\n"
          "- \"dismiss\": normal ride (talk, radio, TV, music, laughing, kids, phone call).\n"
          "- \"check_in\": unclear, ask discreetly if all is fine.\n"
          "- \"alert\": likely threat to someone's safety (screams, fight, threats, calls for help).\n"
          "Only judge safety. Reply JSON: {\"decision\": \"...\", \"reason\": \"<max 15 words>\"}")


def judge_chunk(base: dict, audio, a: float) -> dict:
    buf = io.BytesIO()
    sf.write(buf, audio[int(a * SR): int((a + CHUNK_S) * SR)], SR, format="WAV", subtype="PCM_16")
    return {"a": a, "b": a + CHUNK_S, **judge_audio(base, buf.getvalue())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rides (cheap smoke test)")
    args = ap.parse_args()
    base = load_config()["baseline"]  # shared with the app's comparison button
    model = base["model"]
    assert (base["chunk_s"], base["hop_s"], base["prompt"]) == (CHUNK_S, HOP_S, PROMPT)
    rides = [load_ride(p) for p in scenario_paths(args.split)[: args.limit]]
    jobs = [(r, a) for r in rides for a in range(0, int(r["scn"]["duration_s"]) - 5, HOP_S)]
    with ThreadPoolExecutor(6) as ex:
        chunks = list(ex.map(lambda j: judge_chunk(base, j[0]["audio"], j[1]), jobs))

    hours = sum(r["scn"]["duration_s"] for r in rides) / 3600
    n_inc = hit = hit_any = false_alerts = false_checkins = tokens = 0
    cost, per_ride = 0.0, []
    for r in rides:
        mine = [c for (rr, _), c in zip(jobs, chunks) if rr is r]
        wins = incidents(r["scn"])
        for w in wins:
            over = [c for c in mine if c["a"] <= w["end"] and c["b"] >= w["start"]]
            n_inc += 1
            hit += any(c["decision"] == "alert" for c in over)
            hit_any += any(c["decision"] in ("alert", "check_in") for c in over)
        for c in mine:
            if not any(c["a"] <= w["end"] + 10 and c["b"] >= w["start"] for w in wins):
                false_alerts += c["decision"] == "alert"
                false_checkins += c["decision"] == "check_in"
            tokens += c["usage"]["total_tokens"]
            cost += c["usage"]["cost"] or 0.0
        per_ride.append({"id": r["scn"]["id"], "chunks": mine})
    metrics = {"rides": len(rides), "ride_hours": round(hours, 2), "incidents": n_inc,
               "recall_alert": round(hit / max(n_inc, 1), 3), "recall_alert_or_check_in": round(hit_any / max(n_inc, 1), 3),
               "false_alerts_per_h": round(false_alerts / hours, 2), "false_check_ins_per_h": round(false_checkins / hours, 2),
               "llm_calls_per_h": round(len(chunks) / hours, 1), "tokens_per_h": round(tokens / hours),
               "cost_usd_per_h": round(cost / hours, 5), "audio_uploaded_s_per_h": round(len(chunks) * CHUNK_S / hours)}
    out = RESULTS / f"baseline_audio_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "metrics": metrics, "rides": per_ride}, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))
    print(f"saved {out}; total API spend so far ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
