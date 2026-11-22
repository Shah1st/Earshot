"""Ablation: what each part of the pipeline contributes, and how much is fitted to our own data.

Variants (same rides, same judge model and prompt):
  full         sound head + keyword trigger + transcript + GPS events            (shared/config.json)
  no_keywords  keyword trigger off; Vosk still gives the judge a transcript      -> impact of the trigger words
  no_vosk      no speech recognition at all: the judge sees sound labels + GPS   -> impact of words altogether
  raw_yamnet   no Vosk and native YAMNet class groups instead of the trained head -> nothing trained by us

Reading it:
  * tune vs test for one variant = how much it is fitted to the data it was tuned on;
  * "acted" vs "real sound" incidents = whether detection depends on the acted dialogues
    (acted: incident contains a threat dialogue voiced by the speech model; real: only Freesound recordings).

    uv run python scripts/ablation.py [--splits tune test demo]
"""
from __future__ import annotations

import argparse
import copy
import json

from earshot.common import REPO, RESULTS, load_config
from earshot.detector import Detector
from earshot.judge import Judge
from earshot.llm import spent_usd
from earshot.pipeline import run_ride
from earshot.rides import incidents, load_ride, scenario_paths, score_rides


def variants() -> dict[str, tuple[dict, bool]]:
    v2, v1 = load_config(), load_config(REPO / "shared" / "config_v1.json")
    no_kw = copy.deepcopy(v2)
    no_kw["asr"]["keyword_trigger"] = False
    return {"full": (v2, True), "no_keywords": (no_kw, True), "no_vosk": (v2, False), "raw_yamnet": (v1, False)}


def by_kind(results: list[tuple[dict, list[dict]]]) -> dict:
    """Event opened / alerted, separately for incidents with an acted dialogue and with real sounds only."""
    out = {"acted": [0, 0, 0], "real_sound": [0, 0, 0]}  # [incidents, event opened, alerted]
    for scn, outs in results:
        for w in incidents(scn):
            near = [o for o in outs if o["type"] == "decision" and o["source"] != "gps" and w["start"] - 1 <= o["t"] <= w["end"] + 10]
            k = out["acted" if any(x.startswith("dialog") for x in w["kinds"]) else "real_sound"]
            k[0] += 1
            k[1] += bool(near)
            k[2] += any(o["decision"] == "alert" for o in near)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["tune", "test", "demo"])
    args = ap.parse_args()
    report = {}
    for name, (cfg, use_asr) in variants().items():
        det, judge = Detector(cfg), Judge(cfg)
        for split in args.splits:
            results = []
            for path in scenario_paths(split):
                r = load_ride(path, det, asr_cfg=cfg if use_asr else None)
                results.append((r["scn"], run_ride(cfg, r["route"], r["audio"], r["gps"], det, r["asr"], judge,
                                                   r["scores"], r["emb"])))
            m = score_rides(results)
            m["by_kind"] = by_kind(results)
            m["tokens_total"] = sum(o["usage"]["total_tokens"] for _, outs in results for o in outs if o.get("usage"))
            if split == "demo":
                m["decisions"] = {scn["id"]: [f"{o['t']}s {o['decision']} ({o['source']})" for o in outs if o["type"] == "decision"]
                                  for scn, outs in results}
            report[f"{name}/{split}"] = m
            k = m["by_kind"]
            print(f"{name:12s} {split:5s} alerted {m['recall_alert']:.2f}  opened {m['detector_recall']:.2f}  "
                  f"false alerts/h {m['false_alerts_per_h']:5.1f}  check-ins/h {m['false_check_ins_per_h'] + m['gps_check_ins_per_h']:5.1f}  "
                  f"calls/h {m['llm_calls_per_h']:5.1f}  tokens/h {m['tokens_per_h']:6d}  | "
                  f"acted: opened {k['acted'][1]}/{k['acted'][0]} alerted {k['acted'][2]}/{k['acted'][0]}  "
                  f"real sound: opened {k['real_sound'][1]}/{k['real_sound'][0]} alerted {k['real_sound'][2]}/{k['real_sound'][0]}")
    out = RESULTS / "ablation.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key, m in report.items():
        if "decisions" in m:
            print(f"\n{key}: {m['tokens_total']} tokens")
            for rid, d in m["decisions"].items():
                print(f"   {rid}: {d or 'nothing'}")
    print(f"\nsaved {out}; total API spend so far ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
