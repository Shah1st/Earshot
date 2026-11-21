"""Run the reference pipeline on a split and report the plan 6.2 metrics.

    uv run python scripts/evaluate.py --split test --tag v2                 # detector + Vosk + LLM judge
    uv run python scripts/evaluate.py --split tune --no-llm --tag tune_dry  # free: judge replaced by "check_in"
    uv run python scripts/evaluate.py --config ../shared/config_v1.json --tag v1
    uv run python scripts/evaluate.py --split demo --tag demo -v            # print every decision

Results: results/<tag>.json (metrics + every decision of every ride).
"""
from __future__ import annotations

import argparse
import json

from earshot.common import RESULTS, load_config
from earshot.detector import Detector
from earshot.judge import Judge, NoJudge
from earshot.llm import spent_usd
from earshot.pipeline import run_ride
from earshot.rides import load_ride, scenario_paths, score_rides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det = Detector(cfg)
    judge = NoJudge() if args.no_llm else Judge(cfg)
    results, rides_out = [], []
    for path in scenario_paths(args.split):
        r = load_ride(path, det, asr_cfg=None if args.no_asr else cfg)
        outs = run_ride(cfg, r["route"], r["audio"], r["gps"], det, r["asr"], judge, r["scores"], r["emb"])
        results.append((r["scn"], outs))
        rides_out.append({"id": r["scn"]["id"], "outputs": outs})
        if args.verbose:
            print(f"\n== {r['scn']['id']}  (truth: " + ", ".join(
                f"{e['t']}s {e.get('kind', e.get('gps'))}{'=' + e['expect'] if 'expect' in e else ''}"
                for e in r["scn"]["events"]) + ")")
            for o in outs:
                e = o["event"] or {}
                print(f"  t={o['t']:6.1f}  {o['decision'].upper():8s} [{o['source']}] {o['reason']}")
                if e:
                    print(f"            trigger={e['trigger']} groups={e['groups']} labels={list(e['top_labels'])[:4]} gps={e['gps']}")
                    print(f"            keywords={e['keywords']} transcript=\"...{e['transcript'][-160:]}\"")
    metrics = score_rides(results)
    out = RESULTS / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": args.tag, "split": args.split, "config_version": cfg["version"],
                               "llm": not args.no_llm, "metrics": metrics, "rides": rides_out}, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))
    print(f"saved {out}; total API spend so far ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
