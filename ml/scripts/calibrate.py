"""Calibrate trigger thresholds on the TUNE rides only, then freeze them into the config.

For every triggering group, sweep the threshold (and the rule type) and count on the rendered tune rides:
  hits         = incidents the group is meant to catch where it triggers inside the incident window
  false fires  = merged triggers outside every incident window (each costs one LLM judge call)
Safety first: among settings that reach --min-recall, pick the one with the fewest false fires.

    uv run python scripts/calibrate.py [--config ../shared/config.json] [--min-recall 0.9] [--write]
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from earshot.common import CONFIG_PATH, load_config
from earshot.detector import Detector
from earshot.events import Triggers
from earshot.rides import incidents, load_ride, scenario_paths

# which incident sounds a native group is meant to catch; the head ("distress") is meant to catch everything
GROUP_KINDS = {"scream": {"scream", "dialog_threat"}, "shout": {"shout", "dialog_threat"}, "impact": {"impact"},
               "glass": {"glass"}, "crash": set()}
GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def series(det: Detector, ride: dict, group: str) -> np.ndarray:
    if group in det.head_groups:
        return det.head_series(ride["emb"])
    return ride["scores"][:, det.groups[group]].max(axis=1)


def trigger_times(cfg: dict, group: str, values: np.ndarray) -> list[float]:
    trig = Triggers({**cfg, "groups": {group: cfg["groups"][group]}})
    return [(i + 1) * cfg["audio"]["chunk_s"] for i, v in enumerate(values) if trig.step({group: float(v)})[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--write", action="store_true", help="save the chosen thresholds to the config file")
    args = ap.parse_args()
    cfg = load_config(args.config)
    det = Detector(cfg)
    rides = [load_ride(p, det) for p in scenario_paths("tune")]
    hours = sum(r["scn"]["duration_s"] for r in rides) / 3600
    cool = cfg["event"]["cooldown_s"]
    print(f"{len(rides)} tune rides, {hours:.2f} h, {sum(len(incidents(r['scn'])) for r in rides)} incidents\n")
    chosen = {}
    for group, spec in cfg["groups"].items():
        if not spec.get("trigger", True):
            continue
        vals = [series(det, r, group) for r in rides]
        best_overall = None
        for kind in ("sustained", "impulsive"):
            rows = []
            for thr in GRID:
                c = copy.deepcopy(cfg)
                c["groups"][group].update(threshold=thr, type=kind)
                hits = total = false = 0
                for r, v in zip(rides, vals):
                    wins = incidents(r["scn"])
                    times = trigger_times(c, group, v)
                    for w in wins:
                        if spec.get("source") == "head" or GROUP_KINDS.get(group, set()) & set(w["kinds"]):
                            total += 1
                            hits += any(w["start"] - 1 <= t <= w["end"] + 5 for t in times)
                    last = -1e9
                    for t in times:
                        if not any(w["start"] - 1 <= t <= w["end"] + 10 for w in wins):
                            false += t - last >= cool
                            last = t
                rows.append((thr, hits / max(total, 1), false / hours, total, kind))
            ok = [r for r in rows if r[1] >= args.min_recall] or [max(rows, key=lambda r: r[1])]
            best = min(ok, key=lambda r: (r[2], -r[0]))  # fewest false fires among those that reach the recall
            if best_overall is None or best[2] < best_overall[2]:
                best_overall = best
            print(f"{group} [{kind}] ({rows[0][3]} incidents)   thr: recall / false fires per h")
            print("   " + "  ".join(f"{t:.2f}: {rec:.2f}/{f:.1f}{'*' if t == best[0] else ''}" for t, rec, f, _, _ in rows))
        chosen[group] = {"threshold": best_overall[0], "type": best_overall[4]}
    print("\nchosen:", chosen)
    if args.write:
        for g, c in chosen.items():
            cfg["groups"][g].update(c)
            cfg["groups"][g]["high"] = max(cfg["groups"][g]["high"], round(min(0.97, c["threshold"] + 0.15), 2))
        cfg["version"] += 1
        with open(args.config, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(cfg, indent=2) + "\n")
        print(f"written to {args.config} (version {cfg['version']})")


if __name__ == "__main__":
    main()
