"""Build the five demo rides from the team's own voice recordings instead of the synthesised voices.

Put one audio file per dialogue part in ml/data/recorded/ (wav, m4a, mp3, ogg... anything ffmpeg reads):

    demo_1_forced_stop_part1.*   demo_1_forced_stop_part2.*
    demo_2_roadworks_part1.*     demo_2_roadworks_part2.*
    demo_3_shouting_detour_part1.*  demo_3_shouting_detour_part2.*
    demo_4_normal_part1.*        demo_4_normal_part2.*
    demo_5_engine_stall_part1.*  demo_5_engine_stall_part2.*  demo_5_engine_stall_part3.*

The words are in scripts/make_demo.py (RIDES). Each recording is trimmed, level matched and, unless --dry,
placed in the car with a measured impulse response; the ride then gets the same real driving noise and the
same GPS behaviour as the synthesised demo ride. Rides with a missing part are skipped.
Output: ml/data/demo_recorded/ and shared/scenarios/demo_recorded/ (ride ids end in _rec).

    uv run python scripts/make_recorded_demo.py [--dry]
    uv run python scripts/evaluate.py --split demo_recorded --tag demo_recorded -v
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_demo import DRIVE, RIDES, ROUTE  # noqa: E402
from make_dialogues import trim  # noqa: E402

from earshot.common import CLIPS, DATA, REPO, SR, in_car, normalise, read_audio, write_wav  # noqa: E402
from earshot.rides import load_ride  # noqa: E402

IN, OUT, SCN = DATA / "recorded", DATA / "demo_recorded", REPO / "shared" / "scenarios" / "demo_recorded"


def load_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        return read_audio(path)
    tmp = DATA / "raw" / "recorded_tmp.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", str(SR), "-ac", "1", str(tmp)], check=True)
    return read_audio(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="do not add the car impulse response (recorded in a real car)")
    args = ap.parse_args()
    for d in (OUT, SCN):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for rid, spec in RIDES.items():
        files = [next(iter(sorted(IN.glob(f"{rid}_part{k + 1}.*"))), None) for k in range(len(spec["parts"]))]
        if None in files:
            print(f"{rid}: skipped, missing part(s) {[k + 1 for k, f in enumerate(files) if f is None]}")
            continue
        events, at, t = [], {}, 0.0
        for k, ((gap, turns), f) in enumerate(zip(spec["parts"], files)):
            loud = any(d.startswith("LOUD:") for _, d, _ in turns)
            x = normalise(trim(load_any(f), thr=0.02))
            if not args.dry:
                x = normalise(in_car(x, "passenger", seed=k))
            x = x * (1.6 if loud else 1.0)  # same level convention as the synthesised parts
            t = round(t + gap, 1)
            at[f"p{k + 1}.start"], at[f"p{k + 1}.end"] = t, t + len(x) / SR
            rel = f"recorded/{rid}_part{k + 1}.wav"
            write_wav(CLIPS / rel, x)
            e = {"t": t, "clip": rel, "gain_db": 2 if loud else 0, "kind": "dialog", "len_s": round(len(x) / SR, 2),
                 "text": " ".join(f"{s}: {line}" for s, _, line in turns)}
            if spec["expect"] == "alert" and loud:
                e.update(expect="alert", incident=0)
            events.append(e)
            t += len(x) / SR
        events += [{"t": round(at[ref] + off, 1), "gps": kind, **extra} for kind, ref, off, extra in spec["gps"]]
        scn = {"id": f"{rid}_rec", "split": "demo_recorded", "route": ROUTE, "speed_kmh": 50,
               "duration_s": int(np.ceil(at[spec["end"][0]] + spec["end"][1])), "expected_outcome": spec["expect"] or "nothing",
               "background": spec.get("background", [DRIVE]), "events": sorted(events, key=lambda e: e["t"])}
        (SCN / f"{scn['id']}.json").write_text(json.dumps(scn, indent=1), encoding="utf-8")
        ride = load_ride(SCN / f"{scn['id']}.json", force=True)
        shutil.copy(ride["wav"], OUT / f"{scn['id']}.wav")
        shutil.copy(ride["wav"].with_suffix("").with_suffix(".gps.jsonl"), OUT / f"{scn['id']}.gps.jsonl")
        print(f"{scn['id']}: {scn['duration_s']} s, expected {scn['expected_outcome']}")
    print(f"\nrides in {OUT}; now run: uv run python scripts/evaluate.py --split demo_recorded --tag demo_recorded -v")


if __name__ == "__main__":
    main()
