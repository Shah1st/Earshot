"""Bundle everything the Unity and Android laptops need into ml/demo_pack/ (+ a zip to hand over).

    demo_pack/
      config.json, head.json            thresholds, prompts, v2 head weights (same files as shared/)
      models/                           yamnet tflite + label list (app assets)
      scenarios/*.json, routes/*.geojson
      clips/...                         only the clips the demo scenarios use (Unity mixes them itself), or
      rides/<id>.wav + <id>.gps.jsonl   the same rides pre rendered (simplest: play the wav, replay the GPS lines)
      expected/<id>.json                what the Python reference decided on each ride (for the parity check)

    uv run python scripts/export_demo_pack.py
"""
from __future__ import annotations

import json
import shutil

from earshot.common import CLIPS, ML, MODELS, REPO, RESULTS, RIDES, load_config
from earshot.detector import Detector
from earshot.rides import load_ride, scenario_paths

PACK = ML / "demo_pack"


def main() -> None:
    cfg = load_config()
    shutil.rmtree(PACK, ignore_errors=True)
    for d in ("models", "scenarios", "routes", "rides", "expected"):
        (PACK / d).mkdir(parents=True)
    shutil.copy(REPO / "shared" / "config.json", PACK)
    if (REPO / "shared" / "head.json").exists():
        shutil.copy(REPO / "shared" / "head.json", PACK)
    shutil.copy(MODELS / cfg["audio"]["model"], PACK / "models")
    shutil.copy(MODELS / "yamnet_labels.txt", PACK / "models")
    det = Detector(cfg)
    for path in scenario_paths("demo"):
        ride = load_ride(path, det)
        scn = ride["scn"]
        shutil.copy(path, PACK / "scenarios")
        shutil.copy(REPO / "shared" / scn["route"], PACK / "routes")
        for part in scn["background"] + [e for e in scn["events"] if "clip" in e]:
            dst = PACK / "clips" / part["clip"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(CLIPS / part["clip"], dst)
        for ext in (".wav", ".gps.jsonl"):
            shutil.copy(RIDES / f"{scn['id']}{ext}", PACK / "rides")
    demo = RESULTS / "demo.json"
    if demo.exists():
        for r in json.loads(demo.read_text(encoding="utf-8"))["rides"]:
            (PACK / "expected" / f"{r['id']}.json").write_text(json.dumps(r["outputs"], indent=1), encoding="utf-8")
    shutil.make_archive(str(ML / "demo_pack"), "zip", PACK)
    size = sum(f.stat().st_size for f in PACK.rglob("*") if f.is_file()) / 1e6
    print(f"{PACK} ({size:.0f} MB) and demo_pack.zip written")


if __name__ == "__main__":
    main()
