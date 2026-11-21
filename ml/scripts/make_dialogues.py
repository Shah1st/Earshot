"""Act out the scripted dialogues (scripts/dialogues.py) and place them in the car.

Each turn is performed by an expressive speech model (earshot/voice.py), then convolved with a real
impulse response measured from that seat to the phone mic (driver seat / second row), and the turns
are joined with natural pauses. Shouted turns are 8 dB louder than spoken ones.
Output: data/dry/{dialog_normal,dialog_threat,lines_radio}/*.wav + data/manifest_dialog.csv
(`fx` = "done": car acoustics already applied; `spans` holds the shouted parts and where the threat starts).

    uv run python scripts/make_dialogues.py [--variants 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from dialogues import NORMAL, RADIO, THREAT  # noqa: E402

from earshot.common import DATA, DRY, SR, in_car, normalise, write_wav  # noqa: E402
from earshot.llm import spent_usd  # noqa: E402
from earshot.voice import speak  # noqa: E402

VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
FIELDS = ["path", "category", "role", "source", "source_id", "licence", "uploader", "title", "text", "labels", "split",
          "fx", "spans"]


def trim(x: np.ndarray, thr: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > thr)
    return x[max(0, idx[0] - 800): idx[-1] + 1600] if len(idx) else x


def perform(line: str, direction: str, voice: str, strict: bool = False) -> np.ndarray | None:
    loud = direction.startswith("LOUD:")
    res = speak(line, direction.removeprefix("LOUD:").strip(), voice, strict=strict)
    if res is None:
        return None
    x = trim(resample_poly(res[0], 2, 3).astype(np.float32))  # 24 kHz -> 16 kHz
    return normalise(x, rms_db=-15.0 if loud else -23.0)


def build(kind: str, did: str, turns: list, variant: int) -> dict | None:
    rng = random.Random(f"{did}-{variant}")
    voice = dict(zip("DP", rng.sample(VOICES, 2)))
    seat_seed = rng.randrange(10**6)
    parts, loud, text, t = [], [], [], 0.0
    for spk, direction, line in turns:
        x = perform(line, direction, voice[spk])
        if x is None:
            print(f"  skipped {did} v{variant}: the model would not perform a turn ({line[:40]}...)")
            return None
        is_loud = direction.startswith("LOUD:")
        x = in_car(x, "driver" if spk == "D" else "passenger", seat_seed)
        gap = rng.uniform(0.1, 0.3) if is_loud else rng.uniform(0.3, 0.8)
        parts += [np.zeros(int(gap * SR), np.float32), x]
        t += gap
        if is_loud:
            loud.append([round(t, 2), round(t + len(x) / SR, 2)])
        t += len(x) / SR
        text.append(f"{spk}: {line}")
    audio = np.concatenate(parts)
    audio *= 10 ** (-23 / 20) / float(np.sqrt(np.mean(audio**2)) + 1e-9) * (1.6 if loud else 1.0)
    rel = f"dialog_{kind}/{did}_v{variant}.wav"
    write_wav(DRY / rel, audio)
    alert_from = loud[0][0] if loud else 0.0
    return {"path": rel, "category": f"dialog_{kind}", "role": "alert" if kind == "threat" else "negative",
            "source": "acted (gpt-audio-mini) + McVAMPIRE car IR", "source_id": did, "licence": "synthetic",
            "text": " ".join(text), "fx": "done", "spans": json.dumps({"loud": loud, "alert_from": alert_from})}


def radio(rid: str, direction: str, line: str, variant: int) -> dict | None:
    rng = random.Random(f"{rid}-{variant}")
    x = perform(line, direction, rng.choice(VOICES))
    if x is None:
        return None
    rel = f"lines_radio/{rid}_v{variant}.wav"
    write_wav(DRY / rel, x)
    loud = [[0.0, round(len(x) / SR, 2)]] if direction.startswith("LOUD:") else []
    return {"path": rel, "category": "lines_radio", "role": "negative", "source": "acted (gpt-audio-mini)",
            "source_id": rid, "licence": "synthetic", "text": line, "spans": json.dumps({"loud": loud, "alert_from": 0.0})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2, help="voice pairs per dialogue")
    args = ap.parse_args()
    jobs = [(build, ("normal", d, t, v)) for d, t in NORMAL for v in range(args.variants)]
    jobs += [(build, ("threat", d, t, v)) for d, t in THREAT for v in range(args.variants)]
    jobs += [(radio, (r, d, line, v)) for r, d, line in RADIO for v in range(args.variants)]
    with ThreadPoolExecutor(6) as ex:
        rows = [r for r in ex.map(lambda j: j[0](*j[1]), jobs) if r]
    # split by script: every third dialogue of each kind is test, the rest tune (all voice variants together)
    order = {d: i for group in (NORMAL, THREAT, RADIO) for i, (d, *_) in enumerate(group)}
    for r in rows:
        r["split"] = "test" if order[r["source_id"]] % 3 == 2 else "tune"
    with open(DATA / "manifest_dialog.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)}/{len(jobs)} clips; total spend so far ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
