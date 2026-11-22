"""Generate ride scenarios (plan 1.2) from the frozen split: tune, test, train (dense, for the v2 head).

Tune and train rides only use tune clips, test rides only test clips. Deterministic (seeded).
Levels: clips sit at -23 dBFS RMS; driving noise is level matched so that normal speech has the SNR
measured in real cars (~7 dB at 40 km/h). Output: shared/scenarios/{tune,test,train}/*.json

    uv run python scripts/make_scenarios.py [--n-tune 24] [--n-test 24] [--n-train 60]
"""
from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from earshot.common import MANIFEST, REPO

OUT = REPO / "shared" / "scenarios"
ROUTES = ["routes/epfl_to_gare.geojson", "routes/gare_to_ouchy_loop.geojson", "routes/renens_to_flon.geojson"]
# benign events: category -> (weight, gain range dB)
BENIGN = {"dialog_normal": (6, (-3, 2)), "talk": (2, (-3, 2)), "laughter": (2, (-4, 2)), "cough": (1, (-4, 2)),
          "door": (1, (-6, 0)), "horn": (1, (-6, 0)), "radio_talk": (3, (-6, 0))}
HARD_NEGATIVES = {"laughter", "radio_talk"}  # detector may fire, the LLM should dismiss
INCIDENTS = {"dialog": 3, "dialog+scream": 2, "scream": 1, "shout+impact": 1, "glass": 1}
NOT_HUMAN = ("Animal", "Bird", "Applause", "Clapping", "Fireworks", "Explosion")


class Pool:
    def __init__(self, df: pd.DataFrame, rng: random.Random):
        df = df[~((df.role == "alert") & df.labels.str.contains("|".join(NOT_HUMAN)))]
        self.by_cat = {c: g.to_dict("records") for c, g in df.groupby("category")}
        self.rng = rng

    def pick(self, cat: str) -> dict:
        return self.rng.choice(self.by_cat[cat])

    def cabin(self, speed: int) -> dict:
        """Real driving noise recorded at the closest speed, or a DEMAND in-car segment (quieter file: +10 dB)."""
        drive = [c for c in self.by_cat["cabin"] if c["source"] == "McVAMPIRE" and c["labels"] == str(speed)]
        other = [c for c in self.by_cat["cabin"] if c["source"] == "DEMAND"]
        if other and self.rng.random() < 0.3:
            return {"clip": self.rng.choice(other)["path"], "gain_db": round(self.rng.uniform(8, 12), 1), "loop": True}
        return {"clip": self.rng.choice(drive)["path"], "gain_db": round(self.rng.uniform(-2, 2), 1), "loop": True}


def ev(t: float, clip: dict, gain: float, **kw) -> dict:
    out = {"t": round(t, 1), "clip": clip["path"], "gain_db": round(gain, 1), "kind": clip["category"],
           "len_s": float(clip["duration_s"])}
    if clip.get("text"):
        out["text"] = clip["text"]
    if clip.get("spans"):
        out.update(json.loads(clip["spans"]))  # "loud": shouted parts, "alert_from": where the threat becomes audible
    return {**out, **kw}


def incident(kind: str, t: float, k: int, pool: Pool, rng: random.Random, allow_detour: bool) -> list[dict]:
    """One threat: sound events (all expect=alert, same incident id) plus optional GPS behaviour."""
    u, tag = rng.uniform, {"expect": "alert", "incident": k}
    out = []
    if kind.startswith("dialog"):
        d = pool.pick("dialog_threat")
        out.append(ev(t, d, u(0, 4), **tag))
        if kind == "dialog+scream":
            out.append(ev(t + d["duration_s"] + u(-1.5, 0.5), pool.pick("scream"), u(2, 8), **tag))
    elif kind == "scream":
        out.append(ev(t, pool.pick("scream"), u(2, 8), **tag))
    elif kind == "shout+impact":
        out.append(ev(t, pool.pick("shout"), u(2, 8), **tag))
        out.append(ev(t + u(1, 3), pool.pick("impact"), u(0, 6), **tag))
    elif kind == "glass":
        out.append(ev(t, pool.pick("glass"), u(0, 6), **tag))
        if rng.random() < 0.5:
            out.append(ev(t + u(0.5, 2), pool.pick("scream"), u(2, 8), **tag))
    r = rng.random()
    if allow_detour and kind.startswith("dialog") and r < 0.5:
        out.append({"t": round(t - u(25, 40), 1), "gps": "detour"})
    elif r > 0.7:
        out.append({"t": round(out[0]["t"] + out[0].get("alert_from", 0) + u(-1, 1), 1), "gps": "sudden_stop", "dur": 10})
    return out


def make_ride(rid: str, split: str, pool: Pool, rng: random.Random, duration: int = 300, dense: bool = False) -> dict:
    u = rng.uniform
    speed = rng.choice([30, 40, 50, 60])
    background = [pool.cabin(speed)]
    if rng.random() < 0.6:
        background.append({"clip": pool.pick("radio_music")["path"], "gain_db": round(u(-12, -6), 1), "loop": True,
                           "from": 15, "to": duration})
    if dense:  # training rides for the v2 head: a threat every ~40 s
        times = list(range(30, duration - 30, 40))
        n_inc = len(times)
    else:
        n_inc = rng.choice([0, 0, 1, 1, 2])
        times = sorted(rng.sample(range(70, duration - 50, 75), n_inc))
    events, has_detour, busy = [], False, []
    for k, t in enumerate(times):
        # scored rides skip "shout+impact": its shouts are CREMA-D actors saying a neutral sentence ("It's eleven
        # o'clock"), fine as acoustic training material for the head but not a fair "threat" for the judge
        kinds = {k: w for k, w in INCIDENTS.items() if dense or k != "shout+impact"}
        kind = rng.choices(list(kinds), weights=list(kinds.values()))[0]
        new = incident(kind, t + u(0, 8), k, pool, rng, allow_detour=(k == n_inc - 1))
        has_detour |= any(e.get("gps") == "detour" for e in new)
        sounds = [e for e in new if "clip" in e]
        detour = any(e.get("gps") == "detour" for e in new)  # after a threat on a detour the ride is not "normal" any more
        busy.append((min(e["t"] for e in sounds) - 12,
                     duration if detour else max(e["t"] + e["len_s"] for e in sounds) + (8 if dense else 20)))
        events += new
    t = u(6, 15)
    while t < duration - 25:
        cat = rng.choices(list(BENIGN), weights=[w for w, _ in BENIGN.values()])[0]
        clip = pool.pick(cat)
        if all(t + clip["duration_s"] < a or t > b for a, b in busy):
            events.append(ev(t, clip, u(*BENIGN[cat][1]), **({"expect": "dismiss"} if cat in HARD_NEGATIVES else {})))
            t += clip["duration_s"]
        t += u(4, 10) if dense else u(8, 20)
    if not has_detour and rng.random() < 0.3:  # an ordinary red light: must not raise anything
        events.append({"t": round(u(20, 50), 1), "gps": "stop", "dur": 20})
    return {"id": rid, "split": split, "route": rng.choice(ROUTES), "speed_kmh": speed,
            "duration_s": duration, "background": background, "events": sorted(events, key=lambda e: e["t"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tune", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--n-train", type=int, default=60, help="dense rides from TUNE clips, to train the v2 head")
    args = ap.parse_args()
    df = pd.read_csv(MANIFEST, dtype={"labels": str}).fillna("")
    for split, n in (("tune", args.n_tune), ("test", args.n_test), ("train", args.n_train)):
        rng = random.Random(f"earshot-{split}")
        pool = Pool(df[df.split == ("tune" if split == "train" else split)], rng)
        (OUT / split).mkdir(parents=True, exist_ok=True)
        for f in (OUT / split).glob("*.json"):
            f.unlink()
        for i in range(n):
            scn = make_ride(f"{split}_{i:02d}", split, pool, rng, dense=(split == "train"))
            (OUT / split / f"{scn['id']}.json").write_text(json.dumps(scn, indent=1), encoding="utf-8")
    print(f"wrote {args.n_tune} tune, {args.n_test} test, {args.n_train} train scenarios to {OUT} (demo rides: make_demo.py)")


if __name__ == "__main__":
    main()
