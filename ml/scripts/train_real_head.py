"""Train the sound head on REAL recordings only, then compare it with the alternatives. No API calls.

Training data (tune split only, no acted or synthesised voice anywhere):
  positives  real screams, shouts / yells, breaking glass (FSD50K) and shouted speech by actors (CREMA-D)
  negatives  real speech, laughter, coughs, doors, horns, everyday sounds, music, crowd (FSD50K, CREMA-D)
Each clip is placed in the car (measured impulse response) and mixed over real driving noise at a random level;
positive frames are the loud part of the clip only. Model: logistic regression on [embedding now, 0.5 s ago].

Comparison (thresholds are set on the main TUNE rides for a false open budget, results are from TEST rides):
  native             YAMNet's own distress classes, no training
  head               the current head (its training rides include the acted dialogues)
  head_real          this script's head
  native|head, native|head_real   two triggers together, each at half the budget

Because head_real never hears an acted voice, the acted dialogue incidents are a clean test for it.
Output: shared/head_real.json (not used by the app until the config points to it), results/real_head.json

    uv run python scripts/fetch_extra.py && uv run python scripts/train_real_head.py
"""
from __future__ import annotations

import json
import random
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from earshot.common import CLIPS, DATA, MANIFEST, REPO, RESULTS, SR, in_car, load_config, normalise, read_audio
from earshot.detector import Detector
from earshot.rides import incidents, load_ride, scenario_paths

NATIVE = ["Screaming", "Shout", "Yell", "Children shouting", "Whimper", "Crying, sobbing", "Wail, moan",
          "Shatter", "Glass", "Breaking", "Smash, crash"]
POS_CATS, NEG_CATS = ["scream", "shout", "glass"], ["talk", "laughter", "cough", "door", "horn", "radio_music"]
WIN = 15600


def ctx(e: np.ndarray) -> np.ndarray:
    return np.concatenate([e, np.vstack([e[:1], e[:-1]])], axis=1)


def sig(v: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


def main() -> None:
    cfg = load_config()
    det = Detector(cfg)
    rng = random.Random(0)
    nat_idx = [det.labels.index(c) for c in NATIVE]
    man = pd.read_csv(MANIFEST, dtype={"labels": str}).fillna("")
    tune = man[man.split == "tune"]
    noise = [read_audio(CLIPS / p) * (10 ** (10 / 20) if src == "DEMAND" else 1.0)
             for p, src in tune[tune.category == "cabin"][["path", "source"]].itertuples(index=False)]

    # ---- training clips: (wet audio, positive?) ---------------------------------------------------------
    clips = []
    real = tune[tune.category.isin(POS_CATS + NEG_CATS) & ~tune.source.str.startswith("acted")]
    for r in real.itertuples():
        clips.append((CLIPS / r.path, None, r.category in POS_CATS))
    extra = pd.read_csv(DATA / "extra.csv", dtype=str)
    for r in extra.itertuples():
        clips.append((DATA / "extra" / r.path, zlib.crc32(r.source_id.encode()), r.positive == "1"))
    print(f"training clips: {sum(c[2] for c in clips)} positive, {sum(not c[2] for c in clips)} negative (all real recordings, tune split)")

    X, Y = [], []
    for path, ir_seed, pos in clips:
        clip = read_audio(path)[: 10 * SR]
        if ir_seed is not None:  # extra clips are stored dry: put them in the car
            clip = normalise(in_car(clip, "anyone", ir_seed))[: 10 * SR]
        for _ in range(2 if pos else 1):
            n = rng.choice(noise)
            bed = np.resize(np.roll(n, -rng.randrange(len(n))), len(clip) + SR).copy()
            placed = np.zeros(len(bed), np.float32)
            placed[SR // 2: SR // 2 + len(clip)] = clip * 10 ** ((rng.uniform(0, 8) if pos else rng.uniform(-6, 3)) / 20)
            x = bed * 10 ** (rng.uniform(-3, 3) / 20) + placed
            x = np.concatenate([x, np.zeros((-len(x)) % 8000, np.float32)])
            _, e = det.score_audio(x, with_embeddings=True)
            y = np.zeros(len(e), np.int8)
            if pos:  # positive = frames whose share of the clip is within 12 dB of its loudest frame; quieter ones ignored
                pad = np.concatenate([np.zeros(WIN, np.float32), placed, np.zeros(8000, np.float32)])
                rms = np.array([np.sqrt(np.mean(pad[WIN + (i + 1) * 8000 - WIN: WIN + (i + 1) * 8000] ** 2)) for i in range(len(e))])
                y[rms > 0] = -1
                y[rms >= rms.max() * 10 ** (-12 / 20)] = 1
            X.append(ctx(e))
            Y.append(y)
    x, y = np.concatenate(X), np.concatenate(Y)
    x, y = x[y >= 0], y[y >= 0]
    mu, sd = x.mean(0), x.std(0) + 1e-3
    clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit((x - mu) / sd, y)
    w = (clf.coef_[0] / sd).astype(np.float32)
    b = float(clf.intercept_[0] - (mu * w).sum())
    print(f"frames: {len(y)} ({y.mean():.1%} positive)")
    (REPO / "shared" / "head_real.json").write_text(json.dumps(
        {"input": "concat(yamnet embedding of this window, embedding of the previous window) = 2048 values",
         "output": "sigmoid(w.x + b)", "trained_on": "real recordings only (FSD50K, CREMA-D), car impulse responses and driving noise",
         "weights": [round(float(v), 6) for v in w], "bias": round(b, 6)}), encoding="utf-8")

    # ---- detectors over rides ---------------------------------------------------------------------------
    def series(ride) -> dict[str, np.ndarray]:
        sm = lambda v: np.convolve(v, [0.5, 0.5], mode="same")  # noqa: E731  (two frame average, like "sustained")
        return {"native": sm(ride["scores"][:, nat_idx].max(1)), "head": sm(det.head_series(ride["emb"])),
                "head_real": sm(sig(ctx(ride["emb"]) @ w + b))}

    rides = {s: [load_ride(p, det) for p in scenario_paths(s)] for s in ("tune", "test", "sound_test")}
    ser = {s: [series(r) for r in rs] for s, rs in rides.items()}

    def fire_times(split, i, spec) -> np.ndarray:
        hit = np.zeros(len(next(iter(ser[split][i].values()))), bool)
        for name, thr in spec:
            hit |= ser[split][i][name] >= thr
        return (np.flatnonzero(hit) + 1) * 0.5

    def false_per_h(split, spec) -> float:
        n, hours = 0, 0.0
        for i, r in enumerate(rides[split]):
            wins, last = incidents(r["scn"]), -1e9
            hours += r["scn"]["duration_s"] / 3600
            for t in fire_times(split, i, spec):
                # a detector that fires non stop opens a new event every 15 s (5 s event + 10 s cooldown)
                if not any(wd["from"] - 1 <= t <= wd["end"] + 10 for wd in wins) and t - last >= 15:
                    n += 1
                    last = t
        return n / hours

    grid = np.r_[np.arange(0.02, 0.9, 0.02), np.arange(0.9, 0.9995, 0.005), [0.9995, 0.9999]]

    def pick(name, budget) -> float:  # lowest threshold whose false opens on the TUNE rides fit the budget
        return next((float(t) for t in grid if false_per_h("tune", [(name, t)]) <= budget), 1.0)

    report = {}
    for budget in (15, 30):
        thr = {n: pick(n, budget) for n in ("native", "head", "head_real")}
        half = {n: pick(n, budget / 2) for n in ("native", "head", "head_real")}
        specs = {"native": [("native", thr["native"])], "head": [("head", thr["head"])], "head_real": [("head_real", thr["head_real"])],
                 "native|head": [("native", half["native"]), ("head", half["head"])],
                 "native|head_real": [("native", half["native"]), ("head_real", half["head_real"])]}
        print(f"\nBudget: at most {budget} false opens per hour on the tune rides.  TEST results:")
        print("   detector            acted dialogue   real sound   false opens/h   | isolated real sounds: threats / harmless opened")
        for name, spec in specs.items():
            acted = real_n = a_hit = r_hit = 0
            for i, r in enumerate(rides["test"]):
                times = fire_times("test", i, spec)
                for wd in incidents(r["scn"]):
                    hit = bool(np.any((times >= wd["start"] - 1) & (times <= wd["end"] + 5)))
                    if any(k.startswith("dialog") for k in wd["kinds"]):
                        acted, a_hit = acted + 1, a_hit + hit
                    else:
                        real_n, r_hit = real_n + 1, r_hit + hit
            pos = neg = p_hit = n_hit = 0
            for i, r in enumerate(rides["sound_test"]):
                times = fire_times("sound_test", i, spec)
                for e in r["scn"]["events"]:
                    hit = bool(np.any((times >= e["t"]) & (times <= e["t"] + e["len_s"] + 1.5)))
                    if "expect" in e:
                        pos, p_hit = pos + 1, p_hit + hit
                    else:
                        neg, n_hit = neg + 1, n_hit + hit
            fo = false_per_h("test", spec)
            print(f"   {name:18s}  {a_hit:2d} / {acted}          {r_hit} / {real_n}        {fo:5.1f}           | "
                  f"{p_hit}/{pos} ({p_hit / pos:.0%}) / {n_hit}/{neg} ({n_hit / neg:.0%})    thresholds {[(n, round(t, 3)) for n, t in spec]}")
            report[f"{budget}/{name}"] = {"acted": [a_hit, acted], "real": [r_hit, real_n], "false_opens_per_h": fo,
                                          "isolated_threats": [p_hit, pos], "isolated_harmless": [n_hit, neg],
                                          "thresholds": {n: t for n, t in spec}}
    (RESULTS / "real_head.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
