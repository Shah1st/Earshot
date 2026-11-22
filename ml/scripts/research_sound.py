"""Why are YAMNet's own labels weak in the car, and is a trained head worth it? No API calls.

Part 1  Domain gap. The same real scream / glass clips scored by native YAMNet under four conditions:
        dry (as recorded), with the car impulse response, with car + driving noise, and the noisy mix
        after a 250 Hz high pass (driving noise is mostly low frequency rumble).
Part 2  Detectors compared on the isolated real sound rides (compare_sound_paths.py). Thresholds are set on
        the TUNE rides so that 5% / 10% of harmless sounds would open an event; results are from TEST rides.
          native        max of YAMNet's own distress classes
          native_hp     the same after the high pass
          head          current head (trained on dense rides that include the acted dialogues)
          head_real     head trained only on real recordings + augmentation (random car noise, level, offset)
          head_real_hp  head_real trained and run on high passed audio
Part 3  The same detectors and thresholds on the main test rides: incidents opened (acted dialogue vs real
        sound) and false opens per hour. head_real never saw an acted voice, so this is an independent check.
Part 4  Do YAMNet's laughter / cough labels separate harmless events from threats (for the check-in rule)?

    uv run python scripts/research_sound.py
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.linear_model import LogisticRegression

from earshot.common import CLIPS, DRY, MANIFEST, REPO, RESULTS, SR, load_config, normalise, read_audio
from earshot.detector import Detector
from earshot.rides import incidents, load_ride, scenario_paths

NATIVE = ["Screaming", "Shout", "Yell", "Children shouting", "Whimper", "Crying, sobbing", "Wail, moan",
          "Shatter", "Glass", "Breaking", "Smash, crash"]
BENIGN = ["Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh", "Baby laughter", "Cough",
          "Throat clearing", "Sneeze"]
POS = ["scream", "shout", "glass", "impact"]
NEG = ["talk", "laughter", "cough", "door", "horn", "radio_music", "radio_talk"]
SOS = butter(4, 250, btype="high", fs=SR, output="sos")


def hp(x: np.ndarray) -> np.ndarray:
    return sosfilt(SOS, x).astype(np.float32)


def ctx(e: np.ndarray) -> np.ndarray:
    return np.concatenate([e, np.vstack([e[:1], e[:-1]])], axis=1)


def smooth_peak(v: np.ndarray, a: int, b: int) -> float:
    v = v[max(a, 0):b]
    return float(np.convolve(v, [0.5, 0.5], mode="valid").max()) if len(v) > 1 else float(v.max(initial=0))


def main() -> None:
    cfg = load_config()
    det = Detector(cfg)
    rng = random.Random(0)
    lab = {c: det.labels.index(c) for c in NATIVE + BENIGN + ["Vehicle", "Speech"]}
    nat_idx, ben_idx = [lab[c] for c in NATIVE], [lab[c] for c in BENIGN]
    df = pd.read_csv(MANIFEST, dtype={"labels": str}).fillna("")
    noise = {s: [read_audio(CLIPS / p) for p in df[(df.category == "cabin") & (df.split == s) & (df.source == "McVAMPIRE")
                                                   & df["labels"].isin(["40", "50"])].path] for s in ("tune", "test")}
    report = {}

    def score(x):
        s, e = det.score_audio(np.concatenate([x, np.zeros((-len(x)) % 8000 + 8000, np.float32)]), with_embeddings=True)
        return s, e

    def mix(clip, split, gain_db):
        n = rng.choice(noise[split])
        off = rng.randrange(len(n) - SR)
        bed = np.resize(np.roll(n, -off), len(clip) + SR)
        bed[SR // 2: SR // 2 + len(clip)] += clip * 10 ** (gain_db / 20)
        return bed

    # ---- Part 1: domain gap on real clips (test split, never trained on) --------------------------------
    print("Part 1. Native YAMNet on the same real clips (test split). Share of clips where the best distress class")
    print("        reaches the score, and the mean 'Vehicle' score.")
    for cat in ("scream", "glass"):
        rows = df[(df.category == cat) & (df.split == "test") & (df.source == "FSD50K")]
        cond = {"dry": [], "car IR": [], "car IR + noise": [], "car IR + noise + high pass": []}
        veh = {k: [] for k in cond}
        for r in rows.itertuples():
            wet = read_audio(CLIPS / r.path)
            noisy = mix(wet, "test", 4.0)
            for name, x in (("dry", normalise(read_audio(DRY / r.path))), ("car IR", wet), ("car IR + noise", noisy),
                            ("car IR + noise + high pass", hp(noisy))):
                s, _ = score(x)
                cond[name].append(float(s[:, nat_idx].max()))
                veh[name].append(float(s[:, lab["Vehicle"]].max()))
        for name, v in cond.items():
            v = np.array(v)
            print(f"   {cat:6s} {name:28s} >=0.5: {np.mean(v >= 0.5):.2f}  >=0.3: {np.mean(v >= 0.3):.2f}  >=0.1: {np.mean(v >= 0.1):.2f}"
                  f"   Vehicle {np.mean(veh[name]):.2f}   (n={len(v)})")
            report[f"gap/{cat}/{name}"] = {"ge0.5": float(np.mean(v >= 0.5)), "ge0.3": float(np.mean(v >= 0.3)), "ge0.1": float(np.mean(v >= 0.1))}

    # ---- train head_real on real tune recordings with augmentation ---------------------------------------
    tune = df[(df.split == "tune") & df.category.isin(POS + NEG) & ~df.source.str.startswith("acted")]
    X, Xh, Y = [], [], []
    for r in tune.itertuples():
        clip = read_audio(CLIPS / r.path)[: 8 * SR]
        pos = r.category in POS
        for _ in range(2):
            x = mix(clip, "tune", rng.uniform(0, 8) if pos else rng.uniform(-6, 3))
            a, b = 0.5, 0.5 + min(len(clip) / SR, 6.0)
            for xx, store in ((x, X), (hp(x), Xh)):
                _, e = score(xx)
                store.append(ctx(e))
            n = len(X[-1])
            ends = (np.arange(n) + 1) * 0.5
            ov = np.minimum(ends, b) - np.maximum(ends - 0.975, a)
            y = np.zeros(n, np.int8)
            if pos:
                y[ov > 0] = -1
                y[ov >= min(0.4, 0.6 * (b - a))] = 1
            Y.append(y)
    y = np.concatenate(Y)

    def fit(parts):
        x = np.concatenate(parts)[y >= 0]
        mu, sd = x.mean(0), x.std(0) + 1e-3
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit((x - mu) / sd, y[y >= 0])
        w = clf.coef_[0] / sd
        return w.astype(np.float32), float(clf.intercept_[0] - (mu * w).sum())

    heads = {"head_real": fit(X), "head_real_hp": fit(Xh)}
    print(f"\nhead_real trained on {len(tune)} real tune clips x2 augmentations, {int((y >= 0).sum())} frames, {np.mean(y[y >= 0]):.1%} positive")
    (REPO / "shared" / "head_real.json").write_text(json.dumps(
        {"input": "concat(yamnet embedding of this window, embedding of the previous window) = 2048 values",
         "output": "sigmoid(w.x + b)", "trained_on": "real recordings only (FSD50K, CREMA-D) + car noise augmentation",
         "weights": [round(float(v), 6) for v in heads["head_real"][0]], "bias": round(heads["head_real"][1], 6)}), encoding="utf-8")

    def detectors(ride) -> dict[str, np.ndarray]:
        s, e = ride["scores"], ride["emb"]
        f = RESULTS / "cache" / f"hp_{ride['scn']['id']}.npz"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            sh, eh = det.score_audio(hp(ride["audio"]), with_embeddings=True)
            np.savez_compressed(f, s=sh, e=eh.astype(np.float16))
        z = np.load(f)
        sh, eh = z["s"], z["e"].astype(np.float32)
        sig = lambda v: 1 / (1 + np.exp(-np.clip(v, -30, 30)))  # noqa: E731
        return {"native": s[:, nat_idx].max(1), "native_hp": sh[:, nat_idx].max(1), "head": det.head_series(e),
                "head_real": sig(ctx(e) @ heads["head_real"][0] + heads["head_real"][1]),
                "head_real_hp": sig(ctx(eh) @ heads["head_real_hp"][0] + heads["head_real_hp"][1])}

    # ---- Part 2: isolated real sounds ---------------------------------------------------------------------
    ev = {}
    for split in ("tune", "test"):
        rows = []
        for path in scenario_paths(f"sound_{split}"):
            ride = load_ride(path, det)
            d = detectors(ride)
            for e in ride["scn"]["events"]:
                a, b = int(e["t"] / 0.5), int((e["t"] + e["len_s"] + 1) / 0.5) + 1
                rows.append({"kind": e["kind"], "pos": "expect" in e, **{k: smooth_peak(v, a, b) for k, v in d.items()},
                             "benign": float(ride["scores"][a:b, ben_idx].max())})
        ev[split] = pd.DataFrame(rows)
    names = ["native", "native_hp", "head", "head_real", "head_real_hp"]
    thr = {(n, q): float(np.quantile(ev["tune"][~ev["tune"].pos][n], 1 - q)) for n in names for q in (0.05, 0.10)}
    print("\nPart 2. Isolated real sounds, TEST rides (29 threats, 84 harmless). Threshold set on tune rides.")
    print("        detector        harmless budget 5%: threats / harmless opened     budget 10%: threats / harmless opened")
    t = ev["test"]
    for n in names:
        cells = []
        for q in (0.05, 0.10):
            hit, fa = (t[t.pos][n] >= thr[n, q]).mean(), (t[~t.pos][n] >= thr[n, q]).mean()
            cells.append(f"{hit:.0%} / {fa:.0%}  (thr {thr[n, q]:.2f})")
            report[f"sound/{n}/{q}"] = {"threats": float(hit), "harmless": float(fa), "thr": thr[n, q]}
        k = {kind: f"{(g[n] >= thr[n, 0.10]).mean():.0%}" for kind, g in t.groupby("kind")}
        print(f"        {n:13s}   {cells[0]:34s}   {cells[1]:30s}  by kind at 10%: {k}")

    # ---- Part 3: main test rides ---------------------------------------------------------------------------
    print("\nPart 3. Main test rides (16 incidents: 12 with an acted dialogue, 4 real sound only), same thresholds (10% budget).")
    for n in names:
        acted = real = a_hit = r_hit = false = 0
        hours = 0.0
        for path in scenario_paths("test"):
            ride = load_ride(path, det)
            v = np.convolve(detectors(ride)[n], [0.5, 0.5], mode="same")
            times = (np.flatnonzero(v >= thr[n, 0.10]) + 1) * 0.5
            wins = incidents(ride["scn"])
            hours += ride["scn"]["duration_s"] / 3600
            for w in wins:
                hit = bool(np.any((times >= w["start"] - 1) & (times <= w["end"] + 5)))
                if any(k.startswith("dialog") for k in w["kinds"]):
                    acted, a_hit = acted + 1, a_hit + hit
                else:
                    real, r_hit = real + 1, r_hit + hit
            last = -1e9
            for x in times:
                if not any(w["from"] - 1 <= x <= w["end"] + 10 for w in wins):
                    false += x - last >= 10
                    last = x
        print(f"        {n:13s}  acted dialogue incidents opened {a_hit}/{acted}   real sound {r_hit}/{real}   false opens/h {false / hours:.1f}")
        report[f"main/{n}"] = {"acted": [a_hit, acted], "real": [r_hit, real], "false_opens_per_h": false / hours}

    # ---- Part 4: laughter / cough labels ---------------------------------------------------------------------
    print("\nPart 4. Events the current head opens on the isolated sound rides: is a laughter / cough label present?")
    for split in ("tune", "test"):
        o = ev[split][ev[split]["head"] >= 0.9]
        for q in (0.05, 0.2):
            print(f"        {split}: benign label score >= {q}: {np.mean(o[o.pos].benign >= q):.0%} of {int(o.pos.sum())} threats, "
                  f"{np.mean(o[~o.pos].benign >= q):.0%} of {int((~o.pos).sum())} harmless")
    (RESULTS / "research_sound.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
