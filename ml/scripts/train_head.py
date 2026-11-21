"""v2 head (plan 2.2): logistic regression {distress, normal} on YAMNet's 1,024 d embedding
of the current window concatenated with the previous window's (0.5 s of context).

Training frames come from the dense `train` rides (TUNE clips only, mixed over cabin noise, radio, talk).
A frame is positive when it overlaps an alert sound (scream, shout, impact, glass) or a shouted turn of a
threat dialogue. Calmly spoken threats are NOT positives (their content is the job of Vosk + the LLM), and
harmless shouting (joy, radio commentary) is ignored rather than negative: the head detects "loud distress
like sound", the LLM decides what it means. On device this is one dot product.
Output: shared/head.json  {"weights": [2048], "bias": b}  (the feature scaling is folded in).

    uv run python scripts/train_head.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from earshot.common import REPO, load_config
from earshot.detector import Detector
from earshot.rides import load_ride, scenario_paths

SOUND_KINDS = {"scream", "shout", "impact", "glass"}
WINDOW_S = 0.975


def frame_labels(scn: dict, n: int, chunk_s: float) -> np.ndarray:
    """1 = clearly inside an alert sound, 0 = clearly outside, -1 = edge frame (ignored)."""
    y = np.zeros(n, np.int8)
    ends = (np.arange(n) + 1) * chunk_s
    for e in scn["events"]:
        if "clip" not in e:
            continue
        alert = e.get("expect") == "alert"
        if "loud" in e:  # dialogues and radio lines: only the shouted turns count
            spans = [(e["t"] + a, e["t"] + b) for a, b in e["loud"]]
        elif alert and e.get("kind") in SOUND_KINDS:
            spans = [(e["t"], e["t"] + min(e["len_s"], 6.0))]  # long clips: the tail is often silence
        else:
            continue
        for a, b in spans:
            overlap = np.minimum(ends, b) - np.maximum(ends - WINDOW_S, a)
            y[(overlap > 0) & (y != 1)] = -1
            if alert:  # harmless shouting (joy, radio) stays "ignored": the LLM tells it apart, not the sound head
                y[overlap >= min(0.4, 0.6 * (b - a))] = 1
    return y


def main() -> None:
    cfg = load_config()
    det = Detector({**cfg, "groups": {}}, model="yamnet_tfhub.tflite")
    rides = [load_ride(p, det) for p in scenario_paths("train")]
    X = [np.concatenate([r["emb"], np.vstack([r["emb"][:1], r["emb"][:-1]])], axis=1) for r in rides]  # now + 0.5 s ago
    Y = [frame_labels(r["scn"], len(r["emb"]), cfg["audio"]["chunk_s"]) for r in rides]

    def stack(idx):
        x, y = np.concatenate([X[i] for i in idx]), np.concatenate([Y[i] for i in idx])
        return x[y >= 0], y[y >= 0]

    def fit(x, y):
        mu, sd = x.mean(0), x.std(0) + 1e-3
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit((x - mu) / sd, y)
        w = clf.coef_[0] / sd
        return w.astype(np.float32), float(clf.intercept_[0] - (mu * w).sum())

    hold = list(range(0, len(rides), 5))  # sanity check on held out rides (same clips pool: optimistic)
    xtr, ytr = stack([i for i in range(len(rides)) if i not in hold])
    xho, yho = stack(hold)
    w, b = fit(xtr, ytr)
    p = 1 / (1 + np.exp(-(xho @ w + b)))
    print(f"train frames {len(ytr)} ({ytr.mean():.1%} positive); held out rides: "
          f"AUC {roc_auc_score(yho, p):.3f}, AP {average_precision_score(yho, p):.3f}")

    w, b = fit(*stack(range(len(rides))))
    out = REPO / "shared" / "head.json"
    out.write_text(json.dumps({"input": "concat(yamnet embedding of this window, embedding of the previous window) = 2048 values", "output": "sigmoid(w.x + b) = P(distress)",
                               "weights": [round(float(v), 6) for v in w], "bias": round(b, 6)}), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
