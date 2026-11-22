"""Score every clip on its own with YAMNet: peak group scores and top labels -> data/clip_scores.csv.

Used to see which categories the native classes separate, and to pick strong clips for demo rides.

    uv run python scripts/clip_scores.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from earshot.common import CLIPS, DATA, MANIFEST, SR, load_config, read_audio
from earshot.detector import Detector


def main() -> None:
    cfg = load_config()
    det = Detector(cfg)
    df = pd.read_csv(MANIFEST).fillna("")
    rows = []
    for r in df.itertuples():
        x = read_audio(CLIPS / r.path)
        x = np.concatenate([x, np.zeros((-len(x)) % (SR // 2) + SR // 2, np.float32)])
        s = det.score_audio(x)
        peak = s.max(axis=0)
        top = np.argsort(peak)[::-1][:3]
        rows.append({"path": r.path, "category": r.category, "role": r.role, "split": r.split,
                     **{g: round(float(peak[idx].max()), 3) for g, idx in det.groups.items()},
                     "top": " | ".join(f"{det.labels[i]} {peak[i]:.2f}" for i in top)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "clip_scores.csv", index=False)
    groups = list(det.groups)
    thr = {g: cfg["groups"][g]["threshold"] for g in groups}
    print("share of clips whose peak score is above the current threshold (isolated clips, tune split):")
    tune = out[out.split == "tune"]
    print((tune[groups] >= pd.Series(thr)).groupby(tune.category).mean().round(2).to_string())


if __name__ == "__main__":
    main()
