"""Download more real FSD50K clips, used ONLY to train the real-data sound head (train_real_head.py).

Kept apart from data/manifest.csv on purpose: the evaluation rides and their frozen split are not touched.
Clips already in the manifest are skipped. Same split rule (hash of the Freesound id, 2/3 tune, 1/3 test);
only the tune part is used for training. Only CC0 / CC BY clips.
Output: data/extra/<group>/fsd_<id>.wav (dry, 16 kHz mono) and data/extra.csv

    uv run python scripts/fetch_extra.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree

from earshot.common import DATA, MANIFEST, SR, normalise, read_audio, write_wav

REPO = "Fhrozen/FSD50k"
OUT = DATA / "extra"
BYTES_PER_S = 88200
MUSICAL = {"Music", "Musical_instrument", "Singing"}
LOUD = {"Screaming", "Shout", "Yell"}
# group -> (positive?, FSD50K labels, disqualifying labels, (min_s, max_s), max clips)
GROUPS = {
    "scream": (True, ["Screaming"], MUSICAL, (0.5, 15), 400),
    "shout": (True, ["Shout", "Yell"], MUSICAL | {"Screaming"}, (0.5, 15), 500),
    "glass": (True, ["Shatter"], MUSICAL, (0.3, 10), 400),
    "speech": (False, ["Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
                       "Conversation", "Speech"], MUSICAL | LOUD, (2, 20), 700),
    "laughter": (False, ["Laughter", "Giggle", "Chuckle_and_chortle"], MUSICAL | LOUD, (1, 12), 350),
    "cough": (False, ["Cough", "Sneeze", "Breathing", "Sigh"], MUSICAL | LOUD, (0.5, 10), 300),
    "door": (False, ["Slam", "Knock", "Door", "Thump_and_thud"], MUSICAL | LOUD | {"Shatter"}, (0.3, 8), 300),
    "street": (False, ["Vehicle_horn_and_car_horn_and_honking", "Siren", "Ringtone", "Telephone", "Bark"], MUSICAL | LOUD, (1, 12), 300),
    "everyday": (False, ["Keys_jangling", "Zipper_(clothing)", "Crumpling_and_crinkling", "Coin_(dropping)", "Clapping",
                         "Chink_and_clink", "Tearing", "Packing_tape_and_duct_tape", "Camera", "Tap"], MUSICAL | LOUD | {"Shatter"}, (0.3, 10), 300),
    "media": (False, ["Music", "Singing", "Cheering", "Crowd", "Applause"], LOUD, (4, 25), 400),
}
OK_LICENCES = ("publicdomain/zero", "licenses/by/")


def split_of(source_id: str) -> str:
    return "test" if int(hashlib.md5(source_id.encode()).hexdigest(), 16) % 3 == 0 else "tune"


def main() -> None:
    rng = random.Random(1)
    hub = lambda name: hf_hub_download(REPO, name, repo_type="dataset")  # noqa: E731
    labels, info, sizes = {}, {}, {}
    for split in ("dev", "eval"):
        for r in csv.DictReader(open(hub(f"labels/{split}.csv"), encoding="utf-8")):
            labels[r["fname"]] = (split, set(r["labels"].split(",")))
        info.update(json.load(open(hub(f"metadata/{split}_clips_info_FSD50K.json"), encoding="utf-8")))
        for f in list_repo_tree(REPO, path_in_repo=f"clips/{split}", repo_type="dataset"):
            sizes[f.path.split("/")[-1][:-4]] = f.size
    man = pd.read_csv(MANIFEST, dtype=str).fillna("")
    used = set(man[man.source == "FSD50K"].source_id) | {i for s in man[man.source == "derived"].source_id for i in s.replace("fsd_", "").split("+")}
    picked = []
    for group, (pos, want, ban, (lo, hi), cap) in GROUPS.items():
        pool = [f for f, (_, ls) in sorted(labels.items())
                if ls & set(want) and not ls & ban and f not in used and split_of(f) == "tune"
                and lo <= sizes.get(f, 0) / BYTES_PER_S <= hi
                and any(k in info.get(f, {}).get("license", "") for k in OK_LICENCES)]
        rng.shuffle(pool)
        for f in pool[:cap]:
            used.add(f)
            picked.append((group, pos, f))
        print(f"{group:9s} {'positive' if pos else 'negative'}  pool={len(pool):4d} take={min(len(pool), cap)}")

    def fetch(item):
        group, pos, f = item
        out = OUT / group / f"fsd_{f}.wav"
        try:
            if not out.exists():
                write_wav(out, normalise(read_audio(hub(f"clips/{labels[f][0]}/{f}.wav"))[: SR * 20]))
        except Exception as e:  # one broken download must not stop the rest
            print(f"skip {f}: {e}")
            return None
        return {"path": out.relative_to(OUT).as_posix(), "group": group, "positive": int(pos), "source_id": f,
                "licence": info[f]["license"], "uploader": info[f]["uploader"], "labels": "|".join(sorted(labels[f][1]))}

    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(fetch, picked) if r]
    pd.DataFrame(rows).to_csv(DATA / "extra.csv", index=False)
    print(f"done: {len(rows)} clips ({sum(r['positive'] for r in rows)} positive)")


if __name__ == "__main__":
    main()
