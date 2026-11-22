"""Download the sound clips we need from FSD50K (per clip, no API key) and convert to 16 kHz mono.

FSD50K is Freesound audio with per clip Creative Commons licences; only CC0 and CC-BY clips are
taken. Output (dry, before car acoustics): data/dry/<category>/fsd_<id>.wav and data/manifest_fsd.csv.

    uv run python scripts/fetch_data.py [--per-cat 45]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download, list_repo_tree

from earshot.common import DATA, DRY as CLIPS, SR, normalise, read_audio, write_wav

REPO = "Fhrozen/FSD50k"
BYTES_PER_S = 88200  # FSD50K clips are 44.1 kHz mono PCM16

# category -> (role, FSD50K labels to take, labels that disqualify a clip, (min_s, max_s))
MUSICAL = {"Music", "Musical_instrument", "Singing"}
CATEGORIES = {
    "scream": ("alert", ["Screaming"], MUSICAL, (0.7, 12)),
    "shout": ("alert", ["Shout", "Yell"], MUSICAL | {"Screaming"}, (0.7, 12)),
    "impact": ("alert", ["Thump_and_thud"], MUSICAL, (0.3, 8)),
    "glass": ("alert", ["Shatter"], MUSICAL, (0.3, 8)),
    "laughter": ("negative", ["Laughter", "Giggle", "Chuckle_and_chortle"], MUSICAL, (1, 12)),
    "cough": ("negative", ["Cough", "Sneeze"], MUSICAL, (0.5, 8)),
    "talk": ("negative", ["Conversation", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
                          "Child_speech_and_kid_speaking"], MUSICAL | {"Screaming", "Shout", "Yell"}, (2, 15)),
    "music": ("negative", ["Music"], {"Screaming", "Shout", "Yell"}, (8, 30)),
    "crowd": ("negative", ["Cheering", "Crowd", "Applause"], {"Screaming"}, (3, 20)),
    "door": ("negative", ["Slam", "Knock", "Door"], MUSICAL, (0.3, 8)),
    "horn": ("negative", ["Vehicle_horn_and_car_horn_and_honking", "Siren", "Ringtone"], MUSICAL, (1, 10)),
    "cabin": ("background", ["Car", "Idling", "Engine"], MUSICAL | {"Vehicle_horn_and_car_horn_and_honking",
                                                                   "Race_car_and_auto_racing", "Speech", "Siren"}, (12, 30)),
}
OK_LICENCES = ("publicdomain/zero", "licenses/by/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=45)
    args = ap.parse_args()
    rng = random.Random(0)

    def hub(name):
        return hf_hub_download(REPO, name, repo_type="dataset")

    labels, info, sizes = {}, {}, {}
    for split in ("dev", "eval"):
        for r in csv.DictReader(open(hub(f"labels/{split}.csv"), encoding="utf-8")):
            labels[r["fname"]] = (split, set(r["labels"].split(",")))
        info.update(json.load(open(hub(f"metadata/{split}_clips_info_FSD50K.json"), encoding="utf-8")))
        for f in list_repo_tree(REPO, path_in_repo=f"clips/{split}", repo_type="dataset"):
            sizes[f.path.split("/")[-1][:-4]] = f.size
    print(f"indexed {len(labels)} clips")

    picked, used = [], set()
    for cat, (role, want, ban, (lo, hi)) in CATEGORIES.items():
        pool = [
            f for f, (_, ls) in sorted(labels.items())
            if ls & set(want) and not ls & ban and f not in used
            and lo <= sizes.get(f, 0) / BYTES_PER_S <= hi
            and any(k in info.get(f, {}).get("license", "") for k in OK_LICENCES)
        ]
        rng.shuffle(pool)
        for f in pool[: args.per_cat]:
            used.add(f)
            picked.append((cat, role, f))
        print(f"{cat:9s} {role:10s} pool={len(pool):4d} take={min(len(pool), args.per_cat)}")

    def fetch(item):
        cat, role, f = item
        out = CLIPS / cat / f"fsd_{f}.wav"
        if not out.exists():
            x = read_audio(hub(f"clips/{labels[f][0]}/{f}.wav"))
            x = normalise(x[: SR * 30])
            write_wav(out, x)
        m = info[f]
        return {"path": out.relative_to(CLIPS).as_posix(), "category": cat, "role": role, "source": "FSD50K",
                "source_id": f, "licence": m["license"], "uploader": m["uploader"], "title": m["title"],
                "text": "", "labels": "|".join(sorted(labels[f][1]))}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(fetch, picked))
    with open(DATA / "manifest_fsd.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} clips")


if __name__ == "__main__":
    main()
