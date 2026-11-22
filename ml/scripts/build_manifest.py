"""Put every dry clip "inside the car", freeze the tune/test split, fetch routes, write data/manifest.csv.

* Car acoustics: each clip is convolved with a real impulse response measured in a car (McVAMPIRE),
  from a seat for voices and cabin sounds, from the car loudspeakers for radio. Horns stay outside (muffled).
* Split: by source file, 2/3 tune and 1/3 test, from a hash of the source id unless the source fixed it.
* Radio clips are derived inside one split only (commentary line, optional crowd bed, radio band, car speakers).

    uv run python scripts/build_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zlib

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from scipy.signal import butter, sosfilt

from earshot.common import CLIPS, DATA, DRY, MANIFEST, REPO, SR, in_car, normalise, read_audio, write_wav

ROUTES = {  # name -> (start lat, lon), (end lat, lon)
    "epfl_to_gare": ((46.5191, 6.5668), (46.5167, 6.6291)),
    "gare_to_ouchy_loop": ((46.5167, 6.6291), (46.5370, 6.5950)),
    "renens_to_flon": ((46.5372, 6.5790), (46.5208, 6.6300)),
}
IN_CABIN = {"scream", "shout", "talk", "laughter", "cough", "impact", "glass", "door"}  # get a seat impulse response
INGREDIENTS = {"music", "crowd", "lines_radio"}  # demo clips (category "demo") are simply copied  # only used to build radio clips
SPORT = ("r_goal", "r_save", "r_penalty")


def split_of(source_id: str) -> str:
    return "test" if int(hashlib.md5(source_id.encode()).hexdigest(), 16) % 3 == 0 else "tune"


def band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def seed_of(s: str) -> int:
    return zlib.crc32(s.encode())


def fetch_routes() -> None:
    out = REPO / "shared" / "routes"
    out.mkdir(parents=True, exist_ok=True)
    for name, ((la1, lo1), (la2, lo2)) in ROUTES.items():
        f = out / f"{name}.geojson"
        if f.exists():
            continue
        try:
            r = requests.get(f"https://router.project-osrm.org/route/v1/driving/{lo1},{la1};{lo2},{la2}",
                             params={"overview": "full", "geometries": "geojson"}, timeout=30)
            geom = r.json()["routes"][0]["geometry"]
        except Exception as e:  # offline: straight line, still a valid route
            print(f"OSRM failed for {name} ({e}); using a straight line")
            geom = {"type": "LineString", "coordinates": [[lo1, la1], [lo2, la2]]}
        f.write_text(json.dumps({"type": "Feature", "properties": {"name": name}, "geometry": geom}), encoding="utf-8")
        print(f"route {name}: {len(geom['coordinates'])} points")


def main() -> None:
    fetch_routes()
    df = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(DATA.glob("manifest_*.csv"))], ignore_index=True).fillna("")
    if (df.source != "FSD50K").any():  # real in-car recordings replace the Freesound "car" clips
        df = df[~((df.category == "cabin") & (df.source == "FSD50K"))]
    df["split"] = [s or split_of(i) for s, i in zip(df["split"], df["source_id"])]
    df = df.reset_index(drop=True)

    shutil.rmtree(CLIPS, ignore_errors=True)
    for r in df.itertuples():
        if r.category in INGREDIENTS:
            continue
        x = read_audio(DRY / r.path)
        if r.category in IN_CABIN and r.fx != "done":
            x = normalise(in_car(x, "anyone", seed_of(r.source_id)))
        elif r.category == "horn":
            x = normalise(band(x, 80, 1800))  # heard through the closed car body
        write_wav(CLIPS / r.path, x)

    derived = []
    for split in ("tune", "test"):
        part = df[df.split == split]
        crowd = part[part.category == "crowd"].reset_index(drop=True)
        for i, row in enumerate(part[part.category == "lines_radio"].itertuples()):
            x = read_audio(DRY / row.path)
            lic, src = row.licence, row.source_id
            if row.source_id in SPORT:  # stadium under the commentator
                c = crowd.iloc[i % len(crowd)]
                bed = read_audio(DRY / c.path)
                bed = np.tile(bed, int(np.ceil((len(x) + SR) / len(bed))))[: len(x) + SR] * 0.5
                bed[: len(x)] += x
                x, lic, src = bed, f"{row.licence} + {c.licence}", f"{row.source_id}+fsd_{c.source_id}"
            rel = f"radio_talk/{split}_{i:02d}.wav"
            write_wav(CLIPS / rel, normalise(in_car(band(x, 200, 4500), "speakers")))
            derived.append({**row._asdict(), "path": rel, "category": "radio_talk", "source": "derived", "source_id": src,
                            "licence": lic, "fx": "done"})
        for i, row in enumerate(part[part.category == "music"].head(12).itertuples()):
            rel = f"radio_music/{split}_{i:02d}.wav"
            write_wav(CLIPS / rel, normalise(in_car(band(read_audio(DRY / row.path), 100, 6000), "speakers")))
            derived.append({**row._asdict(), "path": rel, "category": "radio_music", "role": "background",
                            "source": "derived", "fx": "done"})
    df = pd.concat([df[~df.category.isin(INGREDIENTS)], pd.DataFrame(derived).drop(columns="Index")], ignore_index=True)
    df["duration_s"] = [round(sf.info(str(CLIPS / p)).duration, 2) for p in df.path]
    df.to_csv(MANIFEST, index=False)
    print(df.groupby(["category", "split"]).size().unstack(fill_value=0))
    print(f"\n{len(df)} clips -> {MANIFEST}")


if __name__ == "__main__":
    main()
