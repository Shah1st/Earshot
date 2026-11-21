"""Real recordings that make rides sound like a car (no API key needed).

* DEMAND `TCAR` (CC BY 4.0): recorded inside a moving passenger car -> cabin background, 30 s segments.
  Split by time: the first 210 s are tune, the last 90 s are test.
* CREMA-D (ODbL): actors saying "It's eleven o'clock" angry / fearful at HIGH intensity -> real shouted
  speech (alert), and neutral / happy takes -> real talk (negative). Split by actor.

    uv run python scripts/fetch_real.py
"""
from __future__ import annotations

import csv
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

from earshot.common import DATA, MCV, DRY as CLIPS, SR, normalise, read_audio, write_wav

DEMAND_URL = "https://zenodo.org/records/1227121/files/TCAR_16k.zip?download=1"
CREMA_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/{}.wav"
CREMA = {"IEO_ANG_HI": ("shout", "alert"), "IEO_FEA_HI": ("shout", "alert"),
         "IEO_NEU_XX": ("talk", "negative"), "IEO_HAP_HI": ("talk", "negative"), "DFA_NEU_XX": ("talk", "negative")}
TEXT = {"IEO": "It's eleven o'clock.", "DFA": "Don't forget a jacket."}
FIELDS = ["path", "category", "role", "source", "source_id", "licence", "uploader", "title", "text", "labels", "split"]


def demand() -> list[dict]:
    raw = DATA / "raw" / "demand"
    if not (raw / "TCAR").exists():
        raw.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(requests.get(DEMAND_URL, timeout=600).content)).extractall(raw)
    rows = []
    # McVAMPIRE driving noise: recorded in a real car at fixed speeds. Tune and test use different microphones.
    ref = None
    for speed in (100, 0, 30, 40, 50, 60):
        x, fs = sf.read(str(MCV / "driving_noise" / f"{speed}kph.wav"), dtype="float32")
        for mic, split in ((12, "tune"), (4, "tune"), (13, "test")):
            y = resample_poly(x[:, mic], 2, 3).astype(np.float32)
            # level anchor from the dataset documentation: normal speech is ~0 dB SNR at 100 km/h
            ref = ref or 10 ** (-23 / 20) / float(np.sqrt(np.mean(y**2)))
            rel = f"cabin/drive_{speed:03d}kph_mic{mic + 1:02d}.wav"
            write_wav(CLIPS / rel, y * ref)
            rows.append({"path": rel, "category": "cabin", "role": "background", "source": "McVAMPIRE",
                         "source_id": f"noise_{speed}kph_mic{mic + 1}", "licence": "CC BY 4.0", "uploader": "Cerence (Mueller, Wolff)",
                         "title": f"driving noise {speed} km/h", "labels": str(speed), "split": split})
    for ch in (1, 9):
        x = read_audio(raw / "TCAR" / f"ch{ch:02d}.wav")
        for seg in range(len(x) // (30 * SR)):
            rel = f"cabin/tcar_ch{ch:02d}_{seg:02d}.wav"
            write_wav(CLIPS / rel, normalise(x[seg * 30 * SR:(seg + 1) * 30 * SR], rms_db=-40))
            rows.append({"path": rel, "category": "cabin", "role": "background", "source": "DEMAND",
                         "source_id": f"TCAR_ch{ch:02d}_{seg:02d}", "licence": "CC BY 4.0", "uploader": "DEMAND (Thiemann et al.)",
                         "title": "inside a private passenger vehicle", "split": "tune" if seg < 7 else "test"})
    return rows


def crema() -> list[dict]:
    def one(job):
        actor, code = job
        cat, role = CREMA[code]
        name = f"{actor}_{code}"
        out = CLIPS / cat / f"crema_{name}.wav"
        if not out.exists():
            r = requests.get(CREMA_URL.format(name), timeout=60)
            if r.status_code != 200 or r.content[:4] != b"RIFF":
                return None
            tmp = out.with_suffix(".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(r.content)
            write_wav(out, normalise(read_audio(tmp)))
            tmp.unlink()
        return {"path": out.relative_to(CLIPS).as_posix(), "category": cat, "role": role, "source": "CREMA-D",
                "source_id": f"crema_actor_{actor}", "licence": "ODbL 1.0", "uploader": "CREMA-D", "title": code,
                "text": TEXT[code[:3]], "labels": "Shout" if role == "alert" else "Speech", "split": ""}

    jobs = [(a, c) for a in range(1001, 1092) for c in CREMA]
    with ThreadPoolExecutor(8) as ex:
        return [r for r in ex.map(one, jobs) if r]


def main() -> None:
    rows = demand() + crema()
    with open(DATA / "manifest_real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"done: {sum(r['source'] == 'DEMAND' for r in rows)} cabin segments, {sum(r['source'] == 'CREMA-D' for r in rows)} CREMA-D clips")


if __name__ == "__main__":
    main()
