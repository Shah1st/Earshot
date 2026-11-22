"""Paths, config and audio helpers shared by the whole reference pipeline."""
from __future__ import annotations

import json
import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ML = Path(__file__).resolve().parents[1]
REPO = ML.parent
DATA = ML / "data"
CLIPS = DATA / "clips"  # final clips, already "inside the car"
DRY = DATA / "dry"  # source clips before car acoustics
MCV = DATA / "raw" / "mcvampire"  # Cerence In-Car McVAMPIRE: real car impulse responses + driving noise
MODELS = DATA / "models"
RIDES = DATA / "rides"
RESULTS = ML / "results"
CONFIG_PATH = REPO / "shared" / "config.json"
MANIFEST = DATA / "manifest.csv"
SR = 16000


def load_config(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))


def load_env() -> None:
    """Read KEY=value lines from the repo .env into os.environ (no overwrite)."""
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def openrouter_key() -> str:
    load_env()
    for k in ("OPENROUTER_API_KEY", "OpenRouter_api", "OPENROUTER_API"):
        if os.environ.get(k):
            return os.environ[k]
    raise RuntimeError("No OpenRouter key: put OPENROUTER_API_KEY=... in the repo .env")


def read_audio(path: Path | str, sr: int = SR) -> np.ndarray:
    """Any audio file -> mono float32 at `sr`."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    return x


def write_wav(path: Path | str, x: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def normalise(x: np.ndarray, rms_db: float = -23.0, peak_db: float = -1.0) -> np.ndarray:
    """Bring a clip to a common loudness so scenario gain_db values are comparable."""
    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    x = x * (10 ** (rms_db / 20) / rms)
    peak = float(np.abs(x).max() + 1e-9)
    limit = 10 ** (peak_db / 20)
    return (x * (limit / peak) if peak > limit else x).astype(np.float32)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


PHONE_MIC = 12  # McVAMPIRE microphone 13: roof centre between the front seats and the second row
SEATS = {"driver": [1], "passenger": [3, 4, 5], "front_passenger": [2], "anyone": [2, 3, 4, 5, 6, 7, 8]}
ANGLES = [-45, -30, -15, 0, 15, 30, 45]


def car_ir(who: str, seed: int = 0) -> np.ndarray:
    """A measured impulse response from a seat (mouth simulator) or the car loudspeakers to the phone mic, at 16 kHz."""
    rng = np.random.default_rng(seed)
    if who == "speakers":
        ir = sum(sf.read(str(MCV / "ir" / f"ir_ls{k}.wav"), dtype="float32")[0][:, PHONE_MIC] for k in (1, 2, 3, 4))
    else:
        f = MCV / "ir" / f"ir_mouth_pos{rng.choice(SEATS[who])}_{rng.choice(ANGLES)}.wav"
        ir = sf.read(str(f), dtype="float32")[0][:, PHONE_MIC]
    return resample_poly(ir, 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz


def in_car(x: np.ndarray, who: str = "passenger", seed: int = 0) -> np.ndarray:
    """Place a dry sound inside the car: convolve with a real measured car impulse response."""
    from scipy.signal import fftconvolve

    return fftconvolve(x, car_ir(who, seed)).astype(np.float32)
