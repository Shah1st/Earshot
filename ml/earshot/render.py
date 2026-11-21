"""Scenario JSON -> exactly what Unity plays and streams: mixed 16 kHz mono audio + a GPS sample every 0.5 s.

GPS event semantics (Unity must match, or simply replay the pre rendered files from the demo pack):
  "stop"         brake gently at t, stay stopped for `dur` s (default: until the ride ends)
  "sudden_stop"  brake hard at t (40 -> 0 km/h in ~1.3 s), stay stopped `dur` s (default 8), then drive on
  "detour"       at t the car turns off the route and keeps driving away from it (until the ride ends)
  "bypass"       at t the car swings `offset_m` (default 350) to the side of the route and is back on it after
                 `dur` s (default 60): a detour around road works that still reaches the destination
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .common import CLIPS, REPO, SR, read_audio
from .events import EARTH_M, _to_xy

SHARED = REPO / "shared"
DT = 0.5
ACCEL, BRAKE, HARD_BRAKE = 6.0, 6.0, 30.0  # km/h per second


def load_route(rel: str) -> list[list[float]]:
    """GeoJSON LineString ([lon, lat]) -> [[lat, lon], ...]."""
    gj = json.loads((SHARED / rel).read_text(encoding="utf-8"))
    geom = gj["features"][0]["geometry"] if "features" in gj else gj.get("geometry", gj)
    return [[lat, lon] for lon, lat in geom["coordinates"]]


def load_scenario(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fade(x: np.ndarray, n: int = 800) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n:
        x = x.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def render_audio(scn: dict, clips_dir: Path = CLIPS, gps: list[dict] | None = None) -> np.ndarray:
    n = int(scn["duration_s"] * SR)
    mix = np.zeros(n, np.float32)

    def bed(clip: str, gain_db: float, a: int, b: int) -> np.ndarray:
        x = read_audio(clips_dir / clip)
        x = np.tile(_fade(x), int(np.ceil((b - a) / len(x))))
        return x[: b - a] * 10 ** (gain_db / 20)

    for bg in scn.get("background", []):
        a, b = int(bg.get("from", 0) * SR), int(min(bg.get("to", scn["duration_s"]), scn["duration_s"]) * SR)
        x = bed(bg["clip"], bg.get("gain_db", 0), a, b)
        if "idle_clip" in bg and gps:
            # driving noise follows the car: full above 20 km/h, cross faded to the idle clip (engine ticking over,
            # or nearly silent if idle_gain_db is very low = engine off) as the car stops
            speed = np.interp(np.arange(a, b) / SR, [g["t"] for g in gps], [g["speed_kmh"] for g in gps])
            w = np.clip(speed / 20.0, 0, 1).astype(np.float32)
            x = x * w + bed(bg["idle_clip"], bg.get("idle_gain_db", 0), a, b) * (1 - w)
        x = _fade(x, SR // 2)
        mix[a:a + len(x)] += x
    for ev in scn.get("events", []):
        if "clip" not in ev:
            continue
        x = _fade(read_audio(clips_dir / ev["clip"])) * 10 ** (ev.get("gain_db", 0) / 20)
        a = int(ev["t"] * SR)
        x = x[: max(0, n - a)]
        mix[a:a + len(x)] += x
    end = min(n, SR)  # one second fade out so the ride does not end with a click
    mix[n - end:] *= np.linspace(1, 0, end, dtype=np.float32)
    return np.clip(mix, -1.0, 1.0)


def _dist(xy: np.ndarray, p: np.ndarray) -> float:
    """Distance from point p to the route polyline xy (metres)."""
    a, ab = xy[:-1], xy[1:] - xy[:-1]
    u = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    return float(np.linalg.norm(a + u[:, None] * ab - p, axis=1).min())


def render_gps(scn: dict, route: list[list[float]]) -> list[dict]:
    """One sample per 0.5 s: {"t", "lat", "lon", "speed_kmh"}."""
    origin = np.asarray(route[0], float)
    xy = _to_xy(np.asarray(route, float), origin)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    rng = np.random.default_rng(zlib.crc32(scn["id"].encode()))
    gps_events = [e for e in scn.get("events", []) if "gps" in e]
    cruise = scn.get("speed_kmh", 40)

    def along(s: float) -> tuple[np.ndarray, np.ndarray]:
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        d = (xy[i + 1] - xy[i]) / max(seg[i], 1e-9)
        return xy[i] + d * (s - cum[i]), d

    out, v, s, detour_s, detour_dir = [], 0.0, 0.0, None, None
    for k in range(int(scn["duration_s"] / DT)):
        t = k * DT
        target, brake = cruise, BRAKE
        for e in gps_events:
            dur = e.get("dur", 8 if e["gps"] == "sudden_stop" else 1e9)
            if e["gps"] in ("stop", "sudden_stop") and e["t"] <= t < e["t"] + dur:
                target, brake = 0.0, HARD_BRAKE if e["gps"] == "sudden_stop" else BRAKE
            if e["gps"] == "detour" and t >= e["t"] and detour_s is None:
                detour_s = s
                # leave towards open ground: of 8 headings, the one that ends up farthest from the whole route
                p0, d = along(s)
                turn = lambda a: np.array([d[0] * np.cos(a) - d[1] * np.sin(a), d[0] * np.sin(a) + d[1] * np.cos(a)])  # noqa: E731
                far = lambda v: min(_dist(xy, p0 + v * r) for r in (300, 600, 1000))  # noqa: E731
                detour_dir = max((turn(a) for a in np.radians([90, -90, 60, -60, 120, -120, 45, -45])), key=far)
        if detour_s is None and s >= cum[-1]:
            target = 0.0  # arrived
        v = min(target, v + ACCEL * DT) if v < target else max(target, v - brake * DT)
        s += v / 3.6 * DT
        if detour_s is None:
            p, d = along(min(s, cum[-1]))
            for e in gps_events:
                if e["gps"] == "bypass" and e["t"] <= t < e["t"] + e.get("dur", 60):
                    side = e.get("offset_m", 350) * np.sin(np.pi * (t - e["t"]) / e.get("dur", 60)) ** 2
                    p = p + np.array([-d[1], d[0]]) * side
        else:
            p0, _ = along(detour_s)
            p = p0 + detour_dir * (s - detour_s)
        p = p + rng.normal(0, 2.0, 2)  # ~2 m GPS jitter
        lat = origin[0] + p[1] / EARTH_M
        lon = origin[1] + p[0] / (EARTH_M * np.cos(np.radians(origin[0])))
        out.append({"t": t, "lat": round(float(lat), 6), "lon": round(float(lon), 6),
                    "speed_kmh": round(float(max(0.0, v + (rng.normal(0, 0.5) if v > 1 else 0))), 1)})
    return out


def render(scn: dict, clips_dir: Path = CLIPS) -> tuple[np.ndarray, list[dict], list[list[float]]]:
    route = load_route(scn["route"])
    gps = render_gps(scn, route)
    return render_audio(scn, clips_dir, gps), gps, route
