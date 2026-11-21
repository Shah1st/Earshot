"""Trigger logic (plan 2.3) and GPS context rules (plan 2.4). No ML here."""
from __future__ import annotations

from collections import deque

import numpy as np

EARTH_M = 111_320.0


class Triggers:
    """Per group: sustained = moving average above threshold for N consecutive frames,
    impulsive = one frame above threshold."""

    def __init__(self, cfg: dict):
        self.groups = {g: s for g, s in cfg["groups"].items() if s.get("trigger", True)}
        self.smooth = cfg["event"]["smooth_windows"]
        self.sustain = cfg["event"]["sustain_windows"]
        self.hist = {g: deque(maxlen=self.smooth) for g in self.groups}
        self.run = {g: 0 for g in self.groups}

    def step(self, group_scores: dict[str, float]) -> tuple[list[str], dict[str, float]]:
        """Returns (groups that fire on this frame, the value compared to the threshold per group)."""
        fired, values = [], {}
        for g, spec in self.groups.items():
            s = group_scores[g]
            if spec["type"] == "sustained":
                self.hist[g].append(s)
                v = sum(self.hist[g]) / len(self.hist[g])
                self.run[g] = self.run[g] + 1 if v >= spec["threshold"] else 0
                if self.run[g] >= self.sustain:
                    fired.append(g)
            else:
                v = s
                if v >= spec["threshold"]:
                    fired.append(g)
            values[g] = v
        return fired, values


def _to_xy(latlon: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Local metres (equirectangular), good enough at city scale."""
    lat0 = np.radians(origin[0])
    return np.stack([(latlon[..., 1] - origin[1]) * EARTH_M * np.cos(lat0),
                     (latlon[..., 0] - origin[0]) * EARTH_M], axis=-1)


class GpsContext:
    def __init__(self, cfg: dict, route: list[list[float]]):
        self.c = cfg["gps"]
        self.origin = np.asarray(route[0], float)
        self.route = _to_xy(np.asarray(route, float), self.origin)  # [n, 2] metres
        self.speeds: deque[tuple[float, float]] = deque()
        self.off_since: float | None = None
        self.stop_since: float | None = None
        self.sudden_until = -1.0
        self.state = {"speed_kmh": 0.0, "off_route_m": 0.0, "sudden_stop": False, "stopped_s": 0.0,
                      "off_route": False, "long_stop": False, "anomaly": False}

    def _dist_to_route(self, p: np.ndarray) -> float:
        a, b = self.route[:-1], self.route[1:]
        ab = b - a
        u = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return float(np.linalg.norm(a + u[:, None] * ab - p, axis=1).min())

    def update(self, t: float, lat: float, lon: float, speed_kmh: float) -> dict:
        c = self.c
        p = _to_xy(np.asarray([lat, lon], float), self.origin)
        off_m = self._dist_to_route(p)

        self.off_since = (self.off_since if self.off_since is not None else t) if off_m > c["off_route_m"] else None
        off_route = self.off_since is not None and t - self.off_since >= c["off_route_hold_s"]

        self.speeds.append((t, speed_kmh))
        while self.speeds and t - self.speeds[0][0] > c["sudden_stop_window_s"]:
            self.speeds.popleft()
        if max(s for _, s in self.speeds) - speed_kmh > c["sudden_stop_drop_kmh"]:
            self.sudden_until = t + c["sudden_stop_latch_s"]
        sudden = t <= self.sudden_until

        self.stop_since = (self.stop_since if self.stop_since is not None else t) if speed_kmh < c["stopped_speed_kmh"] else None
        stopped_s = t - self.stop_since if self.stop_since is not None else 0.0
        near_end = min(np.linalg.norm(p - self.route[0]), np.linalg.norm(p - self.route[-1])) <= c["endpoint_radius_m"]
        long_stop = stopped_s > c["stopped_hold_s"] and not near_end

        self.state = {"speed_kmh": round(speed_kmh, 1), "off_route_m": round(off_m), "sudden_stop": sudden,
                      "stopped_s": round(stopped_s, 1), "off_route": off_route, "long_stop": long_stop,
                      "anomaly": off_route or sudden or long_stop}
        return self.state
