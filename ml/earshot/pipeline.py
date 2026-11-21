"""Streaming reference pipeline. Mirrors what the Android app does, message by message:

    on_gps(t, lat, lon, speed_kmh)      for every {"type":"gps"} message
    on_audio(t, chunk)                  for every {"type":"audio"} message (0.5 s float chunk)

Both return the list of new outputs (decisions).

Three things can open an event (all on device, 0 tokens):
  "sound"    the sound head hears a loud distress like sound (plan 2.3 trigger logic)
  "keyword"  continuous Vosk hears a safety phrase ("stop the car", "let me out", ...), even spoken calmly
  "gps"      a GPS anomaly starts (left the route, sudden stop, long stop) with no sound event around it
An event is sealed 5 s after it opens and sent to the LLM judge as a short text summary: sound labels,
GPS context, the rolling transcript, earlier decisions of this ride. One exception needs no LLM decision:
a very confident sound together with a GPS anomaly alerts immediately (the LLM only writes the summary).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .asr import find_keywords
from .common import SR
from .events import GpsContext, Triggers
from .judge import NoJudge

GPS_FIELDS = ("speed_kmh", "off_route_m", "sudden_stop", "stopped_s")
GPS_FLAGS = {"off_route": "left the planned route", "sudden_stop": "sudden stop", "long_stop": "long stop away from pickup/drop off"}


class RidePipeline:
    def __init__(self, cfg: dict, route: list[list[float]], detector, asr=None, judge=None):
        self.cfg, self.det, self.asr, self.judge = cfg, detector, asr, judge or NoJudge()
        a = cfg["audio"]
        self.chunk_s = a["chunk_s"]
        self.post_s = a["clip_post_s"]
        self.buffer: deque[np.ndarray] = deque(maxlen=int(a["buffer_s"] / self.chunk_s))  # rolling 30 s, RAM only
        self.recent: deque[np.ndarray] = deque(maxlen=cfg["event"]["smooth_windows"])
        self.triggers = Triggers(cfg)
        self.gps = GpsContext(cfg, route)
        self.event: dict | None = None
        self.prev_emb: np.ndarray | None = None
        self.last_trigger = -1e9
        self.merged_alerted = False  # a rule alert was already sent for the current merged event
        self.prev_flags = {k: False for k in GPS_FLAGS}
        self.outputs: list[dict] = []
        self.n_events = 0

    def _open(self, now: float, trigger: str, note: str = "") -> None:
        self.n_events += 1
        self.merged_alerted = False
        self.event = {"id": self.n_events, "t0": now, "trigger": trigger, "note": note,
                      "groups": {g: 0.0 for g in self.cfg["groups"]}, "active": 0,
                      "class_max": np.max(self.recent, axis=0) if self.recent else np.zeros(len(self.det.labels), np.float32)}
        self.last_trigger = now

    # ---- GPS ---------------------------------------------------------------------------------
    def on_gps(self, t: float, lat: float, lon: float, speed_kmh: float) -> list[dict]:
        st = self.gps.update(t, lat, lon, speed_kmh)
        new = []
        for flag, text in GPS_FLAGS.items():
            rising = st[flag] and not self.prev_flags[flag]
            self.prev_flags[flag] = st[flag]
            if rising and self.event is None and t - self.last_trigger >= self.cfg["event"]["cooldown_s"]:
                if self.cfg["gps"].get("judge_anomalies", True) and self.asr is not None:
                    self._open(t, "gps", text)  # the judge reads what was said around it (road works? engine trouble?)
                else:  # plan 2.5 fallback: no words to go on, ask the rider, no LLM call
                    new.append(self._emit(t, "check_in", "gps", 1, text, None, None))
        return new

    # ---- audio -------------------------------------------------------------------------------
    def on_audio(self, t: float, chunk: np.ndarray, scores: np.ndarray | None = None,
                 emb: np.ndarray | None = None) -> list[dict]:
        now = t + self.chunk_s
        self.buffer.append(np.asarray(chunk, np.float32))
        if scores is None:
            tail = np.concatenate(list(self.buffer)[-2:])[-self.det.window:]
            scores = self.det.scores(np.pad(tail, (self.det.window - len(tail), 0)))
            emb = self.det.embedding() if self.det.head else None
        self.recent.append(scores)
        gs = self.det.group_scores(scores, emb, self.prev_emb)
        self.prev_emb = emb
        fired, _ = self.triggers.step(gs)
        heard = self.asr.feed(now, chunk) if self.asr is not None else []
        if not self.cfg["asr"].get("keyword_trigger", True):
            heard = []  # ablation: Vosk still transcribes for the judge, but a phrase cannot open an event
        groups = self.cfg["groups"]
        new = []

        if fired or heard:
            if self.event is None and now - self.last_trigger >= self.cfg["event"]["cooldown_s"]:
                self._open(now, "sound" if fired else "keyword")
            self.last_trigger = now  # cooldown: later triggers merge into the same event
            # clear case: very confident sound + GPS anomaly -> alert now, do not wait for the LLM
            high = [g for g in fired if gs[g] >= groups[g]["high"]]
            if high and self.gps.state["anomaly"] and not self.merged_alerted:
                self.merged_alerted = True
                flags = ", ".join(v for k, v in GPS_FLAGS.items() if self.gps.state[k])
                new.append(self._emit(now, "alert", "rule", 3, f"{high[0]} {gs[high[0]]:.2f} + {flags}",
                                      None, None, event_id=self.n_events))

        ev = self.event
        if ev is not None:
            for g, s in gs.items():
                ev["groups"][g] = max(ev["groups"][g], s)
            ev["class_max"] = np.maximum(ev["class_max"], scores)
            ev["active"] += any(gs[g] >= spec["threshold"] for g, spec in self.triggers.groups.items())
            if now >= ev["t0"] + self.post_s:
                new.append(self._seal(now, ev))
                self.event = None
        return new

    def _seal(self, now: float, ev: dict) -> dict:
        e, a = self.cfg["event"], self.cfg["asr"]
        top = np.argsort(ev["class_max"])[::-1][: e["top_labels"]]
        payload = {
            "t": round(ev["t0"], 1),
            "trigger": ev["trigger"] + (f": {ev['note']}" if ev["note"] else ""),
            "groups": {g: round(v, 2) for g, v in ev["groups"].items() if v >= 0.1},
            "duration_s": round(ev["active"] * self.chunk_s, 1),
            "top_labels": {self.det.labels[i]: round(float(ev["class_max"][i]), 2) for i in top
                           if ev["class_max"][i] >= e["top_label_min_score"]},
            "gps": {k: self.gps.state[k] for k in GPS_FIELDS},
        }
        if self.asr is not None:
            span = a["gps_context_s"] if ev["trigger"] == "gps" else a["context_s"]
            text = self.asr.transcript(now - span, now)
            payload.update(keywords=list(find_keywords(text, a["keywords"])), transcript=text)
        else:
            payload.update(keywords=[], transcript="")
        earlier = [o for o in self.outputs if o["type"] == "decision"][-3:]
        if earlier:
            payload["earlier"] = [{"t": o["t"], "decision": o["decision"], "reason": o["reason"]} for o in earlier]
        res = self.judge(payload)
        if self.merged_alerted:  # already alerted by rule: the LLM only writes the one line summary
            return self._emit(now, "alert", "rule+summary", 3, res["reason"], payload, res["usage"], ev["id"], "summary")
        return self._emit(now, res["decision"], "llm", res["severity"], res["reason"], payload, res["usage"], ev["id"])

    def _emit(self, t, decision, source, severity, reason, event, usage, event_id=None, kind="decision") -> dict:
        out = {"t": round(t, 1), "type": kind, "decision": decision, "source": source, "severity": severity,
               "reason": reason, "event_id": event_id, "event": event, "usage": usage}
        self.outputs.append(out)
        return out


def run_ride(cfg: dict, route, audio: np.ndarray, gps_track: list[dict], detector, asr=None, judge=None,
             scores: np.ndarray | None = None, emb: np.ndarray | None = None) -> list[dict]:
    """Feed a whole rendered ride through the streaming pipeline (GPS first, then audio, per 0.5 s tick)."""
    pipe = RidePipeline(cfg, route, detector, asr, judge)
    hop = int(cfg["audio"]["chunk_s"] * SR)
    for i in range(len(audio) // hop):
        if i < len(gps_track):
            g = gps_track[i]
            pipe.on_gps(g["t"], g["lat"], g["lon"], g["speed_kmh"])
        pipe.on_audio(i * cfg["audio"]["chunk_s"], audio[i * hop:(i + 1) * hop], None if scores is None else scores[i],
                      None if emb is None else emb[i])
    return pipe.outputs
