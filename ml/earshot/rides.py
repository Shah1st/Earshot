"""Rendered ride cache (audio + GPS + YAMNet frame scores) and the scorer (plan 6.2)."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .common import REPO, RIDES, read_audio, write_wav
from .render import load_route, load_scenario, render_audio, render_gps

SCENARIOS = REPO / "shared" / "scenarios"


def scenario_paths(split: str) -> list[Path]:
    return sorted((SCENARIOS / split).glob("*.json"))


def load_ride(path: Path, detector=None, force: bool = False, asr_cfg: dict | None = None) -> dict:
    """Render once, then reuse: data/rides/<id>.wav, .gps.jsonl and .scores.npy (the slow part)."""
    scn = load_scenario(path)
    route = load_route(scn["route"])
    tag = detector.model.removesuffix(".tflite") if detector is not None else "none"
    wav, gps_f, sc_f, emb_f = (RIDES / f"{scn['id']}{ext}" for ext in (".wav", ".gps.jsonl", f".{tag}.scores.npy", f".{tag}.emb.npy"))
    stamp = RIDES / f"{scn['id']}.json"
    fresh = stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == scn and not force
    if not fresh:
        track = render_gps(scn, route)
        write_wav(wav, render_audio(scn, gps=track))
        gps_f.write_text("\n".join(json.dumps(g) for g in track), encoding="utf-8")
        for f in list(RIDES.glob(f"{scn['id']}.*.npy")) + list(RIDES.glob(f"{scn['id']}.asr*.json")):
            f.unlink()
        stamp.write_text(json.dumps(scn), encoding="utf-8")
    audio = read_audio(wav)
    gps = [json.loads(line) for line in gps_f.read_text(encoding="utf-8").splitlines()]
    scores = emb = None
    if detector is not None:
        if not sc_f.exists():
            if detector.emb_out is not None:
                s, e = detector.score_audio(audio, with_embeddings=True)
                np.save(emb_f, e.astype(np.float16))
            else:
                s = detector.score_audio(audio)
            np.save(sc_f, s)
        scores = np.load(sc_f)
        emb = np.load(emb_f).astype(np.float32) if emb_f.exists() else None
    asr = None
    if asr_cfg is not None:  # continuous Vosk over the ride, cached per keyword list
        from .asr import RecordedAsr
        key = zlib.crc32(json.dumps([asr_cfg["asr"]["model"], asr_cfg["asr"]["keywords"], asr_cfg["asr"]["min_word_conf"]]).encode())
        asr_f = RIDES / f"{scn['id']}.asr{key:08x}.json"
        if not asr_f.exists():
            asr_f.write_text(json.dumps(RecordedAsr.record(asr_cfg, audio)), encoding="utf-8")
        asr = RecordedAsr(json.loads(asr_f.read_text(encoding="utf-8")), asr_cfg["audio"]["chunk_s"])
    return {"scn": scn, "route": route, "audio": audio, "gps": gps, "scores": scores, "emb": emb, "wav": wav, "asr": asr}


def incidents(scn: dict) -> list[dict]:
    """Ground truth threats: sound events sharing an `incident` id -> one time window."""
    out: dict[int, dict] = {}
    for e in scn["events"]:
        if e.get("expect") == "alert":
            start = e["t"] + e.get("alert_from", 0.0)  # dialogues: the threat becomes audible part way in
            w = out.setdefault(e.get("incident", len(out)), {"start": start, "end": start, "from": e["t"], "kinds": []})
            w["from"] = min(w["from"], e["t"])
            w["start"] = min(w["start"], start)
            w["end"] = max(w["end"], e["t"] + e.get("len_s", 0))
            w["kinds"].append(e.get("kind", "?"))
    return list(out.values())


def score_rides(results: list[tuple[dict, list[dict]]], within_s: float = 10.0) -> dict:
    """results: [(scenario, pipeline outputs)]. An incident counts as recalled if an alert is raised
    between its first sound and `within_s` after its last sound ends."""
    hours = sum(s["duration_s"] for s, _ in results) / 3600
    n_inc = hit_alert = hit_any = detected = 0
    false_alerts = false_checkins = gps_checkins = judged_benign = dismissed = llm_calls = tokens = 0
    cost = 0.0
    for scn, outs in results:
        wins = incidents(scn)

        def inside(t, pad=within_s):
            return any(w["from"] - 1 <= t <= w["end"] + pad for w in wins)  # "from": the calm start of a threat dialogue

        decisions = [o for o in outs if o["type"] == "decision"]
        for w in wins:
            near = [o for o in decisions if w["start"] - 1 <= o["t"] <= w["end"] + within_s and o["source"] != "gps"]
            n_inc += 1
            detected += bool(near)
            hit_alert += any(o["decision"] == "alert" for o in near)
            hit_any += any(o["decision"] in ("alert", "check_in") for o in near)
        for o in decisions:
            if o["source"] == "gps":
                gps_checkins += 1
            elif not inside(o["t"]):
                false_alerts += o["decision"] == "alert"
                false_checkins += o["decision"] == "check_in"
                if o["source"] in ("llm", "rule") and o["event"]:
                    judged_benign += 1
                    dismissed += o["decision"] == "dismiss"
        for o in outs:
            if o.get("usage"):
                llm_calls += 1
                tokens += o["usage"]["total_tokens"]
                cost += o["usage"].get("cost") or 0.0
    return {
        "rides": len(results), "ride_hours": round(hours, 2), "incidents": n_inc,
        "recall_alert": round(hit_alert / max(n_inc, 1), 3),
        "recall_alert_or_check_in": round(hit_any / max(n_inc, 1), 3),
        "detector_recall": round(detected / max(n_inc, 1), 3),
        "false_alerts_per_h": round(false_alerts / hours, 2),
        "false_check_ins_per_h": round(false_checkins / hours, 2),
        "gps_check_ins_per_h": round(gps_checkins / hours, 2),
        "benign_events_judged": judged_benign,
        "dismiss_accuracy": round(dismissed / judged_benign, 3) if judged_benign else None,
        "llm_calls_per_h": round(llm_calls / hours, 2),
        "tokens_per_h": round(tokens / hours),
        "cost_usd_per_h": round(cost / hours, 5),
    }
