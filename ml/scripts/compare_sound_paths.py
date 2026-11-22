"""What should happen when the sound model fires and there are no words to read?

Focused rides: real driving noise plus one isolated sound every ~20 s. Positives are real recorded screams and
breaking glass (FSD50K); negatives are real laughter, coughs, doors, horns and short neutral speech. No dialogues.
For every event the sound head opens, three ways of deciding are compared on the same event:

  A  text judge   what the pipeline does now: labels + GPS + transcript as text
  B  audio judge  the last 10 s of audio goes to the audio LLM (audio leaves the phone for that clip)
  C  rule         no LLM: check in with the rider, unless the top sound labels are laughter / cough like
                  (the label list is fixed here, before looking at the test rides)

Rides are built from the tune split and the test split separately; report the test numbers.

    uv run python scripts/compare_sound_paths.py
"""
from __future__ import annotations

import base64
import io
import json
import random

import pandas as pd
import soundfile as sf

from earshot.common import MANIFEST, REPO, RESULTS, SR, load_config
from earshot.detector import Detector
from earshot.judge import Judge
from earshot.llm import chat, parse_json, spent_usd
from earshot.pipeline import RidePipeline
from earshot.rides import load_ride, scenario_paths

SCN = REPO / "shared" / "scenarios"
POSITIVE = {"scream": (2, 8), "glass": (0, 6)}  # category -> gain range dB (same as the evaluation rides)
NEGATIVE = {"laughter": (-4, 2), "cough": (-4, 2), "door": (-6, 0), "horn": (-6, 0), "talk": (-3, 2)}
NOT_HUMAN = ("Animal", "Bird", "Applause", "Clapping", "Fireworks", "Explosion")
BENIGN_LABELS = {"Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh", "Baby laughter",
                 "Cough", "Throat clearing", "Sneeze"}
CLIP_S = 10
AUDIO_PROMPT = ("You judge a safety event detected in a taxi by an on-device sound model. Listen to the clip "
                "(the last 10 seconds before now). Trip context: {ctx}\n"
                "Decide one of:\n"
                "- \"dismiss\": normal ride sound (talk, laughing, coughing, radio, music, a door, a horn outside).\n"
                "- \"check_in\": unclear, ask the rider discreetly if all is fine.\n"
                "- \"alert\": likely threat to someone's safety (a scream, a struggle, breaking glass, calls for help).\n"
                "Only judge safety. Reply JSON: {{\"decision\": \"...\", \"reason\": \"<max 15 words>\"}}")


def make_rides(split: str) -> None:
    out = SCN / f"sound_{split}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    df = pd.read_csv(MANIFEST, dtype={"labels": str}).fillna("")
    df = df[(df.split == split) & ~((df.role == "alert") & df.labels.str.contains("|".join(NOT_HUMAN)))]
    rng = random.Random(f"sound-{split}")
    clips = []
    for cat in list(POSITIVE) + list(NEGATIVE):
        rows = df[df.category == cat].to_dict("records")
        rng.shuffle(rows)
        clips += rows[: 30 if cat == "talk" else len(rows)]  # every real sound clip once; a sample of the speech
    rng.shuffle(clips)
    cabins = [c for c in df[df.category == "cabin"].to_dict("records") if c["source"] == "McVAMPIRE" and c["labels"] in ("40", "50")]
    per_ride = 14
    for k in range(0, len(clips), per_ride):
        events, t = [], 8.0
        for c in clips[k:k + per_ride]:
            pos = c["category"] in POSITIVE
            gain = rng.uniform(*(POSITIVE if pos else NEGATIVE)[c["category"]])
            events.append({"t": round(t, 1), "clip": c["path"], "gain_db": round(gain, 1), "kind": c["category"],
                           "len_s": float(c["duration_s"]), **({"expect": "alert", "incident": len(events)} if pos else {})})
            t += max(20.0, float(c["duration_s"]) + 14.0)
        cabin = rng.choice(cabins)
        scn = {"id": f"sound_{split}_{k // per_ride:02d}", "split": f"sound_{split}", "route": "routes/epfl_to_gare.geojson",
               "speed_kmh": int(cabin["labels"]), "duration_s": int(t + 2),
               "background": [{"clip": cabin["path"], "gain_db": 0, "loop": True}], "events": events}
        (out / f"{scn['id']}.json").write_text(json.dumps(scn, indent=1), encoding="utf-8")


class Compare:
    """Judge stand-in: answers like the text judge (path A) and records what paths B and C would have done."""

    def __init__(self, cfg: dict):
        self.cfg, self.text, self.pipe, self.log = cfg, Judge(cfg), None, []

    def __call__(self, payload: dict) -> dict:
        a = self.text(payload)
        audio = __import__("numpy").concatenate(list(self.pipe.buffer)[-int(CLIP_S / self.cfg["audio"]["chunk_s"]):])
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        ctx = json.dumps({"gps": payload["gps"]}, separators=(",", ":"))
        res = chat([{"role": "user", "content": [
            {"type": "text", "text": AUDIO_PROMPT.format(ctx=ctx)},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}],
            model=self.cfg["judge"]["model"], max_tokens=80, tag="sound_paths_audio")
        b = parse_json(res["text"]).get("decision", "check_in")
        top = next(iter(payload["top_labels"]), "")
        words = len(self.pipe.asr.transcript(payload["t"] - 5, payload["t"] + 6).split()) if self.pipe.asr else 0
        c = "dismiss" if top in BENIGN_LABELS else "check_in"
        self.log.append({"t": payload["t"], "A": a["decision"], "B": b, "C": c, "tokens_A": a["usage"]["total_tokens"],
                         "tokens_B": res["usage"]["total_tokens"], "words": words, "top": top})
        return a


def run(split: str, cfg: dict, det: Detector) -> dict:
    rows, hours, opened_total = [], 0.0, 0
    for path in scenario_paths(f"sound_{split}"):
        r = load_ride(path, det, asr_cfg=cfg)
        cmp = Compare(cfg)
        pipe = RidePipeline(cfg, r["route"], det, r["asr"], cmp)
        cmp.pipe = pipe
        hop = SR // 2
        for i in range(len(r["audio"]) // hop):
            g = r["gps"][i]
            pipe.on_gps(g["t"], g["lat"], g["lon"], g["speed_kmh"])
            pipe.on_audio(i * 0.5, r["audio"][i * hop:(i + 1) * hop], r["scores"][i], r["emb"][i])
        hours += r["scn"]["duration_s"] / 3600
        opened_total += len(cmp.log)
        for e in r["scn"]["events"]:
            hit = [x for x in cmp.log if e["t"] - 1 <= x["t"] <= e["t"] + e["len_s"] + 2]
            rows.append({"kind": e["kind"], "positive": "expect" in e, "opened": bool(hit), **(hit[0] if hit else {})})
    df = pd.DataFrame(rows)
    pos, neg = df[df.positive], df[~df.positive]
    out = {"ride_hours": round(hours, 2), "positives": len(pos), "negatives": len(neg),
           "positives_opened": int(pos.opened.sum()), "negatives_opened": int(neg.opened.sum()), "paths": {}}
    for p in "ABC":
        po, no = pos[pos.opened], neg[neg.opened]
        out["paths"][p] = {
            "real_threat_alert": int((po[p] == "alert").sum()), "real_threat_check_in": int((po[p] == "check_in").sum()),
            "real_threat_dismissed": int((po[p] == "dismiss").sum()),
            "harmless_alert": int((no[p] == "alert").sum()), "harmless_check_in": int((no[p] == "check_in").sum()),
            "harmless_dismissed": int((no[p] == "dismiss").sum()),
            "tokens_per_event": 0 if p == "C" else round(float(df[df.opened][f"tokens_{p}"].mean()))}
    out["opened_by_kind"] = {k: f"{int(g.opened.sum())}/{len(g)}" for k, g in df.groupby("kind")}
    return out


def main() -> None:
    cfg = load_config()
    det = Detector(cfg)
    report = {}
    for split in ("tune", "test"):
        make_rides(split)
        report[split] = m = run(split, cfg, det)
        print(f"\n== {split}: {m['ride_hours']} h, sound head opened {m['positives_opened']}/{m['positives']} real threats "
              f"and {m['negatives_opened']}/{m['negatives']} harmless sounds   {m['opened_by_kind']}")
        for p, name in (("A", "text judge "), ("B", "audio judge"), ("C", "rule      ")):
            x = m["paths"][p]
            print(f"  {p} {name}  threats: alert {x['real_threat_alert']:2d}  check-in {x['real_threat_check_in']:2d}  "
                  f"dismissed {x['real_threat_dismissed']:2d} | harmless: alert {x['harmless_alert']:2d}  "
                  f"check-in {x['harmless_check_in']:2d}  dismissed {x['harmless_dismissed']:2d} | "
                  f"{x['tokens_per_event']} tokens/event")
    (RESULTS / "sound_paths.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nsaved {RESULTS / 'sound_paths.json'}; total API spend so far ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
