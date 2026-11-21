"""Build the five demo rides the team defined, into ml/data/demo/ (+ scenarios in shared/scenarios/demo/).

  1 forced_stop      driver is rudely told to stop and get out; the car stops           -> alert
  2 roadworks        driver announces a detour for road works; leaves the route, returns -> no alert
  3 shouting_detour  passenger shouts at the driver, then the car abruptly leaves route  -> alert
  4 normal           ordinary conversation, on route                                     -> nothing
  5 engine_stall     engine cuts out, driver explains and checks; long stop              -> no alert

Voices are acted (earshot/voice.py) and placed in the car with real impulse responses; if the speech
model refuses a line it is retried with a softer direction, then falls back to Piper TTS (flat voice).

    uv run python scripts/make_demo.py            # build audio + scenarios, then run the pipeline on them
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_dialogues import FIELDS, VOICES, perform  # noqa: E402

from earshot.common import CLIPS, DATA, DRY, MODELS, REPO, SR, in_car, normalise, read_audio, write_wav  # noqa: E402
from earshot.rides import load_ride  # noqa: E402

OUT = DATA / "demo"
SCN = REPO / "shared" / "scenarios" / "demo"
ROUTE = "routes/epfl_to_gare.geojson"
DRIVE = {"clip": "cabin/drive_050kph_mic14.wav", "gain_db": 0, "loop": True}

IDLE = {"idle_clip": "cabin/drive_000kph_mic14.wav"}  # driving noise drops to engine idle when the car stops

# Compact rides: parts follow each other after `gap` seconds; GPS events and the end of the ride are placed
# relative to the parts ("p2.start", "p2.end") so nothing drags on after the action.
#   gps: (kind, anchor, offset_s, extra fields)        end: (anchor, offset_s)
RIDES = {
    "demo_1_forced_stop": {
        "expect": "alert", "gps": [("stop", "p2.end", -4, {})], "end": ("p2.end", 8), "background": [{**DRIVE, **IDLE}],
        "parts": [
            (2, [("P", "polite, a bit tired", "Evening. To the old town, please, near the cathedral."),
                 ("D", "friendly taxi driver", "Sure. Should be about ten minutes at this hour.")]),
            (3, [("P", "LOUD: rude, aggressive, raising his voice", "Hey. Stop the car. Stop the car, now!"),
                 ("D", "caught off guard, uneasy", "What? Sir, we're not there yet."),
                 ("P", "LOUD: furious, shouting orders", "I said stop the car and get out! Get out of the car! Now!"),
                 ("D", "LOUD: frightened, voice shaking", "Okay! Okay! Please, calm down, I'm stopping!")])]},
    "demo_2_roadworks": {
        "expect": "dismiss", "gps": [("bypass", "p1.end", -5, {"dur": 36, "offset_m": 350})], "end": ("p2.end", 8),
        "parts": [
            (2, [("P", "friendly", "Hi. The train station, please."),
                 ("D", "helpful, relaxed", "No problem. Just so you know, there are road works on the main avenue today, "
                                          "so I'll take the side streets. It's a small detour, but we get to the same place."),
                 ("P", "relaxed", "Oh, okay. No problem, thanks for telling me.")]),
            (11, [("P", "curious, calm", "Is it much longer this way?"),
                  ("D", "reassuring", "Two more minutes, that's all. We're nearly back on the main road."),
                  ("P", "relaxed", "Perfect, thank you.")])]},
    "demo_3_shouting_detour": {
        "expect": "alert", "gps": [("detour", "p2.start", 7, {})], "end": ("p2.end", 17),
        "parts": [
            (2, [("P", "impatient", "Could you go a little faster? I'm late."),
                 ("D", "polite, calm", "I'm at the speed limit already, sorry.")]),
            (3, [("P", "LOUD: angry, shouting at the driver", "Are you deaf? I said drive faster! Move!"),
                 ("D", "tense, trying to stay calm", "Sir, please don't shout at me."),
                 ("P", "LOUD: enraged, yelling orders", "Shut up and turn here! Turn right! Now! Do what I say!"),
                 ("D", "LOUD: scared, shouting", "Okay! Okay, I'm turning! Please!")])]},
    "demo_4_normal": {
        "expect": None, "gps": [], "end": ("p2.end", 2),
        "parts": [
            (2, [("P", "friendly", "Morning. Could you take me to the university, the main entrance?"),
                 ("D", "cheerful", "Of course. Nice day for it, finally some sun."),
                 ("P", "relaxed, chatty", "Yeah, after two weeks of rain. I almost forgot what it looks like.")]),
            (2, [("D", "small talk", "Are you a student there, or do you work at the university?"),
                 ("P", "amused", "I teach, actually. First lecture at nine, and I haven't finished my slides."),
                 ("D", "laughing softly", "Ha. Then I'll drive carefully, so you can work in the back.")])]},
    "demo_5_engine_stall": {
        "expect": "dismiss", "gps": [("stop", "p2.start", 1, {})], "end": ("p3.end", 9),
        "background": [{**DRIVE, **IDLE, "idle_gain_db": -30}],  # engine off: almost silent once stopped
        "parts": [
            (2, [("P", "relaxed", "Hi. To the airport, please. No rush, my flight is in three hours.")]),
            (2, [("D", "worried, mumbling", "Uh oh. Hmm. That's not good. The engine just cut out."),
                 ("P", "calm, a little concerned", "Oh. Is everything okay?"),
                 ("D", "apologetic, calm", "Yeah, yeah, nothing dangerous. Sorry about this. Give me a couple of minutes, "
                                          "I need to check what the problem is."),
                 ("P", "relaxed", "Sure, take your time. I'm not in a hurry.")]),
            (12, [("D", "relieved", "Okay, I think it's just the battery cable. One more minute and we can go."),
                  ("P", "friendly", "No worries at all.")])]},
}


def piper(line: str, voice: str) -> np.ndarray:
    """Fallback voice (flat, but never refuses)."""
    from piper import PiperVoice

    tmp = DATA / "raw" / "piper_tmp.wav"
    with wave.open(str(tmp), "wb") as wf:
        PiperVoice.load(str(MODELS / "piper" / f"{voice}.onnx")).synthesize_wav(line, wf)
    return read_audio(tmp)


def act(turns: list, seed: str) -> tuple[np.ndarray, list, str, int]:
    rng = random.Random(seed)
    voice = dict(zip("DP", rng.sample(["ash", "echo", "verse", "ballad", "alloy"], 2)))
    seat = rng.randrange(10**6)
    parts, loud, text, t, fallbacks = [], [], [], 0.0, 0
    for spk, direction, line in turns:
        is_loud = direction.startswith("LOUD:")
        x = perform(line, direction, voice[spk], strict=True)  # strict: no spoken stage directions
        if x is None and is_loud:
            x = perform(line, "LOUD: raised voice, impatient and rude, speaking loudly and firmly", voice[spk], strict=True)
        if x is None:
            fallbacks += 1
            x = normalise(piper(line, "en_US-ryan-medium" if spk == "P" else "en_GB-alan-medium"),
                          rms_db=-15.0 if is_loud else -23.0)
        x = in_car(x, "driver" if spk == "D" else "passenger", seat)
        gap = rng.uniform(0.1, 0.3) if is_loud else rng.uniform(0.3, 0.7)
        parts += [np.zeros(int(gap * SR), np.float32), x]
        t += gap
        if is_loud:
            loud.append([round(t, 2), round(t + len(x) / SR, 2)])
        t += len(x) / SR
        text.append(f"{spk}: {line}")
    audio = np.concatenate(parts)
    audio *= 10 ** (-23 / 20) / float(np.sqrt(np.mean(audio**2)) + 1e-9) * (1.6 if loud else 1.0)
    return audio, loud, " ".join(text), fallbacks


def main() -> None:
    assert VOICES
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.rmtree(SCN, ignore_errors=True)
    OUT.mkdir(parents=True)
    SCN.mkdir(parents=True)
    rows = []
    for rid, spec in RIDES.items():
        events, at, t = [], {}, 0.0
        for k, (gap, turns) in enumerate(spec["parts"]):
            audio, loud, text, fb = act(turns, f"{rid}-{k}")
            t = round(t + gap, 1)
            at[f"p{k + 1}.start"], at[f"p{k + 1}.end"] = t, t + len(audio) / SR
            rel = f"demo/{rid}_part{k + 1}.wav"
            write_wav(DRY / rel, audio)
            write_wav(CLIPS / rel, audio)
            e = {"t": t, "clip": rel, "gain_db": 2 if loud else 0, "kind": "dialog", "len_s": round(len(audio) / SR, 2),
                 "text": text}
            if loud:
                e.update(loud=loud, alert_from=loud[0][0])
            if spec["expect"] == "alert" and loud:
                e.update(expect="alert", incident=0)
            events.append(e)
            rows.append({"path": rel, "category": "demo", "role": "demo", "source": "acted (gpt-audio-mini) + McVAMPIRE car IR",
                         "source_id": f"{rid}_part{k + 1}", "licence": "synthetic", "text": text, "split": "demo", "fx": "done"})
            print(f"{rel}: {len(audio) / SR:.1f} s" + (f"  ({fb} line(s) used the fallback voice)" if fb else ""))
            t += len(audio) / SR
        events += [{"t": round(at[ref] + off, 1), "gps": kind, **extra} for kind, ref, off, extra in spec["gps"]]
        duration = int(np.ceil(at[spec["end"][0]] + spec["end"][1]))
        print(f"  -> {rid}: {duration} s")
        scn = {"id": rid, "split": "demo", "route": ROUTE, "speed_kmh": 50, "duration_s": duration,
               "expected_outcome": spec["expect"] or "nothing", "background": spec.get("background", [DRIVE]),
               "events": sorted(events, key=lambda e: e["t"])}
        (SCN / f"{rid}.json").write_text(json.dumps(scn, indent=1), encoding="utf-8")
        ride = load_ride(SCN / f"{rid}.json", force=True)
        shutil.copy(ride["wav"], OUT / f"{rid}.wav")  # the full ride as the app hears it
        shutil.copy(ride["wav"].with_suffix("").with_suffix(".gps.jsonl"), OUT / f"{rid}.gps.jsonl")
        shutil.copy(SCN / f"{rid}.json", OUT / f"{rid}.scenario.json")
        for e in events:
            if "clip" in e:
                shutil.copy(CLIPS / e["clip"], OUT / Path(e["clip"]).name)  # voices only, no driving noise
    with open(DATA / "manifest_demo.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\ndemo rides in {OUT}")


if __name__ == "__main__":
    main()
