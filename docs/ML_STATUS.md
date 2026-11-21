# Earshot ML: status, data provenance, results, next steps

Last updated: 21 Nov 2026 (evening of day 1). Branch: `ml-pipeline`. Scope: the data + ML part of
`earshot_baseline_plan.md`. The Android app and the Unity scene are not started in this repo yet; their contract
is in `docs/INTEGRATION.md`.

## 1. What exists

| Piece | Where | State |
|---|---|---|
| Shared contract: thresholds, groups, keyword list, GPS limits, judge prompt | `shared/config.json` (v2, current), `shared/config_v1.json` (native YAMNet groups) | frozen after tuning |
| Sound head weights | `shared/head.json` | trained on tune clips only |
| Reference pipeline (what the phone does) | `ml/earshot/pipeline.py`, `detector.py`, `events.py`, `asr.py`, `judge.py` | works end to end |
| OpenRouter client: disk cache, usage log, hard spend cap | `ml/earshot/llm.py` | all calls go through it |
| Data builders | `ml/scripts/fetch_data.py`, `fetch_real.py`, `make_dialogues.py`, `build_manifest.py` | reproducible |
| Ride generator + renderer (audio mix and GPS track from a scenario) | `ml/scripts/make_scenarios.py`, `ml/earshot/render.py` | done |
| Head training, threshold calibration | `ml/scripts/train_head.py`, `calibrate.py` | done |
| Evaluation + audio LLM baseline | `ml/scripts/evaluate.py`, `baseline_audio_llm.py` | done, numbers below |
| Team demo rides (5) | `ml/scripts/make_demo.py` → `ml/data/demo/`, `shared/scenarios/demo/` | all 5 behave as intended |
| Unity stand-in (socket server) | `ml/scripts/stream_ride.py` | written, not yet tested against the app |
| Hand-over bundle | `ml/scripts/export_demo_pack.py` → `ml/demo_pack.zip` | done |

## 2. How the pipeline works now (differences from the plan are marked)

1. **Sound** every 0.5 s: float YAMNet → 1,024 d embedding → logistic regression head on
   `[embedding now, embedding 0.5 s ago]` → `loud_event` probability. Trigger: moving average over 3 chunks above
   the threshold for 2 chunks in a row.
   *Changed:* the plan's v1 (native YAMNet class scores) barely detects anything in driving noise, so the v2 head
   is the default. The MediaPipe `yamnet.tflite` has no embedding output, so the app must use the float model with
   the LiteRT interpreter (16 MB instead of 4 MB).
2. **Speech:** Vosk small model runs **continuously** on device and keeps a rolling transcript as text.
   A safety phrase in it opens an event (`keyword` trigger).
   *Changed:* the plan transcribed only the clip around a sound event. That cannot catch a calm threat
   ("stop the car and get out") or explain a detour, which the team's demo cases need. Cost: more CPU on the phone,
   still 0 tokens. The plan's grammar limited keyword pass was dropped (it hallucinates, see 5).
3. **GPS rules:** off route, sudden stop, long stop (unchanged). A new anomaly with no sound event opens a `gps`
   event. *Changed:* the plan sent a blind check-in; now the judge reads the last 120 s of transcript, so
   "road works, I'll go around" is dismissed.
4. **Judge:** one text call to `google/gemini-3.1-flash-lite` per event (sound labels, GPS, transcript, earlier
   decisions) → dismiss / check_in / alert. Very confident sound + GPS anomaly alerts at once without waiting.
5. **Privacy wording to use:** audio never leaves the phone and only a 30 s rolling audio buffer exists; a rolling
   text transcript of at most 2 minutes is kept in memory; the LLM only ever receives a short text summary of an
   event. A normal quiet ride sends nothing.

## 3. Data: where every sound and text comes from

All clips are 16 kHz mono WAV. Every clip is one row of `ml/data/manifest.csv` with its source, source id,
licence, uploader, text and split. Nothing was recorded by us, and nothing comes from YouTube/TikTok or other
scraped material (copyright and the privacy of real victims).

| Role in a ride | Source | Licence | How it is used |
|---|---|---|---|
| Driving noise | **Cerence In-Car McVAMPIRE** (Zenodo 12806684): noise recorded inside a Mercedes V-class at 0/30/40/50/60 km/h | CC BY 4.0 | background loop, picked by ride speed |
| Driving noise | **DEMAND** `TCAR` (Zenodo 1227121): recording inside a moving private car | CC BY 4.0 | background loop (30% of rides) |
| Car acoustics | **McVAMPIRE** impulse responses: mouth simulator at each seat → roof microphone, and car loudspeakers → microphone | CC BY 4.0 | every voice and sound is convolved with a seat response, radio with the loudspeaker response |
| Screams, shouts, thumps, breaking glass, laughter, coughs, door, horn, music, crowd | **FSD50K** (Freesound clips with human verified labels), downloaded per clip from the `Fhrozen/FSD50k` mirror; only CC0 and CC BY clips | CC0 / CC BY per clip (logged) | real recordings; alert sounds and hard negatives |
| Shouted and neutral speech | **CREMA-D**: 91 actors saying "It's eleven o'clock" angry/fearful at high intensity, and neutral/happy takes | ODbL 1.0 | real shouting (positives for the sound head only) and real talk (negatives) |
| Driver/passenger dialogues | **Written by us** (`ml/scripts/dialogues.py`: 21 normal, 18 threat scripts) and **acted by a speech model** (`openai/gpt-audio-mini` via OpenRouter), 2 voice pairs each | synthetic | 40 normal + 24 threat clips kept (14 of 94 requests were refused by the model, mostly aggressor lines) |
| Radio decoys (football commentary, radio drama "help me!", adverts) | written by us, acted by the same model, crowd bed from FSD50K, band limited, played through the measured car loudspeakers | synthetic + CC | hard negatives |
| Routes | OSRM public router, 3 routes around Lausanne | OpenStreetMap ODbL | GPS tracks |

Why the mix is believable: speech and sounds are placed in the car with **measured** impulse responses, over
**real** driving noise, at the signal to noise ratio the McVAMPIRE authors measured for real in-car speech
(0 dB at 100 km/h, which gives about 7 dB at 40 km/h). What is **not** real: the threat conversations themselves.
No open dataset of real in-car threats exists (the published in-car violence datasets are private or video only),
so these are scripted and acted. This must be said when the numbers are presented.

### 3.1 Split (frozen before any tuning)

* **By source, never by clip:** all clips cut from one source stay together.
  * FSD50K: by Freesound clip id (hash, 2/3 tune, 1/3 test).
  * CREMA-D: by actor (an actor's voice is never in both).
  * Dialogues: by script (every third script is test), so the test dialogues have **different wording and
    situations** from the tune ones; both voice variants of a script stay together.
  * Driving noise: McVAMPIRE tune and test use different microphones; DEMAND is split by time (first 210 s tune,
    last 90 s test).
* Totals: 1,002 clips; tune 59 min, test 29 min of source audio.
* **Tune clips** build: 60 dense `train` rides (sound head training) and 24 `tune` rides (threshold calibration,
  prompt wording). **Test clips** build 24 `test` rides that were only used for the numbers in section 4.
* Rides are 5 minutes, seeded and reproducible (`make_scenarios.py`): real driving noise, optional radio music,
  a harmless event every 8 to 20 s (dialogue, talk, laughter, cough, door, horn, radio), and 0 to 2 threat
  incidents (threat dialogue, with or without a scream, a lone scream, breaking glass), some with a detour or a
  sudden stop. 40% of rides contain no incident.
* An incident counts as caught if an alert is raised between its first threatening sound and 10 s after its end.
* Known limits of the benchmark: small (24 rides, 2 ride hours per split, so one incident is worth
  several points of recall); acted threats; in 7 of the 64 evaluation dialogues (4 tune, 3 test) the speech model
  also spoke a stage direction aloud ("nervous breath", "laughing softly"), found by a Vosk scan; the demo clips
  are generated in a strict mode that rejects such takes, the evaluation clips were left as measured; the same speech model voices appear in tune and test (different
  scripts); FSD50K "scream" clips include some non-human or playful screams.

## 4. Results

Config version 8 for v2 (`shared/config.json`), same prompt and triggers for v1 (`shared/config_v1.json`).
All token and cost figures are the `usage` values returned by OpenRouter. Raw outputs: `ml/results/*.json`
(git ignored; rebuild with `evaluate.py` and `baseline_audio_llm.py`, LLM replies are cached).

### 4.1 Test split (24 rides, 2.0 ride hours, 16 incidents, never used for tuning)

| System | Incidents alerted | False alerts / ride hour | LLM calls / h | Tokens / h | Cost / h | Audio leaving the phone |
|---|---|---|---|---|---|---|
| A. Audio LLM hears everything (30 s chunks, 5 s overlap, same model) | 16 / 16 (100%) | 16.5 | 144 | 126,662 | $0.062 | all of it |
| **Ours v2** (embedding head + keywords + GPS, text judge) | 11 / 16 (69%) | 14.0 | 55.5 | 40,188 | $0.012 | none |
| Ours v1 (native YAMNet groups instead of the head) | 10 / 16 (63%) | 15.5 | 45.0 | 31,888 | $0.010 | none |
| Ours v2, sound trigger only (earlier build, commit `2be0ca0`: no keyword or GPS events) | 8 / 16 (50%) | 1.5 | 27.0 | 13,786 | $0.0045 | none |
| Ours v1, sound trigger only (same earlier build) | 3 / 16 (19%) | 0.5 | 10.5 | 5,260 | $0.0017 | none |

Other v2 figures: an event is opened for 13 of 16 incidents (81%; v1: 69%), harmless events judged: 94,
of which dismissed 69%. About 720 tokens per judged event; the baseline spends about 880 tokens per 30 s chunk.

How to read this honestly:

* **Tokens:** about 3 times fewer than the audio LLM on these rides, which are deliberately busy (something
  happens every 8 to 20 s; no test ride was silent enough to cost 0 tokens). The plan's "~1000x" does not hold
  for busy rides. On a quiet ride the cost is 0: the demo ride `demo_4_normal` makes no LLM call at all.
  The structural difference stands: the baseline uploads every second of audio, ours uploads no audio.
* **Recall is lower than the baseline's** (69% vs 100%, on 16 incidents). The misses are mostly wordless screams
  that neither the head nor the labels identify, and threats the judge found ambiguous.
* **False alerts come almost entirely from one decoy.** 26 of the alerts on the test rides quote the same radio
  line, a radio drama actress screaming "Help! Help me! Somebody call the police! He's getting away!", which as
  text is indistinguishable from a real plea. That clip is only in the test split, so nothing was tuned for it,
  and because the judge sees the last 45 s of transcript one occurrence can raise several alerts. The audio LLM
  has a similar false alert rate (16.5 / h). Without keyword and GPS events (row 4) false alerts were 1.5 / h.
* **Keyword and GPS events are a trade:** +19 points of recall and the ability to handle the team's demo cases,
  for 3 times the tokens and the radio false alerts. Row 4 is the frugal end of the same system.
* **The head matters for sound only detection** (50% vs 19%); once keywords are on, v1 and v2 are close because
  most threat dialogues contain a keyword.

### 4.2 Tune split (24 rides, 2.0 h, 20 incidents; used for calibration, so optimistic)

v2: alerted 13 / 20 (65%; 70% counting check-ins), event opened for 95%, false alerts 1.0 / h, dismissed 94% of
47 harmless events, 34 calls / h, 24,098 tokens / h. The tune radio decoys do not include the radio drama line,
which is why false alerts look so much better here than on test.

### 4.3 Team demo rides (`ml/data/demo/`, not part of the metrics)

| Ride | What happens | System output |
|---|---|---|
| `demo_1_forced_stop` (38 s) | passenger rudely orders the driver to stop and get out; car stops, noise drops to idle | **alert** at 21.5 s (keyword "stop the car") |
| `demo_2_roadworks` (46 s) | driver announces a road works detour, leaves the route, comes back | **dismiss** at 31 s ("driver explained the route change") |
| `demo_3_shouting_detour` (42 s) | passenger shouts orders, car turns off the route | **alert** at 24 s, again at 38.5 s when the route is left |
| `demo_4_normal` (33 s) | ordinary chat on route | nothing, 0 tokens |
| `demo_5_engine_stall` (57 s) | engine cuts out (cabin goes quiet), driver explains, long stop | **dismiss** at 53.5 s ("minor mechanical issue") |

The rides were shortened for the pitch (33 to 57 s each). To make that possible the GPS limits in the config were
tightened **after** the section 4.1 evaluation (config v10: off route hold 10 → 5 s, long stop 60 → 30 s, endpoint
radius 150 → 60 m); the test numbers above were measured with the old limits and have not been re-run.
In demo rides the driving noise now follows the car (idle when stopped, near silence when the engine is off).

These five rides were written after the system existed and the prompt was adjusted with them in view, so they
show the intended behaviour, not an independent test. One aggressor line in ride 3 uses the fallback TTS voice
because the speech model refused it.

## 5. Findings worth telling

* **Native YAMNet classes are not enough in a car.** With the plan's groups and thresholds almost nothing fires
  over driving noise: shouted speech is labelled "Speech", thumps "Knock", screams often "Animal" or "Siren".
  A tiny head on YAMNet's embedding fixes most of it at the cost of one dot product per half second.
* **Vosk keyword grammar hallucinates.** Restricting Vosk to a keyword grammar found a keyword in 48% of harmless
  dialogues. Matching phrases in the ordinary transcript: 100% of threat dialogues, 15% of normal ones (all real
  mentions such as "the police closed the road").
* **The small Vosk model is good enough in driving noise** for the judge to get the gist (demo transcripts are
  nearly verbatim) and runs about 20 times faster than real time on a laptop.
* **The judge must not be told a score means danger.** Calling the head output "distress" made the LLM alert on
  laughter. Renamed to `loud_event` and explained in the prompt.
* **GPS + words beats either alone:** a detour explained in the conversation is dismissed, the same detour after
  shouting is an alert (demo rides 2 and 3).

## 6. What is ahead

Must do for the demo:
1. **Android:** implement `docs/INTEGRATION.md` (LiteRT YAMNet + head, continuous Vosk, GPS rules, judge call).
   First check on the x86_64 emulator: both libraries load, and one inference + Vosk keep up with real time.
   Test against `stream_ride.py` before Unity is ready.
2. **Unity:** play `demo_pack/rides/*.wav` and replay the GPS lines, or mix from the scenario files; send the
   socket protocol. When the real map routes exist, drop them in `shared/routes/` and re-run `make_demo.py`.
3. **Parity check:** app decisions on the 5 demo rides vs `demo_pack/expected/*.json`.
4. **Team's own demo audio:** if the team records or generates its own 5 clips, put the WAVs anywhere and reference
   them in a scenario; run `evaluate.py --split demo -v` to see what the system does before going on stage.

Should do if time allows:
* More test rides and a second seed to tighten the numbers; report recall with a confidence interval.
* Sound head: more varied shouting (real recordings), check the false triggers on laughter and coughs.
* Radio false alerts: give the judge only the transcript since the previous decision plus a short "before" part, so one
  radio line cannot raise several alerts; add a "sounds like the radio" cue (band limited speech, YAMNet `Radio`/`Television`).
* Judge: the operating point is a prompt choice (see section 4); tune on tune rides only.
* Operator Q&A (`Judge.ask`) exists but has not been evaluated.
* Battery/CPU of continuous Vosk on a real phone is not measured.

Not done from the plan: NINA real in-car data (not needed after McVAMPIRE/DEMAND), ESC-50 (FSD50K covers the same
classes with clearer licences), Piper TTS lines (replaced by acted dialogues; Piper remains only as a fallback
voice in `make_demo.py`).

## 7. Budget

All API calls are logged in `ml/results/cache/usage.jsonl`. Spend so far: $0.43 of the $10
(dialogue acting, judge calls during tuning and evaluation, audio baseline). Cached calls are free on re-runs.
