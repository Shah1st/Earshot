# Earshot

**Safety alerts for taxi and ride-hailing trips that listen on the phone and call an LLM only when something
happens. Ride audio is never uploaded.**

Built at LauzHack for the AXA challenge *Frugal use of Multimodal AI* (sound + GPS + text).

## The problem

A ride app knows where the car is, not what is happening inside it. Sending every ride to an audio AI model would
work, but it uploads every conversation and costs the same whether or not anything happens.

## What Earshot does

The listening happens on the phone, with small models and plain rules that cost no API tokens:

| On the phone | What it is | What it notices |
|---|---|---|
| Sound | YAMNet, a public sound classifier, plus a small layer we trained for in-car noise | screams, shouting, breaking glass |
| Speech | Vosk, an offline speech recogniser, keeping the last two minutes as text | what was said; 52 safety phrases ("let me out", "I have a knife") |
| GPS | three rules | leaving the planned route, a sudden stop, a long stop in an odd place |

When one of these fires, the phone collects five more seconds and then decides:

* **Usually: one small LLM call.** The model receives a short text summary (sound labels, what was said, where
  the car is), about 700 tokens, and answers *dismiss*, *check in* or *alert* with a one line reason.
* **Sometimes: no LLM at all.** A loud sound with no words around it, or a GPS anomaly with nobody speaking, is
  handed to a human by rule. We measured that neither a text model nor an audio model handles a wordless scream
  well.

```
 Unity city simulation                        Android app (the phone)                     Laptop
 ┌──────────────────────┐   one TCP socket   ┌───────────────────────────────────┐      ┌──────────────┐
 │ taxi, route, audio   │ ─────────────────▶ │ sound model + speech + GPS rules  │ ───▶ │ judge backend│ ──▶ LLM
 │ plays it out loud    │  audio + GPS,      │ events, rules, operator screen    │ text │ (API key)    │
 └──────────────────────┘  every 0.5 s       └───────────────────────────────────┘ only └──────────────┘
```

The same ride can be sent, for comparison, to the same LLM listening to all of the audio. The app shows both
token counters side by side.

## What you see in the demo

| Scene | What happens | Earshot | Tokens: Earshot / always-listening LLM |
|---|---|---|---|
| Normal ride | two people chatting | nothing | 0 / 1,231 |
| Road works | driver announces a detour, leaves the route, comes back | dismissed: "driver explained the route change" | 690 / 1,557 |
| Shouting, then a turn | passenger shouts orders, car leaves the route | alert | 1,394 / 1,456 |
| Abduction (our own voices) | passenger screams for help, car leaves the route | alert | 655 / 1,305 |
| Quiet knife threat (our own voices) | nobody raises their voice, car brakes hard | alert | 638 / 1,178 |

Two more scenes (a forced stop, an engine stall) are in `demo_pack/`.

## Does it save anything? Measured, not estimated

We built five minute test rides with real driving noise recorded inside a car, real screams and breaking glass,
and conversations placed in the car with measured acoustics. Then we ran Earshot and the always-listening model on
the same 24 test rides (16 incidents), which were never used for tuning.

| System | Incidents alerted | Reached a human | False alerts per hour | Tokens per ride hour | Audio uploaded |
|---|---|---|---|---|---|
| LLM hears the whole ride | 16 of 16 | 16 of 16 | 16.5 | 126,662 | all of it |
| **Earshot** | 13 of 16 | 16 of 16 | 13.5 | 36,357 (71% fewer) | none |
| Earshot, low-cost setting | 9 of 16 | 12 of 16 | 2.0 | 14,159 (89% fewer) | none |

These rides are deliberately busy: something happens every 8 to 20 seconds. A ride where nothing happens costs
Earshot zero tokens. One setting (whether a spoken phrase alone can wake the LLM) moves the system between
cheaper and safer; we run it on the safer side.

The story of how we got there, including the experiments that went against us, is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## What we do not claim

* The threatening conversations are acted (written by us, performed by a speech model, two recorded by us). No
  open dataset of real in-car threats exists.
* The benchmark is small. One incident is worth about six points.
* A radio drama can fool it: a scream from the car speakers reads like a scream in the car. Most of our false
  alerts are one such line.
* A calmly spoken threat on a normal route is only noticed if one of our phrases is recognised through the noise.
* "Audio never leaves the phone" is the design. In the demo an Android emulator plays the phone and the audio
  comes from the simulation.

## What is in this repository

| Folder | What |
|---|---|
| [`emulator/`](emulator/) | The Android app (Kotlin): on-device pipeline, operator screen, model comparison. Its README explains how to run it. |
| [`simulation/`](simulation/) | The Unity city: taxi, routes, scenarios, audio and GPS streaming. Its README explains how to run it. |
| [`ml/`](ml/) | Python reference of the same pipeline, the data builders, training, evaluation and every experiment. |
| [`demo_pack/`](demo_pack/) | Everything the app needs: models, configuration, seven demo rides, expected decisions and token counts. |
| [`shared/`](shared/) | The configuration all three parts read: thresholds, phrases, GPS limits, prompts. |
| [`docs/`](docs/) | [EXPERIMENTS.md](docs/EXPERIMENTS.md) (for a reader), [ML_STATUS.md](docs/ML_STATUS.md) (all tables, data sources, split), [INTEGRATION.md](docs/INTEGRATION.md) (the exact contract between the parts). |

## Running it

1. **Simulation:** open `simulation/` in Unity, press Play, pick a scenario ([details](simulation/README.md)).
2. **App:** open `emulator/` in Android Studio, run it on an emulator, forward the ports, tap **Compare models**
   ([details](emulator/README.md)).
3. **Judge backend:** put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root and start
   `emulator/tools/judge_backend.py`. Without it the local models and rules still run.

No Unity at hand? `emulator/tools/replay_ride.py <ride id>` plays any ride from `demo_pack/` to the app.

To reproduce the measurements, see the rebuild order in [CLAUDE.md](CLAUDE.md) and the scripts in `ml/scripts/`.

## Data and credits

* Driving noise and car acoustics: Cerence In-Car McVAMPIRE (Mueller and Wolff, 2024) and DEMAND, both CC BY 4.0.
* Screams, glass, laughter and other sounds: FSD50K (Freesound clips, CC0 and CC BY only).
* Shouted and neutral speech: CREMA-D (ODbL).
* Models: YAMNet (Google, Apache 2.0), Vosk small English model (Alpha Cephei, Apache 2.0).
* Routes: OpenStreetMap contributors, routed with OSRM.

Every clip, its source and its licence are listed in the data manifest built by `ml/scripts/build_manifest.py`;
sources and the train/test split are described in [docs/ML_STATUS.md](docs/ML_STATUS.md).
