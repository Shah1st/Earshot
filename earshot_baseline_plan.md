# Earshot (working name): baseline spec v2

Frugal, privacy first safety alerts for taxi rides. Text + sound, with GPS context.
AXA "Frugal use of Multimodal AI" hackathon. Updated 21 Nov 2026 after the team Q&A.

**Constraints:** 2 to 3 people, 18 h to deadline, 4 min pitch + 1 to 2 min Q&A, AMD laptop (x86_64 emulator), English only.

---

## 0. One paragraph

A phone app listens for distress sounds (screams, shouting, impacts, breaking glass) with an on device audio model. GPS adds context: sudden stops, long stops in odd places, leaving the planned route. Nothing is kept except a rolling 30 s buffer. When an event fires, the clip around it is transcribed on device with Vosk, and a small LLM receives a short text summary (sound labels, transcript, trip context). It decides: dismiss (false alarm, e.g. football on the radio), check in, or silent alert to an operator. A normal ride costs 0 LLM tokens; we compare against an audio LLM that listens to the whole ride.

---

## 1. Demo setup (decided)

```
 Unity (left half of screen)                      Android emulator (right half)
 ┌───────────────────────────┐   one TCP socket   ┌──────────────────────────────────┐
 │ car scene, master clock   │  (adb reverse)     │ Passenger tab: ride status       │
 │ reads scenario JSON       │ ─────────────────▶ │ Operator tab: alerts, map, Q&A   │
 │ plays audio out loud      │  audio + GPS +     │ pipeline: YAMNet → events →      │
 │ streams the same samples  │  ride start/stop   │ Vosk → LLM judge (OpenRouter)    │
 └───────────────────────────┘                    └──────────────────────────────────┘
```

* All audio is pre loaded. Unity plays it out loud for the audience and streams the same samples to the app, so what people hear is what the app analyses.
* One socket carries everything. `adb reverse tcp:5005 tcp:5005`, the app connects to `localhost:5005`.
* The app calls OpenRouter directly over HTTPS. Key in `local.properties`, never in git.
* A few demo rides (see 6.4). Backup: a recorded video of the full demo.

### 1.1 Socket protocol (newline delimited JSON, easy to debug)

```json
{"type":"start","ride_id":"ride_02","route":[[46.5191,6.5668],[46.5185,6.5701], ...]}
{"type":"gps","t":12.0,"lat":46.5189,"lon":6.5679,"speed_kmh":38.5}
{"type":"audio","t":12.0,"sr":16000,"pcm16_b64":"<0.5 s of mono PCM16, base64>"}
{"type":"stop","ride_id":"ride_02"}
```

* Audio: 0.5 s chunks, 16 kHz mono PCM16 (~43 KB/s after base64, fine on localhost).
* GPS: every 0.5 s together with audio, same `t` clock.
* The route arrives in `start`, so the app computes deviation itself and Unity stays the single source of truth.
* Ground truth (`expect` fields) is **never** sent to the app; Unity writes it to its own log for evaluation.

### 1.2 Scenario file (Unity reads it; Python evaluation reads it too)

```json
{
  "id": "ride_02",
  "route": "routes/epfl_to_gare.geojson",
  "speed_kmh": 40,
  "background": [{"clip": "cabin/123.wav", "gain_db": 0, "loop": true},
                 {"clip": "radio/456.wav", "gain_db": -8, "from": 30, "to": 150}],
  "events": [
    {"t": 60,  "clip": "radio/football_goal.wav", "gain_db": -3, "expect": "dismiss"},
    {"t": 125, "clip": "lines/let_me_out.wav", "expect": "alert"},
    {"t": 128, "clip": "scream/789.wav", "gps": "detour", "expect": "alert"},
    {"t": 160, "gps": "stop"}
  ]
}
```

`expect`: `alert` (should reach the operator), `dismiss` (detector may fire, the LLM should drop it), omitted (should not fire).

---

## 2. Pipeline

| Step | What | Runs where | LLM tokens |
|---|---|---|---|
| 0 | Rolling 30 s audio buffer in memory, nothing stored otherwise | Phone | 0 |
| 1 | YAMNet native class scores every 0.5 s | Phone | 0 |
| 2 | GPS context: sudden stop, long stop off route, leaving the route | Phone | 0 |
| 3 | Event logic: smoothing, thresholds, cooldown | Phone | 0 |
| 4 | On event: seal clip (−10 s … +5 s), Vosk transcript + keyword pass | Phone | 0 |
| 5 | Clear case: alert right away. Ambiguous: LLM judge | Phone → OpenRouter | ~450 per judged event |
| 6 | Output: dismiss / check in / silent operator alert | Phone | 0 |

### 2.1 v1 audio model: YAMNet native scores, no training

* `yamnet.tflite` from the MediaPipe model zoo. Input: 0.975 s windows (15,600 samples at 16 kHz), hop 0.5 s.
* Events to showcase (exact YAMNet display names). Starting thresholds, calibrate then freeze:

| Group | YAMNet classes | Type | Start threshold |
|---|---|---|---|
| scream | `Screaming`, `Crying, sobbing`, `Whimper` | sustained | 0.30 |
| shout | `Shout`, `Yell` | sustained | 0.40 |
| impact | `Slap, smack`, `Thump, thud`, `Bang` | impulsive | 0.35 |
| glass | `Shatter`, `Glass` | impulsive | 0.30 |
| crash (optional) | `Smash, crash`, `Skidding`, `Tire squeal` | impulsive | 0.30 |

Group score = max over its classes in the window.

### 2.2 v2 (after the baseline works): small head

Logistic regression on YAMNet's 1,024 d embedding, `{distress, normal}`, trained on the tuning split including hard negatives (radio, laughter, music, kids). On Android it is one dot product per class, weights exported as JSON. Keep it only if it beats v1 on the test split, and show both numbers.

### 2.3 Event logic

* Sustained groups: moving average over 3 windows, fire if above threshold for ≥ 2 consecutive windows (~1 s).
* Impulsive groups: fire on a single window above threshold.
* Cooldown 10 s: new triggers merge into the same event.

```json
{"t": 128.4, "groups": {"scream": 0.81, "shout": 0.44}, "peak_group": "scream", "duration_s": 2.1,
 "gps": {"speed_kmh": 38, "off_route_m": 220, "sudden_stop": false, "stopped_s": 0}}
```

### 2.4 GPS context (rules, no ML)

* `off_route_m`: distance to the route polyline from `start`. Flag if > 150 m for > 10 s.
* `sudden_stop`: speed drop > 25 km/h within 3 s.
* `stopped_s`: stopped for > 60 s while off route or away from pickup/drop off.

### 2.5 Clear case vs LLM judge

| Audio | GPS anomaly | Decision | LLM call |
|---|---|---|---|
| a group ≥ high threshold (e.g. scream ≥ 0.7) | yes | **silent alert now** | only to write the one line summary |
| above threshold | any | **LLM judges**: dismiss / check in / alert | yes (~450 tokens) |
| below threshold | yes | check in prompt | no |
| below threshold | no | nothing | no |

---

## 3. Vosk, adapted to its quality

| Model | Size | Use |
|---|---|---|
| `vosk-model-small-en-us-0.15` | ~40 MB | start here, supports runtime grammars |
| `vosk-model-en-us-0.22-lgraph` | ~128 MB | switch if transcripts of the test lines are too poor |

1. Two passes on the sealed clip: free transcription with word confidences (`SetWords(true)`), plus a keyword pass with a grammar limited to a phrase list (much more robust than free transcription):
   `["stop the car", "let me out", "help", "help me", "police", "call the police", "don't touch me", "leave me alone", "where are you taking me", "[unk]"]`
2. Words with confidence < 0.5 are dropped; keyword hits are sent as a separate field.
3. The transcript is never required: screams have no words, and clear cases (2.5) alert without it.
4. The judge prompt says the transcript is noisy and to trust sound labels and GPS first.
5. Clip window −10 s … +5 s, because words usually come before the scream.

Python reference (APIs checked against vosk 0.3.45):

```python
import json, numpy as np
from vosk import Model, KaldiRecognizer

model = Model(model_name="vosk-model-small-en-us-0.15")
KEYWORDS = ["stop the car", "let me out", "help", "help me", "police", "call the police",
            "don't touch me", "leave me alone", "where are you taking me", "[unk]"]

def transcribe(pcm16: np.ndarray, sr=16000):
    rec = KaldiRecognizer(model, sr); rec.SetWords(True)
    rec.AcceptWaveform(pcm16.tobytes())
    res = json.loads(rec.FinalResult())
    return " ".join(w["word"] for w in res.get("result", []) if w["conf"] >= 0.5)

def keywords(pcm16: np.ndarray, sr=16000):
    rec = KaldiRecognizer(model, sr, json.dumps(KEYWORDS))
    rec.AcceptWaveform(pcm16.tobytes())
    text = json.loads(rec.FinalResult()).get("text", "")
    return [k for k in KEYWORDS[:-1] if k in text]
```

---

## 4. LLM judge

* `google/gemini-3.1-flash-lite` via OpenRouter, `temperature 0`, JSON output, `max_tokens 80`.
* Log `usage.prompt_tokens`, `usage.completion_tokens`, `usage.cost` for every call.

System prompt (~200 tokens):

```
You judge safety events detected in a taxi by an on-device audio model.
Inputs: sound labels with scores (reliable), GPS context (reliable),
keyword hits (fairly reliable), and a transcript from a small offline
recognizer (often wrong or incomplete: do not rely on exact words).
Decide one of:
- "dismiss": normal ride sound (radio, TV, music, laughing, kids, phone call).
- "check_in": unclear, ask discreetly if all is fine.
- "alert": likely threat to someone's safety, notify the operator silently.
Only judge safety. Never judge emotions or personality.
Reply JSON: {"decision": "...", "severity": 1-3, "reason": "<max 15 words>"}
```

User message (~150 to 250 tokens): the event JSON from 2.3 plus `"keywords"` and `"transcript"`.

Operator Q&A: the operator types a question in the Operator tab; the LLM gets that event's log and transcript only (~300 tokens).

---

## 5. Data (all pre loaded, 16 kHz mono WAV, licence logged in `manifest.csv`)

| Role | Source |
|---|---|
| Cabin background | Freesound CC0 via `fetch_freesound.py` (`cabin`), ESC-50 `engine` |
| Hard negatives (must not alert) | Freesound `radio`, `talk`, `benign`; ESC-50 laughing, coughing, crying baby, door knock |
| Alert sounds | Kaggle Human Screaming Detection; Freesound `scream`, `shout`, `impact`, `glass`, `crash`; ESC-50 glass breaking |
| Spoken lines | **Open datasets first, record only if needed.** See 5.1 |
| Real in car (bonus) | NINA via `rebuild_nina.py`, background job, not on the critical path |

### 5.1 Spoken lines: what open data can and cannot give

* Threat **sentences** ("stop the car, let me out") are unlikely to exist in open datasets.
* Shouted / fearful **speech** exists in acted emotion datasets (CREMA-D, RAVDESS: same neutral sentences spoken angry, fearful, etc.). Useful as realistic shouting for the detector and as hard negatives. Check each licence before use (RAVDESS is non commercial).
* For the content the LLM needs, use offline TTS (e.g. Piper) for a handful of threat and harmless lines. TTS cannot scream, so pair a TTS line with a real scream clip in the scenario.
* If TTS sounds too fake in the demo, record 10 lines from teammates (10 minutes of work).

**Split before tuning:** by source file, 2/3 tune, 1/3 test. Never move clips between them.

---

## 6. Evaluation

### 6.1 Showcase: token efficiency vs an audio LLM

| System | What | How |
|---|---|---|
| A. Audio LLM listens to all | every 30 s chunk (5 s overlap) to an audio model | `baseline_audio_llm.py` |
| Ours | detector + clear case rule + judge on events only | Python reference |

Cost check for A: Gemini 3.1 Flash Lite audio input $0.50 / M tokens, ~25 tokens/s, so ~90k tokens and ~$0.05 per ride hour. Fits easily in the $10.

### 6.2 Metrics

* Event recall on `alert` events (alert within 10 s).
* False alarms per ride hour.
* Dismiss accuracy on `dismiss` events.
* **LLM tokens per ride hour, measured from API usage.**
* Chart: recall vs tokens per ride hour (log x), A vs ours. If time: v1 vs v2 head.

### 6.3 Where evaluation runs: options

| Option | Extra code | Speed | Verdict |
|---|---|---|---|
| Python reference pipeline on WAV files (same `yamnet.tflite`, same thresholds JSON) | ~200 lines, mostly reused for tuning anyway | 30 rides in minutes | **recommended** for tuning and reported numbers |
| App logs events to a JSONL file while Unity plays rides; Python only scores the logs | ~30 lines in the app + ~50 line scorer | real time only (30 rides × 5 min = 2.5 h) | good as a parity check on the 2 to 3 demo rides |

Both read thresholds from one shared `config.json`, so the app and Python never disagree.

### 6.4 Demo rides (few)

1. Normal ride: talk, radio, laughter, nothing fires or the LLM dismisses. Token counter stays at 0 or ~450.
2. False alarm: football commentary peaks, the LLM dismisses it, grey card.
3. Threat: spoken line + scream + detour, red silent alert, operator asks "what was said before?".

---

## 7. App: what is fastest and easiest

**Native Android, Kotlin + Jetpack Compose.** Both key libraries ship official Android packages with demo apps (MediaPipe Audio Classifier, `alphacep/vosk-android-demo`), so there is no plugin layer to debug. Flutter / React Native would add wrappers around the same native libraries.

Dependencies (check latest versions):

```kotlin
implementation("com.google.mediapipe:tasks-audio:<latest>")
implementation("com.alphacephei:vosk-android:<latest>@aar")
implementation("net.java.dev.jna:jna:<latest>@aar")
implementation("com.squareup.okhttp3:okhttp:<latest>")
```

**First hour check on the AMD laptop:** both MediaPipe and Vosk must load on the x86_64 emulator image. If MediaPipe audio does not, run `yamnet.tflite` with the plain LiteRT (TensorFlow Lite) interpreter instead; if Vosk does not, decide fast between another ASR library and dropping on device transcription for the demo.

Components: `SocketClient` (NDJSON), `RingBuffer(30 s)`, `Detector`, `GpsContext`, `Asr` (two Vosk recognizers, background thread), `Judge` (OkHttp → OpenRouter), `Decision` (table 2.5), UI: Passenger tab + Operator tab (alerts, map pin, one line reason, Q&A, token counter).

---

## 8. Plan (H0 = now, deadline H18)

### Roles

| | 3 people | 2 people |
|---|---|---|
| You | ML, LLM judge, data, evaluation, pitch numbers | ML, LLM judge, data, evaluation **+ Android pipeline (Detector, Asr, Judge)** |
| Unity | scene, scenario player, socket sender, screen layout | same **+ app UI (tabs, socket client)** |
| Android | app UI, socket client, Detector/Asr/Judge in Kotlin | (merged above) |

### Checkpoints

| By | ML (you) | Android | Unity |
|---|---|---|---|
| H2 | data downloaded, split frozen, `config.json` v1 | skeleton app; YAMNet + Vosk load on x86_64 emulator | car moves on route; socket sends GPS |
| H5 | Python detector + event logic on tuning clips; judge prompt v1 | socket client; Detector on streamed audio | audio streamed + played aloud in sync |
| H8 | thresholds calibrated; Vosk passes; judge tested on lines | Asr + Judge in app; Passenger tab | scenario JSON drives everything; 3 demo rides |
| H11 | baseline A run; ours on all rides; results table | Operator tab, token counter, Q&A | screen layout, polish |
| H14 | chart; v2 head if time | parity check vs Python on 1 ride | polish |
| H15 | **record backup video**, freeze code | | |
| H16 to H18 | pitch with real numbers, rehearse ×3 | | |

Go / no go at H11: whatever works end to end is what we demo; the rest goes in the pitch as "next".

---

## 9. Risks

| Risk | Plan |
|---|---|
| MediaPipe or Vosk does not load on the x86_64 emulator | found at H2: LiteRT interpreter for YAMNet; for ASR choose fast |
| Socket bridge unstable on stage | backup video |
| YAMNet fires on music / laughter | stricter thresholds per group, then v2 head |
| TTS lines sound fake | record 10 lines |
| OpenRouter audio format rejected in baseline A | switch to `/api/v1/responses` |
| Budget | log `usage.cost`; judge ≈ fractions of a cent per event |

---

## 10. What we say on stage (only what we build)

* Safe: zero LLM tokens on a normal ride; measured tokens vs an audio LLM listening to everything; the LLM filters false alarms; the operator asks questions in text; clear cases alert without waiting for the LLM.
* Say carefully: "no audio leaves the phone" is the design; in the demo the emulator is the phone and audio comes from pre loaded files.
* Frame cost honestly: the audio LLM baseline costs cents per ride hour; the gap is ~1000x tokens, constant upload of every ride, and battery. Multiply by fleet ride hours.
* Q&A one liners: triggered, not continuous recording (what the UK ICO asked for in taxis); sound events, never emotions (EU AI Act Art. 5(1)(f)); life and health purpose only; consent at booking; automatic deletion; EU hosted LLM in production.

Out of the baseline: camera frame, SOS phrase, evidence pack, replay mode in the app.
