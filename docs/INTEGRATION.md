# Earshot: integration contract (Android app, Unity scene)

The Python code in `ml/earshot/` is the reference implementation of what the phone does. This page lists
exactly what the app has to reproduce and what Unity has to send. Every number below lives in
`shared/config.json`; read it from there, do not hard code it. If the app and Python disagree on a ride,
Python is right until we decide otherwise together.

Files you need are all in **`demo_pack/` at the repo root** (tracked in git, produced by
`ml/scripts/export_demo_pack.py`; see `demo_pack/README.md`):

| File | Used by | What |
|---|---|---|
| `config.json` | app | thresholds, keyword list, GPS limits, judge model and system prompt |
| `head.json` | app | sound head: 2,048 weights + bias |
| `models/yamnet_tfhub.tflite`, `models/yamnet_labels.txt` | app | float YAMNet (16 MB) and its 521 class names |
| `models/vosk-model-small-en-us-0.15/` | app | Vosk speech model (68 MB unpacked) |
| `scenarios/*.json`, `routes/*.geojson`, `clips/` | Unity | demo rides as scenario files + the clips they use |
| `rides/<id>.wav`, `rides/<id>.gps.jsonl` | Unity | the same rides pre rendered: one WAV + one GPS line per 0.5 s |
| `expected/<id>.json` | both | what the Python reference decided on each demo ride (parity check) |

---

## 1. Socket protocol (Unity → app)

Unchanged from the plan (section 1.1). Newline delimited JSON over one TCP socket, port 5005,
`adb reverse tcp:5005 tcp:5005`.

```json
{"type":"start","ride_id":"demo_3_shouting_detour","route":[[46.5191,6.5668],[46.5185,6.5701]]}
{"type":"gps","t":12.0,"lat":46.5189,"lon":6.5679,"speed_kmh":38.5}
{"type":"audio","t":12.0,"sr":16000,"pcm16_b64":"<0.5 s of mono PCM16, base64>"}
{"type":"stop","ride_id":"demo_3_shouting_detour"}
```

* `route` is `[lat, lon]` pairs. Audio: exactly 8,000 samples per message, `t` = start of the chunk.
* Per tick send `gps` first, then `audio`, same `t`.
* Never send ground truth (`expect`, `expected_outcome`).

**No Unity yet?** `uv run python scripts/stream_ride.py ../shared/scenarios/demo/<ride>.json --play` (from `ml/`)
serves a ride on port 5005 in real time with this exact protocol.

**Unity coordinates:** any map works; the app only uses the `route` and the GPS points it receives. GPS rules are in
metres, so convert world metres to degrees around any origin:
`lat = lat0 + z / 111320`, `lon = lon0 + x / (111320 * cos(lat0))`. `speed_kmh` must match how fast the point moves.

---

## 2. What the app does per audio chunk (0 LLM tokens)

### 2.1 Sound model

* Model: `yamnet_tfhub.tflite` with the plain **LiteRT / TFLite Interpreter** (not the MediaPipe AudioClassifier:
  we need the embedding output). Resize the input to `[15600]`, then allocate.
* Input: the **last 15,600 samples** (float32 in [-1, 1], i.e. PCM16 / 32768) after each 0.5 s chunk, zero padded at
  ride start. So one inference per chunk, window 0.975 s, hop 0.5 s.
* Outputs: pick by size. 521 values = class scores, 1,024 values = embedding (the third, 64, is unused).
* Head (`head.json`): `p = sigmoid(dot(weights, concat(embedding_now, embedding_previous_chunk)) + bias)`.
  For the first chunk use `embedding_now` twice. This `p` is the group `loud_event`.
* Other groups in `config.groups` (`scream`, `glass`, ...) = max of the listed class scores. They have
  `"trigger": false`: they never open an event, they are only reported to the judge.

### 2.2 Sound trigger (`config.groups.loud_event`, `config.event`)

* `type: "sustained"`: moving average of `p` over the last `smooth_windows` (3) chunks; fire when the average is
  ≥ `threshold` for `sustain_windows` (2) chunks in a row. (`"impulsive"` = single chunk ≥ threshold.)

### 2.3 Speech: continuous Vosk

* One `KaldiRecognizer(model, 16000)` with `setWords(true)`, fed every chunk for the whole ride.
* When `acceptWaveForm` returns true, read `result`: keep words with `conf ≥ asr.min_word_conf` as
  `(word, start, end)` in a rolling list (text only, in memory). Otherwise read `partialResult`.
* **Keyword trigger (`asr.keyword_trigger`, currently `false`):** when enabled, look for the phrases of `asr.keywords` as whole word sequences in the current utterance
  (partial or final), longest phrase first; fire when a phrase appears that was not already counted in this utterance.
* Transcript for the judge = kept words whose time falls in the last `asr.context_s` (45 s), or
  `asr.gps_context_s` (120 s) for GPS events, plus the current partial text.
* Do not use a Vosk grammar for keywords: we measured that it invents keywords in normal talk.

### 2.4 GPS rules (`config.gps`), on every gps message

* `off_route_m`: distance from the point to the route polyline (local metres). Flag `off_route` when it stays
  > `off_route_m` (150) for `off_route_hold_s` (5 s).
* `sudden_stop`: speed dropped by more than `sudden_stop_drop_kmh` (25) within `sudden_stop_window_s` (3 s);
  the flag stays on for `sudden_stop_latch_s` (30 s).
* `stopped_s`: seconds below `stopped_speed_kmh` (3). Flag `long_stop` when `stopped_s` > `stopped_hold_s` (30)
  and the car is farther than `endpoint_radius_m` (60) from both ends of the route.
* `anomaly` = any of the three flags.

### 2.5 Events (`ml/earshot/pipeline.py` is the exact logic)

An event **opens** when no event is open, the last trigger was ≥ `event.cooldown_s` (10 s) ago, and one of:

| Trigger | Condition |
|---|---|
| `sound` | sound trigger fires (2.2) |
| `keyword` | keyword trigger fires (2.3); only when `asr.keyword_trigger` is true (off by default) |
| `gps: <flag text>` | a GPS flag turns on (2.4) |

Any sound or keyword trigger refreshes the cooldown timer (so a long scream is one event).

**Clear case, no LLM decision:** on a chunk where the sound trigger fires with `p ≥ loud_event.high` and
`anomaly` is true → emit `alert` immediately (once per event). The LLM call at sealing then only provides the
one line summary.

While open, track per group the max score, per class the max score. The event **seals** `audio.clip_post_s` (5 s)
after it opened and is sent to the judge.

**Wordless sound, no LLM call (`config.wordless_rule`):** when a `sound` event seals and the transcript from
5 s before it opened until now has at most `max_words` (2) words:

| Condition, checked in this order | Output |
|---|---|
| the clear case alert was already sent for this event | summary "loud sound, no speech" (no LLM) |
| max score of any `benign_labels` class during the event ≥ `benign_min_score` (0.2) | `dismiss` |
| GPS `anomaly` is true | `alert` |
| otherwise | `check_in` |

Source is `rule`. All other events go to the judge.

### 2.6 Judge call (the only thing that costs tokens)

`POST https://openrouter.ai/api/v1/chat/completions`, model `judge.model`, `temperature 0`, `max_tokens 80`,
`response_format {"type":"json_object"}`, `usage {"include": true}`.
System message = `judge.system_prompt`. User message = this JSON, compact:

```json
{"t": 38.5, "trigger": "keyword", "groups": {"loud_event": 0.99}, "duration_s": 2.0,
 "top_labels": {"Speech": 0.98, "Shout": 0.21},
 "gps": {"speed_kmh": 40.7, "off_route_m": 1, "sudden_stop": false, "stopped_s": 0.0},
 "keywords": ["do what i say", "shut up"],
 "transcript": "are you deaf i said drive faster ... shut up and turn here ...",
 "earlier": [{"t": 20.0, "decision": "dismiss", "reason": "..."}]}
```

* `groups`: only values ≥ 0.1. `top_labels`: the `event.top_labels` (5) classes with the highest max score, ≥ 0.10.
* `earlier`: the last 3 decisions of this ride (omit when empty).
* Reply: `{"decision": "dismiss" | "check_in" | "alert", "severity": 1-3, "reason": "..."}`.
  Unreadable reply → treat as `check_in`.
* Show `usage.total_tokens` in the token counter. Measured: about 720 tokens per judged event.
* Operator Q&A: one more call with that event's JSON as context and the operator's question (see `judge.py`).

### 2.7 What the UI gets

Each decision: time, `dismiss` (grey) / `check_in` (ask the rider) / `alert` (silent operator alert), source
(`rule`, `llm`), one line reason, tokens used. For the parity check, write them as JSON lines and compare with
`expected/<id>.json` (same times within 0.5 s, same decisions).

---

## 3. Unity: scenario files

`shared/scenarios/demo/*.json` (format from plan 1.2):

```json
{"id": "demo_2_roadworks", "route": "routes/epfl_to_gare.geojson", "speed_kmh": 50, "duration_s": 46,
 "background": [{"clip": "cabin/drive_050kph_mic14.wav", "gain_db": 0, "loop": true}],
 "events": [{"t": 2.0, "clip": "demo/demo_2_roadworks_part1.wav", "gain_db": 0, "text": "..."},
            {"t": 12.7, "gps": "bypass", "dur": 36, "offset_m": 350}]}
```

* Mix: every clip is 16 kHz mono and already level matched; apply `gain_db`, add, clip to [-1, 1]. Background clips
  loop between `from` and `to` (default whole ride). A background with `idle_clip` follows the car: full driving
  noise above 20 km/h, cross faded to the idle clip (at `idle_gain_db`) as the speed drops to 0.
* GPS events: `stop` (brake gently, stay stopped `dur` s, default until the end), `sudden_stop` (40 → 0 km/h in
  about 1.3 s, default 10 s), `detour` (turn off the route and keep driving away), `bypass` (swing `offset_m` to the
  side and be back on the route after `dur` s). Reference: `ml/earshot/render.py`.
* **Simplest path:** ignore the mixing and GPS logic and replay `rides/<id>.wav` (play aloud + stream in 0.5 s
  chunks) and `rides/<id>.gps.jsonl` (one line per 0.5 s). That is exactly what the Python numbers were measured on.
* Timing that matters for your own rides:
  * leaving the route is flagged ~16 s after the turn at 50 km/h (150 m + 5 s hold); a stop is "long" after 30 s
    and only counts farther than 60 m from the start and end of the route;
  * a spoken explanation (road works, engine trouble) must fall within the 120 s before the flag;
  * new routes: drop a GeoJSON LineString (`[lon, lat]`) in `shared/routes/` and reference it in the scenario.
