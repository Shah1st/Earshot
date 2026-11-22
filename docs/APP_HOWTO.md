# Earshot app: how to use what the ML side provides

Task oriented companion to `docs/INTEGRATION.md` (the full contract). Everything referenced is in `demo_pack/`.

## Run a demo ride and check your result

1. Play `demo_pack/rides/<id>.wav` out loud and stream the same samples in 0.5 s chunks (8,000 samples, PCM16).
2. With each chunk send the line of `demo_pack/rides/<id>.gps.jsonl` that has the same `t` (gps first, then audio).
3. Compare what the app decides with `demo_pack/expected/<id>.json`: same decisions, times within about 0.5 s.
4. If it differs, check `config.json` `version` first (the pack README states the version it was built with).

Seven rides: `demo_1` to `demo_5` use synthesised voices; `demo_6_abduction_rec` and `demo_7_knife_quiet_rec` are our
own recorded voices.

## Show the comparison with an always-listening audio LLM

1. Read `config.baseline` (model, prompt, `chunk_s` 30, `hop_s` 25).
2. Either replay the stored answers: `demo_pack/expected_baseline/<id>.json` is a list of chunks with `a` (start
   second), `decision`, `reason` and `usage.total_tokens`. Reveal each one when the ride reaches `a + 30` s (or the
   end of the ride) and add its tokens to the baseline counter. No API cost, always the same on stage.
3. Or call it live: when 30 s of audio have arrived, then every 25 s, send the last 30 s as a WAV (base64) with
   `baseline.prompt`; at the end send the unfinished chunk. Request format: `docs/INTEGRATION.md` section 2.7.
4. Put the two token counters side by side. Ride 4 is the clearest picture: Earshot 0, audio LLM 1,231.

## Show decisions on the operator screen

* `alert` = red card, `check_in` = amber card ("needs a look"), `dismiss` = grey line in the log. No rider prompt.
* Each decision has a one line `reason`; `source` tells whether the LLM (`llm`) or a rule (`rule`) decided.
* Token counter: add `usage.total_tokens` of every judge call. Rule decisions cost 0.

## Decide without the LLM in two cases

* A sound event with at most 2 transcript words around it: `dismiss` if a laughter or cough label scored ≥ 0.2 during
  the event, `alert` if a GPS anomaly is active, otherwise `check_in`.
* A GPS event with at most 2 words in the last 120 s: `check_in`.

Details and the exact order of checks: `docs/INTEGRATION.md` section 2.5.

## Turn the phrase trigger on or off

`config.asr.keyword_trigger`. On (current): a phrase from `config.asr.keywords` in the transcript opens an event.
Off: Vosk still feeds the judge, but only sounds and GPS anomalies open events (cheaper, fewer false alerts,
catches less). The expected outputs in the pack are for the value in the pack's `config.json`.

## Keep events from piling up

A new event may open 10 s after the previous one **opened** (`config.event.cooldown_s`). Triggers while an event is
open belong to it; triggers do not extend the 10 s.

## Try the other sound head

`shared/config_real.json` + `shared/head_real.json`: a head trained only on real recordings. Same code path, just
load that config and head file. It is more conservative; with it `demo_1` is caught by the phrase, not by the sound.

## Add your own recording as a ride (ML side)

Put `ml/data/recorded/<ride id>.ogg` (one file per scene, any format), run `make_recorded_demo.py`, then
`evaluate.py --split demo_recorded -v`, then `export_demo_pack.py`. Scene ids and their words are in
`ml/scripts/make_demo.py` and `make_recorded_demo.py`.
