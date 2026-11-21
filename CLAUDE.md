# Earshot

Frugal, privacy first safety alerts for taxi rides (AXA "Frugal use of Multimodal AI" hackathon, LauzHack).
An on device sound model listens; only when it hears a possible threat does a small text LLM get a short
summary (sound labels, Vosk transcript, GPS context) and decide: dismiss / check in / silent operator alert.

* Product and demo spec: `earshot_baseline_plan.md` (source of truth for the design).
* What is built, what was found, what is next: `docs/ML_STATUS.md`. **Read it before working on `ml/`.**
* Contract for the Android and Unity teammates: `docs/INTEGRATION.md`.

## Layout

| Path | Owner | What |
|---|---|---|
| `shared/` | everyone | The contract between components: `config.json` (thresholds, groups, prompts), `head.json` (v2 head weights), `scenarios/`, `routes/`. The app, Unity and Python all read these same files. |
| `ml/` | ML | Python reference pipeline, data building, evaluation. `ml/earshot/` is the library, `ml/scripts/` the entry points. |
| `android/` | Android teammate | Kotlin + Compose app (not started here). |
| `unity/` | Unity teammate | Car scene, scenario player, socket sender (not started here). |
| `docs/` | everyone | Status and integration notes. |

Each component lives in its own folder so it can be cloned and launched alone on another laptop.
Work happens on branches (`ml-pipeline` for the ML work); commit or push only when asked.

## ML quick reference

```
cd ml
uv sync                                   # Python 3.12 env in ml/.venv
uv run python scripts/<script>.py         # always run through uv
```

Rebuild order (everything under `ml/data/` is git ignored and reproducible):
`fetch_data` → `fetch_real` → `make_dialogues` → `build_manifest` → `make_scenarios` → `train_head`
→ `calibrate --write` → `make_demo` → `evaluate` → `baseline_audio_llm` → `export_demo_pack`.
Listen to anything: clips in `ml/data/clips/<category>/`, full rides in `ml/data/rides/`, demo rides in `ml/data/demo/`.
Details and current numbers are in `docs/ML_STATUS.md`.

## Rules

* **Secrets:** the OpenRouter key lives in the repo root `.env` (`OPENROUTER_API_KEY=` or `OpenRouter_api=`),
  for Android in `local.properties`. Never commit either, never print the key.
* **Budget:** the OpenRouter account has only $10 in total. All LLM calls go through `ml/earshot/llm.py`
  (disk cache + `usage.jsonl` log + hard cap `EARSHOT_BUDGET_USD`, default $3). Never call the API around it,
  and check `spent_usd()` before any batch job.
* **Tune / test split is frozen** (by source file, in `ml/data/manifest.csv`). Thresholds, the head and prompts are
  tuned on tune rides only; test rides are only for reported numbers. Never move clips between splits.
* **`shared/config.json` is the single source of thresholds and prompts.** Do not hard code them in Python or
  Kotlin. Bump `version` when it changes and tell the teammates.
* **The Python pipeline is the reference for the app.** If `ml/earshot/pipeline.py`, `events.py` or `detector.py`
  change behaviour, update `docs/INTEGRATION.md` in the same change.
* **Only claim what is measured.** Numbers for the pitch come from `ml/results/*.json`; token counts are the
  ones the API reported.
* **Privacy stance of the product:** sound events, never emotions or personality; nothing stored except the
  rolling 30 s buffer; ground truth (`expect`) is never sent to the app.
* No AI attribution in commits, PRs, code or docs.
