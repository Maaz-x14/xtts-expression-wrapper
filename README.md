# XTTS Expression Wrapper

Expressive TTS pipeline wrapping XTTS v2 with emotion-tagged input, prosodic
inline tags, and a dual-model / dual-language comparison harness (stock XTTS v2
vs. a locally fine-tuned Urdu checkpoint, English vs. Urdu reference clips).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The Two Models](#the-two-models)
- [Supported Tags](#supported-tags)
  - [Base emotions](#base-emotions-8)
  - [Sub-emotion aliases](#sub-emotion-aliases)
  - [Inline prosodic tags](#inline-prosodic-tags)
- [Setup](#setup)
- [Running](#running)
  - [CLI (`main.py`)](#cli-mainpy)
  - [Server (`server/server.py`)](#server-serverserverpy)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Known Issues / Open Items](#known-issues--open-items)
- [Roadmap](#roadmap)
- [Dataset Credits](#dataset-credits)

---

## Overview

Input text wrapped in an emotion tag is routed to a pre-recorded reference
clip for that emotion, which XTTS uses as a conditioning reference (speaker
timbre + prosody) for synthesis. Inline tags handle pauses and speed changes
without touching the model.

```
<happy>آپ کیسے ہیں؟</happy>
```

is the whole interface — one tag around the sentence, optional inline tags
inside it.

---

## Architecture

```
<happy>آپ کیسے ہیں؟</happy>
         ↓
    tag_parser.py         → ("happy", [segments])
         ↓
    reference_map.py      → reference_clips/<language>/happy/ref.wav
         ↓
    latent cache           → gpt_cond_latent + speaker_embedding
                              (computed once per emotion, at startup)
         ↓
    XTTS v2 (Coqui)        → model.inference(text, language=<en|ur>, latents=cached)
         ↓
    output/<language>/<filename>.wav
```

### Key design decisions

- **Reference clips**: 4 clips per emotion, concatenated into one composite
  `ref.wav` per emotion per language (`SILENCE_GAP_S = 0.3` gap between
  clips). Concatenation happens once, at build time — not per inference call.
- **Latent cache**: `gpt_cond_latent` / `speaker_embedding` are computed once
  per emotion when the model loads (`ExpressionWrapper._build_cache()`).
  Synthesis calls never re-encode reference audio.
- **One actor per language**: RAVDESS Actor 01 (English), SEMOUR+ Actor 5
  (Urdu) — consistent voice identity across all emotions within a language.
- **Text chunking**: manual sentence-boundary splitting before inference,
  since XTTS's GPT context is ~200–250 chars.
- **Urdu tokenizer patch**: XTTS's stock tokenizer hardcodes a language
  allowlist that doesn't include `ur`. `wrapper.py` monkeypatches
  `VoiceBpeTokenizer.preprocess_text` to route `ur` through `basic_cleaners`
  (lowercase + whitespace collapse, no transliteration) — the same fallback
  Coqui's own maintainers use for `hi` (Hindi), which is also unimplemented
  upstream.

---

## The Two Models

`ExpressionWrapper(model=..., language=...)` takes both independently — any
model × language combination is valid, though `finetuned` + `urdu` and
`base` + `english` are the two combinations actually validated end-to-end so
far.

| `--model`     | Loads                                                              | Source |
| ------------- | ------------------------------------------------------------------ | ------ |
| `base`      | Stock `tts_models/multilingual/multi-dataset/xtts_v2`, auto-downloaded via `TTS.api` | Coqui / HuggingFace |
| `finetuned` | Local checkpoint in `Agri-TTS/` (`config.json`, `model.pth`, `vocab.json`) | Locally fine-tuned on Urdu |

Both converge to the same raw `Xtts` instance on `self.model` — every
downstream call (`get_conditioning_latents`, `inference`) is identical
regardless of which one loaded.

`--language` is independent of `--model` and only decides which
`reference_clips/<language>/` set is used and which XTTS language code
(`en` / `ur`) is passed to `inference()`.

---

## Supported Tags

### Base emotions (8)

| Tag             | English source (RAVDESS) | Urdu source (SEMOUR+) |
| --------------- | ------------------------- | ---------------------- |
| `<neutral>`   | Neutral                   | Neutral                |
| `<calm>`      | Calm                       | Boredom (closest low-arousal proxy) |
| `<happy>`     | Happy                      | Happiness               |
| `<sad>`       | Sad                        | Sadness                  |
| `<angry>`     | Angry                      | Anger                    |
| `<fearful>`   | Fearful                    | Fear                      |
| `<disgust>`   | Disgust                    | Disgust                   |
| `<surprised>` | Surprised                  | Surprise                  |

An unknown or missing emotion tag falls back to `neutral` (logged, not an
error). **Known bug**: the fallback path currently produces near-empty/garbage
audio for unrecognized tags in some cases — see
[Known Issues](#known-issues--open-items).

### Sub-emotion aliases

Resolve to the nearest base emotion above — same clip, different label:

| Alias                                     | Resolves to |
| ------------------------------------------ | ----------- |
| `joy`, `excitement`, `contentment`      | `happy`   |
| `grief`, `loneliness`, `disappointment` | `sad`     |
| `rage`, `frustration`, `irritation`     | `angry`   |
| `anxiety`, `nervousness`, `panic`       | `fearful` |
| `contempt`, `revulsion`, `disdain`      | `disgust` |
| `shock`, `amazement`, `disbelief`       | `surprised` |
| `serenity`, `relaxation`, `boredom`     | `calm`    |
| `whisper`                                 | `neutral` |

### Inline prosodic tags

Used inside an emotion tag, not as top-level tags:

| Tag                   | Effect                               |
| ---------------------- | ------------------------------------- |
| `<pause=300ms>`      | Silence, numpy zeros at 24kHz for Xms |
| `<break>`            | Fixed 500ms silence                   |
| `<silence>`          | Fixed 1000ms silence                  |
| `<fast>text</fast>` | `librosa.effects.time_stretch(rate=1.5)` |
| `<slow>text</slow>` | `librosa.effects.time_stretch(rate=0.7)` |

Example combining tags:

```
<sad>مجھے آپ کی یاد آتی ہے <pause=500ms> بہت زیادہ۔</sad>
```

---

## Setup

### Prerequisites

- Python 3.11
- `Agri-TTS/` checkpoint directory (`config.json`, `model.pth`, `vocab.json`)
  — only required for `--model finetuned`
- `reference_clips/english/` and `reference_clips/urdu/` — pre-built, or
  build from scratch (Urdu requires raw SEMOUR+ data; English downloads
  RAVDESS automatically)

### Install

```bash
git clone https://github.com/Maaz-x14/xtts-expression-wrapper.git
cd xtts-expression-wrapper

./setup.sh
source venv/bin/activate
```

`setup.sh` creates a Python 3.11 venv, installs CPU-only PyTorch, pins
`transformers` (newer versions removed `BeamSearchScorer`, which XTTS needs),
then installs the rest of `requirements.txt`.

### Build reference clips (one-time per language)

```bash
# English — downloads RAVDESS automatically, no-ops if already built
python scripts/build_reference_clips.py --language english

# Urdu — already built on disk for this project (SEMOUR+ Actor 5).
# Only needed if reference_clips/urdu/ is missing or you're rebuilding from
# raw SEMOUR+ data:
python scripts/build_reference_clips.py --language urdu --semour-dir /path/to/SEMOUR+_data

# Force a rebuild of an existing set
python scripts/build_reference_clips.py --language english --force
```

### Set fine-tuned model path (if `Agri-TTS/` isn't in repo root)

```bash
export XTTS_MODEL_DIR=/path/to/Agri-TTS
```

---

## Running

Two ways to run synthesis: the CLI (`main.py`) for one-off tests — reloads
the full model every invocation — or the server (`server/server.py`) for
repeated calls against one already-loaded model.

### CLI (`main.py`)

```bash
python main.py "<happy>السلام علیکم، آپ کیسے ہیں؟</happy>" --model finetuned --language urdu
python main.py "<sad>مجھے آپ کی یاد آتی ہے <pause=500ms> بہت زیادہ۔</sad>" --model finetuned --language urdu
python main.py "<angry>یہ بالکل <pause=300ms> ناقابل قبول ہے!</angry>" --model finetuned --language urdu --output angry_test.wav
python main.py "<happy>Hello there, how are you?</happy>" --model base --language english
```

Flags: `--model {base,finetuned}` (default `finetuned`), `--language
{english,urdu}` (default `urdu`), `--output <filename>` (default
`output.wav`, saved inside `output/<language>/`).

**Note**: CPU inference is slow (~30–90s per sentence). Use a GPU
environment (Kaggle/Colab) for faster iteration — set `--device` isn't
exposed as a flag; the wrapper auto-selects CUDA if available.

### Server (`server/server.py`)

Loads one model **once** at startup; `--model`/`--language` are startup args,
not per-request — restart the server to test a different combination.

```bash
python server/server.py --model finetuned --language urdu
# or
python server/server.py --model base --language english --port 8001
```

Startup does: load model → build latent cache for all 8 emotions → run one
throwaway warmup synthesis (primes first-call overhead so it doesn't skew
your first real timing) → start accepting requests. The server does not
accept connections until this completes.

**Concurrency**: single-worker, fully serialized — a request arriving while
one is in flight waits silently (no queue position, no "busy" response).
This matches sequential test-harness usage; it is not built for concurrent
load.

#### Endpoints

| Method | Path           | Purpose |
| ------ | -------------- | ------- |
| `GET`  | `/health`    | Liveness only — no model access, always 200 once the process is up |
| `GET`  | `/status`    | `{ready, model, language, device, cached_emotions}` |
| `POST` | `/synthesize` | `{"text": "<happy>...</happy>", "output_filename": "test_01.wav"}` → `{"output_path", "emotion", "duration_s"}` |

`/synthesize` saves the wav to `output/<language>/<output_filename>` and
returns the path + metadata as JSON — it does not stream raw audio bytes.

---

## Testing

With the server running:

```bash
# liveness
curl http://127.0.0.1:8000/health

# readiness + which emotions are cached
curl http://127.0.0.1:8000/status

# synthesize
curl -X POST http://127.0.0.1:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "<happy>السلام علیکم، آپ کیسے ہیں؟</happy>", "output_filename": "test_01.wav"}'
```

Expected `/synthesize` response:

```json
{"output_path": "output/urdu/test_01.wav", "emotion": "happy", "duration_s": 12.3}
```

To confirm requests are actually serialized (not just assumed), fire two
requests without waiting between them and check that the second one's
`[WRAPPER] Emotion...` log line only appears after the first finishes:

```bash
curl -X POST http://127.0.0.1:8000/synthesize -H "Content-Type: application/json" \
  -d '{"text": "<happy>test one</happy>", "output_filename": "a.wav"}' &
curl -X POST http://127.0.0.1:8000/synthesize -H "Content-Type: application/json" \
  -d '{"text": "<sad>test two</sad>", "output_filename": "b.wav"}' &
wait
```

---

## Project Structure

```
xtts-expression-wrapper/
├── Agri-TTS/                    # Fine-tuned XTTS v2 checkpoint (not committed)
│   ├── config.json
│   ├── model.pth
│   ├── vocab.json
│   └── ref2.wav                 # for future zero-shot cloning test (unverified, see Known Issues)
├── reference_clips/             # Composite reference clips (not committed, built via script)
│   ├── english/
│   │   ├── neutral.wav          # flat fallback copy
│   │   ├── angry/ref.wav
│   │   ├── calm/ref.wav
│   │   ├── disgust/ref.wav
│   │   ├── fearful/ref.wav
│   │   ├── happy/ref.wav
│   │   ├── neutral/ref.wav
│   │   ├── sad/ref.wav
│   │   └── surprised/ref.wav
│   └── urdu/                    # same layout as english/
├── output/
│   ├── english/
│   └── urdu/
├── src/
│   ├── tag_parser.py            # Emotion tag + inline prosodic tag parsing
│   ├── reference_map.py         # (emotion, language) → reference_clips/<language>/<emotion>/ref.wav
│   └── wrapper.py               # ExpressionWrapper — model load, latent cache, synthesis, tokenizer patch
├── scripts/
│   └── build_reference_clips.py # Builds reference_clips/ for both languages
├── server/
│   └── server.py                # FastAPI server — load once, serve many requests
├── main.py                      # CLI entrypoint — reloads model per invocation
├── requirements.txt
└── setup.sh
```

---

## Known Issues / Open Items

- **Unknown-emotion fallback bug**: an unrecognized tag (e.g. `<madeup>`)
  falls back to `neutral` per the parser, but has been observed to produce
  near-empty/garbage (~0.5s, silent) audio in practice rather than a proper
  neutral synthesis. Not yet root-caused. Deferred.
- **Zero-shot cloning** of the fine-tuned checkpoint (using `Agri-TTS/ref2.wav`)
  — unverified, not yet tested.
- **Temperature/sampling sweep** on `angry`/`disgust` — historically weak
  emotions on English/RAVDESS (~2.8/5 convincingness in Phase 1 subjective
  scoring, even after multi-clip reference improvements). Not yet re-run on
  Urdu/fine-tuned combos.
- **`SEMOUR_SPEAKER = "speaker_05"`** in `build_reference_clips.py` is a
  guessed SEMOUR+ folder-naming convention — dead code unless `urdu/` is
  ever rebuilt from raw SEMOUR+ data with `--force`.
- **CSV test-output harness** (test message + audio path columns, one file
  per model+language combo vs. combined) — not yet built, format not yet
  decided.

---

## Roadmap

```
Phase 1 (English baseline)
  ✅ Level 1 wrapper — emotion via reference-clip swap
  ✅ Prosodic inline tags — pause, break, silence, fast, slow
  ✅ RAVDESS 8-emotion clip set
  ✅ Subjective evaluation — distinctiveness strong (~4.6), convincingness weak (~2.8)
  ✅ Multi-clip A/B test — improved most emotions; angry/disgust ceiling remained
  ✅ Root cause confirmed — XTTS's Perceiver entangles speaker+prosody (architectural ceiling)

Phase 2 (Urdu production)
  ✅ Model swap — base XTTS v2 → Agri-TTS (Urdu fine-tune, language="ur")
  ✅ Reference clips — RAVDESS (English) → SEMOUR+ (Urdu, Actor 5), both retained
  ✅ Dual-model / dual-language wrapper (base|finetuned × english|urdu)
  ✅ Urdu tokenizer allowlist patch
  ✅ First synthesis test on Urdu input — confirmed working end-to-end
  ✅ FastAPI server — single load, serialized requests, warmup, health/status
  ⬜ Subjective evaluation — repeat Phase 1 scoring on Urdu output
  ⬜ Temperature sweep on weak emotions (angry, disgust)
  ⬜ CSV test-output harness

Phase 3 (planned)
  ⬜ Latent mixing (intercept Perceiver, blend α between emotion latents)
  ⬜ Integration with M2M100 transliteration pipeline
       User Speech → STT → Urdu Text → M2M100 → XTTS TTS → Audio
```

---

## Dataset Credits

**SEMOUR+** (Urdu reference clips):

> Scripted EMOtional Speech Repository for Urdu.
> 27,640 utterances, 24 native speakers, 8 emotions.
> https://link.springer.com/article/10.1007/s10579-022-09610-7

**RAVDESS** (English reference clips):

> Livingstone SR, Russo FA (2018). PLOS ONE 13(5): e0196391.
> https://doi.org/10.1371/journal.pone.0196391 — CC BY-NC-SA 4.0
