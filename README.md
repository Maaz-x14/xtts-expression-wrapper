# XTTS Expression Wrapper — Phase 2

Expressive Urdu TTS pipeline built on a locally fine-tuned XTTS v2 checkpoint (`Agri-TTS/`),
served via Auralis with precomputed latent caching and SEMOUR+ Urdu reference clips.

---

## Architecture

```
<happy>آپ کیسے ہیں؟</happy>
         ↓
    tag_parser.py        → ("happy", "آپ کیسے ہیں؟")
         ↓
    reference_map.py     → reference_clips/happy/ref.wav   (SEMOUR+ composite, ~6s)
         ↓
    latent cache         → gpt_cond_latent + speaker_embedding  (precomputed at startup)
         ↓
    Auralis / XTTS v2    → TTSRequest(text, language="ur", latents=cached)
         ↓
    output/output.wav
```

### Key design decisions

- **Reference clips**: 4 longest SEMOUR+ clips per emotion, concatenated into one composite
  `ref.wav` per emotion. Concatenation happens once at prep time, not at inference time.
- **Latent cache**: `gpt_cond_latent` and `speaker_embedding` computed from `ref.wav` at
  server startup. Zero re-encoding per synthesis request.
- **Text chunking**: Auralis splits long input at sentence boundaries, handling XTTS's
  ~250-char GPT context limit transparently.
- **Single actor**: All reference clips sourced from one SEMOUR+ actor for consistent
  voice identity across all emotions.

---

## Supported Tags

### Base emotions (8)

| Tag             | SEMOUR+ Source | Notes                                     |
| --------------- | -------------- | ----------------------------------------- |
| `<neutral>`   | Neutral/       | Also written as neutral.wav flat fallback |
| `<calm>`      | Boredom/       | Closest low-arousal proxy in SEMOUR+      |
| `<happy>`     | Happiness/     |                                           |
| `<sad>`       | Sadness/       |                                           |
| `<angry>`     | Anger/         |                                           |
| `<fearful>`   | Fearful/       |                                           |
| `<disgust>`   | Disgust/       |                                           |
| `<surprised>` | Surprise/      |                                           |

### Sub-emotion aliases (resolve to nearest base emotion)

| Alias                                         | Resolves to   |
| --------------------------------------------- | ------------- |
| `joy`, `excitement`, `contentment`      | `happy`     |
| `grief`, `loneliness`, `disappointment` | `sad`       |
| `rage`, `frustration`, `irritation`     | `angry`     |
| `anxiety`, `nervousness`, `panic`       | `fearful`   |
| `contempt`, `revulsion`, `disdain`      | `disgust`   |
| `shock`, `amazement`, `disbelief`       | `surprised` |
| `serenity`, `relaxation`, `boredom`     | `calm`      |
| `whisper`                                   | `neutral`   |

### Inline prosodic tags

| Tag                   | Mechanism                     |
| --------------------- | ----------------------------- |
| `<pause=300ms>`     | numpy zeros at 24kHz for Xms  |
| `<break>`           | Fixed 500ms silence           |
| `<silence>`         | Fixed 1000ms silence          |
| `<fast>text</fast>` | librosa time_stretch rate=1.5 |
| `<slow>text</slow>` | librosa time_stretch rate=0.7 |

---

## Setup

### Prerequisites

- Python 3.10+
- `Agri-TTS/` checkpoint directory (contains `config.json`, `model.pth`, `vocab.json`)
- `SEMOUR+_data/` dataset directory

### Install

```bash
git clone https://github.com/Maaz-x14/xtts-expression-wrapper.git
cd xtts-expression-wrapper

pip install -r requirements.txt
```

### Prepare reference clips (one-time)

```bash
# Audit candidate actors — listen to audition/ and pick
python scripts/select_clips.py --semour-dir SEMOUR+_data --audit

# Build reference_clips/ from chosen actor (Actor 5 recommended)
python scripts/select_clips.py --semour-dir SEMOUR+_data --build --actor 5
```

### Set model path (if Agri-TTS/ is not in repo root)

1. ```bash
   export XTTS_MODEL_DIR=/path/to/Agri-TTS
   ```

### Run

1. ```bash
   python main.py "<happy>السلام علیکم، آپ کیسے ہیں؟</happy>"
   python main.py "<sad>مجھے آپ کی یاد آتی ہے <pause=500ms> بہت زیادہ۔</sad>"
   python main.py "<angry>یہ بالکل <pause=300ms> ناقابل قبول ہے!</angry>" --output angry_test.wav
   python main.py "<neutral>براہ کرم <slow>آہستہ آہستہ</slow> بولیں۔</neutral>"

   # Force single-clip mode (skip composite, use first clip only)
   python main.py "<happy>السلام علیکم</happy>" --single-clip
   ```

**Note**: CPU inference is slow (~30–90s per sentence). Use Kaggle/Colab for GPU.

---

## Kaggle / Colab Setup (GPU)

```python
!git clone https://github.com/Maaz-x14/xtts-expression-wrapper.git
%cd xtts-expression-wrapper
!pip install -r requirements.txt -q

# Upload Agri-TTS/ checkpoint and reference_clips/ (pre-built) to Kaggle dataset
# Then set the path:
import os
os.environ["XTTS_MODEL_DIR"] = "/kaggle/input/agri-tts/Agri-TTS"

!python main.py "<happy>السلام علیکم</happy>" --output /kaggle/working/output.wav
```

---

## Project Structure

```
xtts-expression-wrapper/
├── Agri-TTS/                    # Fine-tuned XTTS v2 checkpoint (not committed)
│   ├── config.json
│   ├── model.pth
│   └── vocab.json
├── reference_clips/             # SEMOUR+ composites (not committed, built via script)
│   ├── neutral.wav              # flat fallback copy
│   ├── angry/ref.wav
│   ├── calm/ref.wav
│   ├── disgust/ref.wav
│   ├── fearful/ref.wav
│   ├── happy/ref.wav
│   ├── neutral/ref.wav
│   ├── sad/ref.wav
│   └── surprised/ref.wav
├── output/                      # Generated audio (not committed)
├── src/
│   ├── tag_parser.py            # Tag parsing — unchanged from Phase 1
│   ├── reference_map.py         # emotion → reference_clips/<emotion>/ref.wav
│   └── wrapper.py               # Auralis inference + latent cache
├── scripts/
│   ├── select_clips.py          # SEMOUR+ clip selection and composite builder
│   └── download_ravdess.py      # Phase 1 only — not used in Phase 2
├── main.py
├── requirements.txt
└── setup.sh
```

---

## Roadmap

```
Phase 1 (English baseline)
  ✅ Level 1 wrapper — emotion via reference-clip swap (Coqui TTS.api)
  ✅ Prosodic inline tags — pause, break, silence, fast, slow
  ✅ RAVDESS 8-emotion clip set
  ✅ Subjective evaluation — distinctiveness strong (~4.6), convincingness weak (~2.8)
  ✅ Multi-clip A/B test — improved most emotions; angry/disgust ceiling remained
  ✅ Root cause confirmed — Perceiver entangles speaker+prosody (architectural ceiling)

Phase 2 (Urdu production)
  ✅ Backend swap — Coqui TTS.api → Auralis (latent caching, chunking, concurrency)
  ✅ Model swap — base XTTS v2 → Agri-TTS (Urdu fine-tune, language="ur")
  ✅ Reference clips — RAVDESS (English) → SEMOUR+ (Urdu, Actor 5)
  ✅ Clip strategy — concatenated composites (~6s) replacing single clips
  ⬜ First synthesis test on Urdu input
  ⬜ Subjective evaluation — repeat Phase 1 scoring on Urdu output
  ⬜ Temperature sweep on weak emotions (angry, disgust)

Phase 3 (planned)
  ⬜ Level 3 — Latent mixing (intercept Perceiver, blend α between emotion latents)
  ⬜ REST API wrapper (FastAPI, Auralis async backend)
  ⬜ Integration with M2M100 transliteration pipeline
       User Speech → STT → Urdu Text → M2M100 → XTTS TTS → Audio

    
```

---

## TODO — Auralis checkpoint conversion (deferred)

Status: NOT started. Deferred until GPU test (Phase 2.5) confirms whether current
XTTS conditioning-based emotion transfer is viable, or Phase 3 (real emotion-conditioned
fine-tuning) is needed instead.

Auralis can't load `Agri-TTS/model.pth` as-is — it's a Coqui *trainer* checkpoint
(raw state dict + trainer config.json), not the safetensors + HF-config format Auralis expects.

If revisited, conversion requires:

1. Read Auralis's model loader source to confirm exact expected config schema
   (hidden_size, vocab_size, num_layers, etc. — likely different key names than
   Coqui's trainer config.json).
2. `torch.load('Agri-TTS/model.pth')` → extract state dict → remap/rename any
   keys that differ between Coqui's XTTS module naming and Auralis's expected naming
   (if any — needs checking, don't assume 1:1).
3. `safetensors.torch.save_file()` the remapped state dict.
4. Hand-write an HF-style config.json with the right fields translated from
   Coqui's config.json (gpt_cond_len, gpt_layers, etc. → whatever Auralis calls them).
5. Test load in Auralis, verify inference actually runs before assuming conversion succeeded.

Reason to do this: only if Auralis provides a concrete capability current Coqui TTS
setup lacks (e.g. proven better serving perf under real concurrent load). NOT for
emotion quality — Auralis doesn't change conditioning-latent mechanism or ceiling.

## Dataset Credits

**SEMOUR+** (reference clips):

> Scripted EMOtional Speech Repository for Urdu.
> 27,640 utterances, 24 native speakers, 8 emotions.
> https://link.springer.com/article/10.1007/s10579-022-09610-7

**RAVDESS** (Phase 1 baseline, no longer used):

> Livingstone SR, Russo FA (2018). PLOS ONE 13(5): e0196391.
> https://doi.org/10.1371/journal.pone.0196391 — CC BY-NC-SA 4.0
