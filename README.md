# XTTS Expression Wrapper — Phase 1

Proof-of-concept wrapper around XTTS-v2 that adds expression tags via reference audio swapping.
No model weights modified. English testing only at this stage.

---

## Architecture

```
<happy>Hello world</happy>
         ↓
    tag_parser.py        → ("happy", "Hello world")
         ↓
    reference_map.py     → reference_clips/happy.wav
         ↓
    XTTS-v2 inference    → tts_to_file(text, speaker_wav=happy.wav, language="en")
         ↓
    output/output.wav
```

---

## Supported Tags

### Base emotions (8, direct RAVDESS clip + whisper)

| Tag             | Reference Source            | RAVDESS Code |
| --------------- | --------------------------- | ------------ |
| `<neutral>`   | neutral, normal intensity   | `01-01`    |
| `<calm>`      | calm, strong intensity      | `02-02`    |
| `<happy>`     | happy, strong intensity     | `03-02`    |
| `<sad>`       | sad, strong intensity       | `04-02`    |
| `<angry>`     | angry, strong intensity     | `05-02`    |
| `<fearful>`   | fearful, strong intensity   | `06-02`    |
| `<disgust>`   | disgust, strong intensity   | `07-02`    |
| `<surprised>` | surprised, strong intensity | `08-02`    |
| `<whisper>`   | manual clip (not RAVDESS)   | —           |

### Sub-emotion aliases (20, resolve to nearest base emotion above)

No clip of their own — routed to a base emotion's reference clip in `reference_map.py`.
**Unvalidated by listening tests** (that's Step 2 of the roadmap, not done yet).

| Alias tag                                     | Resolves to   |
| --------------------------------------------- | ------------- |
| `joy`, `excitement`, `contentment`      | `happy`     |
| `grief`, `loneliness`, `disappointment` | `sad`       |
| `rage`, `frustration`, `irritation`     | `angry`     |
| `anxiety`, `nervousness`, `panic`       | `fearful`   |
| `contempt`, `revulsion`, `disdain`      | `disgust`   |
| `shock`, `amazement`, `disbelief`       | `surprised` |
| `serenity`, `relaxation`                  | `calm`      |

### Inline prosodic tags

| Tag                   | Mechanism                     |
| --------------------- | ----------------------------- |
| `<pause=300ms>`     | numpy zeros at 24kHz for Xms  |
| `<break>`           | Fixed 500ms silence           |
| `<silence>`         | Fixed 1000ms silence          |
| `<fast>text</fast>` | librosa time_stretch rate=1.5 |
| `<slow>text</slow>` | librosa time_stretch rate=0.7 |

---

## Local Setup (Ubuntu 24.04, CPU)

```bash
# 1. Clone
git clone https://github.com/Maaz-x14/xtts-expression-wrapper.git
cd xtts-expression-wrapper

# 2. Run setup (installs Python 3.11, venv, torch CPU, TTS)
chmod +x setup.sh
./setup.sh

# 3. Activate venv
source venv/bin/activate

# 4. Download RAVDESS reference clips (~199MB, one-time)
python scripts/download_ravdess.py

# 5. Run
python main.py "<happy>Hello, how are you today?</happy>"

# 5b. Run with explicit output path
python main.py "<happy>Hello, how are you today?</happy>" --output custom_name.wav
python main.py "<happy>Hello, how are you today?</happy>" --output /full/path/to/file.wav
```

**Note**: CPU inference takes 30–90 seconds per sentence. Use Kaggle for GPU speed.

> ⚠️ The `--output` flag above assumes `main.py` exposes an `--output`/`-o` argument
> that is passed through to `ExpressionWrapper.synthesize(output_filename=...)`.
> Confirm this is wired up in `main.py` before relying on it — the wrapper method
> itself already accepts a filename, but the CLI layer needs to forward it.

---

## Kaggle Setup (GPU — recommended for inference)

```python
# In a Kaggle notebook cell:
!git clone https://github.com/YOUR_USERNAME/xtts-expression-wrapper.git
%cd xtts-expression-wrapper
!pip install TTS==0.22.0 -q
!pip install requests -q
!python scripts/download_ravdess.py
!python main.py "<happy>Hello, how are you today?</happy>" --output /kaggle/working/output.wav
```

Kaggle T4 GPU reduces inference to ~3–5 seconds per sentence.
Use `--output /kaggle/working/...` so generated audio lands in Kaggle's persisted output directory instead of the ephemeral working copy.

---

## Project Structure

```
xtts-expression-wrapper/
├── reference_clips/         # RAVDESS clips (not committed, download via script)
│   ├── neutral.wav
│   ├── calm.wav
│   ├── happy.wav
│   ├── sad.wav
│   ├── angry.wav
│   ├── fearful.wav
│   ├── disgust.wav
│   ├── surprised.wav
│   └── whisper.wav          # manual, not from download script
├── output/                  # Generated audio (not committed)
├── src/
│   ├── tag_parser.py        # <happy>text</happy> → ("happy", "text"); also handles aliases
│   ├── reference_map.py     # emotion (base or alias) → wav path
│   └── wrapper.py           # XTTS-v2 inference orchestration
├── scripts/
│   └── download_ravdess.py  # One-time RAVDESS download (8 base emotions)
├── main.py                  # CLI entrypoint
├── requirements.txt
├── setup.sh
└── .gitignore
```

---

## Roadmap status

```
Stage 1 ✅ — Level 1 wrapper (emotion via reference swap)
Stage 2 ✅ — Prosodic inline tags (pause, break, silence, fast, slow)
Stage 3 (in progress) — Emotion expansion
  ✅ Step 1 — Tier 1 emotions added (8 RAVDESS base + 20 sub-emotion aliases)
  ⬜ Step 2 — Subjective evaluation (listen + score all 8 base emotions)
  ⬜ Step 3 — Tier 2 emotions (EmoV-DB) — conditional on Step 2 results
  ⬜ Step 4 — Phase 2 measurement before Level 3
Stage 4 — Phase 2 measurement / evaluation
Stage 5 — Level 3: Latent mixing (intercept Perceiver, blend α)
Stage 6 — Urdu adaptation (fine-tuned Urdu XTTS-v2 + SEMOUR+)
```

---

## Dataset Credit

Reference clips sourced from **RAVDESS**:

> Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
> *PLOS ONE* 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391
> License: CC BY-NC-SA 4.0
