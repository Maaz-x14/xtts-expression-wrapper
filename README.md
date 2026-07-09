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

| Tag | Reference Source | RAVDESS Code |
|---|---|---|
| `<neutral>` | neutral, normal intensity | `01-01` |
| `<happy>` | happy, strong intensity | `03-02` |
| `<sad>` | sad, strong intensity | `04-02` |
| `<angry>` | angry, strong intensity | `05-02` |
| `<whisper>` | calm (proxy), normal intensity | `02-01` |

---

## Local Setup (Ubuntu 24.04, CPU)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/xtts-expression-wrapper.git
cd xtts-expression-wrapper

# 2. Run setup (installs Python 3.11, venv, torch CPU, TTS)
chmod +x setup.sh
./setup.sh

# 3. Activate venv
source venv/bin/activate

# 4. Download RAVDESS reference clips (~215MB, one-time)
python scripts/download_ravdess.py

# 5. Run
python main.py "<happy>Hello, how are you today?</happy>"
```

**Note**: CPU inference takes 30–90 seconds per sentence. Use Kaggle for GPU speed.

---

## Kaggle Setup (GPU — recommended for inference)

```python
# In a Kaggle notebook cell:
!git clone https://github.com/YOUR_USERNAME/xtts-expression-wrapper.git
%cd xtts-expression-wrapper

!pip install TTS==0.22.0 -q
!pip install requests -q

!python scripts/download_ravdess.py
!python main.py "<happy>Hello, how are you today?</happy>"
```

Kaggle T4 GPU reduces inference to ~3–5 seconds per sentence.

---

## Project Structure

```
xtts-expression-wrapper/
├── reference_clips/         # RAVDESS clips (not committed, download via script)
│   ├── neutral.wav
│   ├── happy.wav
│   ├── sad.wav
│   ├── angry.wav
│   └── whisper.wav
├── output/                  # Generated audio (not committed)
├── src/
│   ├── tag_parser.py        # <happy>text</happy> → ("happy", "text")
│   ├── reference_map.py     # emotion → wav path
│   └── wrapper.py           # XTTS-v2 inference orchestration
├── scripts/
│   └── download_ravdess.py  # One-time RAVDESS download
├── main.py                  # CLI entrypoint
├── requirements.txt
├── setup.sh
└── .gitignore
```

---

## Dataset Credit

Reference clips sourced from **RAVDESS**:
> Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
> *PLOS ONE* 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391
> License: CC BY-NC-SA 4.0
