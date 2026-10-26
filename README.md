# ARTI5001 Assignment 3: Single-Speaker TTS (LJSpeech)

Private group repository. Due **Mon 2 Nov 2026, 23:59**. Do not make this repository public.

## Layout
```
admin/        task sheet + official notebook template (reference only)
data/         LJSpeech-1.1/ goes here (git-ignored: never commit the dataset)
notebook/     A3_TTS.ipynb: THE submission notebook (sections mirror the report)
configs/      expA.json, expB.json (written by the notebook)
scripts/      synth.py (CLI text → wav), helpers
wav_examples/ generated audio: expA_01.wav … expB_10.wav
results/      splits/, metrics.csv, figures/, environment.json
report/       main.tex + references.bib (Overleaf), figures/
docs/         TASK_SPLIT.md · RUBRIC_HD_MAP.md · RESEARCH.md · GIT_GUIDE.md · miro_board.csv
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download LJSpeech 1.1 from https://keithito.com/LJ-Speech-Dataset/ and extract to data/LJSpeech-1.1/
jupyter notebook notebook/A3_TTS.ipynb
```

## How we work
1. Read `docs/TASK_SPLIT.md` (who owns what) and `docs/RUBRIC_HD_MAP.md` (what HD needs).
2. Every report claim must point to a notebook cell that produced the evidence.
3. One branch per task (`m1/...`, `m2/...`, `m3/...`), with a PR into `main` reviewed by one teammate. See `docs/GIT_GUIDE.md`.

## Submission build (final day)
`<group>_A3.zip` = `<group>_A3.ipynb` + `wav_examples/` · `<group>_A3.pdf` · `<group>_A3.pptx`
