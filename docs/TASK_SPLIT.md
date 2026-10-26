# Task split: 3 members, equal load

**Principle: vertical ownership.** Each member owns one slice end to end: the notebook code, the matching
report section and the matching presentation slides. The viva marks you on *the part you presented*,
so you should present what you built and wrote yourself.

| | **Member 1: Data & Communication** | **Member 2: Model & Pipeline** | **Member 3: Experiments & Evaluation** |
|---|---|---|---|
| **Code (35)** | **1.1** Dataset & pre-processing (9) · **1.4** coding practice, env info, attribution, AI statement (~2) | **1.2** End-to-end TTS pipeline, `synthesize()` entry point (11) | **1.3** Experimental methodology (9) · **1.4** objective metrics: RTF/latency/WER/MCD (~4) |
| **Code share** | ~11 | ~11 | ~13 |
| **Report (35)** | **2.1** Exec summary + intro (7) · data protocol para in 2.3 · conclusion + scaling (part of 2.5) · LaTeX integrator/editor | **2.2** System design + diagram (7) · **2.5** failure modes + improvement proposals (~5) | **2.3** Experimental design + results table (7) · **2.4** objective metrics + trade-offs (~4) |
| **Report share** | ~12 | ~12 | ~11 |
| **Qualitative evidence** | Picks the ≥10 prompts (incl. hard cases), runs the blind listening comparison | Generates the wavs via the pipeline | Spectrogram/alignment figures |
| **Presentation** | Problem, dataset, pre-processing, split | Architecture, pipeline, live demo | Experiments, results table, trade-offs |
| **Viva prep** | sample-rate policy, trimming, normalisation, split leakage | why this model/vocoder, how inference works, failure causes | why this controlled variable, fairness, RTF protocol, metric validity |

**Shared by everyone:** each member runs one of the training/inference jobs at least once (so all three
can answer "how was it trained?"). Everyone reviews everyone else's PRs. Everyone signs the contribution appendix.

## Milestones (due Mon 2 Nov 2026, 23:59)

| Date | Milestone | Owner |
|---|---|---|
| Mon 26 Oct | Repo set up, everyone cloned, split agreed, method chosen | All |
| Tue 27 Oct | 1.1 data pipeline + split files committed (`results/splits/`) | M1 |
| Wed 28 Oct | Baseline `synthesize("text") -> wav` works (pretrained or expA) | M2 |
| Thu 29 Oct | expA and expB trained/fine-tuned; metric code ready | M2 + M3 |
| Fri 30 Oct | Metrics table + ≥10 wavs per experiment + figures | M3 + M1 |
| Sat 31 Oct | Report draft, all sections filled | All |
| Sun 1 Nov | Slides + dry-run presentation, restart-and-run notebook check | All |
| Mon 2 Nov | Zip + PDF + PPTX submitted (one person submits) | M1 |

## Agreed decisions (fill in together)
- Group number: `__`
- Method/toolkit: `__`
- Controlled variable (expA vs expB): `__`
- Compute: `__` (local M4 Mac / Colab GPU / uni GPU)
