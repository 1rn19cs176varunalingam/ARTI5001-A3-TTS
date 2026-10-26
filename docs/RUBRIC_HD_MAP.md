# HD back-propagation map: report → notebook

Method: start from each **report** HD descriptor, list the evidence it needs, and make sure the
**notebook** produces that evidence (table, figure, number, wav). Then the report only *cites* what the
notebook already shows. "Beyond HD" items are the extra work that puts us above the baseline.

Notebook section numbers refer to `notebook/A3_TTS.ipynb`.

---

## 2.1 Executive summary & problem framing (7)
**HD asks:** problem, delivered system, practical value, feasibility/scaling, meaning of results, non-technical, ≤1 page; clear aim + scope.

| Evidence needed | Notebook section |
|---|---|
| One-sentence aim + scope (subset size, single speaker, compute) | §0 Project framing |
| Headline numbers in plain words ("speaks a 5 s sentence in 0.4 s") | §6.4 Results summary cell (auto-generated sentence) |
| Feasibility: hardware, training time, model size | §0.3 Environment + §4 training log (wall-clock time) |
| Scaling: cost/time for full 13,100 utterances, CPU deployment RTF | §8 Scaling & deployment estimates |

**Beyond HD:** a 3-bullet "what this means for a user" box (accessibility, audiobooks, cost per hour of audio).

## 2.2 System design & justification (7)
**HD asks:** a clear diagram, the role of each module, **constraints → choices** (front-end / acoustic / vocoder).

| Evidence needed | Notebook section |
|---|---|
| Pipeline diagram (same one as the report) | §3.1 System design (diagram image + markdown) |
| Module table: input/output shapes per stage (text → ids/phonemes → mel [80×T] → wav) | §3.2 Module walk-through (prints real tensor shapes) |
| Constraint→decision table (compute, time, data, quality goal) | §3.3 Design justification table |
| Alternatives considered and why rejected (Tacotron2 / FastSpeech2 / VITS / Matcha) | §3.3 |

**Beyond HD:** parameter count + model size on disk per module; mel/attention-alignment plot from one real utterance.

## 2.3 Experimental design & results table (7)
**HD asks:** ≥2 controlled experiments, an experiments table (settings + outcomes), split/seed/pre-processing explicit **and consistent with code**, motivated comparison goal.

| Evidence needed | Notebook section |
|---|---|
| Hypothesis: "Changing X will affect Y because Z" | §4.1 Experiment design |
| Config dict per experiment, with the diff printed automatically (proves one variable changed) | §4.2 `CONFIGS` + `diff_configs()` |
| Split ratio + seed + hash of the split files (proves same split) | §1.4 + §4.2 |
| Checkpoint selection rule (best val loss / last step), logging | §4.3 Training protocol |
| Results table generated from `results/metrics.csv` | §6.4 |

**Beyond HD:** a 3rd experiment or a 2nd seed, to show that the difference is bigger than seed noise.

## 2.4 Evaluation & analysis (7)
**HD asks:** objective metric(s) clearly reported **and interpreted**; audio examples used to support **trade-off** discussion (quality vs speed, data size vs quality).

| Evidence needed | Notebook section |
|---|---|
| RTF + latency: warm-up, N repeated runs, mean ± std, device stated | §6.1 Speed metrics |
| Intelligibility metric: WER/CER of ASR (Whisper) on synthesised test sentences | §6.2 Intelligibility |
| Quality metric: MCD vs ground-truth test wavs, and/or UTMOS predicted MOS | §6.3 Quality |
| Embedded audio players, expA vs expB side by side, same prompt | §7.1 Listening panel |
| Spectrogram comparison (GT vs expA vs expB) | §7.2 |
| Trade-off plot: RTF vs WER (or vs MOS), one point per experiment | §6.5 |

**Beyond HD:** a small blind A/B preference test (each member + 2 friends, 10 prompts) with the counts reported; CPU *and* GPU/MPS RTF.

## 2.5 Error analysis & improvement proposals (7)
**HD asks:** ≥2 common failure modes, technically grounded fixes, a clear **recommendation of the preferred variant**, deployment/scaling, future work.

| Evidence needed | Notebook section |
|---|---|
| Stress-test prompts: numbers, abbreviations, long sentences, questions, rare names, OOV words | §7.3 Hard-case prompts |
| Per-prompt WER table, sorted to surface the worst cases | §7.4 Failure table |
| Labelled failure categories (skipping/repeating, mispronunciation, prosody, noise/buzz) with wav refs | §7.4 |
| Recommendation cell: chosen variant + why (from the metrics) | §8.1 |
| Scaling estimates (full-dataset training time, batch inference throughput) | §8.2 |

**Beyond HD:** demonstrate one fix (e.g. better number normalisation) and show the WER drop on the hard set.

---

## Code rubric quick checklist (HD)
- **1.1 (9):** metadata + paths load · split ratio + seed · sample-rate policy · trimming · text normalisation **with justification** · sanity stats (duration histogram, char count, sample audio).
- **1.2 (11):** one `synthesize(text, exp, out_path)` function + a CLI `scripts/synth.py` · consistent output dir · runs top-to-bottom.
- **1.3 (9):** same split · one variable changed · checkpoint selection + validation use + logging documented.
- **1.4 (6):** metric with repeated runs + stated device · clean structure, configurable paths (`PROJECT_ROOT`, `DATA_DIR`) · in-code citation links.
