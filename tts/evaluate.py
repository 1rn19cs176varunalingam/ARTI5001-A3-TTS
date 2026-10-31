"""Objective evaluation (notebook §4): intelligibility, spectral distance, rhythm, speed, step sweep, validation re-score.

Metrics
  - WER / CER: Whisper small.en (Radford et al., 2022) transcribes each clip; reference and transcript go through the
    same text normaliser (tts.text.normalize_text), so "1963" and "nineteen sixty-three" match.
  - MCD: mel-cepstral distortion in dB (Kubichek, 1993) on WORLD/SPTK mel-cepstra (order 24, alpha 0.455, c0 dropped),
    averaged over the DTW alignment of synthesised and recorded speech.
  - duration ratio: synthesised length / recorded length (1.0 = the speaker's own pace).
  - RTF and latency: end-to-end and per stage (front-end, acoustic model, vocoder), medians after warm-up.
All test-set scores are saved per utterance, so a run can be resumed and any table can be rebuilt from the CSV.
"""
import csv, json, time
from pathlib import Path

import jiwer, librosa, numpy as np, pandas as pd, soundfile as sf, torch
from torch.utils.data import DataLoader

from tts.audio import load_wav, mel_spectrogram
from tts.data import TTSDataset, collate
from tts.synth import load_tts, load_vocoder, synthesize, SR
from tts.text import normalize_text, phonemes_to_ids, text_to_ids
from tts.train import evaluate as val_losses

EXPS = ["expA", "expB", "expC", "expD"]


# ---------------------------------------------------------------- data
def load_metadata(data_dir):
    """LJSpeech metadata; QUOTE_NONE so quote characters are not treated as CSV quoting (see §1.6)."""
    return pd.read_csv(Path(data_dir) / "metadata.csv", sep="|", header=None, quoting=csv.QUOTE_NONE,
                       names=["id", "raw", "norm"], dtype=str).set_index("id")


def test_sentences(split_dir, data_dir, n=0, seed=42):
    """A seeded sample of n test clips (n=0: the whole test split) as [(id, normalised transcript)]."""
    meta = load_metadata(data_dir)
    ids = (Path(split_dir) / "test.txt").read_text().split()
    ids = sorted(np.random.default_rng(seed).permutation(sorted(ids))[:n or None])
    return [(i, meta.loc[i, "norm"]) for i in ids]


# ---------------------------------------------------------------- metrics
def mel_cepstrum(wav, order=24, alpha=0.455):
    """WORLD spectral envelope -> SPTK mel-cepstrum (alpha 0.455 suits 22.05 kHz); returns (frames, order), c0 dropped."""
    import pysptk, pyworld
    x = wav.astype(np.float64)
    f0, t = pyworld.dio(x, SR, frame_period=5.0)                       # F0 is only needed to estimate the envelope
    f0 = pyworld.stonemask(x, f0, t, SR)
    sp = pyworld.cheaptrick(x, f0, t, SR, fft_size=1024)
    return pysptk.sp2mc(sp, order=order, alpha=alpha)[:, 1:]


def mcd_dtw(ref, syn):
    """Mel-cepstral distortion (dB) over the DTW alignment: the two utterances have different timing."""
    a, b = mel_cepstrum(ref), mel_cepstrum(syn)
    _, path = librosa.sequence.dtw(a.T, b.T, metric="euclidean")
    d = a[path[:, 0]] - b[path[:, 1]]
    return float(10 / np.log(10) * np.sqrt(2) * np.mean(np.sqrt((d ** 2).sum(1))))


def text_for_wer(s):
    """Normalise (numbers spelled out, lower case), then keep only letters, apostrophes and spaces."""
    s = normalize_text(s)
    return " ".join("".join(c if c.isalpha() or c in " '" else " " for c in s).split())


class Whisper:
    """Automatic listener for WER/CER."""
    def __init__(self, device, name="openai/whisper-small.en"):
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=name, device=device)

    def __call__(self, wav, sr=SR):
        return self.asr({"raw": librosa.resample(wav, orig_sr=sr, target_sr=16000), "sampling_rate": 16000})["text"].strip()


# ---------------------------------------------------------------- systems
def make_systems(device, vocoder, runs):
    """name -> fn(text, recorded wav) -> wav. runs = {"full": ckpt_root}; missing roots are skipped."""
    out = {"ground_truth": lambda text, gt: gt,
           "vocoded_gt": lambda text, gt: vocoder(mel_spectrogram(gt)[None].to(device)).squeeze().clamp(-1, 1).cpu().numpy()}
    for run, root in runs.items():
        for e in EXPS:
            path = Path(root) / e / "best.ckpt"
            if path.exists():
                model, ck = load_tts(path, device)
                out[f"{run}_{e}"] = (lambda m, c: lambda text, gt: synthesize(text, m, vocoder, c["vocab"], c["mel_stats"])[0])(model, ck)
    return out


# ---------------------------------------------------------------- test-set quality
@torch.no_grad()
def score_test_set(sents, runs, data_dir, vocoder_path, out_dir, device, save_wavs=5):
    """Scores every (system, sentence) pair not already in out_dir/per_utterance.csv, saving after each system.
    The noise seed is the sentence index, so every flow model hears the same noise for the same sentence."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_utterance.csv"
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=["system", "id"])
    vocoder = load_vocoder(vocoder_path, device)
    systems, asr = make_systems(device, vocoder, runs), None
    for name, fn in systems.items():
        have = set(done.id[done.system == name])
        todo = [(k, cid, text) for k, (cid, text) in enumerate(sents) if cid not in have]
        if not todo:
            print(f"  {name:14s} all {len(sents)} sentences already scored"); continue
        asr = asr or Whisper(device)
        rows, t0 = [], time.time()
        for k, cid, text in todo:
            torch.manual_seed(k)
            gt = load_wav(Path(data_dir) / "wavs" / f"{cid}.wav")                  # trimmed, exactly as in training
            wav = fn(text, gt)
            hyp = asr(wav)
            r, h = text_for_wer(text), text_for_wer(hyp)
            rows.append({"system": name, "id": cid, "text": text, "whisper": hyp, "wer": jiwer.wer(r, h),
                         "cer": jiwer.cer(r, h), "mcd": mcd_dtw(gt, wav), "dur_ratio": len(wav) / len(gt), "ref_words": len(r.split())})
            if k < save_wavs:
                (out_dir / "test_wavs").mkdir(exist_ok=True)
                sf.write(out_dir / "test_wavs" / f"{name}_{cid}.wav", wav, SR)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(csv_path, index=False)
        print(f"  {name:14s} {len(todo)} sentences scored in {time.time() - t0:.0f} s", flush=True)
    ids = {cid for cid, _ in sents}
    return done[done.id.isin(ids) & done.system.isin(systems)].reset_index(drop=True)


def corpus_wer(g, metric=jiwer.wer):
    """Corpus-level error rate (%): errors summed over all sentences, not an average of per-sentence rates."""
    return 100 * metric([text_for_wer(t) for t in g.text], [text_for_wer(h) for h in g.whisper.fillna("")])


def _boot(g, stat, n_boot, seed):
    rng = np.random.default_rng(seed)
    return np.array([stat(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n_boot)])


def summarise(df, n_boot=1000, seed=0):
    """One row per system: WER, CER, MCD, duration ratio, each with a 95% bootstrap CI over sentences."""
    stats = {"WER_%": corpus_wer, "CER_%": lambda g: corpus_wer(g, jiwer.cer),
             "MCD_dB": lambda g: g.mcd.mean(), "dur_ratio": lambda g: g.dur_ratio.mean()}
    rows = []
    for name, g in df.groupby("system", sort=False):
        g = g.sort_values("id").reset_index(drop=True)
        row = {"system": name, "n": len(g)}
        for k, f in stats.items():
            lo, hi = np.percentile(_boot(g, f, n_boot, seed), [2.5, 97.5])
            row.update({k: f(g), f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows).set_index("system")


def paired_difference(df, a, b, n_boot=1000, seed=0):
    """b minus a for WER and MCD, with a paired bootstrap CI (the same sentences are resampled for both systems).
    If the CI excludes 0, the difference is unlikely to be an accident of which sentences were sampled."""
    ga = df[df.system == a].sort_values("id").reset_index(drop=True)
    gb = df[df.system == b].sort_values("id").reset_index(drop=True)
    assert list(ga.id) == list(gb.id), "systems were scored on different sentences"
    rng, out = np.random.default_rng(seed), {}
    for k, f in {"WER_%": corpus_wer, "MCD_dB": lambda g: g.mcd.mean()}.items():
        diffs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(ga), len(ga))
            diffs.append(f(gb.iloc[i]) - f(ga.iloc[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.update({f"Δ{k}": f(gb) - f(ga), f"Δ{k}_lo": lo, f"Δ{k}_hi": hi})
    return out


# ---------------------------------------------------------------- speed
@torch.no_grad()
def measure_speed(prompts, ckpt_root, vocoder_path, device, n_warm=2, n_rep=3):
    """End-to-end and per-stage timing on the given prompts (warm-up runs are discarded)."""
    sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(device.type, lambda: None)
    vocoder, rows = load_vocoder(vocoder_path, device), []
    for e in EXPS:
        model, ck = load_tts(Path(ckpt_root) / e / "best.ckpt", device)
        st = ck["mel_stats"]

        def run(text):
            """The steps of tts.synth.synthesize, timed separately."""
            t0 = time.perf_counter()
            ids = torch.tensor([text_to_ids(text, ck["vocab"])], device=device)          # normaliser + phonemiser
            t1 = time.perf_counter()
            mel, _ = model.synthesise_mel(ids, n_steps=10, temperature=0.667); sync()     # acoustic model
            t2 = time.perf_counter()
            wav = vocoder(mel * st["mel_std"] + st["mel_mean"]).squeeze().cpu().numpy()  # vocoder (.cpu() waits for it)
            t3 = time.perf_counter()
            return wav, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)

        for t in prompts[:n_warm]:
            run(t)
        for t in prompts:
            for _ in range(n_rep):
                wav, (tf, ta, tv, tt) = run(t)
                a = len(wav) / SR
                rows.append({"device": device.type, "exp": e, "audio_s": a, "latency_s": tt, "frontend_s": tf,
                             "acoustic_s": ta, "vocoder_s": tv, "rtf": tt / a, "rtf_acoustic": ta / a})
    return pd.DataFrame(rows)


def summarise_speed(raw):
    return raw.groupby(["device", "exp"]).agg(
        RTF=("rtf", "median"), RTF_acoustic=("rtf_acoustic", "median"), latency_s=("latency_s", "median"),
        latency_p90_s=("latency_s", lambda x: x.quantile(0.9)), frontend_s=("frontend_s", "median"),
        acoustic_s=("acoustic_s", "median"), vocoder_s=("vocoder_s", "median"), runs=("rtf", "size"))


# ---------------------------------------------------------------- flow-matching step sweep
@torch.no_grad()
def step_sweep(sents, ckpt_path, data_dir, vocoder_path, device, steps=(1, 2, 4, 10, 20)):
    """Number of Euler steps at inference: the quality/speed knob that only flow matching has."""
    vocoder = load_vocoder(vocoder_path, device)
    model, ck = load_tts(ckpt_path, device)
    asr, rows = Whisper(device), []
    for n in steps:
        for k, (cid, text) in enumerate(sents):
            torch.manual_seed(k)
            gt = load_wav(Path(data_dir) / "wavs" / f"{cid}.wav")
            t0 = time.perf_counter()
            wav = synthesize(text, model, vocoder, ck["vocab"], ck["mel_stats"], n_steps=n)[0]
            dt = time.perf_counter() - t0
            rows.append({"steps": n, "id": cid, "text": text, "whisper": asr(wav), "mcd": mcd_dtw(gt, wav), "rtf": dt / (len(wav) / SR)})
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{"steps": n, "WER_%": corpus_wer(g), "MCD_dB": g.mcd.mean(), "RTF": g.rtf.median()}
                            for n, g in df.groupby("steps")]).set_index("steps")
    return summary, df


# ---------------------------------------------------------------- validation re-score
def _records(ids, phonemes, vocab, mel_dir, data_dir):
    """Training-format records for any clips; computes (and caches) the trimmed log-mel of clips not cached yet."""
    recs = []
    for cid in ids:
        p = Path(mel_dir) / f"{cid}.pt"
        if not p.exists():
            torch.save(mel_spectrogram(load_wav(Path(data_dir) / "wavs" / f"{cid}.wav")).half(), p)
        recs.append({"id": cid, "phonemes": phonemes_to_ids(phonemes[cid], vocab), "frames": int(torch.load(p).shape[1])})
    return recs


@torch.no_grad()
def rescore_validation(ckpt_root, val_ids, monitor_ids, phonemes, mel_dir, data_dir, out_dir, device, seed=42, batch=32):
    """Validation loss of every saved checkpoint on ALL validation clips and on the 300-clip monitor set.
    Checks that the monitor set used to pick best.ckpt agrees with the full validation split."""
    rows, tmp = [], Path(out_dir) / "_val_records.json"
    for e in EXPS:
        for path in sorted((Path(ckpt_root) / e).glob("*.ckpt")):
            if path.name == "last.ckpt":
                continue                                                       # identical to epoch_300.ckpt
            model, ck = load_tts(path, device)
            row = {"exp": e, "checkpoint": path.stem, "epoch": ck["epoch"]}
            for name, ids in (("all_val", val_ids), ("monitor", monitor_ids)):
                tmp.write_text(json.dumps(_records(ids, phonemes, ck["vocab"], mel_dir, data_dir)))
                dl = DataLoader(TTSDataset(tmp, mel_dir, ck["mel_stats"]), batch_size=batch, collate_fn=collate)
                parts = val_losses(model, dl, device, seed)
                row.update({f"{name}_total": sum(parts.values()), f"{name}_dur": parts["dur"]})
            rows.append(row)
    tmp.unlink(missing_ok=True)
    return pd.DataFrame(rows)
