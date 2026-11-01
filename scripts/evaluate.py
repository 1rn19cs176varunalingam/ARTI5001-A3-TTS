"""Objective evaluation from the command line (the library is tts/evaluate.py, written by the notebook, §4).

Examples (from the repository root):
    python scripts/evaluate.py --n_test 0        # whole test split; resumes if interrupted
    python scripts/evaluate.py --step_sweep      # expA at 1/2/4/10/20 Euler steps on the whole test split; resumes
Writes results/eval/per_utterance.csv, summary.csv and metrics.json (or step_sweep*.csv).
"""
import argparse, json, os, platform, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
if platform.system() == "Darwin" and Path("/opt/homebrew/lib/libespeak-ng.dylib").exists():
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")

from tts.evaluate import score_test_set, step_sweep, summarise, test_sentences
from tts.train import pick_device

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=0, help="test sentences (0 = the whole test split)")
    ap.add_argument("--full_root", default="checkpoints/runs_full")
    ap.add_argument("--data_dir", default="data/LJSpeech-1.1")
    ap.add_argument("--split_dir", default="results/splits")
    ap.add_argument("--vocoder", default="checkpoints/hifigan_lj_v1")
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--step_sweep", action="store_true", help="run the expA step sweep instead of the main evaluation")
    a = ap.parse_args()
    sents = test_sentences(a.split_dir, a.data_dir, a.n_test)
    if a.step_sweep:
        out = Path(a.out_dir)
        sweep, _ = step_sweep(sents, Path(a.full_root) / "expA" / "best.ckpt", a.data_dir, a.vocoder, pick_device(),
                              out_csv=out / "step_sweep_per_utterance.csv")
        sweep.to_csv(out / "step_sweep.csv")
        print(sweep.round(3).to_string())
        sys.exit(0)
    df = score_test_set(sents, {"full": a.full_root}, a.data_dir, a.vocoder, a.out_dir, pick_device())
    s = summarise(df)
    s.to_csv(Path(a.out_dir) / "summary.csv")
    (Path(a.out_dir) / "metrics.json").write_text(json.dumps(s.round(4).reset_index().to_dict("records"), indent=1))
    print(s.round(3).to_string())
