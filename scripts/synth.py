"""Text -> .wav from a trained experiment.

Example (from the repository root):
    python scripts/synth.py --text "Hello world." --exp expA --out wav_examples/hello.wav

The implementation lives in tts/synth.py, which is written by the notebook (§2).
"""
import os, platform, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
if platform.system() == "Darwin" and Path("/opt/homebrew/lib/libespeak-ng.dylib").exists():
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")

from tts.synth import main

if __name__ == "__main__":
    main()
