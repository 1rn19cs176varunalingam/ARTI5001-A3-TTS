"""Inference: text -> phonemes -> acoustic model -> mel -> HiFi-GAN -> wav.
HiFi-GAN (Kong et al., 2020), code + LJ V1 weights via Matcha-TTS: https://github.com/shivammehta25/Matcha-TTS
CLI: python -m tts.synth --text "Hello world." --exp expA --out wav_examples/hello.wav"""
import argparse, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from tts.hifigan.config import v1
from tts.hifigan.env import AttrDict
from tts.hifigan.models import Generator
from tts.text import text_to_ids, normalize_text
from tts.tts_model import TTSModel
from tts.train import pick_device

SR = 22050
_CACHE = {}


def load_vocoder(path, device):
    voc = Generator(AttrDict(v1)).to(device)
    voc.load_state_dict(torch.load(path, map_location=device, weights_only=False)["generator"])
    voc.eval(); voc.remove_weight_norm()
    return voc


def load_tts(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TTSModel(ck["n_vocab"], decoder=ck["cfg"]["decoder"], dec_blocks=ck["cfg"]["dec_blocks"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


@torch.no_grad()
def synthesize(text, model, vocoder, vocab, mel_stats, n_steps=10, temperature=0.667, length_scale=1.0):
    """Returns (wav float32 numpy, sample rate, normalised mel tensor)."""
    device = next(model.parameters()).device
    ids = torch.tensor([text_to_ids(text, vocab)], device=device)
    mel, _ = model.synthesise_mel(ids, n_steps=n_steps, temperature=temperature, length_scale=length_scale)
    wav = vocoder(mel * mel_stats["mel_std"] + mel_stats["mel_mean"]).squeeze().clamp(-1, 1).cpu().numpy()
    return wav.astype(np.float32), SR, mel


def synthesize_to_wav(text, exp="expA", out_path=None, ckpt="best", ckpt_root="checkpoints/runs",
                      vocoder_path="checkpoints/hifigan_lj_v1", **kw):
    """One call from text to a playable .wav: the notebook / CLI entry point."""
    device = pick_device()
    key = (str(Path(ckpt_root) / exp / f"{ckpt}.ckpt"), str(device))
    if key not in _CACHE:
        model, ck = load_tts(key[0], device)
        if ("voc", str(device)) not in _CACHE:
            _CACHE[("voc", str(device))] = load_vocoder(vocoder_path, device)
        _CACHE[key] = (model, ck)
    model, ck = _CACHE[key]
    wav, sr, mel = synthesize(text, model, _CACHE[("voc", str(device))], ck["vocab"], ck["mel_stats"], **kw)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wav, sr)
    return wav, sr


def main():
    ap = argparse.ArgumentParser(description="Synthesise a .wav from text with a trained experiment.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--exp", default="expA", help="expA | expB | expC | expD")
    ap.add_argument("--ckpt", default="best", help="best | last | epoch_XXX")
    ap.add_argument("--out", default="wav_examples/output.wav")
    ap.add_argument("--ckpt_root", default="checkpoints/runs")
    ap.add_argument("--vocoder", default="checkpoints/hifigan_lj_v1")
    ap.add_argument("--steps", type=int, default=10, help="ODE steps (flow-matching decoders only)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    wav, sr = synthesize_to_wav(a.text, a.exp, a.out, a.ckpt, a.ckpt_root, a.vocoder, n_steps=a.steps)
    print(f"normalised text : {normalize_text(a.text)}")
    print(f"wrote {a.out}: {len(wav) / sr:.2f} s of audio in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
