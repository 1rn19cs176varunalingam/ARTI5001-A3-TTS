"""Audio front-end: load -> (resample) -> trim silence -> 80-band log-mel.
Mel extraction adapted from HiFi-GAN (Kong et al., 2020): https://github.com/jik876/hifi-gan/blob/master/meldataset.py
so features match the pretrained HiFi-GAN LJ V1 vocoder exactly.
"""
import librosa, numpy as np, soundfile as sf, torch

SR, N_FFT, HOP, WIN = 22050, 1024, 256, 1024
N_MELS, FMIN, FMAX = 80, 0, 8000
TOP_DB = 40

MEL_BASIS = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)).float()
WINDOW = torch.hann_window(WIN)


def load_wav(path, trim=True):
    """Returns float32 mono audio in [-1, 1] at SR Hz, leading/trailing silence removed if trim=True."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if trim:
        wav, _ = librosa.effects.trim(wav, top_db=TOP_DB, frame_length=WIN, hop_length=HOP)
    return wav


def mel_spectrogram(wav):
    """wav: 1-D numpy/tensor -> log-mel tensor (80, frames)."""
    wav = torch.as_tensor(wav, dtype=torch.float32)
    pad = (N_FFT - HOP) // 2
    wav = torch.nn.functional.pad(wav[None, None], (pad, pad), mode="reflect")[0]
    spec = torch.stft(wav, N_FFT, hop_length=HOP, win_length=WIN, window=WINDOW, center=False, return_complex=True)
    spec = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
    return torch.log(torch.clamp(MEL_BASIS @ spec, min=1e-5))[0]
