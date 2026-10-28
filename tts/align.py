"""Monotonic Alignment Search (MAS), Glow-TTS (Kim et al., 2020):
https://github.com/jaywalnut310/glow-tts/blob/master/monotonic_align/core.pyx  (ported from Cython to Numba)."""
import numpy as np, torch
from numba import njit


@njit(cache=True)
def _mas_single(value, t_y, t_x):
    path = np.zeros((t_y, t_x), dtype=np.float32)
    neg = -1e9
    for y in range(t_y):                                     # forward: best cumulative score per (frame, phoneme)
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg if x == y else value[y - 1, x]       # stay on the same phoneme
            v_prev = (0.0 if y == 0 else neg) if x == 0 else value[y - 1, x - 1]   # move to the next phoneme
            value[y, x] += max(v_prev, v_cur)
    index = t_x - 1                                          # backward: trace the best path
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1.0
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1
    return path


@torch.no_grad()
def log_likelihood(mu, mel):
    """mu: (B, T_text, 80), mel: (B, 80, T_mel) -> (B, T_mel, T_text) Gaussian log-likelihood (up to a constant)."""
    return -0.5 * torch.cdist(mel.transpose(1, 2), mu) ** 2


@torch.no_grad()
def maximum_path(score, text_lens, mel_lens):
    """0/1 path (B, T_mel, T_text): path[b, j, i] = 1 if mel frame j belongs to phoneme i."""
    score_np = score.float().cpu().numpy()
    path = np.zeros_like(score_np)
    for b in range(score_np.shape[0]):
        ty, tx = int(mel_lens[b]), int(text_lens[b])
        path[b, :ty, :tx] = _mas_single(score_np[b, :ty, :tx].copy(), ty, tx)
    return torch.from_numpy(path).to(score.device)
