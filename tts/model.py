"""Text encoder + duration predictor, after Glow-TTS (Kim et al., 2020) and Matcha-TTS (Mehta et al., 2024)."""
import torch, torch.nn as nn
from tts.text import PAD_ID


class TextEncoder(nn.Module):
    """Phoneme ids -> hidden states h and a mel prior mu for every phoneme."""
    def __init__(self, n_vocab, d_model=192, n_layers=6, n_heads=2, n_mels=80, max_len=1000):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.proj_mu = nn.Linear(d_model, n_mels)

    def forward(self, ids):
        pad_mask = ids == PAD_ID                                             # True on padding
        x = self.embed(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        return h, self.proj_mu(h), pad_mask                                  # mu: (B, T_text, 80)


class DurationPredictor(nn.Module):
    """Predicts log(frames) per phoneme from the (detached) encoder states."""
    def __init__(self, d_model=192, hidden=256, kernel=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel, padding=kernel // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, 1))

    def forward(self, h, pad_mask):
        logw = self.net(h.detach().transpose(1, 2)).squeeze(1)              # (B, T_text)
        return logw.masked_fill(pad_mask, 0.0)
