"""Full acoustic model and losses, following Matcha-TTS (Mehta et al., 2024) and Glow-TTS (Kim et al., 2020).
loss = duration loss (MSE on log-durations) + prior loss (Gaussian NLL of mel under mu_y) + decoder loss."""
import math, torch, torch.nn as nn
from tts.model import TextEncoder, DurationPredictor
from tts.decoders import build_decoder
from tts.align import log_likelihood, maximum_path


def length_mask(lens, max_len):
    """(B,) lengths -> (B, max_len) mask: 1 on real frames, 0 on padding."""
    return (torch.arange(max_len, device=lens.device)[None] < lens[:, None]).float()


def durations_to_path(d, t_mel):
    """d: (T_text,) integer durations -> 0/1 path (T_mel, T_text), the same format MAS produces."""
    end = torch.cumsum(d, 0)
    frames = torch.arange(t_mel, device=d.device)[:, None]
    return ((frames >= (end - d)[None]) & (frames < end[None])).float()


class TTSModel(nn.Module):
    def __init__(self, n_vocab, decoder="flow", dec_blocks=8, n_mels=80, dec_channels=256):
        super().__init__()
        self.encoder = TextEncoder(n_vocab, n_mels=n_mels)
        self.dur = DurationPredictor()
        self.decoder = build_decoder(decoder, n_mels=n_mels, ch=dec_channels, n_blocks=dec_blocks)

    def forward(self, ids, ids_lens, mels, mel_lens):
        """Training step: returns the total loss and its three parts."""
        # 1. Encode text; masks mark real phonemes (x_mask) and real mel frames (y_mask)
        h, mu, pad_mask = self.encoder(ids)
        x_mask = (~pad_mask).float()
        y_mask = length_mask(mel_lens, mels.shape[2])

        # 2. Alignment: which frames belong to which phoneme
        path = maximum_path(log_likelihood(mu, mels), ids_lens, mel_lens)        # alignment, no gradients
        durations = path.sum(dim=1)                                              # frames per phoneme

        # 3. Duration loss: predicted vs MAS log-durations, averaged over real phonemes
        logw = self.dur(h, pad_mask)
        dur_loss = ((logw - torch.log(durations + 1e-8) * x_mask) ** 2).sum() / ids_lens.sum()

        # 4. Prior loss: copy each phoneme's mu to its frames, then Gaussian NLL of the real mel under it
        mu_y = (path @ mu).transpose(1, 2)                                       # (B, 80, T_mel)
        prior_loss = (0.5 * ((mels - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask[:, None]).sum() \
                     / (y_mask.sum() * mels.shape[1])
        # 5. Decoder loss: flow matching or regression, conditioned on mu_y
        dec_loss = self.decoder.loss(mels, y_mask[:, None], mu_y)
        total = dur_loss + prior_loss + dec_loss
        return total, {"dur": dur_loss.item(), "prior": prior_loss.item(), "dec": dec_loss.item()}

    @torch.no_grad()
    def synthesise_mel(self, ids, n_steps=10, temperature=0.667, length_scale=1.0):
        """ids: (1, T_text) -> normalised mel (1, 80, T_mel) and integer durations (T_text,)."""
        h, mu, pad_mask = self.encoder(ids)
        logw = self.dur(h, pad_mask)[0]
        d = torch.ceil(torch.exp(logw) * length_scale).long().clamp(min=1)      # frames per phoneme, at least 1
        t_mel = int(d.sum())
        mu_y = (durations_to_path(d, t_mel) @ mu[0]).T[None]
        mask = torch.ones(1, 1, t_mel, device=ids.device)
        return self.decoder.sample(mu_y, mask, n_steps=n_steps, temperature=temperature), d
