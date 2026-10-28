"""Two mel decoders sharing one backbone, so the ONLY difference between them is the training objective.

FlowMatchingDecoder: OT-CFM (Lipman et al., 2023) as in Matcha-TTS (Mehta et al., 2024): learns a velocity field that
    moves Gaussian noise to the mel, conditioned on mu_y; sampled with n Euler ODE steps.
RegressionDecoder: FastSpeech-style (Ren et al., 2021): predicts the mel directly from mu_y with an L1 loss, one pass.
Backbone: 1-D residual conv blocks with dilations 1,2,4,8 (a simplification of Matcha's U-Net + Transformer).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the flow time t in [0, 1] -> vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = 1000 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class ResBlock(nn.Module):
    """Pre-norm residual block: norm -> SiLU -> dilated conv (+ time embedding) -> norm -> SiLU -> conv, plus skip.
    Multiplying by the mask after every conv keeps padded frames at exactly zero."""
    def __init__(self, ch, dilation, time_cond):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.time = nn.Linear(ch, ch) if time_cond else None

    def forward(self, x, mask, temb=None):
        y = self.conv1(F.silu(self.norm1(x))) * mask
        if self.time is not None:                                                # flow decoder only: tell the block "where" on the path it is
            y = y + self.time(temb)[:, :, None]
        y = self.conv2(F.silu(self.norm2(y))) * mask
        return (x + y) * mask


class _Backbone(nn.Module):
    """Shared by both decoders: 1x1 conv in -> n_blocks ResBlocks (dilation 1,2,4,8 repeating) -> 1x1 conv out."""
    def __init__(self, in_ch, n_mels, ch, n_blocks, time_cond):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.ModuleList([ResBlock(ch, 2 ** (i % 4), time_cond) for i in range(n_blocks)])
        self.out = nn.Conv1d(ch, n_mels, 1)

    def forward(self, x, mask, temb=None):
        h = self.inp(x) * mask
        for blk in self.blocks:
            h = blk(h, mask, temb)
        return self.out(h) * mask


class FlowMatchingDecoder(nn.Module):
    kind = "flow"

    def __init__(self, n_mels=80, ch=256, n_blocks=8, sigma_min=1e-4):
        super().__init__()
        self.sigma_min = sigma_min
        self.time = TimeEmbedding(ch)
        self.net = _Backbone(2 * n_mels, n_mels, ch, n_blocks, time_cond=True)   # input: [x_t ; mu_y]

    def forward(self, x, mask, mu_y, t):
        return self.net(torch.cat([x, mu_y], dim=1), mask, self.time(t))

    def loss(self, x1, mask, mu_y):
        """OT-CFM loss: MSE between the predicted and the true velocity at a random point of the noise->mel path."""
        x0 = torch.randn_like(x1)                                                # noise
        t = torch.rand(x1.shape[0], device=x1.device)[:, None, None]             # one time per clip
        x_t = (1 - (1 - self.sigma_min) * t) * x0 + t * x1                       # point on the straight path
        u = x1 - (1 - self.sigma_min) * x0                                       # target velocity
        v = self(x_t, mask, mu_y, t[:, 0, 0])
        return ((v - u) ** 2 * mask).sum() / (mask.sum() * x1.shape[1])

    @torch.no_grad()
    def sample(self, mu_y, mask, n_steps=10, temperature=0.667):
        """Integrate the learned velocity from t=0 (noise) to t=1 (mel). temperature < 1 reduces sampling noise."""
        x = torch.randn_like(mu_y) * temperature
        dt = 1.0 / n_steps
        for i in range(n_steps):                                                 # Euler ODE solver
            t = torch.full((mu_y.shape[0],), i * dt, device=mu_y.device)
            x = x + dt * self(x, mask, mu_y, t)
        return x * mask


class RegressionDecoder(nn.Module):
    kind = "regression"

    def __init__(self, n_mels=80, ch=256, n_blocks=8):
        super().__init__()
        self.net = _Backbone(n_mels, n_mels, ch, n_blocks, time_cond=False)      # input: mu_y only

    def forward(self, mask, mu_y):
        return self.net(mu_y, mask)

    def loss(self, x1, mask, mu_y):
        return ((self(mask, mu_y) - x1).abs() * mask).sum() / (mask.sum() * x1.shape[1])   # masked L1

    @torch.no_grad()
    def sample(self, mu_y, mask, **_):
        return self(mask, mu_y)                                                  # deterministic, one pass


DECODERS = {"flow": FlowMatchingDecoder, "regression": RegressionDecoder}   # config value "decoder" -> class


def build_decoder(kind, **kwargs):
    return DECODERS[kind](**kwargs)
