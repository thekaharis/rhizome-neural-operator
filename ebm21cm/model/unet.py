"""2-D U-Net with periodic padding and adaptive-GroupNorm conditioning.

Every spatial op pads circularly, so the network is exactly equivariant to
transverse rolls by multiples of 2**(levels-1) -- matching the periodic
21cmFAST transverse plane. Attention is written out explicitly (no fused
SDPA kernel) because the energy parameterization differentiates the network
twice, and fused attention kernels do not support double backward.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, k=3, stride=1):
    return nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2,
                     padding_mode="circular" if k > 1 else "zeros")


def norm(ch):
    return nn.GroupNorm(min(32, ch // 4) if ch >= 8 else 1, ch)


def zero(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class FourierEmbedding(nn.Module):
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        x = x[:, None] * self.freqs[None, :] * (2 * math.pi)
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = conv(cin, cout)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.n2 = norm(cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = zero(conv(cout, cout))
        self.skip = conv(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return (h + self.skip(x)) / math.sqrt(2)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.n = norm(ch)
        self.qkv = conv(ch, 3 * ch, 1)
        self.proj = zero(conv(ch, ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        w = torch.einsum("bhcn,bhcm->bhnm", q, k) / math.sqrt(C // self.heads)
        h = torch.einsum("bhnm,bhcm->bhcn", w.softmax(dim=-1), v)
        return (x + self.proj(h.reshape(B, C, H, W))) / math.sqrt(2)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c = conv(ch, ch, 3, stride=2)

    def forward(self, x):
        return self.c(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c = conv(ch, ch)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """F(x_in, c_noise, scalars) -> field with ``out_ch`` channels."""

    def __init__(self, in_ch, out_ch, n_scalars, ch=64, mults=(1, 2, 4), num_res=2,
                 attn_levels=(2,), heads=4, dropout=0.0):
        super().__init__()
        self.levels = len(mults)
        emb_dim = 4 * ch
        self.noise_emb = nn.Sequential(FourierEmbedding(ch), nn.Linear(ch, emb_dim), nn.SiLU(),
                                       nn.Linear(emb_dim, emb_dim))
        self.scalar_emb = nn.Sequential(nn.Linear(n_scalars, emb_dim), nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim)) if n_scalars else None
        self.emb_out = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inp = conv(in_ch, ch)

        self.down = nn.ModuleList()
        skips = [ch]
        cur = ch
        for lvl, m in enumerate(mults):
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(cur, ch * m, emb_dim, dropout)])
                cur = ch * m
                if lvl in attn_levels:
                    blk.append(Attention(cur, heads))
                self.down.append(blk)
                skips.append(cur)
            if lvl < self.levels - 1:
                self.down.append(nn.ModuleList([Down(cur)]))
                skips.append(cur)

        self.mid = nn.ModuleList([ResBlock(cur, cur, emb_dim, dropout), Attention(cur, heads),
                                  ResBlock(cur, cur, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mults))):
            for _ in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(cur + skips.pop(), ch * m, emb_dim, dropout)])
                cur = ch * m
                if lvl in attn_levels:
                    blk.append(Attention(cur, heads))
                self.up.append(blk)
            if lvl > 0:
                self.up.append(nn.ModuleList([Up(cur)]))
        self.out = nn.Sequential(norm(cur), nn.SiLU(), zero(conv(cur, out_ch)))

    @property
    def multiple(self):
        return 2 ** (self.levels - 1)

    def forward(self, x, c_noise, scalars=None):
        if x.shape[-1] % self.multiple or x.shape[-2] % self.multiple:
            raise ValueError(f"spatial size {tuple(x.shape[-2:])} must be divisible by {self.multiple}")
        emb = self.noise_emb(c_noise)
        if self.scalar_emb is not None:
            emb = emb + self.scalar_emb(scalars)
        emb = self.emb_out(emb)

        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            for layer in blk:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            for layer in blk:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.out(h)
