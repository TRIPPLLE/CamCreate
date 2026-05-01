"""
+------------------------------------------------------------------------------+
|              SketchFlowGAN - Real-Time Sketch-to-Color GAN                 |
|         A Novel Architecture for Parallel Drawing Colorization              |
+------------------------------------------------------------------------------+
|  Generator  : ChromaFlowNet                                                 |
|  Discriminator: HierarchicalFieldCritic                                     |
+------------------------------------------------------------------------------+
|  Novel Components (all designed from scratch):                              |
|  1. Directional Stroke Analyzer (DSA)                                       |
|     |-- Orientation-aware convolutions (8 Gabor-init directions, then        |
|       learned). Extracts stroke directionality unavailable to plain conv.   |
|  2. Ring Attention Bottleneck (RAB)                                         |
|     |-- Each position attends to positions at fixed-radius rings              |
|       O(N*R) vs O(N2) standard self-attention. Captures spatial context    |
|       without quadratic cost.                                               |
|  3. Gated Residual Bridges (GRB)                                            |
|     |-- Learned sigmoid gates decide how much sketch structure passes         |
|       through skip connections - lets the network suppress sketch noise.    |
|  4. Frequency-Decomposed Upsampler (FDU)                                    |
|     |-- Each upsample stage has a LOW path (bilinear+conv) and HIGH path     |
|       (pixel-shuffle + learnable high-pass). Merged via learned alpha weights.  |
|  5. Stroke-Adaptive Normalization (SAN)                                     |
|     |-- Normalization gamma,beta conditioned on a "stroke code" from the sketch,   |
|       not on a random style vector - ensures structural coherence.          |
|  6. Multi-Field Critic (MFC)                                                |
|     |-- Three parallel discriminator paths: 16x16, 64x64, full-image patches |
|       + a Frequency Critic on FFT magnitude spectrum.                       |
+------------------------------------------------------------------------------+
|  Loss Functions:                                                            |
|  * Sketch-Conditioned LSGAN (least-squares, stable training)                |
|  * Spectral Perceptual Loss (L1 on log FFT magnitudes)                      |
|  * Color Coherence Loss (L1 on LAB color histograms)                        |
|  * Sketch Fidelity Loss (edge-map consistency, Sobel from scratch)          |
+------------------------------------------------------------------------------+

Usage:
    python sketchflow_gan.py --mode train  --data_dir ./dataset
    python sketchflow_gan.py --mode infer  --sketch_path ./my_sketch.png
    python sketchflow_gan.py --mode demo   # live side-by-side drawing demo

Dataset layout expected:
    dataset/
        sketches/   *.png  (grayscale drawings)
        colored/    *.png  (paired color images, same filenames)

Requirements:
    pip install torch torchvision opencv-python pillow matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CUSTOM PRIMITIVE LAYERS
# ─────────────────────────────────────────────────────────────────────────────

class SpectralNorm(nn.Module):
    """
    Spectral normalization implemented from scratch (no nn.utils.spectral_norm).
    Constrains the largest singular value of each weight matrix to ≤ 1,
    stabilising GAN discriminator training.
    """
    def __init__(self, module, n_power_iterations=1):
        super().__init__()
        self.module = module
        self.n_iter = n_power_iterations

        # Move original weight out of the way so we can control normalization
        w = module.weight
        del module.weight
        self.register_parameter('weight_orig', nn.Parameter(w.data))
        
        # Original bias stays as is
        if hasattr(module, 'bias') and module.bias is not None:
            self.bias = module.bias
        else:
            self.bias = None

        h, w_dim = w.shape[0], w.view(w.shape[0], -1).shape[1]
        self.register_buffer('u', F.normalize(torch.randn(h), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(w_dim), dim=0))

    def _power_iter(self, w_mat):
        with torch.no_grad():
            u, v = self.u, self.v
            for _ in range(self.n_iter):
                v = F.normalize(w_mat.t() @ u, dim=0)
                u = F.normalize(w_mat @ v, dim=0)
        return u, v, (u @ (w_mat @ v))

    def forward(self, x):
        w = self.weight_orig
        w_mat = w.view(w.shape[0], -1)
        u, v, sigma = self._power_iter(w_mat)
        if self.training:
            with torch.no_grad():
                self.u.copy_(u)
                self.v.copy_(v)
        
        # Calculate normalized weight and manually inject into module
        w_norm = w / sigma
        
        # We temporarily set the module's weight to the normalized tensor.
        # Since we deleted the Parameter 'weight' in __init__, we can now
        # freely assign a Tensor to this attribute name.
        setattr(self.module, 'weight', w_norm)
        out = self.module(x)
        return out


def sn_conv(in_c, out_c, k=4, s=2, p=1, bias=True):
    """Spectral-normalized Conv2d helper."""
    return SpectralNorm(nn.Conv2d(in_c, out_c, k, s, p, bias=bias))


class PixelShuffle2x(nn.Module):
    """2× upscale via pixel-shuffle (sub-pixel convolution) from scratch."""
    def __init__(self, in_c):
        super().__init__()
        # Expand channels by 4 then rearrange into 2× spatial
        self.expand = nn.Conv2d(in_c, in_c * 4, 1)
        self.scale = 2

    def forward(self, x):
        x = self.expand(x)              # [B, C*4, H, W]
        B, C4, H, W = x.shape
        C = C4 // (self.scale ** 2)
        # Manual pixel-shuffle: reshape → permute → reshape
        x = x.view(B, C, self.scale, self.scale, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C, H * self.scale, W * self.scale)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DIRECTIONAL STROKE ANALYZER (DSA)
# ─────────────────────────────────────────────────────────────────────────────

class GaborInitConv(nn.Module):
    """
    Convolutional layer whose weights are initialised with Gabor filters
    spanning 8 orientations (0°→157.5°). After init the weights are fully
    learnable — Gabor only provides a semantically meaningful starting point
    for stroke detection rather than random noise.
    """
    def __init__(self, in_c, out_c, k=7):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False)
        self._gabor_init(k, out_c)

    def _gabor_kernel(self, theta, sigma=2.0, lam=5.0, gamma=0.5, k=7):
        """Single Gabor kernel as numpy array."""
        half = k // 2
        y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
        x_t =  x * math.cos(theta) + y * math.sin(theta)
        y_t = -x * math.sin(theta) + y * math.cos(theta)
        kernel = np.exp(-(x_t**2 + gamma**2 * y_t**2) / (2 * sigma**2))
        kernel *= np.cos(2 * math.pi * x_t / lam)
        kernel /= (kernel.std() + 1e-8)
        return kernel

    def _gabor_init(self, k, out_c):
        n_angles = 8
        angles = [i * math.pi / n_angles for i in range(n_angles)]
        with torch.no_grad():
            for out_idx in range(out_c):
                theta = angles[out_idx % n_angles]
                kernel = self._gabor_kernel(theta, k=k)
                kernel_t = torch.from_numpy(kernel)
                # broadcast across in_channels
                for in_idx in range(self.conv.weight.shape[1]):
                    self.conv.weight[out_idx, in_idx] = kernel_t

    def forward(self, x):
        return self.conv(x)


class DirectionalStrokeAnalyzer(nn.Module):
    """
    DSA Block — Novel stroke-aware feature extractor.

    Instead of a single convolution, we run THREE parallel branches:
      • Gabor branch: Orientation-selective stroke detection
      • Laplacian branch: Edge / corner detection (fixed Laplacian of Gaussian)
      • Learned branch: Standard conv for content features
    All three outputs are fused with a learned 1×1 projection.

    This gives the generator rich structural priors from the sketch before
    any learned features are computed.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        third = out_c // 3
        rest  = out_c - 2 * third

        # Branch 1: Gabor (orientation-selective)
        self.gabor = GaborInitConv(in_c, third, k=7)

        # Branch 2: Laplacian of Gaussian (LoG) — fixed, not learned
        self.log_conv = nn.Conv2d(in_c, third, 5, padding=2, bias=False)
        self._init_log(5, third)
        self.log_conv.weight.requires_grad = False  # Fixed edge detector

        # Branch 3: Learned residual content
        self.content = nn.Conv2d(in_c, rest, 3, padding=1)

        # Fusion
        self.fuse = nn.Conv2d(out_c, out_c, 1)
        self.norm = nn.GroupNorm(min(32, out_c), out_c)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def _init_log(self, k, out_c):
        """Laplacian-of-Gaussian kernel (isotropic edge detector)."""
        sigma = 1.4
        half  = k // 2
        y, x  = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
        r2    = x**2 + y**2
        kernel = -(1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
        kernel -= kernel.mean()
        kernel /= (np.abs(kernel).sum() + 1e-8)
        kt = torch.from_numpy(kernel)
        with torch.no_grad():
            for i in range(out_c):
                for j in range(self.log_conv.weight.shape[1]):
                    self.log_conv.weight[i, j] = kt

    def forward(self, x):
        g = self.gabor(x)
        l = F.leaky_relu(self.log_conv(x), 0.2)
        c = self.content(x)
        out = self.fuse(torch.cat([g, l, c], dim=1))
        return self.act(self.norm(out))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — STROKE-ADAPTIVE NORMALIZATION (SAN)
# ─────────────────────────────────────────────────────────────────────────────

class StrokeAdaptiveNorm(nn.Module):
    """
    SAN — Novel normalization conditioned on a stroke-code vector extracted
    from the sketch.

    Unlike AdaIN (which uses an arbitrary style vector), SAN derives γ and β
    from a compact "stroke summary" of the current sketch region. This keeps
    the normalization structurally coherent with the drawing.

    stroke_code_dim: dimension of the stroke descriptor passed from encoder.
    """
    def __init__(self, num_features, stroke_code_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # Two separate MLPs to predict γ and β from the stroke code
        self.gamma_proj = nn.Sequential(
            nn.Linear(stroke_code_dim, num_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 2, num_features)
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(stroke_code_dim, num_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 2, num_features)
        )

    def forward(self, x, stroke_code):
        """
        x          : [B, C, H, W]  feature map to normalize
        stroke_code: [B, D]        stroke descriptor from sketch encoder
        """
        normed = self.norm(x)
        gamma  = self.gamma_proj(stroke_code).unsqueeze(-1).unsqueeze(-1)
        beta   = self.beta_proj(stroke_code).unsqueeze(-1).unsqueeze(-1)
        return normed * (1 + gamma) + beta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RING ATTENTION BOTTLENECK (RAB)
# ─────────────────────────────────────────────────────────────────────────────

class RingAttentionBottleneck(nn.Module):
    """
    RAB — Novel O(N·R) spatial attention mechanism.

    Standard self-attention is O(N²) in sequence length (= H·W pixels).
    RAB instead samples R rings of radius r₁, r₂, … around each position
    and computes attention only within each ring.

    For a 32×32 feature map with 3 rings of 8 samples each:
      Standard: 1024 × 1024 = 1M dot-products
      RAB:      1024 × 24   = 24K dot-products   (42× cheaper)

    This makes it practical even in the bottleneck of a generator.

    Implementation uses grid_sample for differentiable ring sampling.
    """
    def __init__(self, channels, n_heads=4, rings=(4, 8, 16)):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads  = n_heads
        self.rings    = rings           # radii in pixels
        self.n_samples = 8              # samples per ring (evenly spaced angles)
        self.head_dim  = channels // n_heads

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.scale  = self.head_dim ** -0.5

        # Precompute ring offsets (angular samples at each radius)
        offsets = []
        for r in rings:
            for i in range(self.n_samples):
                angle = 2 * math.pi * i / self.n_samples
                offsets.append((r * math.cos(angle), r * math.sin(angle)))
        # [R*S, 2]
        self.register_buffer(
            'ring_offsets',
            torch.tensor(offsets, dtype=torch.float32)
        )

    def _sample_ring_features(self, feat, offsets):
        """
        Sample feat at positions (each pixel + each ring offset).
        Returns [B, C, H, W, K] where K = number of offsets.
        """
        B, C, H, W = feat.shape
        K = offsets.shape[0]

        # Build base grid: [B, H, W, 2]  (normalized coords in [-1, 1])
        gy = torch.linspace(-1, 1, H, device=feat.device)
        gx = torch.linspace(-1, 1, W, device=feat.device)
        grid = torch.stack(torch.meshgrid(gx, gy, indexing='xy'), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)                     # [B, H, W, 2]

        # Offsets in normalised coords
        off_norm = offsets.clone()
        off_norm[:, 0] /= (W / 2)
        off_norm[:, 1] /= (H / 2)

        sampled = []
        for k in range(K):
            shifted = grid + off_norm[k].view(1, 1, 1, 2)
            shifted = shifted.clamp(-1, 1)
            # grid_sample expects [B, C, H_out, W_out] ← grid [B, H_out, W_out, 2]
            s = F.grid_sample(feat, shifted, mode='bilinear',
                              padding_mode='border', align_corners=True)  # [B, C, H, W]
            sampled.append(s)
        return torch.stack(sampled, dim=-1)  # [B, C, H, W, K]

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.q_proj(x)  # [B, C, H, W]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Sample keys and values at ring positions
        K_ring = self._sample_ring_features(K, self.ring_offsets)  # [B, C, H, W, K]
        V_ring = self._sample_ring_features(V, self.ring_offsets)

        # Reshape for multi-head attention
        n_k = self.ring_offsets.shape[0]
        Q = Q.view(B, self.n_heads, self.head_dim, H, W)
        K_ring = K_ring.view(B, self.n_heads, self.head_dim, H, W, n_k)
        V_ring = V_ring.view(B, self.n_heads, self.head_dim, H, W, n_k)

        # Attention scores: Q · K^T over ring dimension
        # [B, heads, head_dim, H, W] × [B, heads, head_dim, H, W, K]
        Q_exp = Q.unsqueeze(-1)                    # [B, heads, d, H, W, 1]
        scores = (Q_exp * K_ring).sum(dim=2)       # [B, heads, H, W, K]
        scores = scores * self.scale
        attn   = F.softmax(scores, dim=-1)         # [B, heads, H, W, K]

        # Weighted sum of values
        # attn: [B, heads, H, W, K]  V_ring: [B, heads, d, H, W, K]
        attn_exp = attn.unsqueeze(2)               # [B, heads, 1, H, W, K]
        out = (attn_exp * V_ring).sum(dim=-1)      # [B, heads, d, H, W]
        out = out.view(B, C, H, W)

        # Residual connection
        return x + self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GATED RESIDUAL BRIDGE (GRB)
# ─────────────────────────────────────────────────────────────────────────────

class GatedResidualBridge(nn.Module):
    """
    GRB — Novel learnable skip connection for sketch→color generators.

    Problem with standard U-Net skip connections: sketch noise, irrelevant
    strokes, and texture artifacts all pass through equally.

    GRB solution: a learned gate G ∈ [0,1] (per-channel, per-spatial-location)
    controls how much of the encoder (sketch) feature passes vs. decoder feature.

        output = G * encoder_feat + (1 - G) * decoder_feat

    G is computed from the concatenation of both features → the network
    learns WHERE sketch structure is important and WHERE to let color
    information dominate.
    """
    def __init__(self, channels):
        super().__init__()
        # Gate: learns from both encoder and decoder context
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.Sigmoid()                   # Gate ∈ [0, 1]
        )
        # Light transform on encoder feature before merging
        self.enc_transform = nn.Conv2d(channels, channels, 1)

    def forward(self, enc_feat, dec_feat):
        """
        enc_feat: skip-connection feature from encoder (sketch-side)
        dec_feat: upsampled feature from decoder (color-side)
        """
        gate = self.gate_conv(torch.cat([enc_feat, dec_feat], dim=1))
        enc_t = self.enc_transform(enc_feat)
        return gate * enc_t + (1 - gate) * dec_feat


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FREQUENCY-DECOMPOSED UPSAMPLER (FDU)
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyDecomposedUpsampler(nn.Module):
    """
    FDU — Novel 2× upsampling with explicit frequency decomposition.

    Most generators use bilinear or transposed conv for upsampling.
    FDU instead splits into two paths per stage:
      • LOW path : Bilinear + conv → smooth, global color regions
      • HIGH path: PixelShuffle + learnable high-pass → sharp edges/details

    A per-channel learned α blends the two paths:
        out = α·high + (1-α)·low

    This forces the network to explicitly separate "what color goes here"
    (low-freq) from "where are the edges" (high-freq), matching how humans
    perceive colorized sketches.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # LOW frequency path
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # HIGH frequency path (pixel-shuffle based)
        self.pixel_shuffle = PixelShuffle2x(in_c)
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Learnable blend coefficient per output channel
        self.alpha = nn.Parameter(torch.zeros(1, out_c, 1, 1) + 0.5)

    def forward(self, x):
        # LOW: bilinear upsample then conv
        x_up_low  = F.interpolate(x, scale_factor=2, mode='bilinear',
                                   align_corners=False)
        low  = self.low_conv(x_up_low)

        # HIGH: pixel-shuffle upsample then conv
        x_up_high = self.pixel_shuffle(x)
        high = self.high_conv(x_up_high)

        # Blend with learned α (sigmoid ensures [0,1])
        alpha = torch.sigmoid(self.alpha)
        return alpha * high + (1 - alpha) * low


class FrequencyDecomposedRefiner(nn.Module):
    """
    Refiner variant of FDU that preserves spatial resolution.
    Used for final full-resolution passes where upsampling is not required.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # LOW frequency path (blur then conv)
        self.low_path = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # HIGH frequency path (residual details)
        self.high_path = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.alpha = nn.Parameter(torch.zeros(1, out_c, 1, 1) + 0.5)

    def forward(self, x):
        low   = self.low_path(x)
        high  = self.high_path(x)
        alpha = torch.sigmoid(self.alpha)
        return alpha * high + (1 - alpha) * low


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — GENERATOR: ChromaFlowNet
# ─────────────────────────────────────────────────────────────────────────────

class ChromaFlowNet(nn.Module):
    """
    ChromaFlowNet — The generator for SketchFlowGAN.

    Architecture overview:
    ┌─────────────────────────────────────────────────────────────┐
    │  INPUT: Sketch [B, 1, H, W]                                 │
    │                                                             │
    │  ENCODER (sketch analysis)                                  │
    │    E0: DSA(1→64)      → [B, 64, H,   W]    + GRB skip      │
    │    E1: DSA+down(64→128)→ [B,128, H/2, W/2] + GRB skip      │
    │    E2: conv(128→256)  → [B,256, H/4, W/4]  + GRB skip      │
    │    E3: conv(256→512)  → [B,512, H/8, W/8]  + GRB skip      │
    │                                                             │
    │  STROKE CODE: Global avg-pool of E0 → stroke descriptor     │
    │              [B, 64] used to drive SAN in decoder           │
    │                                                             │
    │  BOTTLENECK                                                 │
    │    RAB(512) × 2       → [B,512, H/8, W/8]                  │
    │                                                             │
    │  DECODER (color synthesis)                                  │
    │    D3: FDU(512→256) + GRB(E3) → [B,256, H/4, W/4]  + SAN  │
    │    D2: FDU(256→128) + GRB(E2) → [B,128, H/2, W/2]  + SAN  │
    │    D1: FDU(128→64)  + GRB(E1) → [B, 64, H,   W]   + SAN  │
    │    D0: FDU(64→32)   + GRB(E0) → [B, 32, H,   W]   + SAN  │
    │                                                             │
    │  OUTPUT HEAD: Conv(32→3) + Tanh → [B, 3, H, W]  (RGB)     │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, img_size=256):
        super().__init__()
        stroke_code_dim = 64

        # ── Encoder ────────────────────────────────────────────────────
        self.enc0 = DirectionalStrokeAnalyzer(1, 64)           # [B,64,H,W]
        self.enc1 = nn.Sequential(
            DirectionalStrokeAnalyzer(64, 128),
            nn.AvgPool2d(2)                                    # [B,128,H/2,W/2]
        )
        self.enc2 = nn.Sequential(
            self._enc_block(128, 256),
            nn.AvgPool2d(2)                                    # [B,256,H/4,W/4]
        )
        self.enc3 = nn.Sequential(
            self._enc_block(256, 512),
            nn.AvgPool2d(2)                                    # [B,512,H/8,W/8]
        )

        # ── Stroke code extractor ──────────────────────────────────────
        self.stroke_code_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → [B, 64, 1, 1]
            nn.Flatten(),              # → [B, 64]
            nn.Linear(64, stroke_code_dim),
            nn.ReLU(inplace=True)
        )

        # ── Bottleneck: two stacked Ring Attention Blocks ─────────────
        self.bottleneck = nn.Sequential(
            RingAttentionBottleneck(512, n_heads=4, rings=(2, 4, 6)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            RingAttentionBottleneck(512, n_heads=4, rings=(2, 4, 6)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Decoder: FDU upsampling ────────────────────────────────────
        self.fdu3 = FrequencyDecomposedUpsampler(512, 256)     # → H/4
        self.fdu2 = FrequencyDecomposedUpsampler(256, 128)     # → H/2
        self.fdu1 = FrequencyDecomposedUpsampler(128, 64)      # → H
        self.fdu0 = FrequencyDecomposedRefiner(64,  32)        # → H (refinement at full res)

        # ── Gated Residual Bridges ────────────────────────────────────
        self.grb3 = GatedResidualBridge(256)
        self.grb2 = GatedResidualBridge(128)
        self.grb1 = GatedResidualBridge(64)
        self.grb0 = GatedResidualBridge(32)   # bridge e0 (64) → need proj

        # Bridge channel mismatch: enc0 is 64-ch, dec0 needs 32-ch bridge
        self.enc0_proj = nn.Conv2d(64, 32, 1)

        # ── Stroke-Adaptive Normalization in decoder ──────────────────
        self.san3 = StrokeAdaptiveNorm(256, stroke_code_dim)
        self.san2 = StrokeAdaptiveNorm(128, stroke_code_dim)
        self.san1 = StrokeAdaptiveNorm(64,  stroke_code_dim)
        self.san0 = StrokeAdaptiveNorm(32,  stroke_code_dim)

        # ── Output head ───────────────────────────────────────────────
        self.out_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 1),
            nn.Tanh()
        )

    def _enc_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, sketch):
        """
        sketch: [B, 1, H, W]  grayscale drawing, values ∈ [-1, 1]
        returns: [B, 3, H, W]  colorized image,   values ∈ [-1, 1]
        """
        # ── Encode ────────────────────────────────────────────────────
        e0 = self.enc0(sketch)    # [B, 64, H,   W]
        e1 = self.enc1(e0)        # [B,128, H/2, W/2]
        e2 = self.enc2(e1)        # [B,256, H/4, W/4]
        e3 = self.enc3(e2)        # [B,512, H/8, W/8]

        # ── Stroke code ───────────────────────────────────────────────
        sc = self.stroke_code_proj(e0)   # [B, 64]

        # ── Bottleneck ────────────────────────────────────────────────
        z = self.bottleneck(e3)          # [B, 512, H/8, W/8]

        # ── Decode with GRB + SAN ─────────────────────────────────────
        # D3
        d3 = self.fdu3(z)                        # [B, 256, H/4, W/4]
        d3 = self.grb3(e2, d3)                   # GRB with encoder skip
        d3 = self.san3(d3, sc)                   # SAN conditioning

        # D2
        d2 = self.fdu2(d3)                       # [B, 128, H/2, W/2]
        d2 = self.grb2(e1, d2)
        d2 = self.san2(d2, sc)

        # D1
        d1 = self.fdu1(d2)                       # [B, 64, H, W]
        d1 = self.grb1(e0, d1)
        d1 = self.san1(d1, sc)

        # D0 — additional refinement pass at full resolution
        d0 = self.fdu0(d1)                       # [B, 32, H, W]
        e0_proj = self.enc0_proj(e0)             # 64 → 32 channels
        d0 = self.grb0(e0_proj, d0)
        d0 = self.san0(d0, sc)

        return self.out_head(d0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — DISCRIMINATOR: HierarchicalFieldCritic (MFC)
# ─────────────────────────────────────────────────────────────────────────────

class PatchCritic(nn.Module):
    """
    Single-scale patch discriminator path.
    Operates on patches of size determined by the receptive field,
    using spectral normalization throughout.
    """
    def __init__(self, in_c=4, base_c=64, n_layers=3):
        super().__init__()
        layers = [
            sn_conv(in_c, base_c, k=4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        c = base_c
        for _ in range(n_layers - 1):
            layers += [
                sn_conv(c, c * 2, k=4, s=2, p=1),
                nn.InstanceNorm2d(c * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c *= 2
        layers += [
            sn_conv(c, c * 2, k=4, s=1, p=1),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 2, 1, k=4, s=1, p=1)   # per-patch score
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrequencyCritic(nn.Module):
    """
    Novel discriminator branch operating on FFT magnitude spectra.
    Penalises frequency artifacts (checkerboard, blurring) that spatial
    discriminators miss.

    Takes [B, C, H, W], computes log |FFT| → channels-last → small ConvNet.
    """
    def __init__(self, in_c=3):
        super().__init__()
        self.net = nn.Sequential(
            sn_conv(in_c, 64, k=4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(64, 128, k=4, s=2, p=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(128, 256, k=4, s=2, p=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(256, 1, k=4, s=1, p=1)
        )

    def _fft_magnitude(self, x):
        """Log-magnitude of the 2D FFT, shifted so DC is centred."""
        # x: [B, C, H, W]
        fft  = torch.fft.fft2(x, norm='ortho')
        mag  = torch.abs(fft)
        mag  = torch.fft.fftshift(mag, dim=(-2, -1))
        return torch.log(mag + 1e-8)

    def forward(self, x):
        freq = self._fft_magnitude(x)
        return self.net(freq)


class HierarchicalFieldCritic(nn.Module):
    """
    HFC — Multi-Field Critic (MFC).

    Runs FOUR parallel critics on the (sketch, image) pair:
      1. Fine   critic: 4-layer PatchCritic → 16×16 patch discrimination
      2. Mid    critic: 3-layer PatchCritic → 64×64 patch discrimination
      3. Coarse critic: 2-layer PatchCritic → full-image discrimination
      4. Freq   critic: FrequencyCritic    → spectral artifact detection

    The generator must fool ALL four simultaneously.
    Each critic receives (sketch ‖ image) as input — this is the paired
    conditioning used in pix2pix-style training.
    """
    def __init__(self):
        super().__init__()
        # in_c = 1 (sketch) + 3 (image) = 4
        self.fine   = PatchCritic(in_c=4, base_c=64, n_layers=4)
        self.mid    = PatchCritic(in_c=4, base_c=64, n_layers=3)
        self.coarse = PatchCritic(in_c=4, base_c=64, n_layers=2)
        self.freq   = FrequencyCritic(in_c=3)   # operates on image only

    def forward(self, sketch, image):
        """
        sketch: [B, 1, H, W]
        image : [B, 3, H, W]   (either real or generated)
        Returns: list of patch-score maps (each a [B, 1, h, w] tensor)
        """
        pair = torch.cat([sketch, image], dim=1)  # [B, 4, H, W]
        return [
            self.fine(pair),
            self.mid(pair),
            self.coarse(pair),
            self.freq(image)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CUSTOM LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def sobel_edges(x):
    """
    Sobel edge detector implemented as a fixed convolution (no import needed).
    x: [B, C, H, W] → returns [B, C, H, W] edge magnitudes.
    """
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    B, C, H, W = x.shape
    x_flat = x.view(B * C, 1, H, W)
    kx = kx.expand(1, 1, 3, 3)
    ky = ky.expand(1, 1, 3, 3)
    gx = F.conv2d(x_flat, kx, padding=1)
    gy = F.conv2d(x_flat, ky, padding=1)
    # Use in-place ops and fewer intermediates to save VRAM
    mag = gx.pow(2)
    mag.add_(gy.pow(2))
    mag.add_(1e-8)
    mag.sqrt_()
    return mag.view(B, C, H, W)


class SketchFlowLoss(nn.Module):
    """
    Combined loss for SketchFlowGAN training.

    Lossₜₒₜₐₗ = λ_adv · L_adv + λ_spl · L_spl + λ_ccl · L_ccl + λ_sfl · L_sfl

      • L_adv: Multi-field LSGAN adversarial loss
      • L_spl: Spectral Perceptual Loss (L1 on log-FFT of real vs. fake)
      • L_ccl: Color Coherence Loss (histogram matching in RGB)
      • L_sfl: Sketch Fidelity Loss (Sobel edge agreement sketch↔generated)
    """
    def __init__(self, lambda_adv=1.0, lambda_spl=5.0,
                 lambda_ccl=10.0, lambda_sfl=2.0):
        super().__init__()
        self.w_adv = lambda_adv
        self.w_spl = lambda_spl
        self.w_ccl = lambda_ccl
        self.w_sfl = lambda_sfl

    def lsgan_loss(self, preds, target_is_real):
        """Least-squares GAN loss — more stable than BCE."""
        target = 1.0 if target_is_real else 0.0
        return sum(F.mse_loss(p, torch.full_like(p, target)) for p in preds)

    def spectral_perceptual_loss(self, fake, real):
        """L1 distance between log-magnitude FFT spectra."""
        def log_fft(x):
            f = torch.fft.fft2(x, norm='ortho')
            return torch.log(torch.abs(f) + 1e-8)
        return F.l1_loss(log_fft(fake), log_fft(real))

    def color_coherence_loss(self, fake, real, bins=32):
        """
        Soft histogram matching loss.
        Encourages matching color distributions across the image,
        preventing the generator from using wrong global palettes.
        """
        loss = 0.0
        for c in range(3):
            # Flatten spatial dims
            f_c = fake[:, c].reshape(fake.shape[0], -1)   # [B, H*W]
            r_c = real[:, c].reshape(real.shape[0], -1)
            # Soft histogram via KDE with fixed bandwith
            edges = torch.linspace(-1, 1, bins + 1, device=fake.device)
            f_hist = self._soft_histogram(f_c, edges)
            r_hist = self._soft_histogram(r_c, edges)
            loss += F.l1_loss(f_hist, r_hist)
        return loss / 3

    def _soft_histogram(self, x, edges):
        """Differentiable histogram via triangle kernel."""
        width = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2   # [bins]
        # x: [B, N],  centers: [bins]
        # distance of each sample to each bin center
        diff = (x.unsqueeze(-1) - centers.view(1, 1, -1)).abs()  # [B, N, bins]
        weights = F.relu(1 - diff / width)
        hist = weights.mean(dim=1)   # [B, bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        return hist

    def sketch_fidelity_loss(self, generated, sketch):
        """
        Structural consistency: edges of generated image should align with
        edges of the input sketch. Prevents the generator from ignoring sketch.
        """
        gen_gray = generated.mean(dim=1, keepdim=True)   # RGB → gray
        gen_edges = sobel_edges(gen_gray)
        sk_edges  = sobel_edges(sketch)
        return F.l1_loss(gen_edges, sk_edges)

    def generator_loss(self, D_fake_preds, fake, real, sketch):
        """Total generator loss."""
        adv  = self.lsgan_loss(D_fake_preds, target_is_real=True)
        spl  = self.spectral_perceptual_loss(fake, real)
        ccl  = self.color_coherence_loss(fake, real)
        sfl  = self.sketch_fidelity_loss(fake, sketch)
        return (self.w_adv * adv + self.w_spl * spl +
                self.w_ccl * ccl + self.w_sfl * sfl)

    def discriminator_loss(self, D_real_preds, D_fake_preds):
        """Discriminator: real → 1, fake → 0."""
        real_loss = self.lsgan_loss(D_real_preds, target_is_real=True)
        fake_loss = self.lsgan_loss(D_fake_preds, target_is_real=False)
        return (real_loss + fake_loss) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SketchColorDataset(Dataset):
    """
    Handles two modes:
    1. Split Folders: data_dir/sketches/*.png and data_dir/colored/*.png
    2. Concatenated: side-by-side images in data_dir (common in Pix2Pix datasets)
    """
    def __init__(self, data_dir, img_size=256, augment=True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment  = augment
        
        # Check for split mode
        self.sketch_dir  = self.data_dir / 'sketches'
        self.colored_dir = self.data_dir / 'colored'
        self.is_split = self.sketch_dir.exists() and self.colored_dir.exists()

        if self.is_split:
            self.files = sorted(list(self.sketch_dir.glob('*.png')) + list(self.sketch_dir.glob('*.jpg')))
            print(f"[Dataset] Split mode detected. Found {len(self.files)} samples.")
        else:
            # Check for concatenated mode (train/val/test folders)
            # Find all images recursively
            self.files = sorted(list(self.data_dir.rglob('*.jpg')) + list(self.data_dir.rglob('*.png')))
            print(f"[Dataset] Concatenated mode detected. Found {len(self.files)} files in {self.data_dir}")

        if not self.files:
            raise FileNotFoundError(f"No images found in {self.data_dir}")

        self.to_tensor   = T.ToTensor()
        self.normalize   = T.Normalize([0.5], [0.5])
        self.norm_rgb    = T.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        
        if self.is_split:
            sketch  = Image.open(path).convert('L')
            cl_path = self.colored_dir / path.name
            colored = Image.open(cl_path).convert('RGB')
        else:
            # Side-by-side mode: Split image down the middle
            img = Image.open(path).convert('RGB')
            w, h = img.size
            # Pix2Pix: Sketch is usually LEFT, Photo is RIGHT
            # But sometimes it's reversed. We assume [Sketch | Photo] based on edges2shoes
            sketch  = img.crop((0, 0, w // 2, h)).convert('L')
            colored = img.crop((w // 2, 0, w, h))

        # Resize
        sketch  = sketch.resize((self.img_size, self.img_size), Image.BILINEAR)
        colored = colored.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Augment
        if self.augment and torch.rand(1).item() > 0.5:
            sketch  = sketch.transpose(Image.FLIP_LEFT_RIGHT)
            colored = colored.transpose(Image.FLIP_LEFT_RIGHT)

        sk_t = self.normalize(self.to_tensor(sketch))    # [1, H, W]
        cl_t = self.norm_rgb(self.to_tensor(colored))    # [3, H, W]
        return sk_t, cl_t


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class SketchFlowTrainer:
    def __init__(self, G, D, loss_fn, device, lr=2e-4, beta=(0.5, 0.999)):
        self.G      = G.to(device)
        self.D      = D.to(device)
        self.loss   = loss_fn
        self.device = device

        # Two-timescale update rule: D learns slightly faster than G
        self.opt_G = optim.Adam(G.parameters(), lr=lr,       betas=beta)
        self.opt_D = optim.Adam(D.parameters(), lr=lr * 1.5, betas=beta)

        # Cosine annealing with warm restarts
        self.sched_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_G, T_0=50, T_mult=2)
        self.sched_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_D, T_0=50, T_mult=2)

        self.history = {'G_loss': [], 'D_loss': []}

    def train_step(self, sketch, real):
        sketch = sketch.to(self.device)
        real   = real.to(self.device)

        # ── Train Discriminator ──────────────────────────────────────
        self.opt_D.zero_grad()
        with torch.no_grad():
            fake = self.G(sketch)

        D_real = self.D(sketch, real)
        D_fake = self.D(sketch, fake.detach())
        d_loss = self.loss.discriminator_loss(D_real, D_fake)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.opt_D.step()

        # ── Train Generator ──────────────────────────────────────────
        self.opt_G.zero_grad()
        fake     = self.G(sketch)
        D_fake_g = self.D(sketch, fake)
        g_loss   = self.loss.generator_loss(D_fake_g, fake, real, sketch)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.opt_G.step()

        return d_loss.item(), g_loss.item()

    def train(self, dataloader, epochs=100, save_dir='./checkpoints', accum_steps=4):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/samples", exist_ok=True)
        
        scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Starting training on {self.device}")
        
        sample_every = 1
        log_every    = 1
        
        epoch = 0
        try:
            for epoch in range(1, epochs + 1):
                self.G.train()
                self.D.train()
                epoch_d, epoch_g = [], []
                
                self.opt_G.zero_grad()
                self.opt_D.zero_grad()

                for i, (sketch, real) in enumerate(dataloader):
                    sketch, real = sketch.to(self.device), real.to(self.device)

                    # --- Train Discriminator ---
                    self.opt_D.zero_grad()
                    with torch.amp.autocast("cuda"):
                        fake = self.G(sketch)
                        D_real = self.D(sketch, real)
                        D_fake = self.D(sketch, fake.detach())
                        loss_D = self.loss.discriminator_loss(D_real, D_fake)
                    
                    # Scale loss for accumulation
                    scaler.scale(loss_D / accum_steps).backward()
                    
                    if (i + 1) % accum_steps == 0:
                        scaler.step(self.opt_D)
                        scaler.update()
                        self.opt_D.zero_grad()
                    
                    # --- Train Generator ---
                    with torch.amp.autocast("cuda"):
                        D_fake_news = self.D(sketch, fake)
                        # We need to pass fake.detach() or just fake? 
                        # In generator loss, we want gradients to flow to G.
                        loss_G = self.loss.generator_loss(D_fake_news, fake, real, sketch)
                    
                    # Scale loss for accumulation
                    scaler.scale(loss_G / accum_steps).backward()
                    
                    if (i + 1) % accum_steps == 0:
                        scaler.step(self.opt_G)
                        scaler.update()
                        self.opt_G.zero_grad()

                    epoch_d.append(loss_D.item())
                    epoch_g.append(loss_G.item())

                    if i % 10 == 0:
                        print(f"Batch {i}/{len(dataloader)} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}", end='\r')

                self.sched_G.step()
                self.sched_D.step()

                mean_d = np.mean(epoch_d)
                mean_g = np.mean(epoch_g)
                self.history['D_loss'].append(mean_d)
                self.history['G_loss'].append(mean_g)

                print(f"\nEpoch {epoch:4d}/{epochs} | D: {mean_d:.4f} | G: {mean_g:.4f}")

                if epoch % sample_every == 0:
                    self._save_samples(dataloader, epoch, save_dir)
                    self._save_checkpoint(epoch, save_dir)
        except KeyboardInterrupt:
            print("\n[Trainer] Training interrupted by user. Saving current state...")
            self._save_checkpoint(epoch, save_dir)
            return

        self._plot_losses(save_dir)

    @torch.no_grad()
    def _save_samples(self, dataloader, epoch, save_dir):
        self.G.eval()
        sketch, real = next(iter(dataloader))
        sketch = sketch[:4].to(self.device)
        real   = real[:4].to(self.device)
        fake   = self.G(sketch)

        # Denormalize: [-1,1] → [0,1]
        denorm = lambda t: (t * 0.5 + 0.5).clamp(0, 1)
        sk_show   = denorm(sketch.expand(-1, 3, -1, -1))
        fake_show = denorm(fake)
        real_show = denorm(real)

        # Side-by-side grid: [sketch | generated | real]
        grid = torch.cat([sk_show, fake_show, real_show], dim=3)
        img  = grid.permute(0, 2, 3, 1).cpu().numpy()

        fig, axes = plt.subplots(1, len(img), figsize=(15, 5))
        for ax, im in zip(axes, img):
            ax.imshow(im)
            ax.axis('off')
        fig.suptitle(f'Epoch {epoch}  |  Sketch → Generated → Real')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/samples/epoch_{epoch:04d}.png', dpi=80)
        plt.close()

    def _save_checkpoint(self, epoch, save_dir):
        torch.save({
            'epoch':   epoch,
            'G_state': self.G.state_dict(),
            'D_state': self.D.state_dict(),
            'optG':    self.opt_G.state_dict(),
            'optD':    self.opt_D.state_dict(),
            'history': self.history
        }, f'{save_dir}/checkpoint_epoch{epoch:04d}.pt')

    def _plot_losses(self, save_dir):
        plt.figure(figsize=(10, 4))
        plt.plot(self.history['G_loss'], label='Generator')
        plt.plot(self.history['D_loss'], label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SketchFlowGAN Training Losses')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_losses.png')
        plt.close()
        print(f"Loss plot saved to {save_dir}/training_losses.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — REAL-TIME INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeInferenceEngine:
    """
    Designed for side-by-side drawing + generation (the "parallel" use case).

    Usage pattern:
        engine = RealTimeInferenceEngine(checkpoint_path, device)
        while user_is_drawing:
            sketch_frame  = capture_canvas()       # [H, W] numpy uint8
            colored_frame = engine.infer(sketch_frame)  # [H, W, 3] numpy
            display_side_by_side(sketch_frame, colored_frame)

    Features:
        • TorchScript-exportable for faster runtime
        • Optional half-precision (fp16) for 2× speed on GPU
        • Frame-level debouncing: only re-runs when sketch has changed enough
    """
    def __init__(self, checkpoint_path=None, device='cpu',
                 img_size=256, use_fp16=False):
        self.device   = torch.device(device)
        self.img_size = img_size
        self.use_fp16 = use_fp16 and (device != 'cpu')

        self.G = ChromaFlowNet(img_size=img_size).to(self.device)
        self.G.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            ck = torch.load(checkpoint_path, map_location=self.device)
            self.G.load_state_dict(ck['G_state'])
            print(f"[Engine] Loaded checkpoint from epoch {ck['epoch']}")
        else:
            print("[Engine] No checkpoint — running with random weights (for testing)")

        if self.use_fp16:
            self.G = self.G.half()

        # Preprocessing pipeline
        self.preprocess = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self._last_sketch = None
        self._cached_out = None

    def infer(self, sketch_np, change_threshold=0.02):
        """
        sketch_np: numpy [H, W] uint8 grayscale OR [H, W, 3] BGR (from OpenCV)
        returns  : numpy [H, W, 3] uint8 RGB colorized image
        """
        # Convert to PIL grayscale
        if sketch_np.ndim == 3:
            from PIL import Image as PIL_Image
            pil = PIL_Image.fromarray(sketch_np).convert('L')
        else:
            from PIL import Image as PIL_Image
            pil = PIL_Image.fromarray(sketch_np)

        # Debounce: skip expensive forward pass if sketch barely changed
        # We compute a fast mean difference for thresholding
        if self._last_sketch is not None:
            diff = np.abs(sketch_np.astype(np.float32) - self._last_sketch.astype(np.float32)).mean() / 255.0
            if diff < change_threshold and self._cached_out is not None:
                return self._cached_out

        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        if self.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            out = self.G(tensor)                     # [1, 3, H, W]

        # Denormalize
        out = (out.squeeze(0).float() * 0.5 + 0.5).clamp(0, 1)
        out_np = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        self._last_sketch = sketch_np.copy()
        self._cached_out  = out_np
        return out_np

    def export_torchscript(self, save_path='sketchflow_generator.pt'):
        """Export the generator to TorchScript for production deployment."""
        dummy = torch.randn(1, 1, self.img_size, self.img_size).to(self.device)
        scripted = torch.jit.trace(self.G, dummy)
        scripted.save(save_path)
        print(f"TorchScript exported → {save_path}")
        return scripted


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — LIVE DEMO (OpenCV side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def run_live_demo(checkpoint_path=None, img_size=256):
    """
    Live drawing demo using OpenCV.
    Left canvas: you draw with the mouse.
    Right panel: SketchFlowGAN output updates in real time.

    Controls:
        Left-click + drag : draw
        'c'               : clear canvas
        'q'               : quit
    """
    try:
        import cv2
    except ImportError:
        print("Install opencv-python: pip install opencv-python")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = RealTimeInferenceEngine(checkpoint_path, device, img_size)

    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255
    drawing = [False]

    def mouse_callback(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            cv2.circle(canvas, (x, y), 3, 0, -1)

    win = 'SketchFlowGAN — Draw (left) | Generated (right)  [c=clear, q=quit]'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)

    print(f"\nLive Demo active | Device: {device}")
    print("Draw on the LEFT canvas. Right side updates in real time.")

    while True:
        colored = engine.infer(canvas)
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        sketch_bgr  = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Divider
        divider = np.ones((img_size, 4, 3), dtype=np.uint8) * 80
        combined = np.hstack([sketch_bgr, divider, colored_bgr])
        cv2.imshow(win, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 255

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — ARCHITECTURE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_architecture_summary(img_size=256):
    device = 'cpu'
    G = ChromaFlowNet(img_size=img_size)
    D = HierarchicalFieldCritic()

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())

    print("=" * 60)
    print("  SketchFlowGAN — Architecture Summary")
    print("=" * 60)
    print(f"  Generator  (ChromaFlowNet):         {g_params:>10,} params")
    print(f"  Discriminator (HierarchFieldCritic):{d_params:>10,} params")
    print(f"  Total:                              {g_params+d_params:>10,} params")
    print()
    print("  Novel Components:")
    print("  |-- DSA  : Directional Stroke Analyzer")
    print("  |         (8-orientation Gabor init + LoG + learned branch)")
    print("  |-- RAB  : Ring Attention Bottleneck")
    print("  |         (O(N·R) attention at fixed-radius rings)")
    print("  |-- GRB  : Gated Residual Bridge")
    print("  |         (learned sigmoid gate on skip connections)")
    print("  |-- FDU  : Frequency-Decomposed Upsampler")
    print("  |         (low-freq bilinear + high-freq pixel-shuffle + alpha blend)")
    print("  |-- SAN  : Stroke-Adaptive Normalization")
    print("  |         (gamma,beta predicted from stroke-code descriptor)")
    print("  |-- MFC  : Multi-Field Critic")
    print("            (4-layer, 3-layer, 2-layer patch + FFT critic)")
    print()
    print("  Loss Functions:")
    print("  |-- LSGAN adversarial (multi-field)")
    print("  |-- Spectral Perceptual Loss (log-FFT L1)")
    print("  |-- Color Coherence Loss (soft histogram matching)")
    print("  |-- Sketch Fidelity Loss (Sobel edge consistency)")
    print("=" * 60)

    # Quick forward pass test
    dummy_sk = torch.randn(1, 1, img_size, img_size)
    dummy_co = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        out_g = G(dummy_sk)
        out_d = D(dummy_sk, dummy_co)
    print(f"\n  Forward pass OK:")
    print(f"  G: {list(dummy_sk.shape)} -> {list(out_g.shape)}")
    print(f"  D: outputs from {len(out_d)} critics, "
          f"shapes: {[list(o.shape) for o in out_d]}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SketchFlowGAN')
    parser.add_argument('--mode', choices=['train', 'infer', 'demo', 'summary'],
                        default='summary')
    parser.add_argument('--data_dir',    type=str, default='./dataset')
    parser.add_argument('--checkpoint',  type=str, default=None)
    parser.add_argument('--sketch_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./output.png')
    parser.add_argument('--img_size',    type=int, default=256)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--epochs',      type=int, default=200)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--save_dir',    type=str, default='./checkpoints')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch size = batch_size * accum_steps)')
    parser.add_argument('--fp16',        action='store_true',
                        help='Half-precision inference (GPU only)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.mode == 'summary':
        print_architecture_summary(args.img_size)

    elif args.mode == 'train':
        dataset = SketchColorDataset(args.data_dir, img_size=args.img_size)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=2, pin_memory=True)
        G       = ChromaFlowNet(img_size=args.img_size)
        D       = HierarchicalFieldCritic()
        loss_fn = SketchFlowLoss()
        trainer = SketchFlowTrainer(G, D, loss_fn, device, lr=args.lr)

        if args.checkpoint:
            ck = torch.load(args.checkpoint, map_location=device)
            G.load_state_dict(ck['G_state'])
            D.load_state_dict(ck['D_state'])
            print(f"Resumed from epoch {ck['epoch']}")

        trainer.train(loader, epochs=args.epochs, save_dir=args.save_dir, accum_steps=args.accum_steps)

    elif args.mode == 'infer':
        if not args.sketch_path:
            print("Provide --sketch_path for inference mode.")
            return
        engine = RealTimeInferenceEngine(
            args.checkpoint, device, args.img_size, use_fp16=args.fp16)
        sketch = np.array(Image.open(args.sketch_path).convert('L'))
        result = engine.infer(sketch)
        Image.fromarray(result).save(args.output_path)
        print(f"Saved colorized output → {args.output_path}")

    elif args.mode == 'demo':
        run_live_demo(args.checkpoint, args.img_size)


if __name__ == '__main__':
    main()
