"""
networks.py

Neural Network Building Blocks for AdS/CFT Flow Matching
=========================================================

This module provides all neural network components for the UV-stabilized
generative flow matching framework, including:

Core Data Structures
--------------------
- BulkState: Container for bulk field state (Φ̃, Π̃)
- BulkField: Channelwise bulk scalar specification
- DiscretizationConfig: Slice discretization configuration

Neural Networks
---------------
- ResidualVelocityNet: MLP residual network for point data
- ResidualVelocityNet2D: CNN residual network for 2D field data
- AdSBackbone: Klein-Gordon backbone in UV-stabilized variables

UV Boundary Operations
----------------------
- UVDictionaryLift: Dictionary-consistent UV lift (Section 5.1)
- UVReadout: UV coefficient extraction (Section 5.2)
- UVCoefficientOperator: Two-term coefficient fitting

Path Interpolation
------------------
- AdSAffineHermiteCoupling: AdS-affine cubic Hermite path (Section 9)

Factory Functions
-----------------
- build_slice_laplacian: Build geometry-appropriate Laplacian
- slice_quadrature_weights: Compute L² quadrature weights
- sample_isotropic_noise_from_quadrature: Sample L²-isotropic noise

Document References
-------------------
- Section 3: Klein-Gordon backbone
- Section 4: UV-stabilized variables and residual structure
- Section 5: UV lift and readout
- Section 7: IR prior and Sobolev weighting
- Section 9: Path interpolation (linear and Hermite)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_cft.registry import register_network
from ads_cft.geometry import AdSGeometry, SliceGeometry
from ads_cft.config import DiscretizationConfig

Tensor = torch.Tensor


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class BulkState:
    """
    Stabilized bulk state at a radius: (Φ̃, Π̃).

    Document notation: S(r) = (Φ̃(r,·), Π̃(r,·))

    The UV-stabilized variables are defined as:
        Φ̃ = e^{Δr} Φ
        Π̃ = e^{Δr}(Π + ΔΦ)

    where Δ is the conformal dimension and r is the radial coordinate.

    Attributes
    ----------
    phi_tilde : Tensor
        UV-stabilized field Φ̃, shape (B, C, *spatial) or (B, C)
    pi_tilde : Tensor
        UV-stabilized momentum Π̃, same shape as phi_tilde

    Example
    -------
    >>> state = BulkState(phi_tilde=torch.randn(32, 2), pi_tilde=torch.randn(32, 2))
    >>> print(state.phi_tilde.shape)  # (32, 2)
    """

    phi_tilde: Tensor
    pi_tilde: Tensor


@dataclass
class CanonicalState:
    """
    Canonical bulk state at a radius: (Φ, P) where P = ω(r)Π.
    
    This is the Hamiltonian formulation where the system becomes:
        Φ' = ∂H/∂P = P/ω(r)
        P' = -∂H/∂Φ = ω(r)(Δ_ĝΦ/f² - m²Φ)
    
    The Hamiltonian is:
        H(r) = ∫_Σ [P²/(2ω) + ω/2 (|∇Φ|²/f² + m²Φ²)] dvol_ĝ
    
    This form enables symplectic integrators (leapfrog, implicit midpoint)
    which conserve energy and phase-space volume.
    
    Attributes
    ----------
    phi : Tensor
        Physical field Φ (not UV-stabilized), shape (B, C, *spatial) or (B, C)
    P : Tensor
        Canonical momentum P = ω(r)Π, same shape as phi
        
    Note
    ----
    For symplectic integration, we evolve (Φ, P) internally, then convert
    back to (Φ̃, Π̃) at the boundaries for UV stability in I/O.
    """
    
    phi: Tensor
    P: Tensor


# =============================================================================
# Note: DiscretizationConfig is imported from config.py
# =============================================================================


# =============================================================================
# Bulk Field Specification (Document Section 5)
# =============================================================================


class BulkField(nn.Module):
    """
    Channelwise bulk scalar specification via UV dimensions Δ_i.

    Defines the bulk field theory parameters for multi-channel fields.

    Document eq (mass-dimension):
        m_i² = Δ_i(Δ_i - d)

    Defines α_i = 2Δ_i - d for asymptotic expansions:
        Φ ~ A e^{-Δr} + B e^{-(d-Δ)r}  as r → ∞

    Attributes
    ----------
    d : int
        Boundary dimension
    deltas : Tensor
        Conformal dimensions Δ_i for each channel
    m2 : Tensor
        Mass squared m² = Δ(Δ-d) for each channel
    alpha : Tensor
        Asymptotic exponents α = 2Δ - d

    Example
    -------
    >>> bulk = BulkField(d=2, deltas=[1.5, 2.0])
    >>> print(bulk.n_fields)  # 2
    >>> print(bulk.m2)  # tensor([-0.75, 0.0])
    """

    def __init__(self, d: int, deltas: Sequence[float]) -> None:
        """
        Initialize bulk field specification.

        Args:
            d: Boundary dimension (must be positive)
            deltas: Conformal dimensions Δ_i > d/2 for each channel

        Raises:
            ValueError: If d <= 0 or deltas is empty
        """
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        if len(deltas) == 0:
            raise ValueError("deltas must be non-empty")

        self.d = int(d)
        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)
        self.register_buffer("m2", deltas_t * (deltas_t - float(d)))
        self.register_buffer("alpha", 2.0 * deltas_t - float(d))

    @property
    def n_fields(self) -> int:
        """Return number of field channels."""
        return int(self.deltas.numel())

    def _broadcast_deltas(self, x: Tensor) -> Tensor:
        """
        Broadcast deltas to shape (1, C, 1, 1, ...) matching x.

        Args:
            x: Tensor with shape (B, C, ...)

        Returns:
            Deltas broadcasted to match x

        Raises:
            ValueError: If x has wrong shape or channel mismatch
        """
        if x.ndim < 2:
            raise ValueError("Expected x to have shape (B, C, ...)")
        if x.shape[1] != self.n_fields:
            raise ValueError(
                f"Channel mismatch: x has C={x.shape[1]}, BulkField has {self.n_fields}"
            )
        shape = [1, self.n_fields] + [1] * (x.ndim - 2)
        return self.deltas.to(device=x.device, dtype=x.dtype).view(*shape)

    def _broadcast_r(self, r: Tensor, target_ndim: int) -> Tensor:
        """
        Broadcast r to shape (B, 1, 1, ...) matching target_ndim.

        Args:
            r: Radius tensor (scalar or (B,))
            target_ndim: Target number of dimensions

        Returns:
            Radius broadcasted to target shape
        """
        if r.ndim == 0:
            r = r.view(1, 1, *([1] * (target_ndim - 2)))
        elif r.ndim == 1:
            r = r.view(-1, 1, *([1] * (target_ndim - 2)))
        return r

    def stabilize(self, phi: Tensor, pi: Tensor, r: Tensor) -> BulkState:
        """
        Convert physical (Φ, Π) to UV-stabilized (Φ̃, Π̃).

        Document eq (uv-stable-def):
            Φ̃ = e^{Δr} Φ
            Π̃ = e^{Δr}(Π + ΔΦ)

        Args:
            phi: Physical field Φ, shape (B, C, ...)
            pi: Physical momentum Π, shape (B, C, ...)
            r: Radius, scalar or (B,)

        Returns:
            BulkState with stabilized variables
        """
        deltas = self._broadcast_deltas(phi)
        r_b = self._broadcast_r(r, phi.ndim)
        exp_factor = torch.exp(deltas * r_b)

        phi_tilde = exp_factor * phi
        pi_tilde = exp_factor * (pi + deltas * phi)

        return BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)

    def destabilize(self, state: BulkState, r: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert UV-stabilized (Φ̃, Π̃) back to physical (Φ, Π).

        Inverse of eq (uv-stable-def):
            Φ = e^{-Δr} Φ̃
            Π = e^{-Δr} Π̃ - Δ Φ

        Args:
            state: BulkState with stabilized variables
            r: Radius, scalar or (B,)

        Returns:
            Tuple of (phi, pi) physical variables
        """
        deltas = self._broadcast_deltas(state.phi_tilde)
        r_b = self._broadcast_r(r, state.phi_tilde.ndim)
        exp_neg = torch.exp(-deltas * r_b)

        phi = exp_neg * state.phi_tilde
        pi = exp_neg * state.pi_tilde - deltas * phi

        return phi, pi


# =============================================================================
# Time Embedding
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time/radius embedding for neural networks.

    Uses the positional encoding from "Attention Is All You Need":
        PE(pos, 2i) = sin(pos / 10000^{2i/d})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

    This provides a continuous embedding that the network can use to
    condition on the current time/radius.

    Attributes
    ----------
    dim : int
        Embedding dimension
    max_period : float
        Maximum period for sinusoidal functions

    Example
    -------
    >>> embed = SinusoidalTimeEmbedding(dim=128)
    >>> t = torch.linspace(0, 1, 32)  # (32,)
    >>> emb = embed(t)  # (32, 128)
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        """
        Initialize sinusoidal embedding.

        Args:
            dim: Embedding dimension
            max_period: Maximum period (default 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Compute sinusoidal embedding.

        Args:
            t: Time/radius values, shape (B,) or scalar

        Returns:
            Embeddings of shape (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =============================================================================
# Residual Velocity Networks (Document Section 4.2)
# =============================================================================


@register_network("residual_mlp")
class ResidualVelocityNet(nn.Module):
    """
    MLP residual velocity network for point data.

    Implements the neural residual R_θ for the full velocity field:
        V_θ(S, r) = V_KG(S, r) + R_θ(S, r)

    Document Section 4.2:
        - HERMITE path: R_θ(Φ̃, Π̃, r) = (0, N_θ(...))
        - LINEAR path: R_θ(...) = (N_θ^φ, N_θ^π)

    Architecture:
        [Φ̃, Π̃, time_emb] → MLP → [residual output]

    Attributes
    ----------
    channels : int
        Number of field channels
    output_both : bool
        If True, output both (n_phi, n_pi); else only n_pi

    Example
    -------
    >>> net = ResidualVelocityNet(channels=2, hidden=256, depth=4)
    >>> phi = torch.randn(32, 2)
    >>> pi = torch.randn(32, 2)
    >>> t = torch.rand(32)
    >>> n_pi = net(phi, pi, t)  # (32, 2)
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 256,
        depth: int = 4,
        time_emb_dim: int = 128,
        output_both: bool = False,
    ) -> None:
        """
        Initialize MLP residual network.

        Args:
            channels: Number of field channels C
            hidden: Hidden layer dimension
            depth: Number of hidden layers
            time_emb_dim: Time embedding dimension
            output_both: If True, output (n_phi, n_pi); else only n_pi
        """
        super().__init__()
        self.channels = channels
        self.output_both = output_both

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        input_dim = 2 * channels + time_emb_dim
        output_dim = 2 * channels if output_both else channels

        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self, phi_tilde: Tensor, pi_tilde: Tensor, t: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute residual velocity.

        Args:
            phi_tilde: UV-stabilized field, shape (B, C)
            pi_tilde: UV-stabilized momentum, shape (B, C)
            t: Normalized time in [0, 1], shape (B,) or scalar

        Returns:
            If output_both: (n_phi, n_pi) each of shape (B, C)
            Else: n_pi of shape (B, C)
        """
        if t.ndim == 0:
            t = t.expand(phi_tilde.shape[0])

        t_emb = self.time_embed(t)
        x = torch.cat([phi_tilde, pi_tilde, t_emb], dim=-1)
        out = self.net(x)

        if self.output_both:
            return out[:, : self.channels], out[:, self.channels :]
        return out


@register_network("residual_cnn")
class ResidualVelocityNet2D(nn.Module):
    """
    CNN residual velocity network for 2D field data.

    Uses a U-Net-like architecture with:
    - Time conditioning at every block (standard practice)
    - Concatenative skip connections (better information flow)
    - GroupNorm normalization
    - SiLU activations

    Architecture:
        Input: [Φ̃, Π̃] concatenated → Conv → Encoder → Middle → Decoder → Output
        Time embedding added at every residual block

    Attributes
    ----------
    in_channels : int
        Number of input channels (C for phi_tilde)
    output_both : bool
        If True, output both (n_phi, n_pi); else only n_pi

    Example
    -------
    >>> net = ResidualVelocityNet2D(in_channels=2, hidden=64, depth=3)
    >>> phi = torch.randn(32, 2, 16, 16)
    >>> pi = torch.randn(32, 2, 16, 16)
    >>> t = torch.rand(32)
    >>> n_pi = net(phi, pi, t)  # (32, 2, 16, 16)
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
        output_both: bool = False,
    ) -> None:
        """
        Initialize CNN residual network.

        Args:
            in_channels: Number of input field channels
            hidden: Base hidden channel count
            depth: Number of encoder/decoder stages
            time_emb_dim: Time embedding dimension
            output_both: If True, output (n_phi, n_pi); else only n_pi
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_both = output_both
        self.depth = depth

        # Time embedding network
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Input: concatenate phi_tilde and pi_tilde
        self.input_conv = nn.Conv2d(2 * in_channels, hidden, 3, padding=1)

        # Track channel sizes for skip connections
        self.encoder_channels = []
        
        # Encoder blocks with time conditioning
        self.down_blocks = nn.ModuleList()
        self.down_time_mlps = nn.ModuleList()
        ch = hidden
        for i in range(depth):
            self.encoder_channels.append(ch)
            out_ch = ch * 2
            self.down_blocks.append(
                self._make_down_block(ch, out_ch)
            )
            # Time projection for this level
            self.down_time_mlps.append(nn.Linear(time_emb_dim, out_ch))
            ch = out_ch

        # Middle block with time conditioning
        self.middle_block = self._make_middle_block(ch)
        self.middle_time_mlp = nn.Linear(time_emb_dim, ch)

        # Decoder blocks with time conditioning
        # Note: concatenative skips - after upsample we have out_ch, then concat with skip_ch
        self.up_blocks = nn.ModuleList()
        self.up_time_mlps = nn.ModuleList()
        self.skip_convs = nn.ModuleList()  # To reduce concatenated channels
        
        for i in range(depth):
            skip_ch = self.encoder_channels[depth - 1 - i]
            out_ch = ch // 2
            
            # After upsample (conv1): x has out_ch channels
            # After concat with skip: out_ch + skip_ch channels
            # Reduce back to out_ch
            self.skip_convs.append(nn.Conv2d(out_ch + skip_ch, out_ch, 1))
            self.up_blocks.append(
                self._make_up_block(ch, out_ch)
            )
            # Time projection for this level
            self.up_time_mlps.append(nn.Linear(time_emb_dim, out_ch))
            ch = out_ch

        # Output
        out_channels = 2 * in_channels if output_both else in_channels
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def _make_down_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create encoder block: downsample + two convs."""
        return nn.ModuleList([
            # First conv with stride 2 for downsampling
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            # Second conv
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        ])

    def _make_middle_block(self, ch: int) -> nn.Module:
        """Create middle block: two convs at bottleneck."""
        return nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        ])

    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create decoder block: upsample + two convs."""
        return nn.ModuleList([
            # Upsample with transposed conv
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            # Second conv
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        ])

    def forward(
        self, phi_tilde: Tensor, pi_tilde: Tensor, t: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute residual velocity for 2D fields.

        Args:
            phi_tilde: UV-stabilized field, shape (B, C, H, W)
            pi_tilde: UV-stabilized momentum, shape (B, C, H, W)
            t: Normalized time in [0, 1], shape (B,) or scalar

        Returns:
            If output_both: (n_phi, n_pi) each of shape (B, C, H, W)
            Else: n_pi of shape (B, C, H, W)
        """
        B = phi_tilde.shape[0]

        if t.ndim == 0:
            t = t.expand(B)

        # Compute time embedding once
        t_emb = self.time_mlp(self.time_embed(t))  # (B, time_emb_dim)

        # Input
        x = torch.cat([phi_tilde, pi_tilde], dim=1)  # (B, 2C, H, W)
        x = self.input_conv(x)  # (B, hidden, H, W)

        # Encoder with skip connections
        skips = []
        for i, (down_block, time_mlp) in enumerate(zip(self.down_blocks, self.down_time_mlps)):
            skips.append(x)
            conv1, norm1, conv2, norm2 = down_block
            
            # First conv + norm + activation
            x = conv1(x)
            x = norm1(x)
            x = F.silu(x)
            
            # Add time embedding
            t_proj = time_mlp(t_emb)  # (B, out_ch)
            x = x + t_proj.view(B, -1, 1, 1)
            
            # Second conv + norm + activation
            x = conv2(x)
            x = norm2(x)
            x = F.silu(x)

        # Middle block with time conditioning
        conv1, norm1, conv2, norm2 = self.middle_block
        
        # First conv
        x = conv1(x)
        x = norm1(x)
        x = F.silu(x)
        
        # Add time embedding
        t_proj = self.middle_time_mlp(t_emb)
        x = x + t_proj.view(B, -1, 1, 1)
        
        # Second conv
        x = conv2(x)
        x = norm2(x)
        x = F.silu(x)

        # Decoder with concatenative skip connections
        for i, (up_block, time_mlp, skip_conv) in enumerate(
            zip(self.up_blocks, self.up_time_mlps, self.skip_convs)
        ):
            skip = skips[self.depth - 1 - i]
            conv1, norm1, conv2, norm2 = up_block
            
            # Upsample
            x = conv1(x)
            x = norm1(x)
            x = F.silu(x)
            
            # Handle size mismatch before concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            
            # Concatenate skip connection (before second conv)
            # We need to handle this differently - concat then project
            x_with_skip = torch.cat([x, skip], dim=1)
            x = skip_conv(x_with_skip)  # Reduce channels back
            
            # Add time embedding
            t_proj = time_mlp(t_emb)
            x = x + t_proj.view(B, -1, 1, 1)
            
            # Second conv
            x = conv2(x)
            x = norm2(x)
            x = F.silu(x)

        # Output
        out = self.output_conv(x)

        if self.output_both:
            return out[:, : self.in_channels], out[:, self.in_channels :]
        return out


@register_network("residual_cnn_simple")
class ResidualVelocityNet2DSimple(nn.Module):
    """
    Simplified CNN residual velocity network for 2D field data.
    
    This is the simpler architecture used in the original implementation:
    - Time embedding added once at input (not at every block)
    - Additive skip connections (not concatenative)
    - Fewer parameters (~10M vs ~13M for the full version)
    
    Use this for toy datasets with spectral encoding to match original results.
    
    Architecture:
        Input: [Φ̃, Π̃] → Conv + time → Encoder → Middle → Decoder → Output
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
        output_both: bool = False,
    ) -> None:
        """
        Initialize simplified CNN residual network.
        
        Args:
            in_channels: Number of input field channels
            hidden: Base hidden channel count
            depth: Number of encoder/decoder stages
            time_emb_dim: Time embedding dimension
            output_both: If True, output (n_phi, n_pi); else only n_pi
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_both = output_both
        
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )
        
        # Input: concatenate phi_tilde and pi_tilde
        self.input_conv = nn.Conv2d(2 * in_channels, hidden, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = hidden
        for i in range(depth):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.GroupNorm(8, ch * 2),
                nn.SiLU(),
                nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
                nn.GroupNorm(8, ch * 2),
                nn.SiLU(),
            ))
            ch = ch * 2
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                nn.GroupNorm(8, ch // 2),
                nn.SiLU(),
                nn.Conv2d(ch // 2, ch // 2, 3, padding=1),
                nn.GroupNorm(8, ch // 2),
                nn.SiLU(),
            ))
            ch = ch // 2
        
        # Output
        out_channels = 2 * in_channels if output_both else in_channels
        self.output_conv = nn.Conv2d(hidden, out_channels, 3, padding=1)
    
    def forward(
        self, 
        phi_tilde: Tensor, 
        pi_tilde: Tensor, 
        t: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute residual velocity for 2D fields.
        
        Args:
            phi_tilde: (B, C, H, W)
            pi_tilde: (B, C, H, W)
            t: (B,) normalized time
            
        Returns:
            If output_both: (n_phi, n_pi) each (B, C, H, W)
            Else: n_pi of shape (B, C, H, W)
        """
        B = phi_tilde.shape[0]
        
        if t.ndim == 0:
            t = t.expand(B)
        
        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (B, hidden)
        
        # Input
        x = torch.cat([phi_tilde, pi_tilde], dim=1)  # (B, 2C, H, W)
        x = self.input_conv(x)  # (B, hidden, H, W)
        
        # Add time embedding
        x = x + t_emb.view(B, -1, 1, 1)
        
        # Encoder with skip connections
        skips = []
        for down in self.down_blocks:
            skips.append(x)
            x = down(x)
        
        # Middle
        x = self.middle(x)
        
        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x)
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        # Output
        out = self.output_conv(x)
        
        if self.output_both:
            return out[:, :self.in_channels], out[:, self.in_channels:]
        return out


# =============================================================================
# Potential-Based Residual Networks (for Symplectic Integration)
# =============================================================================


@register_network("potential_mlp")
class PotentialResidualMLP(nn.Module):
    """
    MLP residual that outputs a scalar potential U_θ(Φ, r).
    
    For symplectic integration, instead of directly outputting a momentum
    correction, we output a scalar potential. The force is then computed as:
        F = -ω(r) ∂U_θ/∂Φ
    
    This keeps the full dynamics Hamiltonian:
        H_total = H_KG + ΔH_θ,  where ΔH_θ = ∫ ω(r) U_θ(Φ,r) dvol
    
    The momentum equation becomes:
        P' = -∂H_total/∂Φ = P'_KG - ω ∂U_θ/∂Φ
    
    Architecture:
        [Φ, time_emb] → MLP → scalar U_θ (per mode/point)
        Then compute -ω ∂U_θ/∂Φ via autodiff
    """
    
    def __init__(
        self,
        channels: int,
        hidden: int = 256,
        depth: int = 4,
        time_emb_dim: int = 128,
    ) -> None:
        """
        Initialize potential residual MLP.
        
        Args:
            channels: Number of field channels C
            hidden: Hidden layer dimension
            depth: Number of hidden layers  
            time_emb_dim: Time embedding dimension
        """
        super().__init__()
        self.channels = channels
        
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        
        # Input: Φ and time embedding
        # Output: scalar potential per channel
        input_dim = channels + time_emb_dim
        
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        # Output one scalar per channel
        layers.append(nn.Linear(hidden, channels))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, phi: Tensor, t: Tensor) -> Tensor:
        """
        Compute potential U_θ(Φ, t).
        
        Args:
            phi: Field values, shape (B, C)
            t: Normalized time in [0, 1], shape (B,) or scalar
            
        Returns:
            U_theta: Potential values, shape (B, C)
        """
        if t.ndim == 0:
            t = t.expand(phi.shape[0])
        
        t_emb = self.time_embed(t)
        x = torch.cat([phi, t_emb], dim=-1)
        return self.net(x)
    
    def compute_force(self, phi: Tensor, t: Tensor, omega: Tensor) -> Tensor:
        """
        Compute the force F = -ω ∂U_θ/∂Φ via autodiff.
        
        Args:
            phi: Field values, shape (B, C), requires_grad=True
            t: Normalized time, shape (B,) or scalar
            omega: Slice weight ω(r), shape (B,) or scalar
            
        Returns:
            force: -ω ∂U_θ/∂Φ, shape (B, C)
        """
        # Enable gradients even during inference (e.g., sampling)
        with torch.enable_grad():
            phi_for_grad = phi.detach().requires_grad_(True)
            U = self.forward(phi_for_grad, t)
            
            # Sum over batch and channels to get scalar for backward
            U_sum = U.sum()
            
            # Compute gradient (don't create graph during inference)
            grad_U = torch.autograd.grad(U_sum, phi_for_grad, create_graph=self.training)[0]
        
        # Expand omega if needed
        if omega.ndim == 0:
            omega = omega.expand(phi.shape[0])
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)  # (B, 1)
        
        return -omega * grad_U


@register_network("potential_cnn")
class PotentialResidualCNN(nn.Module):
    """
    CNN residual that outputs a scalar potential field U_θ(Φ, r).
    
    For 2D field data, outputs a scalar field that represents the
    learned potential correction. The force is -ω ∂U_θ/∂Φ.
    
    Architecture:
        [Φ] + time_emb → Conv layers → scalar field U_θ(x)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
    ) -> None:
        """
        Initialize potential residual CNN.
        
        Args:
            in_channels: Number of input field channels
            hidden_channels: Hidden channel dimension
            depth: Number of conv blocks
            time_emb_dim: Time embedding dimension
        """
        super().__init__()
        self.in_channels = in_channels
        
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_channels)
        
        # Input convolution (only Φ, not Π)
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Middle blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
            ))
        
        # Output: one scalar per channel
        self.output_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
    
    def forward(self, phi: Tensor, t: Tensor) -> Tensor:
        """
        Compute potential U_θ(Φ, t).
        
        Args:
            phi: Field values, shape (B, C, H, W)
            t: Normalized time, shape (B,) or scalar
            
        Returns:
            U_theta: Potential field, shape (B, C, H, W)
        """
        B = phi.shape[0]
        if t.ndim == 0:
            t = t.expand(B)
        
        # Time embedding
        t_emb = self.time_embed(t)
        t_proj = self.time_proj(t_emb)[:, :, None, None]  # (B, H, 1, 1)
        
        # Process
        x = self.input_conv(phi)
        x = x + t_proj
        
        for block in self.blocks:
            x = x + block(x)
        
        return self.output_conv(x)
    
    def compute_force(self, phi: Tensor, t: Tensor, omega: Tensor) -> Tensor:
        """
        Compute the force F = -ω ∂U_θ/∂Φ via autodiff.
        
        Args:
            phi: Field values, shape (B, C, H, W), requires_grad=True
            t: Normalized time, shape (B,) or scalar
            omega: Slice weight ω(r), shape (B,) or scalar
            
        Returns:
            force: -ω ∂U_θ/∂Φ, shape (B, C, H, W)
        """
        # Enable gradients even during inference (e.g., sampling)
        with torch.enable_grad():
            phi_for_grad = phi.detach().requires_grad_(True)
            U = self.forward(phi_for_grad, t)
            
            # Sum to get scalar
            U_sum = U.sum()
            
            # Compute gradient (don't create graph during inference)
            grad_U = torch.autograd.grad(U_sum, phi_for_grad, create_graph=self.training)[0]
        
        # Expand omega
        if omega.ndim == 0:
            omega = omega.expand(phi.shape[0])
        while omega.ndim < phi.ndim:
            omega = omega.unsqueeze(-1)
        
        return -omega * grad_U


# =============================================================================
# AdS Backbone (Document Section 3-4, eq uv-stable-ode)
# =============================================================================


class AdSBackbone(nn.Module):
    """
    Klein-Gordon backbone in UV-stabilized variables.

    Implements the deterministic part of the velocity field from the
    Klein-Gordon equation in the warped AdS geometry.

    Document eq (uv-stable-ode):
        ∂_r Φ̃ = Π̃
        ∂_r Π̃ = -(d·f'/f - 2Δ)Π̃ - (1/f²)Δ_ĝΦ̃ + dΔ(f'/f - 1)Φ̃

    For planar slicing (f'/f = 1):
        ∂_r Φ̃ = Π̃
        ∂_r Π̃ = -(d - 2Δ)Π̃ - (1/f²)Δ_ĝΦ̃

    For point data (no spatial structure, Δ_ĝ = 0):
        ∂_r Φ̃ = Π̃
        ∂_r Π̃ = (2Δ - d)Π̃

    Attributes
    ----------
    geometry : AdSGeometry
        AdS geometry with warp factor f(r)
    d : int
        Boundary dimension
    n_channels : int
        Number of field channels
    slice_op : Optional[SliceLaplacian]
        Slice Laplacian operator (None for point data)

    Example
    -------
    >>> backbone = AdSBackbone(geometry, d=2, deltas=[1.5, 2.0])
    >>> state = BulkState(phi_tilde=torch.randn(32, 2), pi_tilde=torch.randn(32, 2))
    >>> r = torch.tensor(0.5)
    >>> velocity = backbone(state, r)
    """

    def __init__(
        self,
        geometry: AdSGeometry,
        d: int,
        deltas: Sequence[float],
        slice_op: Optional["SliceLaplacian"] = None,
    ) -> None:
        """
        Initialize AdS backbone.

        Args:
            geometry: AdS geometry object
            d: Boundary dimension
            deltas: Conformal dimensions per channel
            slice_op: Optional slice Laplacian operator
        """
        super().__init__()
        self.geometry = geometry
        self.d = d
        self.n_channels = len(deltas)
        self.slice_op = slice_op

        deltas_t = torch.tensor(deltas, dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)
        self.register_buffer("two_delta", 2.0 * deltas_t)
        self.register_buffer("d_delta", float(d) * deltas_t)

    def forward(self, state: BulkState, r: Tensor) -> BulkState:
        """
        Compute backbone velocity V_KG(S, r).

        Args:
            state: Current bulk state (Φ̃, Π̃)
            r: Current radius

        Returns:
            Backbone velocity (dΦ̃/dr, dΠ̃/dr)
        """
        phi_tilde = state.phi_tilde
        pi_tilde = state.pi_tilde

        # Geometry factors
        f_prime_over_f = self.geometry.log_f_prime(r)
        f_inv_sq = self.geometry.f_inv2(r)

        if not isinstance(f_prime_over_f, Tensor):
            f_prime_over_f = torch.tensor(
                f_prime_over_f, device=phi_tilde.device, dtype=phi_tilde.dtype
            )
        if not isinstance(f_inv_sq, Tensor):
            f_inv_sq = torch.tensor(
                f_inv_sq, device=phi_tilde.device, dtype=phi_tilde.dtype
            )

        # Broadcast to match state shape
        ndim = phi_tilde.ndim
        if f_prime_over_f.ndim == 0:
            pass
        else:
            for _ in range(ndim - 1):
                f_prime_over_f = f_prime_over_f.unsqueeze(-1)
                f_inv_sq = f_inv_sq.unsqueeze(-1)

        # Coefficients
        d_fpf = self.d * f_prime_over_f
        two_delta_b = self.two_delta.view(1, -1, *([1] * (ndim - 2)))
        pi_coeff = two_delta_b - d_fpf

        f_prime_minus_1 = f_prime_over_f - 1.0
        d_delta_b = self.d_delta.view(1, -1, *([1] * (ndim - 2)))
        phi_coeff = d_delta_b * f_prime_minus_1

        # ∂_r Φ̃ = Π̃
        dphi_dr = pi_tilde

        # ∂_r Π̃ = ...
        dpi_dr = pi_coeff * pi_tilde + phi_coeff * phi_tilde

        # Laplacian term
        if self.slice_op is not None:
            lap_phi = self.slice_op.apply_minus_laplacian(phi_tilde)
            dpi_dr = dpi_dr + f_inv_sq * lap_phi

        return BulkState(phi_tilde=dphi_dr, pi_tilde=dpi_dr)


# =============================================================================
# UV Dictionary Lift (Document Section 5.1)
# =============================================================================


class UVDictionaryLift(nn.Module):
    """
    Dictionary-consistent UV lift.

    Lifts boundary data y to UV bulk state using the holographic dictionary.

    Document eq (lift-dict):
        (A, B) = (E_src(y), E_vev(y))
        Φ̃_UV = B + e^{(2Δ-d)r_UV} A
        Π̃_UV = (2Δ-d) e^{(2Δ-d)r_UV} A + ξ

    Document eq (lift-identity) special case (A≡0):
        Φ̃_UV = E(y)
        Π̃_UV = ξ

    Attributes
    ----------
    geometry : AdSGeometry
        AdS geometry
    bulk_field : BulkField
        Bulk field specification
    noise_sigma : float
        Standard deviation for momentum noise ξ

    Example
    -------
    >>> lift = UVDictionaryLift(geometry, bulk_field, disc, noise_sigma=0.1)
    >>> y = torch.randn(32, 2)
    >>> r_uv = torch.tensor(1.0)
    >>> state = lift(y, r_uv)
    """

    def __init__(
        self,
        geometry: AdSGeometry,
        bulk_field: BulkField,
        disc: DiscretizationConfig,
        embed_src: Optional[Callable[[Tensor], Tensor]] = None,
        embed_vev: Optional[Callable[[Tensor], Tensor]] = None,
        noise_sigma: float = 0.1,
    ) -> None:
        """
        Initialize UV lift.

        Args:
            geometry: AdS geometry
            bulk_field: Bulk field specification
            disc: Discretization configuration
            embed_src: Optional source embedding E_src (A coefficient)
            embed_vev: Optional VEV embedding E_vev (B coefficient)
            noise_sigma: Standard deviation for momentum noise
        """
        super().__init__()
        self.geometry = geometry
        self.bulk_field = bulk_field
        self.disc = disc
        self.embed_src = embed_src
        self.embed_vev = embed_vev
        self.noise_sigma = noise_sigma

    def forward(
        self,
        y: Tensor,
        r_uv: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> BulkState:
        """
        Lift data y to UV bulk state.

        Args:
            y: Boundary data, shape (B, *data_shape)
            r_uv: UV radius
            generator: Optional random generator for noise

        Returns:
            BulkState at UV boundary
        """
        B = y.shape[0]
        device = y.device
        dtype = y.dtype

        # Compute coefficient embeddings
        if self.embed_vev is not None:
            B_coeff = self.embed_vev(y)
        else:
            B_coeff = y

        if self.embed_src is not None:
            A_coeff = self.embed_src(y)
        else:
            A_coeff = None

        # Build Φ̃_UV
        if A_coeff is not None:
            # Full dictionary: Φ̃ = B + e^{α r} A where α = 2Δ - d
            alpha = self.bulk_field.alpha.view(1, -1, *([1] * (B_coeff.ndim - 2)))
            exp_factor = torch.exp(alpha * r_uv)
            phi_tilde = B_coeff + exp_factor * A_coeff

            # Π̃ = α e^{α r} A + ξ
            pi_deterministic = alpha * exp_factor * A_coeff
        else:
            # Minimal lift: Φ̃ = E(y), Π̃ = ξ
            phi_tilde = B_coeff
            pi_deterministic = torch.zeros_like(phi_tilde)

        # Sample noise ξ
        if generator is not None:
            # Use randn with explicit args for generator support
            pi_noise = self.noise_sigma * torch.randn(
                phi_tilde.shape, device=phi_tilde.device, dtype=phi_tilde.dtype, generator=generator
            )
        else:
            pi_noise = self.noise_sigma * torch.randn_like(phi_tilde)
        pi_tilde = pi_deterministic + pi_noise

        return BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)

    def encode_field(self, y: Tensor) -> Tensor:
        """
        Return field embedding without full lift.

        Args:
            y: Boundary data

        Returns:
            Field embedding
        """
        if self.embed_vev is not None:
            return self.embed_vev(y)
        return y

    def sample_pi_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Sample momentum noise ξ.

        Args:
            shape: Output shape
            device: Target device
            dtype: Target dtype
            generator: Optional random generator

        Returns:
            Sampled noise
        """
        return self.noise_sigma * torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )


# =============================================================================
# UV Readout (Document Section 5.2)
# =============================================================================


class UVReadout(nn.Module):
    """
    UV coefficient extraction.

    Extracts boundary data from UV bulk state.

    Document Section 5.2:
        - "direct": y = Φ̃(r_UV)
        - "coefficient": two-term fit for A, B coefficients

    Attributes
    ----------
    d : int
        Boundary dimension
    mode : str
        Readout mode: 'direct' or 'coefficient'

    Example
    -------
    >>> readout = UVReadout(d=2, deltas=[1.5, 2.0], mode="direct")
    >>> y = readout(state)
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        mode: Literal["direct", "coefficient"] = "direct",
    ) -> None:
        """
        Initialize UV readout.

        Args:
            d: Boundary dimension
            deltas: Conformal dimensions per channel
            mode: Readout mode
        """
        super().__init__()
        self.d = d
        self.mode = mode

        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)
        self.register_buffer("alpha", 2.0 * deltas_t - float(d))

    def forward(
        self, state: BulkState, r_uv: Optional[Tensor] = None
    ) -> Tensor:
        """
        Extract boundary data from UV state.

        Args:
            state: UV bulk state
            r_uv: Optional UV radius (for coefficient mode)

        Returns:
            Extracted boundary data
        """
        if self.mode == "direct":
            return state.phi_tilde
        else:
            # Would implement two-term fit here
            raise NotImplementedError("Coefficient extraction not implemented")


# =============================================================================
# UV Coefficient Operator (Document Section 5.2)
# =============================================================================


class UVCoefficientOperator(nn.Module):
    """
    Extract (A, B) coefficients from UV slices.

    Fits the asymptotic expansion:
        Φ̃(r, x) ≈ B(x) + A(x) e^{(2Δ-d)r}

    to extract the source (A) and VEV (B) coefficients.

    Attributes
    ----------
    d : int
        Boundary dimension
    alpha : Tensor
        Asymptotic exponents 2Δ - d
    """

    def __init__(self, d: int, deltas: Sequence[float]) -> None:
        """
        Initialize coefficient operator.

        Args:
            d: Boundary dimension
            deltas: Conformal dimensions per channel
        """
        super().__init__()
        self.d = d
        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)
        self.register_buffer("alpha", 2.0 * deltas_t - float(d))

    def forward(
        self,
        phi_tilde_samples: Tensor,
        r_samples: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Fit coefficients from multiple radius samples.

        Args:
            phi_tilde_samples: (B, C, M, *spatial) where M is number of r samples
            r_samples: (M,) radius values

        Returns:
            (A, B): source and VEV coefficients

        Raises:
            NotImplementedError: Full coefficient extraction not yet implemented
        """
        raise NotImplementedError("Full coefficient extraction not implemented")


# =============================================================================
# Hermite Path Helpers (Document Section 9)
# NOTE: These helpers are duplicated in paths.py to avoid circular imports.
# Both implementations are identical. paths.py is the canonical source.
# =============================================================================


def _hermite_basis(u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Cubic Hermite basis polynomials on [0, 1].

    These are the standard Hermite interpolation basis functions:
        h00(u) = 2u³ - 3u² + 1
        h10(u) = u³ - 2u² + u
        h01(u) = -2u³ + 3u²
        h11(u) = u³ - u²

    Args:
        u: Parameter in [0, 1]

    Returns:
        Tuple of (h00, h10, h01, h11)
    """
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00, h10, h01, h11


def _hermite_basis_deriv(u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    First derivative of cubic Hermite basis.

    Args:
        u: Parameter in [0, 1]

    Returns:
        Tuple of (dh00/du, dh10/du, dh01/du, dh11/du)
    """
    dh00 = 6 * u ** 2 - 6 * u
    dh10 = 3 * u ** 2 - 4 * u + 1
    dh01 = -6 * u ** 2 + 6 * u
    dh11 = 3 * u ** 2 - 2 * u
    return dh00, dh10, dh01, dh11


def _hermite_basis_second_deriv(u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Second derivative of cubic Hermite basis.

    Args:
        u: Parameter in [0, 1]

    Returns:
        Tuple of (d²h00/du², d²h10/du², d²h01/du², d²h11/du²)
    """
    d2h00 = 12 * u - 6
    d2h10 = 6 * u - 4
    d2h01 = -12 * u + 6
    d2h11 = 6 * u - 2
    return d2h00, d2h10, d2h01, d2h11


def _broadcast_time(t: Tensor, like: Tensor) -> Tensor:
    """
    Broadcast time t across spatial/channel dims of like.

    Args:
        t: Time tensor (scalar or (B,))
        like: Target tensor to match shape

    Returns:
        Broadcasted time tensor
    """
    if t.ndim == 0:
        t = t.expand(like.shape[0])
    while t.ndim < like.ndim:
        t = t.unsqueeze(-1)
    return t


# =============================================================================
# AdS-Affine Hermite Coupling (Document Section 9)
# NOTE: This class is equivalent to AdSAffineHermitePath in paths.py.
# Both implementations exist to avoid circular imports (paths.py imports BulkState
# from this module). For new code, prefer using paths.py directly.
# =============================================================================


class AdSAffineHermiteCoupling(nn.Module):
    """
    AdS-affine cubic Hermite path in UV-stable variables.

    Implements the geometric path interpolation from Document eq (rectified-path).

    NOTE: This class is equivalent to AdSAffineHermitePath in paths.py.

    The key idea is to use the affine parameter:
        u(r) = (1/Z) ∫_{r_IR}^r 1/ω(s) ds

    for cubic Hermite interpolation between endpoints. This naturally accounts
    for the AdS geometry.

    The induced Π̃(t) and target velocity are computed via chain rule
    as specified in eq (target-velocity).

    Attributes
    ----------
    geometry : AdSGeometry
        AdS geometry with warp factor
    r_ir : float
        IR boundary radius
    r_uv : float
        UV boundary radius

    Example
    -------
    >>> coupling = AdSAffineHermiteCoupling(geometry, r_ir=0.0, r_uv=1.0)
    >>> state_t, target_vel = coupling.state_and_target_velocity(S0, S1, t)
    """

    def __init__(
        self, geometry: AdSGeometry, r_ir: float, r_uv: float,
        use_radial_hermite: bool = False
    ) -> None:
        """
        Initialize Hermite coupling.

        Args:
            geometry: AdS geometry
            r_ir: IR boundary radius
            r_uv: UV boundary radius
            use_radial_hermite: If True, use radial Hermite (u=t, no ω factors).
                Auto-enabled for HSV geometries. Default False for standard AdS.
        """
        super().__init__()
        self.geometry = geometry
        self.r_ir = float(r_ir)
        self.r_uv = float(r_uv)
        self.use_radial_hermite = use_radial_hermite

    @property
    def r_span(self) -> float:
        """Return radial span r_UV - r_IR."""
        return self.r_uv - self.r_ir

    def state_and_target_velocity(
        self, state0: BulkState, state1: BulkState, t: Tensor
    ) -> Tuple[BulkState, BulkState]:
        """
        Compute interpolated state and target velocity at time t.

        Document eq (rectified-path) and (target-velocity).
        
        For HSV geometries, uses "radial Hermite" (u=t, dr/du=r_span) instead
        of affine-parameter Hermite to avoid numerical instabilities from
        exploding/vanishing ω factors.

        Args:
            state0: IR state (Φ̃_0, Π̃_0)
            state1: UV state (Φ̃_1, Π̃_1)
            t: Normalized time in [0, 1]

        Returns:
            (state_t, target_velocity_t)
        """
        device = state0.phi_tilde.device
        dtype = state0.phi_tilde.dtype
        t = torch.as_tensor(t, device=device, dtype=dtype)

        # Physical radius r(t) = r_IR + t * (r_UV - r_IR)
        t_b = _broadcast_time(t, state0.phi_tilde)
        r_ir_t = torch.tensor(self.r_ir, device=device, dtype=dtype)
        r_uv_t = torch.tensor(self.r_uv, device=device, dtype=dtype)
        r_t = r_ir_t + self.r_span * t_b
        rspan = torch.tensor(self.r_span, device=device, dtype=dtype)

        # Check if HSV geometry or use_radial_hermite - use radial Hermite instead of affine Hermite
        is_hsv = getattr(self.geometry, 'is_hsv', False)
        use_radial = is_hsv or self.use_radial_hermite

        if use_radial:
            # =================================================================
            # HSV: RADIAL HERMITE PATH
            # =================================================================
            # For HSV, ω(r) has inverted behavior (decreases toward UV),
            # making affine-parameter formulation numerically unstable.
            # Instead, use t directly as the Hermite parameter:
            #   - u = t (normalized radius IS the parameter)
            #   - dr/du = r_span (constant)
            #   - m₀ = r_span * Π̃₀, m₁ = r_span * Π̃₁
            # This removes all ω factors and stays stable for any p.
            # =================================================================
            
            u_b = t_b  # u = t for radial Hermite
            
            # Constant tangent scaling (no ω factors!)
            m0 = state0.pi_tilde * rspan
            m1 = state1.pi_tilde * rspan

            # Hermite interpolation in t-space
            h00, h10, h01, h11 = _hermite_basis(u_b)
            phi_t = (
                h00 * state0.phi_tilde
                + h10 * m0
                + h01 * state1.phi_tilde
                + h11 * m1
            )

            # First derivative: dΦ̃/dt
            dh00, dh10, dh01, dh11 = _hermite_basis_deriv(u_b)
            phi_dot = (
                dh00 * state0.phi_tilde
                + dh10 * m0
                + dh01 * state1.phi_tilde
                + dh11 * m1
            )

            # Second derivative: d²Φ̃/dt²
            d2h00, d2h10, d2h01, d2h11 = _hermite_basis_second_deriv(u_b)
            phi_ddot = (
                d2h00 * state0.phi_tilde
                + d2h10 * m0
                + d2h01 * state1.phi_tilde
                + d2h11 * m1
            )

            # Π̃_t = dΦ̃/dr = (dΦ̃/dt) / (dr/dt) = φ̇ / r_span
            pi_t = phi_dot / rspan

            # Target velocities
            # dΦ̃/dt = φ̇ (direct from Hermite)
            dphi_dt = phi_dot

            # dΠ̃/dt = d(φ̇/r_span)/dt = φ̈ / r_span
            dpi_dt = phi_ddot / rspan

        else:
            # =================================================================
            # STANDARD AdS: AFFINE HERMITE PATH
            # =================================================================
            # Affine parameter u(r_t) ∈ [0, 1]
            u = self.geometry.affine_parameter(r_t, self.r_ir, self.r_uv)
            if hasattr(u, "to"):
                u = u.to(device=device, dtype=dtype)
            elif not isinstance(u, Tensor):
                u = torch.tensor(u, device=device, dtype=dtype)
            u_b = _broadcast_time(u, state0.phi_tilde)

            # Endpoint tangents: dΦ̃/du = (dr/du) Π̃ = Z ω(r) Π̃
            drdu_ir = self.geometry.dr_du(r_ir_t, self.r_ir, self.r_uv)
            drdu_uv = self.geometry.dr_du(r_uv_t, self.r_ir, self.r_uv)

            # Ensure dr_du results are on correct device
            if hasattr(drdu_ir, "to"):
                drdu_ir = drdu_ir.to(device=device, dtype=dtype)
            else:
                drdu_ir = torch.tensor(drdu_ir, device=device, dtype=dtype)
            if hasattr(drdu_uv, "to"):
                drdu_uv = drdu_uv.to(device=device, dtype=dtype)
            else:
                drdu_uv = torch.tensor(drdu_uv, device=device, dtype=dtype)

            m0 = state0.pi_tilde * drdu_ir  # dΦ̃/du at u=0
            m1 = state1.pi_tilde * drdu_uv  # dΦ̃/du at u=1

            # Hermite interpolation in u
            h00, h10, h01, h11 = _hermite_basis(u_b)
            phi_t = (
                h00 * state0.phi_tilde
                + h10 * m0
                + h01 * state1.phi_tilde
                + h11 * m1
            )

            # First u-derivative
            dh00, dh10, dh01, dh11 = _hermite_basis_deriv(u_b)
            phi_u = (
                dh00 * state0.phi_tilde
                + dh10 * m0
                + dh01 * state1.phi_tilde
                + dh11 * m1
            )

            # Second u-derivative
            d2h00, d2h10, d2h01, d2h11 = _hermite_basis_second_deriv(u_b)
            phi_uu = (
                d2h00 * state0.phi_tilde
                + d2h10 * m0
                + d2h01 * state1.phi_tilde
                + d2h11 * m1
            )

            # Geometry factors
            omega = self.geometry.slice_weight(r_t)
            omega_prime = self.geometry.slice_weight_prime(r_t)
            Z = self.geometry.affine_norm_const(
                self.r_ir, self.r_uv, device=device, dtype=dtype
            )

            # Ensure geometry factors are on correct device
            if hasattr(omega, "to"):
                omega = omega.to(device=device, dtype=dtype)
            elif not isinstance(omega, Tensor):
                omega = torch.tensor(omega, device=device, dtype=dtype)
            if hasattr(omega_prime, "to"):
                omega_prime = omega_prime.to(device=device, dtype=dtype)
            elif not isinstance(omega_prime, Tensor):
                omega_prime = torch.tensor(omega_prime, device=device, dtype=dtype)

            # Broadcast scalars
            omega = _broadcast_time(omega, state0.phi_tilde)
            omega_prime = _broadcast_time(omega_prime, state0.phi_tilde)

            # Π̃_t = ∂_r Φ̃ = (du/dr) φ_u = φ_u / (Z ω)
            pi_t = phi_u / (Z * omega)

            # Target velocity in t-time (Document eq target-velocity)
            # dΦ̃/dt = (r_UV - r_IR) Π̃_t
            dphi_dt = rspan * pi_t

            # dΠ̃/dt = (r_UV - r_IR) [φ̈/(Z²ω²) - ω' φ̇/(Z ω²)]
            dpi_dt = rspan * (
                phi_uu / (Z * Z * omega * omega)
                - omega_prime * phi_u / (Z * omega * omega)
            )

        state_t = BulkState(phi_tilde=phi_t, pi_tilde=pi_t)
        target_vel = BulkState(phi_tilde=dphi_dt, pi_tilde=dpi_dt)

        return state_t, target_vel

    def forward(
        self, state0: BulkState, state1: BulkState, t: Tensor
    ) -> BulkState:
        """
        Return interpolated state at time t.

        Args:
            state0: IR state
            state1: UV state
            t: Normalized time

        Returns:
            Interpolated state
        """
        state_t, _ = self.state_and_target_velocity(state0, state1, t)
        return state_t


# =============================================================================
# Factory Functions
# =============================================================================


def build_slice_laplacian(
    geometry: AdSGeometry,
    disc: DiscretizationConfig,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Optional["SliceLaplacian"]:
    """
    Build appropriate slice Laplacian for geometry and discretization.

    Args:
        geometry: AdS geometry
        disc: Discretization configuration
        device: Target device
        dtype: Target dtype

    Returns:
        SliceLaplacian instance or None if not applicable
    """
    # Import here to avoid circular imports
    from ads_cft.laplacian_base import (
        SliceLaplacian,
        PlanarSpectralLaplacian,
    )

    if geometry.slice_geometry == SliceGeometry.PLANAR:
        if disc.grid_shape is not None and disc.box_lengths is not None:
            return PlanarSpectralLaplacian(
                grid_shape=disc.grid_shape,
                box_lengths=disc.box_lengths,
                device=device,
                dtype=dtype,
            )

    elif geometry.slice_geometry == SliceGeometry.FLAT:
        # ABLATION: Same slice Laplacian as planar (flat ℝ^d)
        if disc.grid_shape is not None and disc.box_lengths is not None:
            return PlanarSpectralLaplacian(
                grid_shape=disc.grid_shape,
                box_lengths=disc.box_lengths,
                device=device,
                dtype=dtype,
            )

    # No Laplacian for point data or unsupported configs
    return None


def slice_quadrature_weights(
    geometry: AdSGeometry,
    disc: DiscretizationConfig,
    r: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Optional[Tensor]:
    """
    Compute quadrature weights for L² inner product at radius r.

    The slice measure is dvol_{g_r} = ω(r) dvol_ĝ where ω(r) = f(r)^d.

    Args:
        geometry: AdS geometry
        disc: Discretization configuration
        r: Radius value
        device: Target device
        dtype: Target dtype

    Returns:
        Quadrature weights tensor or None if not applicable
    """
    omega = geometry.slice_weight(torch.tensor(r, device=device, dtype=dtype))

    if geometry.slice_geometry == SliceGeometry.PLANAR:
        if disc.grid_shape is not None and disc.box_lengths is not None:
            # Uniform weights: dx₁ × ... × dx_d × ω(r)
            vol_per_cell = 1.0
            for n, L in zip(disc.grid_shape, disc.box_lengths):
                vol_per_cell *= L / n
            return (
                omega
                * vol_per_cell
                * torch.ones(disc.grid_shape, device=device, dtype=dtype)
            )

    elif geometry.slice_geometry == SliceGeometry.FLAT:
        # ABLATION: Same quadrature as planar, but ω(r)=1 for flat
        if disc.grid_shape is not None and disc.box_lengths is not None:
            vol_per_cell = 1.0
            for n, L in zip(disc.grid_shape, disc.box_lengths):
                vol_per_cell *= L / n
            return (
                omega
                * vol_per_cell
                * torch.ones(disc.grid_shape, device=device, dtype=dtype)
            )

    # For other geometries, would need specific quadrature rules
    return None


def sample_isotropic_noise_from_quadrature(
    weights: Tensor,
    shape: Tuple[int, ...],
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Sample L²-isotropic Gaussian noise given quadrature weights.

    The noise is scaled so that ||ξ||²_{L²} has the desired variance.

    Formula: ξ_p = σ η_p / √w_p where η ~ N(0,1)

    Args:
        weights: Quadrature weights for L² inner product
        shape: Output shape (B, C, *spatial)
        sigma: Target L² standard deviation
        generator: Optional random generator

    Returns:
        L²-isotropic noise samples
    """
    eta = torch.randn(
        shape, device=weights.device, dtype=weights.dtype, generator=generator
    )
    # Broadcast weights to match shape
    w = weights
    while w.ndim < eta.ndim:
        w = w.unsqueeze(0)
    return sigma * eta / torch.sqrt(w + 1e-10)


# =============================================================================
# State Arithmetic Helpers
# =============================================================================


def state_add(a: BulkState, b: BulkState) -> BulkState:
    """
    Element-wise addition of BulkStates.

    Args:
        a: First state
        b: Second state

    Returns:
        Sum state
    """
    return BulkState(
        phi_tilde=a.phi_tilde + b.phi_tilde,
        pi_tilde=a.pi_tilde + b.pi_tilde,
    )


def state_scale(a: BulkState, s: Union[float, Tensor]) -> BulkState:
    """
    Scalar multiplication of BulkState.

    Args:
        a: State to scale
        s: Scalar or tensor multiplier

    Returns:
        Scaled state
    """
    return BulkState(
        phi_tilde=s * a.phi_tilde,
        pi_tilde=s * a.pi_tilde,
    )


def state_sub(a: BulkState, b: BulkState) -> BulkState:
    """
    Element-wise subtraction of BulkStates.

    Args:
        a: First state
        b: Second state

    Returns:
        Difference state
    """
    return BulkState(
        phi_tilde=a.phi_tilde - b.phi_tilde,
        pi_tilde=a.pi_tilde - b.pi_tilde,
    )


def state_lerp(
    a: BulkState, b: BulkState, t: Union[float, Tensor]
) -> BulkState:
    """
    Linear interpolation: (1-t)*a + t*b.

    Args:
        a: State at t=0
        b: State at t=1
        t: Interpolation parameter

    Returns:
        Interpolated state
    """
    if isinstance(t, Tensor):
        while t.ndim < a.phi_tilde.ndim:
            t = t.unsqueeze(-1)
    return BulkState(
        phi_tilde=(1 - t) * a.phi_tilde + t * b.phi_tilde,
        pi_tilde=(1 - t) * a.pi_tilde + t * b.pi_tilde,
    )

# =============================================================================
# Canonical State Conversion and Arithmetic (for Symplectic Integration)
# =============================================================================


def bulk_to_canonical(
    state: BulkState,
    r: Tensor,
    delta: Tensor,
    omega: Tensor,
    geometry=None,
) -> CanonicalState:
    """
    Convert from UV-stabilized (Φ̃, Π̃) to canonical (Φ, P).
    
    The UV-stabilization relations are geometry-dependent:
        Φ̃ = f(r)^Δ Φ           →  Φ = f(r)^{-Δ} Φ̃
        Π̃ = f(r)^Δ (Π + Δ(f'/f)Φ)  →  Π = f(r)^{-Δ} Π̃ - Δ(f'/f)Φ
        P = ω(r) Π
    
    For standard AdS: f(r) = e^r, so f^Δ = e^{Δr} and f'/f = 1.
    For HSV: f(r) = [(1-p)r]^{-p/(1-p)}, with geometry-specific f'/f.
    
    Args:
        state: UV-stabilized state (Φ̃, Π̃)
        r: Radial coordinate, shape (B,) or scalar
        delta: Conformal dimensions, shape (C,) or scalar
        omega: Slice weight ω(r), shape (B,) or scalar
        geometry: Optional AdS geometry object. If None, uses AdS formulas.
        
    Returns:
        Canonical state (Φ, P)
    """
    device = state.phi_tilde.device
    dtype = state.phi_tilde.dtype
    
    # Convert to tensors
    if isinstance(r, (int, float)):
        r = torch.tensor(r, device=device, dtype=dtype)
    if isinstance(delta, (int, float)):
        delta = torch.tensor(delta, device=device, dtype=dtype)
    if isinstance(omega, (int, float)):
        omega = torch.tensor(omega, device=device, dtype=dtype)
    
    # Keep scalar r for geometry calls
    r_scalar = r.clone()
    
    # Reshape for broadcasting: (B, C, ...) or (B, C)
    while r.ndim < state.phi_tilde.ndim:
        r = r.unsqueeze(-1)
    while delta.ndim < state.phi_tilde.ndim:
        delta = delta.unsqueeze(0)
        if delta.ndim < state.phi_tilde.ndim:
            delta = delta.unsqueeze(-1) if state.phi_tilde.ndim > 2 else delta
    while omega.ndim < state.phi_tilde.ndim:
        omega = omega.unsqueeze(-1)
    
    # Compute geometry-dependent factors
    if geometry is not None and getattr(geometry, 'is_hsv', False):
        # HSV geometry: use f(r)^{-Δ} and f'/f from geometry
        f_val = geometry.f(r_scalar)
        if not isinstance(f_val, Tensor):
            f_val = torch.tensor(f_val, device=device, dtype=dtype)
        f_prime_over_f = geometry.log_f_prime(r_scalar)
        if not isinstance(f_prime_over_f, Tensor):
            f_prime_over_f = torch.tensor(f_prime_over_f, device=device, dtype=dtype)
        
        # Broadcast f and f'/f
        while f_val.ndim < state.phi_tilde.ndim:
            f_val = f_val.unsqueeze(-1)
        while f_prime_over_f.ndim < state.phi_tilde.ndim:
            f_prime_over_f = f_prime_over_f.unsqueeze(-1)
        
        # f^{-Δ} = 1 / f^Δ
        f_safe = torch.clamp(f_val.abs(), min=1e-10)
        inv_stab_factor = f_safe ** (-delta)
        
        # Φ = f^{-Δ} Φ̃
        phi = inv_stab_factor * state.phi_tilde
        
        # Π = f^{-Δ} Π̃ - Δ(f'/f)Φ
        pi = inv_stab_factor * state.pi_tilde - delta * f_prime_over_f * phi
    else:
        # Standard AdS: f(r) = e^r, f'/f = 1
        # f^{-Δ} = e^{-Δr}
        inv_stab_factor = torch.exp(-delta * r)
        
        # Φ = e^{-Δr} Φ̃
        phi = inv_stab_factor * state.phi_tilde
        
        # Π = e^{-Δr} Π̃ - Δ·Φ  (since f'/f = 1 for AdS)
        pi = inv_stab_factor * state.pi_tilde - delta * phi
    
    # P = ω Π
    P = omega * pi
    
    return CanonicalState(phi=phi, P=P)


def canonical_to_bulk(
    state: CanonicalState,
    r: Tensor,
    delta: Tensor,
    omega: Tensor,
    geometry=None,
) -> BulkState:
    """
    Convert from canonical (Φ, P) to UV-stabilized (Φ̃, Π̃).
    
    The UV-stabilization relations are geometry-dependent:
        Π = P / ω(r)
        Φ̃ = f(r)^Δ Φ
        Π̃ = f(r)^Δ (Π + Δ(f'/f)Φ)
    
    For standard AdS: f(r) = e^r, so f^Δ = e^{Δr} and f'/f = 1.
    For HSV: f(r) = [(1-p)r]^{-p/(1-p)}, with geometry-specific f'/f.
    
    Args:
        state: Canonical state (Φ, P)
        r: Radial coordinate, shape (B,) or scalar
        delta: Conformal dimensions, shape (C,) or scalar
        omega: Slice weight ω(r), shape (B,) or scalar
        geometry: Optional AdS geometry object. If None, uses AdS formulas.
        
    Returns:
        UV-stabilized state (Φ̃, Π̃)
    """
    device = state.phi.device
    dtype = state.phi.dtype
    
    # Convert to tensors
    if isinstance(r, (int, float)):
        r = torch.tensor(r, device=device, dtype=dtype)
    if isinstance(delta, (int, float)):
        delta = torch.tensor(delta, device=device, dtype=dtype)
    if isinstance(omega, (int, float)):
        omega = torch.tensor(omega, device=device, dtype=dtype)
    
    # Keep scalar r for geometry calls
    r_scalar = r.clone()
    
    # Reshape for broadcasting
    while r.ndim < state.phi.ndim:
        r = r.unsqueeze(-1)
    while delta.ndim < state.phi.ndim:
        delta = delta.unsqueeze(0)
        if delta.ndim < state.phi.ndim:
            delta = delta.unsqueeze(-1) if state.phi.ndim > 2 else delta
    while omega.ndim < state.phi.ndim:
        omega = omega.unsqueeze(-1)
    
    # Π = P / ω
    omega_safe = torch.clamp(omega, min=1e-10)
    pi = state.P / omega_safe
    
    # Compute geometry-dependent factors
    if geometry is not None and getattr(geometry, 'is_hsv', False):
        # HSV geometry: use f(r)^Δ and f'/f from geometry
        f_val = geometry.f(r_scalar)
        if not isinstance(f_val, Tensor):
            f_val = torch.tensor(f_val, device=device, dtype=dtype)
        f_prime_over_f = geometry.log_f_prime(r_scalar)
        if not isinstance(f_prime_over_f, Tensor):
            f_prime_over_f = torch.tensor(f_prime_over_f, device=device, dtype=dtype)
        
        # Broadcast f and f'/f
        while f_val.ndim < state.phi.ndim:
            f_val = f_val.unsqueeze(-1)
        while f_prime_over_f.ndim < state.phi.ndim:
            f_prime_over_f = f_prime_over_f.unsqueeze(-1)
        
        # f^Δ
        f_safe = torch.clamp(f_val.abs(), min=1e-10)
        stab_factor = f_safe ** delta
        
        # Φ̃ = f^Δ Φ
        phi_tilde = stab_factor * state.phi
        
        # Π̃ = f^Δ (Π + Δ(f'/f)Φ)
        pi_tilde = stab_factor * (pi + delta * f_prime_over_f * state.phi)
    else:
        # Standard AdS: f(r) = e^r, f'/f = 1
        # f^Δ = e^{Δr}
        stab_factor = torch.exp(delta * r)
        
        # Φ̃ = e^{Δr} Φ
        phi_tilde = stab_factor * state.phi
        
        # Π̃ = e^{Δr}(Π + Δ·Φ)  (since f'/f = 1 for AdS)
        pi_tilde = stab_factor * (pi + delta * state.phi)
    
    return BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)


def canonical_add(a: CanonicalState, b: CanonicalState) -> CanonicalState:
    """Element-wise addition of CanonicalStates."""
    return CanonicalState(phi=a.phi + b.phi, P=a.P + b.P)


def canonical_scale(a: CanonicalState, s: Union[float, Tensor]) -> CanonicalState:
    """Scalar multiplication of CanonicalState."""
    return CanonicalState(phi=s * a.phi, P=s * a.P)


def canonical_sub(a: CanonicalState, b: CanonicalState) -> CanonicalState:
    """Element-wise subtraction of CanonicalStates."""
    return CanonicalState(phi=a.phi - b.phi, P=a.P - b.P)