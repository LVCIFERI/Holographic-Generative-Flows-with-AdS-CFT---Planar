"""
encoding_image.py

Image Encoding and Dataset Support for AdS/CFT Flow Matching
=============================================================

This module provides image-specific encoding that treats images as boundary
SOURCES in the CFT, consistent with the AdS/CFT dictionary. The image is
interpreted as the source J(x), and the bulk field φ is obtained by
convolving with the bulk-boundary propagator K.

Key Concepts (Source-Based Encoding)
------------------------------------
- Images are boundary SOURCES J(x, y), not fields directly
- FFT gives the source spectral representation Ĵ(kx, ky)
- Bulk field: φ̂(k) = K̂(r_UV, k) * Ĵ(k) (propagator times source)
- Momentum: π̂(k) = ∂_r K̂(r_UV, k) * Ĵ(k)
- Laplacian is diagonal: Δφ̂_k = -|k|²φ̂_k

This is consistent with the AdS/CFT dictionary (Section 3 of the paper):
    Φ(r,x) = ∫ K(r,x;x') J(x') d^d x'

In Fourier space:
    φ̂_k(r) = K̂(r, k) × Ĵ_k

The bulk-boundary propagator for planar AdS is:
    K̂_Δ(r, k) ∝ (|k|z)^ν K_ν(|k|z)

where ν = Δ - d/2, z = e^{-r}, and K_ν is the modified Bessel function.

Supported Datasets
------------------
- MNIST: 28×28 grayscale, 10 classes
- FashionMNIST: 28×28 grayscale, 10 classes
- CIFAR10: 32×32 RGB, 10 classes
- CIFAR100: 32×32 RGB, 100 classes

Image Encoding Pipeline (Source-Based)
--------------------------------------
1. Image (source J) → FFT → Ĵ(k) source spectral coefficients
2. Ĵ(k) * K̂(r_UV, k) → φ̂(k) field coefficients (propagator lift)
3. Ĵ(k) * ∂_r K̂(r_UV, k) → π̂(k) momentum coefficients
4. Laplacian is diagonal: -Δφ̂ = |k|²φ̂
5. Inverse: φ̂(k) / K̂(r_UV, k) → Ĵ(k) → IFFT → reconstructed image

Document References
-------------------
- Section 3: Bulk-boundary propagator and spectral representation
- Section 5: UV stabilization for image data
- Appendix B: Image dataset preprocessing
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from ads_cft.registry import register_encoder
from ads_cft.encoding_base import _bessel_k_nu

Tensor = torch.Tensor


# =============================================================================
# Dataset Registry
# =============================================================================


IMAGE_DATASETS: Dict[str, str] = {
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
}


def is_image_dataset(name: str) -> bool:
    """
    Check if a dataset name corresponds to an image dataset.

    Args:
        name: Dataset name

    Returns:
        True if name is a known image dataset
    """
    return name.lower() in IMAGE_DATASETS


# =============================================================================
# Image Dataset Wrapper
# =============================================================================


class ImageDatasetWrapper(Dataset):
    """
    Wrapper for image datasets that prepares them for AdS/CFT flow matching.

    Key features:
    - Normalizes images to [-1, 1] or [0, 1] range
    - Optionally flattens for compatibility checks
    - Provides image shape and channel info

    Attributes
    ----------
    dataset : Dataset
        Base torchvision dataset
    normalize_range : Tuple[float, float]
        Target range for pixel values
    flatten : bool
        If True, flatten images to 1D
    image_shape : Tuple[int, int, int]
        (C, H, W) image dimensions
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width

    Example
    -------
    >>> dataset = load_mnist()
    >>> image, label = dataset[0]
    >>> print(image.shape)  # (1, 28, 28)
    """

    def __init__(
        self,
        dataset: Dataset,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        flatten: bool = False,
    ) -> None:
        """
        Initialize image dataset wrapper.

        Args:
            dataset: Base torchvision dataset
            normalize_range: Target range for pixel values
            flatten: If True, flatten images to 1D (for debugging)
        """
        self.dataset = dataset
        self.normalize_range = normalize_range
        self.flatten = flatten

        # Get a sample to determine shape
        sample, _ = dataset[0]
        if isinstance(sample, Tensor):
            self.image_shape = tuple(sample.shape)
        else:
            # PIL Image
            sample = transforms.ToTensor()(sample)
            self.image_shape = tuple(sample.shape)

        self.n_channels = self.image_shape[0]
        self.height = self.image_shape[1]
        self.width = self.image_shape[2]

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Get a single item.

        Args:
            idx: Item index

        Returns:
            Tuple of (image_tensor, label)
        """
        image, label = self.dataset[idx]

        # Convert to tensor if needed
        if not isinstance(image, Tensor):
            image = transforms.ToTensor()(image)

        # Normalize to target range
        # Input is [0, 1] from ToTensor
        low, high = self.normalize_range
        image = image * (high - low) + low

        if self.flatten:
            image = image.flatten()

        return image, label

    def get_data_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of a single data sample.

        Returns:
            (C, H, W) or (C*H*W,) if flattened
        """
        if self.flatten:
            return (self.n_channels * self.height * self.width,)
        return self.image_shape

    def get_image_shape(self) -> Tuple[int, int, int]:
        """
        Return (C, H, W) regardless of flatten setting.

        Returns:
            Image shape tuple
        """
        return self.image_shape


# =============================================================================
# Dataset Loaders
# =============================================================================


def load_mnist(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    flatten: bool = False,
) -> ImageDatasetWrapper:
    """
    Load MNIST dataset.

    MNIST: 28×28 grayscale images of handwritten digits (0-9)
    - 60,000 training images
    - 10,000 test images
    - 1 channel (grayscale)

    For AdS/CFT: d=2 boundary, field has 1 channel, 28×28 spatial resolution

    Args:
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        normalize_range: Target pixel value range
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing MNIST
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    return ImageDatasetWrapper(dataset, normalize_range, flatten)


def load_fashion_mnist(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    flatten: bool = False,
) -> ImageDatasetWrapper:
    """
    Load Fashion-MNIST dataset.

    Fashion-MNIST: 28×28 grayscale images of clothing items
    - 60,000 training images
    - 10,000 test images
    - 1 channel (grayscale)
    - 10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
                  Sandal, Shirt, Sneaker, Bag, Ankle boot

    Args:
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        normalize_range: Target pixel value range
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing Fashion-MNIST
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    return ImageDatasetWrapper(dataset, normalize_range, flatten)


def load_cifar10(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    flatten: bool = False,
) -> ImageDatasetWrapper:
    """
    Load CIFAR-10 dataset.

    CIFAR-10: 32×32 RGB images of objects
    - 50,000 training images
    - 10,000 test images
    - 3 channels (RGB)
    - 10 classes: airplane, automobile, bird, cat, deer,
                  dog, frog, horse, ship, truck

    For AdS/CFT: d=2 boundary, field has 3 channels, 32×32 spatial resolution

    Args:
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        normalize_range: Target pixel value range
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing CIFAR-10
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    return ImageDatasetWrapper(dataset, normalize_range, flatten)


def load_cifar100(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    flatten: bool = False,
) -> ImageDatasetWrapper:
    """
    Load CIFAR-100 dataset.

    CIFAR-100: 32×32 RGB images with 100 fine-grained classes
    - 50,000 training images
    - 10,000 test images
    - 3 channels (RGB)
    - 100 classes grouped into 20 superclasses

    Args:
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        normalize_range: Target pixel value range
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing CIFAR-100
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    return ImageDatasetWrapper(dataset, normalize_range, flatten)


def load_image_dataset(
    name: str,
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    flatten: bool = False,
    n_samples: Optional[int] = None,
) -> Union[ImageDatasetWrapper, Dataset]:
    """
    Load an image dataset by name.

    Args:
        name: Dataset name (mnist, fashion_mnist, cifar10, cifar100)
        root: Data directory
        train: If True, load training set; else test set
        download: If True, download if not present
        normalize_range: Target range for pixel values
        flatten: If True, flatten images
        n_samples: If specified, use only this many samples

    Returns:
        ImageDatasetWrapper containing the dataset

    Raises:
        ValueError: If dataset name is unknown
    """
    name_lower = name.lower()

    if name_lower == "mnist":
        dataset = load_mnist(root, train, download, normalize_range, flatten)
    elif name_lower == "fashion_mnist":
        dataset = load_fashion_mnist(root, train, download, normalize_range, flatten)
    elif name_lower == "cifar10":
        dataset = load_cifar10(root, train, download, normalize_range, flatten)
    elif name_lower == "cifar100":
        dataset = load_cifar100(root, train, download, normalize_range, flatten)
    else:
        raise ValueError(
            f"Unknown image dataset: {name}. Available: {list(IMAGE_DATASETS.keys())}"
        )

    # Subsample if requested
    if n_samples is not None and n_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:n_samples].tolist()

        # Create a new wrapper that subsamples
        class SubsampledDataset(Dataset):
            """Subsampled version of an image dataset."""

            def __init__(
                self, base_dataset: ImageDatasetWrapper, indices: list
            ) -> None:
                self.base = base_dataset
                self.indices = indices
                self.image_shape = base_dataset.image_shape
                self.n_channels = base_dataset.n_channels
                self.height = base_dataset.height
                self.width = base_dataset.width
                self.normalize_range = base_dataset.normalize_range
                self.flatten = base_dataset.flatten

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
                return self.base[self.indices[idx]]

            def get_data_shape(self) -> Tuple[int, ...]:
                return self.base.get_data_shape()

            def get_image_shape(self) -> Tuple[int, int, int]:
                return self.base.get_image_shape()

        return SubsampledDataset(dataset, indices)

    return dataset


# =============================================================================
# Image Spectral Encoder
# =============================================================================


@register_encoder("image_spectral")
class ImageSpectralEncoder(nn.Module):
    """
    Source-based spectral encoding for images using the AdS/CFT bulk-boundary propagator.

    This encoder treats images as boundary SOURCES J(x), not fields directly.
    The bulk field φ is obtained by convolving with the bulk-boundary propagator K:

        φ̂(k) = K̂(r_UV, k) * Ĵ(k)
        π̂(k) = ∂_r K̂(r_UV, k) * Ĵ(k)

    where Ĵ(k) = FFT(image) is the source in Fourier space.

    This is consistent with the AdS/CFT dictionary (paper Section 3, eq. 6):
        Φ(r,x) = ∫ K(r,x;x') J(x') d^d x'

    For planar AdS, the UV-stabilized bulk-boundary propagator in Fourier space is:
        K̃_Δ(r, k) = (2/Γ(ν)) (e^{-r}|k|/2)^ν K_ν(|k|e^{-r})

    where ν = Δ - d/2 and K_ν is the modified Bessel function of the second kind.

    Attributes
    ----------
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width
    normalize : bool
        Whether to use orthonormal FFT
    delta : float
        Conformal dimension for propagator
    r_uv : float
        UV boundary radius
    k_sq : Tensor
        Precomputed |k|² values
    envelope_phi : Tensor
        Precomputed field propagator envelope K̂(r_UV, k)
    envelope_pi : Tensor
        Precomputed momentum propagator envelope ∂_r K̂(r_UV, k)

    Example
    -------
    >>> encoder = ImageSpectralEncoder(image_shape=(1, 28, 28), delta=2.0)
    >>> phi_hat, pi_hat = encoder.encode_with_pi(images)  # (B, 2, 28, 28) each
    >>> images_decoded = encoder.decode(phi_hat)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        normalize: bool = True,
        delta: float = 2.0,
        r_uv: float = 1.0,
        regularization: float = 1e-2,
    ) -> None:
        """
        Initialize source-based image spectral encoder.

        Args:
            image_shape: (n_channels, height, width)
            normalize: If True, normalize FFT by sqrt(H*W)
            delta: Conformal dimension Δ for bulk-boundary propagator
            r_uv: UV boundary radial coordinate
            regularization: Small epsilon for numerical stability at k=0
        """
        print(
            f"[DEBUG] ImageSpectralEncoder.__init__ (SOURCE-BASED) started with shape={image_shape}",
            flush=True,
        )
        super().__init__()

        self.n_channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.normalize = normalize
        self.delta = delta
        self.r_uv = r_uv
        self.d = 2  # Boundary dimension for images
        self.regularization = regularization

        print(f"[DEBUG] Computing wave numbers...", flush=True)
        # Precompute wave numbers for Laplacian
        # For images, we use L = H or W (assuming unit spacing, so L = size)
        kx = torch.fft.fftfreq(self.height) * 2 * torch.pi
        ky = torch.fft.fftfreq(self.width) * 2 * torch.pi

        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        k_sq = KX ** 2 + KY ** 2
        k_mag = torch.sqrt(k_sq + regularization ** 2)

        # CRITICAL: clone() required because meshgrid returns views with shared memory
        self.register_buffer("k_sq", k_sq.clone())
        self.register_buffer("KX", KX.clone())
        self.register_buffer("KY", KY.clone())
        self.register_buffer("k_mag", k_mag.clone())
        print(f"[DEBUG] Wave numbers computed, k_sq shape={k_sq.shape}", flush=True)

        # Compute propagator envelopes using Bessel functions
        # This matches the point encoder in encoding_spectral.py
        print(
            f"[DEBUG] Computing Bessel propagator envelopes with delta={delta}, r_uv={r_uv}...",
            flush=True,
        )
        envelope_phi, envelope_pi = self._compute_bessel_envelopes(
            k_mag, delta, r_uv, regularization
        )
        self.register_buffer("envelope_phi", envelope_phi)
        self.register_buffer("envelope_pi", envelope_pi)
        
        # Also store the ratio for backward compatibility
        # π/φ ratio at each k
        pi_phi_ratio = envelope_pi / (envelope_phi + 1e-10)
        self.register_buffer("pi_phi_ratio", pi_phi_ratio)
        
        print(f"[DEBUG] ImageSpectralEncoder.__init__ complete", flush=True)

    def _compute_bessel_envelopes(
        self,
        k_mag: Tensor,
        delta: float,
        r_uv: float,
        regularization: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the UV-stabilized bulk-boundary propagator envelopes.

        For planar AdS, the UV-stabilized propagator in Fourier space is:
            K̃_Δ(r, k) = (2/Γ(ν)) (e^{-r}|k|/2)^ν K_ν(|k|e^{-r})

        where ν = Δ - d/2, and the momentum envelope is:
            ∂_r K̃_Δ ~ ξ^{ν+1} K_{ν-1}(ξ)

        with ξ = |k|e^{-r} = |k|z where z is the Poincaré coordinate.

        This matches SpectralHolographicEncoder._init_planar_fourier() exactly.

        Args:
            k_mag: |k| magnitude tensor (regularized)
            delta: Conformal dimension Δ
            r_uv: UV boundary radius
            regularization: Small epsilon for stability

        Returns:
            envelope_phi: Field propagator envelope (H, W)
            envelope_pi: Momentum propagator envelope (H, W)
        """
        nu = delta - self.d / 2.0  # ν = Δ - d/2

        # ξ = |k| * z = |k| * e^{-r} is the argument for Bessel functions
        z_uv = math.exp(-r_uv)
        xi = k_mag * z_uv

        # Compute Bessel functions K_ν(ξ) and K_{ν-1}(ξ)
        K_nu = _bessel_k_nu(nu, xi, eps=regularization)
        K_nu_minus_1 = _bessel_k_nu(abs(nu - 1.0), xi, eps=regularization)

        # UV-stabilized propagator: K̃ = e^{(d-Δ)r} K
        # Starting from κ_{|k|}(r) = (2/Γ(ν)) e^{-dr/2} (|k|/2)^ν K_ν(|k|e^{-r})
        # Multiplying by e^{(d-Δ)r} gives:
        #   κ̃_{|k|}(r) = (2/Γ(ν)) (ξ/2)^ν K_ν(ξ)
        xi_power = torch.pow(xi, nu)
        envelope_phi_raw = xi_power * K_nu

        # Normalize so that envelope_phi[0, 0] = 1 (k=0 mode)
        norm_val = envelope_phi_raw[0, 0].item()
        if abs(norm_val) > 1e-10:
            envelope_phi = envelope_phi_raw / norm_val
        else:
            envelope_phi = envelope_phi_raw

        # Momentum envelope: Π̃ = e^{(d-Δ)r}(Π + (d-Δ)Φ) = ∂_r Φ̃
        # ∂_r[ξ^ν K_ν(ξ)] where ξ = |k|e^{-r}
        # = ξ^{ν+1} K_{ν-1}(ξ)
        envelope_pi_raw = xi_power * xi * K_nu_minus_1

        if abs(norm_val) > 1e-10:
            envelope_pi = envelope_pi_raw / norm_val
        else:
            envelope_pi = envelope_pi_raw

        print(
            f"[DEBUG] Bessel envelopes computed: phi range [{envelope_phi.min():.4f}, {envelope_phi.max():.4f}], "
            f"pi range [{envelope_pi.min():.4f}, {envelope_pi.max():.4f}]",
            flush=True,
        )

        return envelope_phi, envelope_pi

    def encode(self, images: Tensor) -> Tensor:
        """
        Encode images (sources) to bulk field spectral coefficients.

        The image is treated as a source J, and the field is obtained by
        multiplying by the propagator envelope:
            φ̂(k) = K̂(r_UV, k) * Ĵ(k)

        Args:
            images: (B, C, H, W) image tensor (source J)

        Returns:
            phi_hat: (B, 2*C, H, W) field spectral coefficients (real/imag interleaved)
        """
        B, C, H, W = images.shape

        # 2D FFT for each channel: image → source spectral coefficients Ĵ(k)
        J_hat_complex = torch.fft.fft2(
            images, norm="ortho" if self.normalize else None
        )

        # Apply propagator envelope: φ̂ = K̂ × Ĵ
        envelope = self.envelope_phi.to(device=J_hat_complex.device, dtype=J_hat_complex.real.dtype)
        phi_hat_complex = J_hat_complex * envelope

        # Split into real and imaginary, interleave channels
        # Output shape: (B, 2*C, H, W)
        phi_real = phi_hat_complex.real  # (B, C, H, W)
        phi_imag = phi_hat_complex.imag  # (B, C, H, W)

        # Interleave: [real_0, imag_0, real_1, imag_1, ...]
        phi_hat = torch.stack([phi_real, phi_imag], dim=2)  # (B, C, 2, H, W)
        phi_hat = phi_hat.view(B, 2 * C, H, W)

        return phi_hat

    def encode_with_pi(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode images (sources) to bulk field and momentum spectral coefficients.

        The image is treated as a source J, and field/momentum are obtained by
        multiplying by the respective propagator envelopes:
            φ̂(k) = K̂(r_UV, k) * Ĵ(k)
            π̂(k) = ∂_r K̂(r_UV, k) * Ĵ(k)

        This matches the point encoder physics exactly.

        Args:
            images: (B, C, H, W) image tensor (source J)

        Returns:
            phi_hat: (B, 2*C, H, W) field spectral coefficients
            pi_hat: (B, 2*C, H, W) momentum spectral coefficients
        """
        B, C, H, W = images.shape

        # 2D FFT for each channel: image → source spectral coefficients Ĵ(k)
        J_hat_complex = torch.fft.fft2(
            images, norm="ortho" if self.normalize else None
        )

        # Apply propagator envelopes: φ̂ = K̂ × Ĵ, π̂ = ∂_r K̂ × Ĵ
        envelope_phi = self.envelope_phi.to(device=J_hat_complex.device, dtype=J_hat_complex.real.dtype)
        envelope_pi = self.envelope_pi.to(device=J_hat_complex.device, dtype=J_hat_complex.real.dtype)
        
        phi_hat_complex = J_hat_complex * envelope_phi
        pi_hat_complex = J_hat_complex * envelope_pi

        # Split into real and imaginary, interleave channels
        phi_real = phi_hat_complex.real
        phi_imag = phi_hat_complex.imag
        pi_real = pi_hat_complex.real
        pi_imag = pi_hat_complex.imag

        # Interleave: [real_0, imag_0, real_1, imag_1, ...]
        phi_hat = torch.stack([phi_real, phi_imag], dim=2).view(B, 2 * C, H, W)
        pi_hat = torch.stack([pi_real, pi_imag], dim=2).view(B, 2 * C, H, W)

        return phi_hat, pi_hat

    def decode(self, phi_hat: Tensor) -> Tensor:
        """
        Decode field spectral coefficients back to images (sources).

        Since encoding applied: φ̂ = K̂ * Ĵ
        Decoding inverts: Ĵ = φ̂ / K̂ → IFFT → image

        Args:
            phi_hat: (B, 2*C, H, W) field spectral coefficients

        Returns:
            images: (B, C, H, W) reconstructed images (sources)
        """
        B = phi_hat.shape[0]
        C = self.n_channels
        H, W = self.height, self.width

        # Reshape: (B, 2*C, H, W) -> (B, C, 2, H, W) -> complex (B, C, H, W)
        phi_hat_ri = phi_hat.view(B, C, 2, H, W)
        phi_hat_complex = torch.complex(phi_hat_ri[:, :, 0], phi_hat_ri[:, :, 1])

        # Invert propagator: Ĵ = φ̂ / K̂
        envelope = self.envelope_phi.to(device=phi_hat_complex.device, dtype=phi_hat_complex.real.dtype)
        # Add small epsilon to avoid division by zero
        J_hat_complex = phi_hat_complex / (envelope + 1e-10)

        # Inverse 2D FFT: Ĵ → J (source/image)
        images = torch.fft.ifft2(
            J_hat_complex, norm="ortho" if self.normalize else None
        )

        # Take real part (imaginary should be ~0 for real inputs)
        return images.real

    def get_k_sq(self) -> Tensor:
        """Return |k|² for Laplacian computation."""
        return self.k_sq


# =============================================================================
# Image Spectral Laplacian
# =============================================================================


class ImageSpectralLaplacian(nn.Module):
    """
    Laplacian operator for image spectral coefficients.

    In Fourier space: Δφ̂(x) = -|k|² φ̂(k)

    This is diagonal - just multiply by -|k|²!

    Attributes
    ----------
    minus_k_sq : Tensor
        Precomputed -|k|² values

    Example
    -------
    >>> laplacian = ImageSpectralLaplacian(encoder.get_k_sq())
    >>> lap_phi = laplacian.apply_minus_laplacian(phi_hat)
    """

    def __init__(self, k_sq: Tensor) -> None:
        """
        Initialize image spectral Laplacian.

        Args:
            k_sq: (H, W) tensor of |k|² values
        """
        super().__init__()
        self.register_buffer("minus_k_sq", -k_sq)

    def apply_minus_laplacian(self, phi_hat: Tensor) -> Tensor:
        """
        Apply (-Δ) in spectral space.

        Args:
            phi_hat: (B, 2*C, H, W) spectral coefficients

        Returns:
            result: (B, 2*C, H, W) = |k|² φ̂
        """
        k_sq = (-self.minus_k_sq).to(device=phi_hat.device, dtype=phi_hat.dtype)
        # Expand for batch and channel dimensions
        return phi_hat * k_sq.unsqueeze(0).unsqueeze(0)

    def diag_eigs(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Return diagonal eigenvalues (|k|²) for Sobolev prior.

        Args:
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor of shape (H, W) containing |k|² values
        """
        eigs = -self.minus_k_sq
        if device is not None:
            eigs = eigs.to(device=device)
        if dtype is not None:
            eigs = eigs.to(dtype=dtype)
        return eigs


# =============================================================================
# Image Spectral Codec (Combined Encoder + Decoder + Laplacian)
# =============================================================================


@register_encoder("image_codec")
class ImageSpectralCodec(nn.Module):
    """
    Full encoder-decoder for images with spectral Laplacian.

    This is the correct way to apply AdS/CFT to images:
    - Images ARE boundary fields (d=2)
    - FFT gives spectral representation
    - Laplacian is diagonal multiplication
    - Momentum π̃ from bulk-boundary propagator derivative (like toys)

    Attributes
    ----------
    encoder : ImageSpectralEncoder
        Spectral encoder
    laplacian : ImageSpectralLaplacian
        Diagonal Laplacian operator
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width
    image_shape : Tuple[int, int, int]
        (C, H, W) image dimensions
    field_shape : Tuple[int, int, int]
        (2*C, H, W) for real/imag coefficients

    Example
    -------
    >>> codec = ImageSpectralCodec(image_shape=(1, 28, 28))
    >>> phi_hat, pi_hat = codec.encode_with_pi(images)
    >>> laplacian = codec.get_laplacian()
    >>> images_decoded = codec.decode(phi_hat)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        normalize: bool = True,
        delta: float = 2.0,
        r_uv: float = 1.0,
    ) -> None:
        """
        Initialize image spectral codec.

        Args:
            image_shape: (C, H, W) image dimensions
            normalize: Use orthonormal FFT
            delta: Conformal dimension for propagator
            r_uv: UV boundary location
        """
        super().__init__()

        self.encoder = ImageSpectralEncoder(
            image_shape, normalize, delta=delta, r_uv=r_uv
        )
        self.laplacian = ImageSpectralLaplacian(self.encoder.get_k_sq())

        self.n_channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.image_shape = image_shape

        # Field shape for bulk state: (2*C, H, W) for real/imag
        self.field_shape = (2 * self.n_channels, self.height, self.width)

    def encode(self, images: Tensor) -> Tensor:
        """Encode images to spectral coefficients (phi only)."""
        return self.encoder.encode(images)

    def encode_with_pi(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode images to spectral coefficients with physical momentum.

        Args:
            images: (B, C, H, W) image tensor

        Returns:
            phi_hat: (B, 2*C, H, W) field spectral coefficients
            pi_hat: (B, 2*C, H, W) momentum from propagator derivative
        """
        return self.encoder.encode_with_pi(images)

    def decode(self, phi_hat: Tensor) -> Tensor:
        """Decode spectral coefficients to images."""
        return self.encoder.decode(phi_hat)

    def get_laplacian(self) -> ImageSpectralLaplacian:
        """Return the spectral Laplacian operator."""
        return self.laplacian


# =============================================================================
# Utility Functions
# =============================================================================


def get_image_data_info(dataset_name: str) -> Dict[str, int]:
    """
    Get information about an image dataset.

    Args:
        dataset_name: Dataset name

    Returns:
        dict with keys: n_channels, height, width, n_classes, d (boundary dim)

    Raises:
        ValueError: If dataset name is unknown
    """
    info = {
        "mnist": {
            "n_channels": 1,
            "height": 28,
            "width": 28,
            "n_classes": 10,
            "d": 2,
        },
        "fashion_mnist": {
            "n_channels": 1,
            "height": 28,
            "width": 28,
            "n_classes": 10,
            "d": 2,
        },
        "cifar10": {
            "n_channels": 3,
            "height": 32,
            "width": 32,
            "n_classes": 10,
            "d": 2,
        },
        "cifar100": {
            "n_channels": 3,
            "height": 32,
            "width": 32,
            "n_classes": 100,
            "d": 2,
        },
    }

    name_lower = dataset_name.lower()
    if name_lower not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return info[name_lower]


def create_image_dataloaders(
    dataset_name: str,
    batch_size: int,
    root: str = "./data",
    n_train_samples: Optional[int] = None,
    n_val_samples: Optional[int] = None,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Tuple[int, ...]]:
    """
    Create train and validation dataloaders for an image dataset.

    Args:
        dataset_name: Name of dataset (mnist, cifar10, etc.)
        batch_size: Batch size
        root: Data directory
        n_train_samples: Limit training samples (None = use all)
        n_val_samples: Limit validation samples (None = use all)
        normalize_range: Pixel value range
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, data_shape)
    """
    # Load datasets
    train_dataset = load_image_dataset(
        dataset_name,
        root,
        train=True,
        download=True,
        normalize_range=normalize_range,
        n_samples=n_train_samples,
    )
    val_dataset = load_image_dataset(
        dataset_name,
        root,
        train=False,
        download=True,
        normalize_range=normalize_range,
        n_samples=n_val_samples,
    )

    # Get data shape
    data_shape = train_dataset.get_data_shape()

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, data_shape


def create_image_encoder(
    dataset_name: str,
    delta: float = 2.0,
    r_uv: float = 1.0,
) -> ImageSpectralCodec:
    """
    Create an image encoder for a given dataset.

    Args:
        dataset_name: Name of dataset
        delta: Conformal dimension
        r_uv: UV boundary radius

    Returns:
        ImageSpectralCodec configured for the dataset
    """
    info = get_image_data_info(dataset_name)
    image_shape = (info["n_channels"], info["height"], info["width"])
    return ImageSpectralCodec(image_shape, delta=delta, r_uv=r_uv)
