"""
data_image.py

Image Dataset Support for AdS/CFT Flow Matching
================================================

Images are treated as boundary fields Φ(x, y) on a 2D domain (d=2).
This is the physically correct interpretation: the image IS the CFT field,
not a collection of points.

Supported Datasets
------------------
- MNIST: 28×28 grayscale, 10 classes
- FashionMNIST: 28×28 grayscale, 10 classes
- CIFAR10: 32×32 RGB, 10 classes
- CIFAR100: 32×32 RGB, 100 classes

Physics Interpretation
----------------------
For images, the boundary dimension is d=2 (spatial image dimensions).
The image channels represent different field components. The FFT provides
the spectral representation, and the momentum π̃ is computed from the
bulk-boundary propagator derivative, exactly matching the physics used
for toy datasets.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

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
        name: Dataset name to check

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
    - Normalizes images to specified range (default: [-1, 1])
    - Optionally flattens for compatibility checks
    - Provides image shape and channel info

    Attributes
    ----------
    dataset : Dataset
        Underlying torchvision dataset
    normalize_range : Tuple[float, float]
        Target range for pixel values
    flatten : bool
        Whether to flatten images to 1D
    image_shape : Tuple[int, int, int]
        Image shape (C, H, W)
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width

    Example
    -------
    >>> dataset = ImageDatasetWrapper(mnist_dataset, normalize_range=(-1, 1))
    >>> image, label = dataset[0]
    >>> print(image.shape)  # torch.Size([1, 28, 28])
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
            normalize_range: Target range for pixel values (default: (-1, 1))
            flatten: If True, flatten images to 1D (default: False)
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
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
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
            Tuple of (channels, height, width)
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
        normalize_range: Target range for pixel values
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing MNIST
    """
    transform = transforms.Compose([transforms.ToTensor()])

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
        normalize_range: Target range for pixel values
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing Fashion-MNIST
    """
    transform = transforms.Compose([transforms.ToTensor()])

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
        normalize_range: Target range for pixel values
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing CIFAR-10
    """
    transform = transforms.Compose([transforms.ToTensor()])

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
        normalize_range: Target range for pixel values
        flatten: If True, flatten images

    Returns:
        ImageDatasetWrapper containing CIFAR-100
    """
    transform = transforms.Compose([transforms.ToTensor()])

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
) -> ImageDatasetWrapper:
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
            def __init__(self, base_dataset, indices):
                self.base = base_dataset
                self.indices = indices
                self.image_shape = base_dataset.image_shape
                self.n_channels = base_dataset.n_channels
                self.height = base_dataset.height
                self.width = base_dataset.width
                self.normalize_range = base_dataset.normalize_range
                self.flatten = base_dataset.flatten

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.base[self.indices[idx]]

            def get_data_shape(self):
                return self.base.get_data_shape()

            def get_image_shape(self):
                return self.base.get_image_shape()

        return SubsampledDataset(dataset, indices)

    return dataset


# =============================================================================
# Image Field Encoder (Direct FFT)
# =============================================================================


class ImageSpectralEncoder(nn.Module):
    """
    Encode images directly to spectral coefficients using FFT.

    For images, the boundary field IS the image - we take the 2D FFT of each channel.
    The momentum π̃ is computed from the bulk-boundary propagator derivative,
    matching the physics used for toy datasets.

    This is the physically correct encoding for treating images as CFT fields:
        Φ(x, y) → Φ̂(kx, ky) via FFT
        Π̂(kx, ky) = (∂_r K / K)|_{r=r_UV} × Φ̂(kx, ky)

    The spectral coefficients then evolve via the Klein-Gordon equation in the bulk.

    Attributes
    ----------
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width
    normalize : bool
        Whether to normalize FFT
    delta : float
        Conformal dimension Δ
    r_uv : float
        UV boundary location
    d : int
        Boundary dimension (always 2 for images)
    k_sq : Tensor
        Squared wave numbers |k|²
    pi_phi_ratio : Tensor
        Ratio π/φ from propagator derivative

    Example
    -------
    >>> encoder = ImageSpectralEncoder(image_shape=(1, 28, 28))
    >>> phi_hat = encoder.encode(images)
    >>> phi_hat, pi_hat = encoder.encode_with_pi(images)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # (C, H, W)
        normalize: bool = True,
        delta: float = 2.0,
        r_uv: float = 1.0,
    ) -> None:
        """
        Initialize image spectral encoder.

        Args:
            image_shape: (n_channels, height, width)
            normalize: If True, normalize FFT by sqrt(H*W) (default: True)
            delta: Conformal dimension Δ for bulk-boundary propagator (default: 2.0)
            r_uv: UV boundary radial coordinate (default: 1.0)
        """
        super().__init__()

        self.n_channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.normalize = normalize
        self.delta = delta
        self.r_uv = r_uv
        self.d = 2  # Boundary dimension for images

        # Precompute wave numbers for Laplacian
        # k = 2π n / L where n is FFT index and L is domain size
        kx = torch.fft.fftfreq(self.height) * 2 * torch.pi
        ky = torch.fft.fftfreq(self.width) * 2 * torch.pi

        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        k_sq = KX ** 2 + KY ** 2

        # CRITICAL: clone() required because meshgrid returns views with shared memory
        self.register_buffer("k_sq", k_sq.clone())
        self.register_buffer("KX", KX.clone())
        self.register_buffer("KY", KY.clone())

        # Compute propagator ratio for momentum: π/φ = (∂_r K / K)|_{r=r_UV}
        pi_phi_ratio = self._compute_propagator_ratio(k_sq, delta, r_uv)
        self.register_buffer("pi_phi_ratio", pi_phi_ratio)

    def _compute_propagator_ratio(
        self, k_sq: Tensor, delta: float, r_uv: float
    ) -> Tensor:
        """
        Compute the ratio π/φ = (∂_r K / K)|_{r=r_UV} from the Bessel propagator.

        For planar AdS with conformal dimension Δ:
            K_Δ(r, k) = r^{d/2} |k|^ν K_ν(|k|r)  where ν = Δ - d/2

        The derivative ratio gives the momentum-field relationship at the UV boundary.
        Uses asymptotic approximation for numerical stability.

        Args:
            k_sq: Squared wave numbers
            delta: Conformal dimension
            r_uv: UV boundary location

        Returns:
            Tensor of π/φ ratios
        """
        k = torch.sqrt(k_sq).numpy()
        nu = delta - self.d / 2  # ν = Δ - d/2
        z = r_uv  # UV boundary

        # Regularize k=0
        k_reg = np.maximum(k, 1e-6)
        arg = k_reg * z

        try:
            import scipy.special

            # Compute Bessel function ratio carefully
            large_arg_mask = arg > 100
            small_arg_mask = arg < 0.01
            mid_mask = ~large_arg_mask & ~small_arg_mask

            ratio_bessel = np.zeros_like(arg)

            # Large argument: asymptotic ratio → 1
            ratio_bessel[large_arg_mask] = 1.0

            # Small argument: use series expansion
            if nu > 1:
                ratio_bessel[small_arg_mask] = 2.0 / (
                    arg[small_arg_mask] * (nu - 1) + 1e-10
                )
            else:
                ratio_bessel[small_arg_mask] = 1.0

            # Middle range: direct computation
            if np.any(mid_mask):
                K_nu = scipy.special.kv(nu, arg[mid_mask])
                K_nu_m1 = scipy.special.kv(nu - 1, arg[mid_mask])

                # Safe division
                valid = (K_nu > 1e-30) & np.isfinite(K_nu) & np.isfinite(K_nu_m1)
                ratio_bessel[mid_mask] = np.where(
                    valid, K_nu_m1 / (K_nu + 1e-30), 1.0
                )

            # Clip to reasonable range
            ratio_bessel = np.clip(ratio_bessel, 0.0, 100.0)

        except Exception:
            # Fallback: use simple asymptotic approximation
            ratio_bessel = np.ones_like(k)

        # Final ratio: π/φ = (d/2 - ν) - |k| × K_{ν-1}/K_ν
        pi_phi = (self.d / 2 - nu) - k_reg * ratio_bessel

        # At k=0, use limiting behavior: π/φ → -Δ (from AdS/CFT)
        pi_phi[k < 1e-5] = -delta

        # Ensure finite values
        pi_phi = np.clip(pi_phi, -100.0, 100.0)

        return torch.from_numpy(pi_phi.astype(np.float32))

    def encode(self, images: Tensor) -> Tensor:
        """
        Encode images to spectral coefficients.

        Args:
            images: (B, C, H, W) image tensor

        Returns:
            phi_hat: (B, 2*C, H, W) spectral coefficients (real/imag interleaved)
        """
        B, C, H, W = images.shape

        # 2D FFT for each channel
        phi_hat_complex = torch.fft.fft2(
            images, norm="ortho" if self.normalize else None
        )

        # Split into real and imaginary, interleave channels
        phi_real = phi_hat_complex.real  # (B, C, H, W)
        phi_imag = phi_hat_complex.imag  # (B, C, H, W)

        # Interleave: [real_0, imag_0, real_1, imag_1, ...]
        phi_hat = torch.stack([phi_real, phi_imag], dim=2)  # (B, C, 2, H, W)
        phi_hat = phi_hat.view(B, 2 * C, H, W)

        return phi_hat

    def encode_with_pi(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode images to spectral coefficients with physical momentum.

        The momentum π̃ is computed from the bulk-boundary propagator derivative,
        exactly matching the physics used for toy datasets.

        Args:
            images: (B, C, H, W) image tensor

        Returns:
            phi_hat: (B, 2*C, H, W) field spectral coefficients
            pi_hat: (B, 2*C, H, W) momentum spectral coefficients
        """
        B, C, H, W = images.shape

        # 2D FFT for each channel
        phi_hat_complex = torch.fft.fft2(
            images, norm="ortho" if self.normalize else None
        )

        # Compute momentum from propagator: π = (π/φ ratio) × φ
        pi_hat_complex = phi_hat_complex * self.pi_phi_ratio.to(phi_hat_complex.device)

        # Split into real and imaginary, interleave channels
        phi_real = phi_hat_complex.real
        phi_imag = phi_hat_complex.imag
        pi_real = pi_hat_complex.real
        pi_imag = pi_hat_complex.imag

        # Interleave
        phi_hat = torch.stack([phi_real, phi_imag], dim=2).view(B, 2 * C, H, W)
        pi_hat = torch.stack([pi_real, pi_imag], dim=2).view(B, 2 * C, H, W)

        return phi_hat, pi_hat

    def decode(self, phi_hat: Tensor) -> Tensor:
        """
        Decode spectral coefficients back to images.

        Args:
            phi_hat: (B, 2*C, H, W) spectral coefficients

        Returns:
            images: (B, C, H, W) reconstructed images
        """
        B = phi_hat.shape[0]
        C = self.n_channels
        H, W = self.height, self.width

        # Reshape: (B, 2*C, H, W) -> (B, C, 2, H, W) -> complex (B, C, H, W)
        phi_hat_ri = phi_hat.view(B, C, 2, H, W)
        phi_hat_complex = torch.complex(phi_hat_ri[:, :, 0], phi_hat_ri[:, :, 1])

        # Inverse 2D FFT
        images = torch.fft.ifft2(
            phi_hat_complex, norm="ortho" if self.normalize else None
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

    In Fourier space: Δφ̂(k) = -|k|² φ̂(k)

    Attributes
    ----------
    minus_k_sq : Tensor
        Buffer containing -|k|² values

    Example
    -------
    >>> laplacian = ImageSpectralLaplacian(k_sq)
    >>> result = laplacian.apply_minus_laplacian(phi_hat)  # = |k|² φ̂
    """

    def __init__(self, k_sq: Tensor) -> None:
        """
        Initialize spectral Laplacian.

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
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
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
# Image Spectral Codec (Encoder + Decoder + Laplacian)
# =============================================================================


class ImageSpectralCodec(nn.Module):
    """
    Full encoder-decoder for images with spectral Laplacian.

    This is the correct way to apply AdS/CFT to images:
    - Images ARE boundary fields (d=2)
    - FFT gives spectral representation
    - Laplacian is diagonal multiplication
    - Momentum π̃ from bulk-boundary propagator derivative

    Attributes
    ----------
    encoder : ImageSpectralEncoder
        FFT-based encoder
    laplacian : ImageSpectralLaplacian
        Spectral Laplacian operator
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width
    image_shape : Tuple[int, int, int]
        Image shape (C, H, W)
    field_shape : Tuple[int, int, int]
        Field shape for bulk state (2*C, H, W)

    Example
    -------
    >>> codec = ImageSpectralCodec(image_shape=(1, 28, 28))
    >>> phi_hat = codec.encode(images)
    >>> images_rec = codec.decode(phi_hat)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # (C, H, W)
        normalize: bool = True,
        delta: float = 2.0,
        r_uv: float = 1.0,
    ) -> None:
        """
        Initialize image spectral codec.

        Args:
            image_shape: (n_channels, height, width)
            normalize: If True, normalize FFT
            delta: Conformal dimension Δ
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
        dataset_name: Name of the dataset

    Returns:
        Dictionary with keys: n_channels, height, width, n_classes, d

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