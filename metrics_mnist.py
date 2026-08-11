"""
metrics_mnist.py

MNIST-appropriate distributional metrics (referee request)
==========================================================

Referee request (Report 1, on the MNIST comparison):

    "the FID comparison should be supplemented with an MNIST-appropriate
     classifier-feature FID, kernel-based metrics such as KID,
     precision/recall-style coverage metrics, or another suitable
     distributional measure."

This module implements ALL of the suggested alternatives so that the revised
manuscript can report, per model and per seed:

  1. Classifier-feature FID
       Frechet distance in the 128-dimensional penultimate feature space of a
       small MNIST CNN classifier trained (deterministically) to ~99% test
       accuracy on the SAME pixel normalisation used by the generative
       pipeline ([-1, 1]).  This is the standard MNIST-appropriate substitute
       for the ImageNet Inception features.

  2. KID (Kernel Inception Distance; Binkowski, Sutherland, Arbel & Gretton,
     "Demystifying MMD GANs", ICLR 2018)
       Unbiased squared MMD with the polynomial kernel
           k(x, y) = ( x . y / D + 1 )^3 ,     D = feature dimension,
       estimated over ``n_subsets`` random subsets of size ``subset_size``
       (the standard block estimator; mean and std over blocks are reported).
       Computed BOTH in the classifier feature space and in the SAME
       InceptionV3 feature space used by the published FID pipeline, so the
       kernel metric can be compared like-for-like with the reported FID.

  3. Improved Precision and Recall (Kynkaanniemi, Karras, Laine, Lehtinen &
     Aila, NeurIPS 2019)
       Manifold estimated by k-nearest-neighbour balls (default k = 3) in the
       classifier feature space:
           precision = fraction of generated features inside the real manifold
           recall    = fraction of real features inside the generated manifold

Conventions (kept deliberately identical to the published pipeline)
-------------------------------------------------------------------
* Images are in [-1, 1] (the training pipeline maps MNIST to [-1, 1]); both
  real and generated tensors are clamped to [-1, 1] before feature extraction
  (a no-op for real data; removes tiny overshoots in generated data).
* The InceptionV3 feature extractor reproduces metrics.compute_fid EXACTLY:
  torchvision inception_v3(weights='IMAGENET1K_V1', transform_input=False),
  fc replaced by Identity, grayscale replicated to 3 channels, bilinear resize
  to 299x299, ImageNet mean/std normalisation, inputs fed as-is (no range
  remapping) — so KID-Inception lives in the same feature space as the FID
  values already reported in the paper.
* Every stochastic element (classifier training, KID subset selection) is
  seeded; results are bit-reproducible given the same hardware/library stack.

All functions accept torch tensors of shape (N, 1, 28, 28) (or (N, 28, 28))
in [-1, 1].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# =============================================================================
# 0. Small utilities
# =============================================================================


def _as_nchw(images: Tensor) -> Tensor:
    """(N, 28, 28) -> (N, 1, 28, 28); pass through (N, C, H, W)."""
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor of rank 3 or 4, got {images.shape}")
    return images


def _prepare_images(images: Tensor) -> Tensor:
    """Standard preparation: NCHW float32, clamped to the valid range [-1, 1]."""
    return _as_nchw(images).float().clamp_(-1.0, 1.0) if images.requires_grad is False \
        else _as_nchw(images).float().clamp(-1.0, 1.0)


# =============================================================================
# 1. MNIST feature classifier
# =============================================================================


class MNISTFeatureClassifier(nn.Module):
    """
    Small MNIST CNN classifier used as an MNIST-appropriate feature extractor.

    Architecture (the widely used PyTorch-examples MNIST CNN):
        Conv(1->32, 3x3) - ReLU - Conv(32->64, 3x3) - ReLU - MaxPool(2)
        - Dropout(0.25) - Flatten - Linear(9216->128) - ReLU  <- FEATURES (128-d)
        - Dropout(0.5) - Linear(128->10)

    ``features(x)`` returns the 128-dimensional post-ReLU penultimate
    activations used for classifier-feature FID / KID / precision-recall.
    Inputs are expected in [-1, 1] (the pipeline's pixel normalisation).
    """

    FEATURE_DIM = 128

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x  # (N, 128)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.dropout2(x)
        return self.fc2(x)


def train_mnist_classifier(
    data_root: str = "./data",
    device: Optional[torch.device] = None,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> Tuple[MNISTFeatureClassifier, float]:
    """
    Deterministically train the feature classifier on MNIST (pixels in [-1,1]).

    Returns (trained model on CPU in eval mode, test accuracy in [0, 1]).
    Typical result: >= 0.989 test accuracy after 3 epochs.
    """
    import torchvision
    import torchvision.transforms as T

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    tfm = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])  # -> [-1, 1]
    train_set = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_set = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=g,
        num_workers=2, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=2,
    )

    model = MNISTFeatureClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        print(f"[MNIST-CLF] epoch {epoch + 1}/{epochs} done (last loss {loss.item():.4f})",
              flush=True)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += int(y.numel())
    acc = correct / max(total, 1)
    print(f"[MNIST-CLF] test accuracy: {acc:.4f}", flush=True)
    return model.cpu().eval(), acc


def get_mnist_classifier(
    cache_path: str = "mnist_classifier.pt",
    data_root: str = "./data",
    device: Optional[torch.device] = None,
    epochs: int = 3,
    seed: int = 0,
) -> Tuple[MNISTFeatureClassifier, float]:
    """
    Load the cached feature classifier, training (and caching) it if absent.

    The cache stores the state dict together with the recorded test accuracy
    so the revised manuscript can quote it ("features from a CNN classifier
    with XX.X% MNIST test accuracy").
    """
    path = Path(cache_path)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        model = MNISTFeatureClassifier()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        acc = float(payload.get("test_accuracy", float("nan")))
        print(f"[MNIST-CLF] loaded cached classifier from {path} "
              f"(test accuracy {acc:.4f})", flush=True)
        return model, acc

    model, acc = train_mnist_classifier(
        data_root=data_root, device=device, epochs=epochs, seed=seed
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "test_accuracy": acc,
            "epochs": epochs,
            "seed": seed,
            "input_range": "[-1, 1]",
            "feature_dim": MNISTFeatureClassifier.FEATURE_DIM,
        },
        path,
    )
    print(f"[MNIST-CLF] cached classifier to {path}", flush=True)
    return model, acc


@torch.no_grad()
def extract_classifier_features(
    classifier: MNISTFeatureClassifier,
    images: Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Penultimate 128-d features for images in [-1, 1]. Returns (N, 128) float64."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).eval()
    images = _prepare_images(images)
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i: i + batch_size].to(device)
        feats.append(classifier.features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


# =============================================================================
# 2. InceptionV3 features (byte-compatible with metrics.compute_fid)
# =============================================================================


def _load_inception(device: torch.device) -> nn.Module:
    import torchvision.models as tv_models

    inception = tv_models.inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    inception.fc = nn.Identity()
    return inception.to(device).eval()


@torch.no_grad()
def extract_inception_features(
    images: Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 50,
) -> np.ndarray:
    """
    2048-d InceptionV3 pool features with EXACTLY the preprocessing of the
    published metrics.compute_fid: grayscale replicated to 3 channels,
    bilinear resize to 299, ImageNet mean/std, inputs fed as-is (the pipeline
    supplies [-1, 1] images). Returns (N, 2048) float64.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = _load_inception(device)
    images = _prepare_images(images)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i: i + batch_size].to(device)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear",
                                  align_corners=False)
        batch = (batch - mean) / std
        act = inception(batch)
        if isinstance(act, tuple):  # pragma: no cover (eval mode returns tensor)
            act = act[0]
        if act.ndim == 4 and (act.shape[2] != 1 or act.shape[3] != 1):
            act = F.adaptive_avg_pool2d(act, (1, 1))
        feats.append(act.view(act.shape[0], -1).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


# =============================================================================
# 3. Frechet distance in any feature space
# =============================================================================


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """
    ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}), with the same numerical
    safeguards (eps jitter, real part) as the published metrics.compute_fid.
    """
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(0.0, fid)


def compute_feature_fid(feat_real: np.ndarray, feat_gen: np.ndarray) -> float:
    """Frechet distance between Gaussian fits of two feature sets."""
    mu_r, sig_r = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
    mu_g, sig_g = feat_gen.mean(axis=0), np.cov(feat_gen, rowvar=False)
    return frechet_distance(mu_r, sig_r, mu_g, sig_g)


# =============================================================================
# 4. KID (unbiased polynomial-kernel MMD^2, block estimator)
# =============================================================================


def _poly3_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """k(x, y) = (x . y / D + 1)^3 with D the feature dimension."""
    D = x.shape[1]
    return (x @ y.T / D + 1.0) ** 3


def _mmd2_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased MMD^2 with the degree-3 polynomial kernel (Binkowski 2018)."""
    m, n = x.shape[0], y.shape[0]
    kxx = _poly3_kernel(x, x)
    kyy = _poly3_kernel(y, y)
    kxy = _poly3_kernel(x, y)
    sum_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    sum_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    sum_xy = kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def compute_kid(
    feat_real: np.ndarray,
    feat_gen: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    KID block estimator: mean and std of the unbiased MMD^2 over ``n_subsets``
    random subsets of size ``subset_size`` (capped at the available sample
    counts), each drawn without replacement within a subset.

    Returns:
        (kid_mean, kid_std) — the raw MMD^2 scale; multiply by 10^3 when
        quoting "KID x 10^3" as is conventional.
    """
    rng = np.random.default_rng(seed)
    m = min(subset_size, feat_real.shape[0], feat_gen.shape[0])
    if m < 2:
        raise ValueError("KID needs at least 2 samples per subset")
    vals = np.empty(n_subsets, dtype=np.float64)
    for s in range(n_subsets):
        idx_r = rng.choice(feat_real.shape[0], m, replace=False)
        idx_g = rng.choice(feat_gen.shape[0], m, replace=False)
        vals[s] = _mmd2_unbiased(feat_real[idx_r], feat_gen[idx_g])
    return float(vals.mean()), float(vals.std())


# =============================================================================
# 5. Improved Precision and Recall (Kynkaanniemi et al., 2019)
# =============================================================================


def _knn_radii(feats: Tensor, k: int, chunk: int) -> Tensor:
    """
    Distance from each point to its k-th nearest OTHER point in the same set.

    Args:
        feats: (N, D) float tensor on the compute device.
        k: neighbour index (k = 3 is the paper default).
        chunk: row-chunk size for the pairwise-distance computation.

    Returns:
        (N,) tensor of radii.
    """
    N = feats.shape[0]
    if N <= k:
        raise ValueError(f"Need more than k={k} points, got N={N}")
    radii = torch.empty(N, device=feats.device, dtype=feats.dtype)
    for i in range(0, N, chunk):
        rows = feats[i: i + chunk]
        d = torch.cdist(rows, feats)  # (chunk, N)
        # exclude self-distance
        r_idx = torch.arange(i, min(i + chunk, N), device=feats.device)
        d[torch.arange(rows.shape[0], device=feats.device), r_idx] = float("inf")
        radii[i: i + chunk] = d.kthvalue(k, dim=1).values
    return radii


def _fraction_inside_manifold(
    query: Tensor, support: Tensor, support_radii: Tensor, chunk: int
) -> float:
    """Fraction of query points within any k-NN ball of the support set."""
    inside = 0
    for i in range(0, query.shape[0], chunk):
        d = torch.cdist(query[i: i + chunk], support)  # (chunk, N_support)
        inside += int((d <= support_radii.unsqueeze(0)).any(dim=1).sum())
    return inside / query.shape[0]


def compute_precision_recall(
    feat_real: np.ndarray,
    feat_gen: np.ndarray,
    k: int = 3,
    chunk: int = 1024,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Improved precision & recall in a feature space (Kynkaanniemi et al. 2019).

        precision = fraction of GENERATED features inside the REAL manifold
        recall    = fraction of REAL features inside the GENERATED manifold

    with each manifold the union of k-NN balls of its own point set.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr = torch.from_numpy(np.ascontiguousarray(feat_real)).float().to(device)
    fg = torch.from_numpy(np.ascontiguousarray(feat_gen)).float().to(device)

    radii_r = _knn_radii(fr, k=k, chunk=chunk)
    radii_g = _knn_radii(fg, k=k, chunk=chunk)

    precision = _fraction_inside_manifold(fg, fr, radii_r, chunk)
    recall = _fraction_inside_manifold(fr, fg, radii_g, chunk)
    return float(precision), float(recall)


# =============================================================================
# 6. One-call convenience wrapper
# =============================================================================


def compute_all_mnist_metrics(
    real_images: Tensor,
    gen_images: Tensor,
    device: Optional[torch.device] = None,
    classifier_path: str = "mnist_classifier.pt",
    data_root: str = "./data",
    classifier_epochs: int = 3,
    k_nn: int = 3,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    kid_seed: int = 0,
    include_inception: bool = True,
) -> Dict[str, float]:
    """
    Compute every referee-requested MNIST metric for one (real, generated) pair.

    Both tensors: (N, 1, 28, 28) in [-1, 1] (clamped internally).

    Returns a flat dict:
        fid_classifier, kid_classifier_mean, kid_classifier_std,
        precision, recall, classifier_test_accuracy,
        [fid_inception, kid_inception_mean, kid_inception_std]
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_images = _prepare_images(real_images)
    gen_images = _prepare_images(gen_images)

    out: Dict[str, float] = {
        "n_real": float(real_images.shape[0]),
        "n_gen": float(gen_images.shape[0]),
    }

    # ---- classifier feature space ------------------------------------------
    clf, acc = get_mnist_classifier(
        cache_path=classifier_path, data_root=data_root,
        device=device, epochs=classifier_epochs,
    )
    out["classifier_test_accuracy"] = float(acc)

    fr_c = extract_classifier_features(clf, real_images, device=device)
    fg_c = extract_classifier_features(clf, gen_images, device=device)

    out["fid_classifier"] = compute_feature_fid(fr_c, fg_c)
    kid_c_mean, kid_c_std = compute_kid(
        fr_c, fg_c, n_subsets=kid_subsets, subset_size=kid_subset_size, seed=kid_seed
    )
    out["kid_classifier_mean"] = kid_c_mean
    out["kid_classifier_std"] = kid_c_std

    prec, rec = compute_precision_recall(fr_c, fg_c, k=k_nn, device=device)
    out["precision"] = prec
    out["recall"] = rec

    # ---- Inception feature space (same space as the published FID) ---------
    if include_inception:
        fr_i = extract_inception_features(real_images, device=device)
        fg_i = extract_inception_features(gen_images, device=device)
        out["fid_inception"] = compute_feature_fid(fr_i, fg_i)
        kid_i_mean, kid_i_std = compute_kid(
            fr_i, fg_i, n_subsets=kid_subsets, subset_size=kid_subset_size,
            seed=kid_seed,
        )
        out["kid_inception_mean"] = kid_i_mean
        out["kid_inception_std"] = kid_i_std

    return out
