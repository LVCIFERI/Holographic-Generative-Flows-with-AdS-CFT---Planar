#!/usr/bin/env python3
"""
evaluate_mnist_extended.py

Extended MNIST evaluation (referee request): classifier-feature FID, KID
(classifier and Inception feature spaces), and improved precision/recall,
computed per model and per seed, then aggregated as mean ± std.

No re-training is required: the script consumes either

  (1) the saved generated-sample tensors ``samples_final.pt`` written by the
      updated train.py at the final EMA evaluation (preferred: metrics are
      then recomputed on EXACTLY the reported sample sets), or
  (2) saved model checkpoints (``checkpoints/ema_model.pt`` by preference,
      falling back to ``best_model.pt`` / ``final_model.pt``), from which
      10,000 fresh samples are drawn with a fixed generator seed.

Usage
-----
# Scan a results directory produced by run_mnist_experiments.sh:
#   <results_dir>/{ads_hermite,ads_linear,no_kg_linear,cnn_baseline}/seed_{1,2,3}/
PYTHONPATH=.. python evaluate_mnist_extended.py \\
    --results_dir ./results_mnist_comparison_YYYYMMDD_HHMMSS \\
    --out extended_metrics_summary.json

# Single run directory (repeatable):
python evaluate_mnist_extended.py --run_dir path/to/seed_1 --label ads_hermite

# Direct samples file(s):
python evaluate_mnist_extended.py --samples s1.pt --samples s2.pt --label ads_linear

# Explicit checkpoint with explicit model configuration (for runs that predate
# config.json; the preset must match how the checkpoint was trained):
python evaluate_mnist_extended.py --checkpoint ema_model.pt --preset ads_hermite

Conventions
-----------
* Real reference: the full 10,000-image MNIST TEST split (held out; default) or
  a seeded 10,000-image subset of the training split (--real train), both
  normalised to [-1, 1] exactly as in training.
* Generated and real images are clamped to [-1, 1] before feature extraction.
* Generation uses a fixed torch.Generator seed (--gen_seed, default 12345) and
  batched sampling. Results are deterministic for a fixed (--gen_seed,
  --gen_batch) pair; both values are recorded in every per-run JSON. (The IR
  prior draws phi then pi per batch, so different --gen_batch values interleave
  the generator stream differently — keep the default 1000 for like-for-like
  reruns.)
* Every run directory gets an ``extended_metrics.json``; the aggregate summary
  (mean ± std across seeds per experiment) is written to --out and printed as
  an aligned table plus ready-to-paste LaTeX rows.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# Experiment presets: EXACT training configurations of run_mnist_experiments.sh
# =============================================================================

_MNIST_COMMON: Dict[str, object] = dict(
    dataset="mnist",
    slice_geometry="planar",
    deltas=(1.5,),
    r_ir=0.0,
    r_uv=1.0,
    ode_solver="rk4",
    ode_n_steps=100,
    lift_noise_sigma=0.1,
    spectral_n_modes=28,  # run_mnist_experiments.sh uses 28 (28x28 images)
    residual_hidden=64,
    residual_depth=3,
    epochs=1500,
    batch_size=128,
    n_train_samples=10000,
    use_ema=True,
)

EXPERIMENT_PRESETS: Dict[str, Dict[str, object]] = {
    "ads_hermite": dict(_MNIST_COMMON, path_type="hermite", use_image_encoding=True),
    "ads_linear": dict(_MNIST_COMMON, path_type="linear", use_image_encoding=True),
    "no_kg_linear": dict(
        _MNIST_COMMON, path_type="linear", backbone_scale=0.0, use_image_encoding=True
    ),
    "cnn_baseline": dict(
        _MNIST_COMMON, path_type="linear", backbone_scale=0.0, use_vanilla_cnn=True
    ),
}


# =============================================================================
# Model loading / sampling
# =============================================================================


def _config_from_dict(raw: Dict[str, object], device: str):
    """Build an ExperimentConfig from a raw dict, ignoring unknown keys."""
    from ads_cft.config import ExperimentConfig

    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    filtered = {k: v for k, v in raw.items() if k in fields}
    if "deltas" in filtered and isinstance(filtered["deltas"], list):
        filtered["deltas"] = tuple(filtered["deltas"])
    filtered["device"] = device
    return ExperimentConfig(**filtered)


def load_model(
    checkpoint_path: Path,
    config_source: Dict[str, object],
    device: str,
):
    """Rebuild the model exactly as train.py does and load the checkpoint."""
    from ads_cft.train import create_model

    cfg = _config_from_dict(config_source, device)
    model = create_model(cfg, data_shape=(1, 28, 28))

    state = torch.load(checkpoint_path, map_location="cpu")
    # Checkpoints store data normalisation alongside the state dict
    data_mean = state.pop("data_mean", None)
    data_std = state.pop("data_std", None)
    model.load_state_dict(state, strict=True)
    if data_mean is not None:
        try:
            model.data_mean = data_mean
            model.data_std = data_std
        except Exception:
            pass
    model = model.to(torch.device(device)).eval()
    return model


@torch.no_grad()
def generate_samples(
    model, n: int, device: str, seed: int, batch: int = 1000
) -> torch.Tensor:
    """Draw n samples with a fixed generator; batched, deterministic."""
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    chunks: List[torch.Tensor] = []
    remaining = n
    while remaining > 0:
        b = min(batch, remaining)
        chunks.append(model.sample(b, generator=g).detach().cpu())
        remaining -= b
    return torch.cat(chunks, dim=0)


# =============================================================================
# Real reference data
# =============================================================================


def load_real_mnist(split: str, n: int, data_root: str) -> torch.Tensor:
    """MNIST images in [-1, 1], shape (n, 1, 28, 28)."""
    import torchvision
    import torchvision.transforms as T

    tfm = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train = split == "train"
    ds = torchvision.datasets.MNIST(data_root, train=train, download=True, transform=tfm)
    if train:
        g = torch.Generator()
        g.manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    else:
        idx = list(range(min(n, len(ds))))
    imgs = torch.stack([ds[i][0] for i in idx], dim=0)
    return imgs


# =============================================================================
# Run resolution
# =============================================================================


def resolve_run(run_dir: Path) -> Tuple[str, Path]:
    """
    Locate the evaluation source inside a run directory.

    Preference order:
        samples/samples_final.pt, samples_final.pt          -> ("samples", path)
        checkpoints/ema_model.pt, ema_model.pt,
        checkpoints/best_model.pt, best_model.pt,
        checkpoints/final_model.pt, final_model.pt          -> ("checkpoint", path)
    """
    # train.py nests results one level below --output_dir as
    # <output_dir>/<auto_experiment_name>_<timestamp>/..., so search both the
    # run_dir itself and its immediate subdirectories (newest first).
    def _candidates(rel: str):
        cands = sorted(run_dir.glob(rel), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        cands += sorted(run_dir.glob(f"*/{rel}"), key=lambda p: p.stat().st_mtime,
                        reverse=True)
        return cands

    for rel in ("samples/samples_final.pt", "samples_final.pt"):
        for p in _candidates(rel):
            return "samples", p
    for rel in (
        "checkpoints/ema_model.pt", "ema_model.pt",
        "checkpoints/best_model.pt", "best_model.pt",
        "checkpoints/final_model.pt", "final_model.pt",
    ):
        for p in _candidates(rel):
            return "checkpoint", p
    raise FileNotFoundError(
        f"No samples_final.pt or model checkpoint found under {run_dir}. "
        f"If this seed's weights were deleted (SAVE_ALL_WEIGHTS=false in the "
        f"original scripts), re-train this seed — the updated train.py will "
        f"then persist samples_final.pt so this never recurs."
    )


def find_config_for_run(run_dir: Path, preset: Optional[str]) -> Dict[str, object]:
    """config.json inside the run dir (or any single subdir), else a preset."""
    candidates = [run_dir / "config.json"] + sorted(run_dir.glob("*/config.json"))
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    if preset is not None:
        if preset not in EXPERIMENT_PRESETS:
            raise KeyError(
                f"Unknown preset '{preset}'. Known: {sorted(EXPERIMENT_PRESETS)}"
            )
        return dict(EXPERIMENT_PRESETS[preset])
    # Infer from directory naming (results_dir/<experiment>/seed_k)
    for name in (run_dir.name, run_dir.parent.name):
        if name in EXPERIMENT_PRESETS:
            return dict(EXPERIMENT_PRESETS[name])
    raise FileNotFoundError(
        f"No config.json under {run_dir} and the directory name does not match a "
        f"known preset {sorted(EXPERIMENT_PRESETS)}; pass --preset explicitly."
    )


# =============================================================================
# Main evaluation
# =============================================================================


def evaluate_one(
    source_kind: str,
    source_path: Path,
    config_source: Optional[Dict[str, object]],
    real_images: torch.Tensor,
    args,
) -> Dict[str, float]:
    from ads_cft.metrics_mnist import compute_all_mnist_metrics

    if source_kind == "samples":
        gen = torch.load(source_path, map_location="cpu")
        if not isinstance(gen, torch.Tensor):
            raise TypeError(f"{source_path} does not contain a tensor")
    else:
        model = load_model(source_path, config_source, args.device)
        gen = generate_samples(
            model, n=args.n_gen, device=args.device,
            seed=args.gen_seed, batch=args.gen_batch,
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    gen = gen.float()
    if gen.ndim == 3:
        gen = gen.unsqueeze(1)
    n = min(args.n_gen, gen.shape[0])
    gen = gen[:n]

    metrics = compute_all_mnist_metrics(
        real_images, gen,
        device=torch.device(args.device),
        classifier_path=args.classifier_path,
        data_root=args.data_root,
        classifier_epochs=args.classifier_epochs,
        k_nn=args.k_nn,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        include_inception=not args.skip_inception,
    )
    metrics["source_kind"] = source_kind
    metrics["source_path"] = str(source_path)
    metrics["gen_seed"] = float(args.gen_seed) if source_kind == "checkpoint" else -1.0
    metrics["gen_batch"] = float(args.gen_batch) if source_kind == "checkpoint" else -1.0
    metrics["real_split"] = args.real
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("evaluation sources (choose one style)")
    src.add_argument("--results_dir", type=str, default=None,
                     help="Directory laid out as <exp_name>/seed_k (run_mnist_experiments.sh layout)")
    src.add_argument("--run_dir", action="append", default=[],
                     help="Individual run directory (repeatable)")
    src.add_argument("--samples", action="append", default=[],
                     help="Path to a generated-samples .pt tensor (repeatable)")
    src.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a model checkpoint (use with --preset)")
    src.add_argument("--preset", type=str, default=None,
                     choices=sorted(EXPERIMENT_PRESETS),
                     help="Training configuration preset for --checkpoint / config-less runs")
    src.add_argument("--label", type=str, default=None,
                     help="Experiment label for --run_dir/--samples/--checkpoint sources")

    parser.add_argument("--real", type=str, default="test", choices=["test", "train"],
                        help="Real reference split (default: the held-out 10k test set)")
    parser.add_argument("--n_gen", type=int, default=10000)
    parser.add_argument("--gen_seed", type=int, default=12345)
    parser.add_argument("--gen_batch", type=int, default=1000)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--classifier_path", type=str, default="mnist_classifier.pt")
    parser.add_argument("--classifier_epochs", type=int, default=3)
    parser.add_argument("--k_nn", type=int, default=3)
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--skip_inception", action="store_true",
                        help="Skip InceptionV3 metrics (e.g. no network for weight download)")
    parser.add_argument("--out", type=str, default="extended_metrics_summary.json")
    args = parser.parse_args()

    # ---- collect (label, run_dir_or_None, source_kind, source_path, config) --
    jobs: List[Tuple[str, Optional[Path], str, Path, Optional[Dict]]] = []

    if args.results_dir:
        root = Path(args.results_dir)
        if not root.exists():
            sys.exit(f"results_dir not found: {root}")
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            seed_dirs = sorted(exp_dir.glob("seed_*")) or [exp_dir]
            for sd in seed_dirs:
                try:
                    kind, path = resolve_run(sd)
                except FileNotFoundError as e:
                    print(f"[SKIP] {e}", flush=True)
                    continue
                cfg = None
                if kind == "checkpoint":
                    cfg = find_config_for_run(sd, preset=exp_dir.name
                                              if exp_dir.name in EXPERIMENT_PRESETS
                                              else args.preset)
                jobs.append((exp_dir.name, sd, kind, path, cfg))

    for rd in args.run_dir:
        rdp = Path(rd)
        kind, path = resolve_run(rdp)
        cfg = None
        if kind == "checkpoint":
            cfg = find_config_for_run(rdp, preset=args.preset)
        jobs.append((args.label or rdp.parent.name or rdp.name, rdp, kind, path, cfg))

    for sp in args.samples:
        jobs.append((args.label or Path(sp).stem, None, "samples", Path(sp), None))

    if args.checkpoint:
        if args.preset is None:
            sys.exit("--checkpoint requires --preset (training configuration).")
        jobs.append((args.label or args.preset, None, "checkpoint",
                     Path(args.checkpoint), dict(EXPERIMENT_PRESETS[args.preset])))

    if not jobs:
        sys.exit("No evaluation sources given. See --help.")

    # ---- real reference ------------------------------------------------------
    print(f"[REAL] Loading MNIST {args.real} split ({args.n_gen} images, "
          f"normalised to [-1, 1]) ...", flush=True)
    real_images = load_real_mnist(args.real, args.n_gen, args.data_root)

    # ---- evaluate ------------------------------------------------------------
    per_run: List[Dict[str, object]] = []
    for label, run_dir, kind, path, cfg in jobs:
        print(f"\n[EVAL] {label}  <-  {path}  ({kind})", flush=True)
        m = evaluate_one(kind, path, cfg, real_images, args)
        m["experiment"] = label
        m["run_dir"] = str(run_dir) if run_dir is not None else ""
        per_run.append(m)
        out_dir = run_dir if run_dir is not None else path.parent
        try:
            with open(Path(out_dir) / "extended_metrics.json", "w") as f:
                json.dump(m, f, indent=2)
        except Exception as e:  # read-only source locations etc.
            print(f"[WARN] Could not write per-run JSON next to source: {e}")

    # ---- aggregate -----------------------------------------------------------
    metric_keys = [
        "fid_classifier", "kid_classifier_mean", "precision", "recall",
        "fid_inception", "kid_inception_mean",
    ]
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for m in per_run:
        grouped[str(m["experiment"])].append(m)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for exp, runs in grouped.items():
        summary[exp] = {}
        for k in metric_keys:
            vals = [float(r[k]) for r in runs if k in r]
            if vals:
                summary[exp][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_seeds": len(vals),
                    "values": vals,
                }

    payload = {"per_run": per_run, "summary": summary,
               "real_split": args.real, "n_gen": args.n_gen}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OUT] Wrote {args.out}")

    # ---- table ---------------------------------------------------------------
    def fmt(exp: str, key: str, scale: float = 1.0, prec: int = 3) -> str:
        s = summary.get(exp, {}).get(key)
        if s is None:
            return "--"
        return f"{s['mean'] * scale:.{prec}f}±{s['std'] * scale:.{prec}f}"

    print("\n===================== EXTENDED MNIST METRICS (mean ± std over seeds) =====================")
    header = (f"{'experiment':<16} {'FID_clf':>14} {'KIDx1e3_clf':>14} "
              f"{'Precision':>12} {'Recall':>12} {'FID_Inc':>14} {'KIDx1e3_Inc':>14}")
    print(header)
    print("-" * len(header))
    for exp in sorted(summary):
        print(f"{exp:<16} {fmt(exp, 'fid_classifier'):>14} "
              f"{fmt(exp, 'kid_classifier_mean', 1e3):>14} "
              f"{fmt(exp, 'precision'):>12} {fmt(exp, 'recall'):>12} "
              f"{fmt(exp, 'fid_inception', 1.0, 2):>14} "
              f"{fmt(exp, 'kid_inception_mean', 1e3):>14}")

    print("\nLaTeX rows (Model & FID_clf & KID_clf x 10^3 & Precision & Recall \\\\):")
    for exp in sorted(summary):
        row = " & ".join([
            exp.replace("_", r"\_"),
            fmt(exp, "fid_classifier").replace("±", r" $\pm$ "),
            fmt(exp, "kid_classifier_mean", 1e3).replace("±", r" $\pm$ "),
            fmt(exp, "precision").replace("±", r" $\pm$ "),
            fmt(exp, "recall").replace("±", r" $\pm$ "),
        ])
        print(f"  {row} \\\\")


if __name__ == "__main__":
    main()
