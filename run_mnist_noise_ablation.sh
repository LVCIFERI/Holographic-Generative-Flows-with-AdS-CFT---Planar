#!/bin/bash
# =============================================================================
# run_mnist_noise_ablation.sh
#
# OPTIONAL: MNIST arm of the lift-noise ablation (Referee 1, implementation
# details: "quantify how the reported results change when the lift noise is
# varied or set to zero").
#
# The checkerboard grid in run_referee_experiments.sh is the primary
# deliverable for this request. This script adds the MNIST spot-check at
# sigma = 0 for the two image models where Pi-tilde matters most:
#   * ads_hermite  (Hermite endpoint slopes consume Pi-tilde directly ->
#                   the single most informative sigma = 0 run)
#   * ads_linear
# at the FULL published budget (1500 epochs, batch 128, 10k train, 3 seeds).
# sigma = 0.1 is the published setting; reuse the existing 3-seed numbers.
#
# COST WARNING: 2 models x 3 seeds x 1500 epochs. Depending on your GPU this
# is roughly 2-4 GPU-days total. Trim with:
#   MODELS="hermite" N_SEEDS=3 ./run_mnist_noise_ablation.sh
#   SIGMAS="0.0"                                     (default)
#
# Every run saves config.json, git_commit.txt, run_command.txt, train.log and
# samples/samples_final.pt, so the extended MNIST metrics
# (evaluate_mnist_extended.py) can be computed afterwards with no retraining.
# =============================================================================

set -eo pipefail

# Hyperparameters (VERBATIM from run_mnist_experiments.sh)
EPOCHS=${EPOCHS:-1500}
SPECTRAL_MODES=28
HIDDEN=64
DEPTH=3
BATCH_SIZE=128
N_TRAIN=10000
N_VIZ=10000
VIZ_EVERY=50
DATASET="mnist"
DELTAS="1.5"
R_IR=0.0
R_UV=1.0
ODE_SOLVER="rk4"
ODE_N_STEPS=100

N_SEEDS=${N_SEEDS:-3}
SIGMAS=${SIGMAS:-"0.0"}
MODELS=${MODELS:-"hermite linear"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=${RESULTS_DIR:-"results_mnist_noise_${TIMESTAMP}"}
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "MNIST LIFT-NOISE ABLATION (optional, full 1500-epoch budget)"
echo "Results: $RESULTS_DIR   Seeds: $N_SEEDS"
echo "Sigmas:  $SIGMAS   Models: $MODELS"
echo "============================================================"

aggregate_dir() {
    local output_dir=$1
    python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("$output_dir")
all_metrics = []
for seed_dir in sorted(output_dir.glob("seed_*")):
    candidates = sorted(seed_dir.glob("final_metrics.json")) \
               + sorted(seed_dir.glob("*/final_metrics.json"))
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            all_metrics.append(json.load(f))
    else:
        print(f"  Warning: no final_metrics.json under {seed_dir}")

if not all_metrics:
    print("  No metrics found to aggregate")
    raise SystemExit(0)

summary = {}
keys = set()
for m in all_metrics:
    keys.update(m.keys())
for key in keys:
    numeric = [m[key] for m in all_metrics
               if key in m and isinstance(m[key], (int, float))
               and not isinstance(m[key], bool)]
    if key == "fid":
        numeric = [v for v in numeric if v >= 0]
    if numeric:
        summary[key] = {"mean": float(np.mean(numeric)),
                        "std": float(np.std(numeric)),
                        "values": numeric, "n_seeds": len(numeric)}

with open(output_dir / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for key in ["fid", "swd", "nll", "gen_time"]:
    if key in summary:
        s = summary[key]
        print(f"    {key}: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_seeds']})")
EOF
}

for sigma in $SIGMAS; do
    stag=$(echo "$sigma" | tr '.' 'p')
    for variant in $MODELS; do
        case "$variant" in
            hermite) PATH_TYPE="hermite" ;;
            linear)  PATH_TYPE="linear" ;;
            *) echo "Unknown model variant: $variant"; exit 1 ;;
        esac
        exp_base="${RESULTS_DIR}/ads_${variant}_sigma${stag}"
        echo ""
        echo "=== ads_${variant}, sigma = ${sigma}  (${N_SEEDS} seeds) ==="
        for seed in $(seq 1 $N_SEEDS); do
            seed_dir="${exp_base}/seed_${seed}"
            mkdir -p "$seed_dir"
            echo ""
            echo "  Seed ${seed}/${N_SEEDS} -> ${seed_dir}"
            CMD=(python train.py \
                --dataset "$DATASET" \
                --slice_geometry planar \
                --path_type "$PATH_TYPE" \
                --deltas $DELTAS \
                --r_ir $R_IR \
                --r_uv $R_UV \
                --ode_solver $ODE_SOLVER \
                --ode_n_steps $ODE_N_STEPS \
                --lift_noise_sigma $sigma \
                --use_image_encoding \
                --spectral_n_modes $SPECTRAL_MODES \
                --residual_hidden $HIDDEN \
                --residual_depth $DEPTH \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --n_train_samples $N_TRAIN \
                --n_viz_real $N_VIZ \
                --n_viz_gen $N_VIZ \
                --viz_every_epochs $VIZ_EVERY \
                --checkpoint_every 999999999 \
                --use_ema \
                --output_dir "$seed_dir" \
                --name "ads_mnist_${variant}_sigma${stag}_seed${seed}" \
                --seed $seed)
            printf '%q ' "${CMD[@]}" > "${seed_dir}/run_command.txt"; echo "" >> "${seed_dir}/run_command.txt"
            git rev-parse HEAD > "${seed_dir}/git_commit.txt" 2>/dev/null || true
            "${CMD[@]}" 2>&1 | tee "${seed_dir}/train.log"
        done
        echo ""
        echo "  Aggregating ads_${variant}, sigma = ${sigma} ..."
        aggregate_dir "$exp_base"
    done
done

echo ""
echo "============================================================"
echo "MNIST NOISE ABLATION COMPLETED -> $RESULTS_DIR"
echo "Extended metrics afterwards (no retraining needed):"
echo "  PYTHONPATH=.. python evaluate_mnist_extended.py --results_dir $RESULTS_DIR"
echo "============================================================"
