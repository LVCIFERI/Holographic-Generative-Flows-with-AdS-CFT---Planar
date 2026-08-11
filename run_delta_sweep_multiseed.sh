#!/bin/bash
# =============================================================================
# run_delta_sweep_multiseed.sh
#
# Multi-seed rerun of the conformal-dimension (Delta) scan.
#
# Referee 1: "the Delta and HSV scans should be reported over multiple seeds
# with uncertainties, or clearly labeled as exploratory single-run studies."
#
# This script reruns the ORIGINAL run_delta_sweep.sh configuration verbatim
# (planar AdS, hermite path, checkerboard, 100 epochs, spectral 16 modes,
# residual 64x3, EMA, default lift noise 0.1) for Delta in {1.5, 2.0, 2.5,
# 3.0}, but over SEEDS (default: 1 2 3), and aggregates BV / WED_norm /
# JS_cell / CQS as mean +/- std per Delta.
#
# Usage:
#   ./run_delta_sweep_multiseed.sh
#   SEEDS="42 1 2" ./run_delta_sweep_multiseed.sh   # include the original seed
#
# Cost: 4 Deltas x 3 seeds = 12 checkerboard runs ~ 4 GPU-hours on an A100.
# =============================================================================

set -eo pipefail

# Hyperparameters (VERBATIM from run_delta_sweep.sh)
EPOCHS=${EPOCHS:-100}
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000

DELTA_VALUES=(1.5 2.0 2.5 3.0)
SEEDS=(${SEEDS:-1 2 3})

OUTPUT_BASE=${OUTPUT_BASE:-"./outputs/delta_sweep_multiseed_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_BASE"

echo "============================================"
echo "DELTA SWEEP (multi-seed): Planar AdS, checkerboard"
echo "============================================"
echo "Deltas: ${DELTA_VALUES[*]}"
echo "Seeds:  ${SEEDS[*]}"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_BASE"
echo "============================================"

aggregate_delta() {
    local delta_dir=$1
    python3 << EOF
import json
import numpy as np
from pathlib import Path

delta_dir = Path("$delta_dir")
all_metrics = []
for seed_dir in sorted(delta_dir.glob("seed_*")):
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
    if numeric:
        summary[key] = {"mean": float(np.mean(numeric)),
                        "std": float(np.std(numeric)),
                        "values": numeric, "n_seeds": len(numeric)}

with open(delta_dir / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for key in ["BV", "WED_norm", "JS_cell", "CQS"]:
    if key in summary:
        s = summary[key]
        print(f"    {key}: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_seeds']})")
EOF
}

for delta in "${DELTA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        seed_dir="${OUTPUT_BASE}/delta_${delta}/seed_${seed}"
        mkdir -p "$seed_dir"
        echo ""
        echo "----------------------------------------------"
        echo "Running: Delta = $delta, seed = $seed"
        echo "----------------------------------------------"
        CMD=(python train.py \
            --dataset checkerboard \
            --slice_geometry planar \
            --path_type hermite \
            --lift_noise_sigma ${LIFT_NOISE_SIGMA:-0.1} \
            --use_ema \
            --use_spectral_encoding \
            --spectral_n_modes $SPECTRAL_MODES \
            --residual_depth $DEPTH \
            --residual_hidden $HIDDEN \
            --deltas $delta \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --n_train_samples $N_TRAIN \
            --n_viz_real $N_VIZ \
            --n_viz_gen $N_VIZ \
            --output_dir "$seed_dir" \
            --name "planar_checkerboard_delta_${delta}_seed${seed}" \
            --seed $seed)
        printf '%q ' "${CMD[@]}" > "${seed_dir}/run_command.txt"; echo "" >> "${seed_dir}/run_command.txt"
        git rev-parse HEAD > "${seed_dir}/git_commit.txt" 2>/dev/null || true
        "${CMD[@]}" 2>&1 | tee "${seed_dir}/train.log"
    done
    echo ""
    echo "  Aggregating Delta = $delta ..."
    aggregate_delta "${OUTPUT_BASE}/delta_${delta}"
done

echo ""
echo "============================================"
echo "DELTA SWEEP (multi-seed) COMPLETED"
echo "Per-Delta metrics_summary.json written under: $OUTPUT_BASE/delta_*/"
echo "Report BV and WED_norm as mean +/- std over seeds (referee request)."
echo "============================================"
