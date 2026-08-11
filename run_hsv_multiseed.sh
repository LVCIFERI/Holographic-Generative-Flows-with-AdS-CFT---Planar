#!/bin/bash
# =============================================================================
# run_hsv_multiseed.sh
#
# Multi-seed rerun of the hyperscaling-violating (HSV) geometry scan.
#
# Referee 1: "the Delta and HSV scans should be reported over multiple seeds
# with uncertainties, or clearly labeled as exploratory single-run studies."
#
# Reruns the ORIGINAL run_hsv_experiments.sh configuration verbatim
# (planar_hsv geometry, u-bounds with u_uv = e^{-1} = 0.3678794, hermite path,
# spectral 16 modes, cnn 64x3, checkerboard, 100 epochs) for
# p in {0.1, 0.25, 0.5, 1.0}, over SEEDS (default: 1 2 3), and aggregates
# BV / WED_norm / JS_cell / CQS as mean +/- std per p.
#
# Usage:
#   ./run_hsv_multiseed.sh
#   SEEDS="42 1 2" ./run_hsv_multiseed.sh    # include the original seed 42
#
# Cost: 4 p-values x 3 seeds = 12 checkerboard runs ~ 4 GPU-hours on an A100.
# =============================================================================

set -eo pipefail

DATASET=${DATASET:-checkerboard}
EPOCHS=${EPOCHS:-100}
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000
SEEDS=(${SEEDS:-1 2 3})
PLANAR_P_VALUES=(0.1 0.25 0.5 1.0)

OUTPUT_BASE=${OUTPUT_BASE:-"outputs/hsv_multiseed_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_BASE"

echo "============================================"
echo "HSV GEOMETRY SCAN (multi-seed)"
echo "============================================"
echo "p values: ${PLANAR_P_VALUES[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Epochs:   $EPOCHS   Dataset: $DATASET"
echo "Output:   $OUTPUT_BASE"
echo "============================================"

aggregate_p() {
    local p_dir=$1
    python3 << EOF
import json
import numpy as np
from pathlib import Path

p_dir = Path("$p_dir")
all_metrics = []
for seed_dir in sorted(p_dir.glob("seed_*")):
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

with open(p_dir / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for key in ["BV", "WED_norm", "JS_cell", "CQS"]:
    if key in summary:
        s = summary[key]
        print(f"    {key}: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_seeds']})")
EOF
}

for P in "${PLANAR_P_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        seed_dir="${OUTPUT_BASE}/planar_hsv_p${P}/seed_${seed}"
        mkdir -p "$seed_dir"
        echo ""
        echo "----------------------------------------------"
        echo "Running: planar_hsv, p = $P, seed = $seed"
        echo "----------------------------------------------"
        CMD=(python train.py \
            --dataset "$DATASET" \
            --slice_geometry planar_hsv \
            --hsv_p "$P" \
            --hsv_use_u_bounds \
            --hsv_u_uv 0.3678794 \
            --path_type hermite \
            --lift_noise_sigma ${LIFT_NOISE_SIGMA:-0.1} \
            --use_spectral_encoding \
            --spectral_n_modes $SPECTRAL_MODES \
            --cnn_hidden $HIDDEN \
            --cnn_depth $DEPTH \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --n_train $N_TRAIN \
            --n_viz_gen $N_VIZ \
            --output_dir "$seed_dir" \
            --seed "$seed")
        printf '%q ' "${CMD[@]}" > "${seed_dir}/run_command.txt"; echo "" >> "${seed_dir}/run_command.txt"
        git rev-parse HEAD > "${seed_dir}/git_commit.txt" 2>/dev/null || true
        "${CMD[@]}" 2>&1 | tee "${seed_dir}/train.log"
    done
    echo ""
    echo "  Aggregating p = $P ..."
    aggregate_p "${OUTPUT_BASE}/planar_hsv_p${P}"
done

echo ""
echo "============================================"
echo "HSV SCAN (multi-seed) COMPLETED"
echo "Per-p metrics_summary.json written under: $OUTPUT_BASE/planar_hsv_p*/"
echo "Report BV and WED_norm as mean +/- std over seeds, and phrase the"
echo "conclusion as: the AdS model outperforms every TESTED HSV model"
echo "(referee request: no continuous-limit claim from four p values)."
echo "============================================"
