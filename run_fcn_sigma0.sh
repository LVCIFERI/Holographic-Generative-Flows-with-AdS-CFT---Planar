#!/bin/bash
# =============================================================================
# run_fcn_sigma0.sh
#
# FCN (MLP) baseline at lift_noise_sigma = 0: the one missing cell of the
# restated sigma=0 main checkerboard table.
#
# Flags are VERBATIM the published mlp_baseline_linear block from
# run_fair_experiments.sh (10.6M-param MLP, hidden 1442, depth 6), with only
# the lift noise changed to 0 and periodic checkpoints disabled.
#
# Resumable: seeds that already have a final_metrics.json are skipped, so the
# script can be re-run safely after a crash.
#
# Usage (from the repo root):
#   ./run_fcn_sigma0.sh
#   RESULTS_DIR=results_referee_20260807_073359 N_SEEDS=3 ./run_fcn_sigma0.sh
#
# Cost: 3 seeds x 100 epochs ~ 1 GPU-hour.
# =============================================================================

set -eo pipefail

RESULTS_DIR=${RESULTS_DIR:-"results_referee_20260807_073359"}
N_SEEDS=${N_SEEDS:-3}
SIGMA=${SIGMA:-0.0}

# Published FCN settings (verbatim from run_fair_experiments.sh)
EPOCHS=${EPOCHS:-100}
MLP_HIDDEN=1442
MLP_DEPTH=6
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000

STAG=$(echo "$SIGMA" | tr '.' 'p')
OUT_BASE="${RESULTS_DIR}/lift_noise/fcn_sigma${STAG}"
mkdir -p "$OUT_BASE"

echo "============================================================"
echo "FCN BASELINE @ sigma=${SIGMA}  ->  ${OUT_BASE}   (${N_SEEDS} seeds)"
echo "============================================================"

seed_done() {
    compgen -G "$1/final_metrics.json" > /dev/null 2>&1 && return 0
    compgen -G "$1/*/final_metrics.json" > /dev/null 2>&1 && return 0
    return 1
}

for seed in $(seq 1 $N_SEEDS); do
    seed_dir="${OUT_BASE}/seed_${seed}"
    if seed_done "$seed_dir"; then
        echo "  Seed ${seed}: already complete -> skipping"
        continue
    fi
    mkdir -p "$seed_dir"
    # remove partial run dirs from a previous crash
    for d in "$seed_dir"/*/; do
        [ -d "$d" ] && [ ! -f "${d}final_metrics.json" ] && \
            { echo "  Removing partial: $d"; rm -rf "$d"; }
    done
    echo ""
    echo "  Seed ${seed}/${N_SEEDS} -> ${seed_dir}"
    CMD=(python train.py \
        --dataset checkerboard \
        --slice_geometry planar \
        --path_type linear \
        --backbone_scale 0.0 \
        --deltas 1.5 \
        --r_ir 0.0 \
        --r_uv 1.0 \
        --ode_solver rk4 \
        --ode_n_steps 100 \
        --lift_noise_sigma $SIGMA \
        --residual_hidden $MLP_HIDDEN \
        --residual_depth $MLP_DEPTH \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_train_samples $N_TRAIN \
        --n_viz_real $N_VIZ \
        --n_viz_gen $N_VIZ \
        --use_ema \
        --checkpoint_every 999999999 \
        --output_dir "$seed_dir" \
        --name "fcn_baseline_sigma${STAG}_seed${seed}" \
        --seed $seed)
    printf '%q ' "${CMD[@]}" > "${seed_dir}/run_command.txt"; echo "" >> "${seed_dir}/run_command.txt"
    git rev-parse HEAD > "${seed_dir}/git_commit.txt" 2>/dev/null || true
    "${CMD[@]}" 2>&1 | tee "${seed_dir}/train.log"
done

echo ""
echo "  Aggregating ..."
python3 << EOF
import json
import numpy as np
from pathlib import Path

out_base = Path("$OUT_BASE")
n_seeds = $N_SEEDS

all_metrics = []
for seed in range(1, n_seeds + 1):
    seed_dir = out_base / f"seed_{seed}"
    candidates = sorted(seed_dir.glob("final_metrics.json")) \
               + sorted(seed_dir.glob("*/final_metrics.json"))
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            all_metrics.append(json.load(f))
    else:
        print(f"  Warning: no final_metrics.json under {seed_dir}")

if not all_metrics:
    raise SystemExit("  No metrics found to aggregate")

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

with open(out_base / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Key metrics (mean +/- std over seeds):")
for key in ["BV", "WED", "WED_norm", "JS_cell", "CQS", "gen_time"]:
    if key in summary:
        s = summary[key]
        print(f"    {key}: {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
EOF

echo ""
echo "Done -> ${OUT_BASE}/metrics_summary.json"
