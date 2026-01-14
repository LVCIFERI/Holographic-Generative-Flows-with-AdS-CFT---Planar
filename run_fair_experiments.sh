#!/bin/bash
# =============================================================================
# run_fair_comparisons.sh
#
# Fair comparison experiments:
#   1. AdS + Planar + Hermite (full model)
#   2. AdS + Planar + Linear (full model)
#   3. No KG + Linear (backbone_scale=0, spectral encoding)
#   4. MLP Baseline + Linear (no spectral encoding, raw points)
#
# Usage:
#   ./run_fair_comparisons.sh              # Single seed (default)
#   N_SEEDS=3 ./run_fair_comparisons.sh    # 3 seeds per experiment
#
# =============================================================================

set -e

# Hyperparameters (matching run_experiments.sh defaults)
EPOCHS=100
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
MLP_HIDDEN=1442
MLP_DEPTH=6
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000
DATASET="checkerboard"

# Physics parameters
DELTAS="1.5"
R_IR=0.0
R_UV=1.0

# ODE integration
ODE_SOLVER="rk4"
ODE_N_STEPS=100

# UV lift noise
LIFT_NOISE_SIGMA=0.1

# Seed configuration
N_SEEDS=${N_SEEDS:-1}
SAVE_ALL_WEIGHTS=${SAVE_ALL_WEIGHTS:-false}

# =============================================================================
# Multi-Seed Functions
# =============================================================================

aggregate_seed_metrics() {
    local output_dir=$1
    local n_seeds=$2
    
    python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("$output_dir")
n_seeds = $n_seeds

# Collect metrics from each seed
all_metrics = []
for seed in range(1, n_seeds + 1):
    seed_dir = output_dir / f"seed_{seed}"
    metrics_file = seed_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            all_metrics.append(json.load(f))

if not all_metrics:
    print("  No metrics found to aggregate")
    exit(0)

# Compute mean and std for each metric
summary = {}
metric_keys = set()
for m in all_metrics:
    metric_keys.update(m.keys())

for key in metric_keys:
    values = [m.get(key) for m in all_metrics if key in m]
    numeric_values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric_values:
        summary[key] = {
            "mean": float(np.mean(numeric_values)),
            "std": float(np.std(numeric_values)),
            "values": numeric_values,
            "n_seeds": len(numeric_values)
        }

summary_file = output_dir / "metrics_summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved metrics summary to {summary_file}")
EOF
}

run_with_seeds() {
    local exp_name=$1
    local output_dir=$2
    shift 2
    local train_cmd="$@"
    
    if [[ $N_SEEDS -eq 1 ]]; then
        echo "Running single seed experiment..."
        $train_cmd --output_dir "$output_dir" --name "$exp_name"
    else
        echo "Running $N_SEEDS seeds..."
        
        for seed in $(seq 1 $N_SEEDS); do
            local seed_dir="${output_dir}/seed_${seed}"
            echo ""
            echo "  Seed $seed/$N_SEEDS -> $seed_dir"
            
            $train_cmd --output_dir "$seed_dir" --name "${exp_name}_seed${seed}" --seed $seed
            
            if [[ "$SAVE_ALL_WEIGHTS" == "false" && $seed -gt 1 ]]; then
                echo "  Removing weights from seed $seed..."
                rm -f "${seed_dir}/final_model.pt" "${seed_dir}/best_model.pt" "${seed_dir}/ema_model.pt"
            fi
        done
        
        echo ""
        echo "  Aggregating metrics..."
        aggregate_seed_metrics "$output_dir" $N_SEEDS
    fi
}

# =============================================================================
# Main
# =============================================================================

# Results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_fair_comparison_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "FAIR COMPARISON EXPERIMENTS"
echo "============================================"
echo "Results: $RESULTS_DIR"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Seeds: $N_SEEDS"
echo ""
echo "Physics parameters:"
echo "  Deltas: $DELTAS"
echo "  Radial: r_ir=$R_IR, r_uv=$R_UV"
echo "  ODE: $ODE_SOLVER, $ODE_N_STEPS steps"
echo "  Lift noise: $LIFT_NOISE_SIGMA"
echo ""

# -------------------------------------------------------------------------
# 1. AdS + Planar + Hermite (full KG backbone)
# -------------------------------------------------------------------------
echo "[1/4] AdS + Planar + Hermite"
run_with_seeds "ads_planar_hermite" "${RESULTS_DIR}/ads_planar_hermite" \
    python train.py \
    --dataset "$DATASET" \
    --slice_geometry planar \
    --path_type hermite \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_spectral_encoding \
    --spectral_n_modes $SPECTRAL_MODES \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --use_ema

echo "Completed: AdS + Planar + Hermite"
echo ""

# -------------------------------------------------------------------------
# 2. AdS + Planar + Linear (full KG backbone)
# -------------------------------------------------------------------------
echo "[2/4] AdS + Planar + Linear"
run_with_seeds "ads_planar_linear" "${RESULTS_DIR}/ads_planar_linear" \
    python train.py \
    --dataset "$DATASET" \
    --slice_geometry planar \
    --path_type linear \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_spectral_encoding \
    --spectral_n_modes $SPECTRAL_MODES \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --use_ema

echo "Completed: AdS + Planar + Linear"
echo ""

# -------------------------------------------------------------------------
# 3. No KG + Linear (backbone_scale=0, spectral encoding)
#    Fair comparison: same encoding/network, no physics backbone
# -------------------------------------------------------------------------
echo "[3/4] No KG + Linear (spectral baseline)"
run_with_seeds "no_kg_linear" "${RESULTS_DIR}/no_kg_linear" \
    python train.py \
    --dataset "$DATASET" \
    --slice_geometry planar \
    --path_type linear \
    --backbone_scale 0.0 \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_spectral_encoding \
    --spectral_n_modes $SPECTRAL_MODES \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --use_ema

echo "Completed: No KG + Linear"
echo ""

# -------------------------------------------------------------------------
# 4. MLP Baseline + Linear (no spectral encoding, raw point data)
#    Pure neural ODE baseline
# -------------------------------------------------------------------------
echo "[4/4] MLP Baseline + Linear"
run_with_seeds "mlp_baseline_linear" "${RESULTS_DIR}/mlp_baseline_linear" \
    python train.py \
    --dataset "$DATASET" \
    --slice_geometry planar \
    --path_type linear \
    --backbone_scale 0.0 \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --residual_hidden $MLP_HIDDEN \
    --residual_depth $MLP_DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --use_ema

echo "Completed: MLP Baseline + Linear"
echo ""

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo "============================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================"
echo "Results saved to: $RESULTS_DIR"
echo "Seeds per experiment: $N_SEEDS"
echo ""
echo "Experiments:"
echo "  1. ads_planar_hermite   - Full AdS model with Hermite path"
echo "  2. ads_planar_linear    - Full AdS model with Linear path"
echo "  3. no_kg_linear         - Spectral encoding, no KG backbone"
echo "  4. mlp_baseline_linear  - Raw points, MLP only"
