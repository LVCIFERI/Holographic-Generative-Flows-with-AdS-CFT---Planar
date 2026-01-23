#!/bin/bash
# =============================================================================
# run_rcutoff_test.sh
#
# r_uv & r_ir (cutoff) sweep for planar AdS on checkerboard.
# Tests r_ir = 0, r_uv = 2.0, 5.0, 10.0
# =============================================================================

set -e

# Hyperparameters (matching run_experiments.sh)
EPOCHS=100
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000

# rUV values to sweep
RUV=(2.0 5.0 10.0)
# rIR values to sweep
RIR=(0.0 0.0 0.0)

# Output directory
OUTPUT_BASE="./outputs/dr_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

echo "============================================"
echo "R_UV/R_IR SWEEP: Planar AdS on Checkerboard"
echo "============================================"
echo "r_uv: ${RUV[*]}"
echo "r_ir: ${RIR[*]}"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_BASE"
echo "============================================"

for i in "${!RUV[@]}"; do
    r_uv="${RUV[$i]}"
    r_ir="${RIR[$i]}"

    echo ""
    echo "----------------------------------------------"
    echo "Running: r_uv = $r_uv, r_ir = $r_ir"
    echo "----------------------------------------------"

    python train.py \
        --dataset checkerboard \
        --slice_geometry planar \
        --path_type hermite \
        --use_ema \
        --use_spectral_encoding \
        --spectral_n_modes $SPECTRAL_MODES \
        --residual_depth $DEPTH \
        --residual_hidden $HIDDEN \
        --r_uv="$r_uv" \
        --r_ir="$r_ir" \
        --delta 1.5 \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_train_samples $N_TRAIN \
        --n_viz_real $N_VIZ \
        --n_viz_gen $N_VIZ \
        --output_dir "${OUTPUT_BASE}/ruv_${r_uv}_rir_${r_ir}" \
        --name "planar_checkerboard_ruv_${r_uv}_rir_${r_ir}"

    echo "Completed: r_uv = $r_uv, r_ir = $r_ir"
done

echo ""
echo "============================================"
echo "R_UV/R_IR SWEEP COMPLETED"
echo "============================================"
echo "Results saved to: $OUTPUT_BASE"
