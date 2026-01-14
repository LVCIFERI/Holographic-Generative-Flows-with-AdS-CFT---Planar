#!/bin/bash
# =============================================================================
# run_delta_sweep.sh
#
# Delta (conformal dimension) sweep for planar AdS on checkerboard.
# Tests Δ = 1.5, 2.0, 2.5, 3.0
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

# Delta values to sweep
DELTA_VALUES=(1.5 2.0 2.5 3.0)

# Output directory
OUTPUT_BASE="./outputs/delta_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

echo "============================================"
echo "DELTA SWEEP: Planar AdS on Checkerboard"
echo "============================================"
echo "Deltas: ${DELTA_VALUES[*]}"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_BASE"
echo "============================================"

for delta in "${DELTA_VALUES[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Running: Delta = $delta"
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
        --deltas $delta \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_train_samples $N_TRAIN \
        --n_viz_real $N_VIZ \
        --n_viz_gen $N_VIZ \
        --output_dir "${OUTPUT_BASE}/delta_${delta}" \
        --name "planar_checkerboard_delta_${delta}"
    
    echo "Completed: Delta = $delta"
done

echo ""
echo "============================================"
echo "DELTA SWEEP COMPLETED"
echo "============================================"
echo "Results saved to: $OUTPUT_BASE"
