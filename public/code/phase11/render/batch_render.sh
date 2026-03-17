#!/bin/bash
# Batch render 300 frames on 4 GPUs
# Usage: bash batch_render.sh <meshes_dir> <output_dir> [samples]
cd /data/yinshaoxuan/fluid/phases/phase11

MESHES=${1:-./export/flood/meshes_v1}
OUTPUT=${2:-./renders_v01}
SAMPLES=${3:-128}
META=${4:-./export/flood/meta.json}
RENDER_SCRIPT=${5:-blender_render.py}
BLENDER=/data/yinshaoxuan/blender-5.0.1-linux-x64/blender

mkdir -p "$OUTPUT"

echo "Rendering 300 frames from $MESHES -> $OUTPUT (${SAMPLES} samples, 4 GPUs)"
t0=$(date +%s)

# Split 300 frames across 4 GPUs
for gpu in 0 1 2 3; do
    start=$((gpu * 75))
    end=$((start + 74))
    # Build comma-separated frame list
    frames=""
    for f in $(seq $start $end); do
        if [ -n "$frames" ]; then frames="${frames},"; fi
        frames="${frames}${f}"
    done
    echo "  GPU $gpu: frames $start-$end"
    CUDA_VISIBLE_DEVICES=$gpu $BLENDER --background --python "$RENDER_SCRIPT" -- \
        --meshes "$MESHES" --meta "$META" \
        --frame "$frames" --output "$OUTPUT" --samples $SAMPLES --gpus 1 \
        > "${OUTPUT}/log_gpu${gpu}.txt" 2>&1 &
done

echo "Waiting for all 4 GPUs to finish..."
wait

now=$(date +%s)
elapsed=$((now - t0))

# Count rendered frames
n_rendered=$(ls -1 "${OUTPUT}"/frame_*.png 2>/dev/null | wc -l)
echo "Done! $n_rendered/300 frames rendered in ${elapsed}s"
