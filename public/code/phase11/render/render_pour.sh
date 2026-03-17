#!/bin/bash
# Batch render water pour frames with sx-engine
RENDERER="/c/Users/djmax/Desktop/work/fluid/sx_renderer/sx_renderer.exe"
MESH_DIR="/c/Users/djmax/Desktop/work/fluid/phases/phase11/export/water_pour/meshes"
OUT_DIR="/c/Users/djmax/Desktop/work/fluid/phases/phase11/export/water_pour/renders"
mkdir -p "$OUT_DIR"

# Camera looking down at the water from an angle
# Domain is 2m cube, center at (1,1,1)
CAM_POS="2.5 2.5 2.0"
CAM_TARGET="1.0 1.0 0.3"

for i in $(seq 0 249); do
    FRAME=$(printf "%06d" $i)
    PLY="$MESH_DIR/water_${FRAME}.ply"
    OUT="$OUT_DIR/frame_${FRAME}.png"
    if [ -f "$PLY" ] && [ ! -f "$OUT" ]; then
        "$RENDERER" --water "$PLY" \
            --output "$OUT" \
            --samples 32 --width 1280 --height 720 \
            --cam_pos $CAM_POS --cam_target $CAM_TARGET \
            --fov 45 --ground_z 0.0 \
            --water-preset pool \
            --exposure 1.2 2>/dev/null
        if [ $((i % 25)) -eq 0 ]; then
            echo "Rendered frame $i/249"
        fi
    fi
done
echo "Done rendering!"
