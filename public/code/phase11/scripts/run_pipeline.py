"""Master pipeline: run after V1 rendering completes.
Handles: foam bake, V2/V9/V10/V11 rendering, FLIP variants, Warp PBF, video compilation.

Usage: nohup python run_pipeline.py > pipeline.log 2>&1 &
"""
import subprocess, os, sys, time, glob

PYTHON = "/data/yinshaoxuan/miniconda3/envs/taichi/bin/python"
BLENDER = "/data/yinshaoxuan/blender-5.0.1-linux-x64/blender"
FFMPEG = "/data/yinshaoxuan/miniconda3/envs/taichi/bin/ffmpeg"
SPLASHSURF = os.path.expanduser("~/.cargo/bin/splashsurf")
BASE = "/data/yinshaoxuan/fluid/phases/phase11"


def run(cmd, timeout=7200):
    print(f"  CMD: {cmd[:120]}...", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            print(f"  ERROR (rc={p.returncode}): {p.stderr[-300:]}", flush=True)
        return p.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s", flush=True)
        return -1


def render_4gpu(meshes, output, meta, script="blender_render.py", samples=128):
    """Render 300 frames on 4 GPUs in parallel."""
    os.makedirs(output, exist_ok=True)
    procs = []
    for gpu in range(4):
        start = gpu * 75
        end = start + 74
        frames = ",".join(str(f) for f in range(start, end + 1))
        cmd = (f'CUDA_VISIBLE_DEVICES={gpu} {BLENDER} --background '
               f'--python {script} -- '
               f'--meshes {meshes} --meta {meta} '
               f'--frame {frames} --output {output} --samples {samples} --gpus 1')
        log = os.path.join(output, f"log_gpu{gpu}.txt")
        with open(log, "w") as logf:
            p = subprocess.Popen(cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"    GPU {gpu}: frames {start}-{end} (PID {p.pid})", flush=True)

    for p in procs:
        p.wait()
    n = len(glob.glob(os.path.join(output, "frame_*.png")))
    print(f"    Rendered: {n}/300 frames", flush=True)
    return n


def splashsurf_mesh(sim_dir, mesh_dir):
    """Mesh all PLY files in sim_dir with SplashSurf."""
    os.makedirs(mesh_dir, exist_ok=True)
    for i in range(300):
        f = f"{i:06d}"
        ply = os.path.join(sim_dir, f"water_{f}.ply")
        obj = os.path.join(mesh_dir, f"water_{f}.obj")
        if os.path.exists(obj) and os.path.getsize(obj) > 100:
            continue
        if not os.path.exists(ply):
            continue
        subprocess.run([SPLASHSURF, "reconstruct", ply,
                        f"--output-file={obj}",
                        "--particle-radius=0.003",
                        "--smoothing-length=2.0",
                        "--cube-size=0.8",
                        "--normals=on"],
                       capture_output=True)
    n = len(glob.glob(os.path.join(mesh_dir, "water_*.obj")))
    print(f"    SplashSurf meshed: {n}/300 frames", flush=True)
    return n


def bake_foam_dir(sim_dir, mesh_dir):
    """Bake foam from NPZ particles onto OBJ mesh vertices via KDTree."""
    sys.path.insert(0, BASE)
    from mesh_surface import bake_foam_to_mesh
    ok = 0
    for fr in range(300):
        npz = os.path.join(sim_dir, f"water_{fr:06d}.npz")
        obj = os.path.join(mesh_dir, f"water_{fr:06d}.obj")
        foam = os.path.join(mesh_dir, f"foam_{fr:06d}.npy")
        if not os.path.exists(obj) or not os.path.exists(npz):
            continue
        if os.path.exists(foam) and os.path.getsize(foam) > 100:
            ok += 1
            continue
        if bake_foam_to_mesh(npz, obj, foam):
            ok += 1
    print(f"    Foam baked: {ok}/300 frames", flush=True)


def compile_video(render_dir, video_path):
    """Compile PNG frames to MP4."""
    os.makedirs(os.path.dirname(video_path) if "/" in video_path else ".", exist_ok=True)
    n = len(glob.glob(os.path.join(render_dir, "frame_*.png")))
    if n < 10:
        print(f"    SKIP {video_path} — only {n} frames", flush=True)
        return
    cmd = (f'{FFMPEG} -y -framerate 30 '
           f'-i {render_dir}/frame_%06d.png '
           f'-c:v libx264 -crf 18 -pix_fmt yuv420p {video_path}')
    run(cmd, timeout=300)
    if os.path.exists(video_path):
        sz = os.path.getsize(video_path) // 1024
        print(f"    Video: {video_path} ({sz}KB)", flush=True)
    else:
        print(f"    FAILED: {video_path}", flush=True)


def main():
    os.chdir(BASE)
    os.makedirs("videos", exist_ok=True)
    t0 = time.time()

    print("=" * 60, flush=True)
    print("Phase 11 — Pipeline Master", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)

    # ---- 1. Wait for V1 rendering to complete ----
    print("\n[1] Waiting for V1 rendering...", flush=True)
    while True:
        n = len(glob.glob("renders_v01/frame_*.png"))
        if n >= 295:
            break
        print(f"    V1: {n}/300 rendered, waiting 60s...", flush=True)
        time.sleep(60)
    print(f"    V1 rendering done: {n} frames", flush=True)

    # ---- 2. Compile V1 video ----
    print("\n[2] Compiling V1 video...", flush=True)
    compile_video("renders_v01", "videos/v01_baseline.mp4")

    # ---- 3. Ensure SplashSurf meshing is complete ----
    print("\n[3] Checking SplashSurf meshing...", flush=True)
    n_ss = len(glob.glob("export/flood/meshes_ss/water_*.obj"))
    if n_ss < 290:
        print(f"    SS at {n_ss}/300, running remaining...", flush=True)
        splashsurf_mesh("export/flood", "export/flood/meshes_ss")
    else:
        print(f"    SS meshing complete: {n_ss}/300", flush=True)

    # ---- 4. Bake foam on SS meshes ----
    print("\n[4] Baking foam on SplashSurf meshes...", flush=True)
    bake_foam_dir("export/flood", "export/flood/meshes_ss")

    # ---- 5. Render V2: SplashSurf + flood base ----
    print("\n[5] Rendering V2 (SplashSurf)...", flush=True)
    render_4gpu("export/flood/meshes_ss", "renders_v02",
                "export/flood/meta.json", "blender_render.py")
    compile_video("renders_v02", "videos/v02_splashsurf.mp4")

    # ---- 6. Render V9/V10/V11/V12 in PARALLEL (1 GPU each) ----
    print("\n[6] Rendering V9/V10/V11/V12 in parallel (1 GPU each)...", flush=True)
    parallel_renders = [
        (0, "renders_v09", "blender_render_v09_clear.py", "v09_clear_water"),
        (1, "renders_v10", "blender_render_v10_closeup.py", "v10_closeup"),
        (2, "renders_v11", "blender_render_v11_topdown.py", "v11_topdown"),
        (3, "renders_v12", "blender_render_v12_sunset.py", "v12_sunset"),
    ]
    procs = []
    for gpu, output, script, vname in parallel_renders:
        os.makedirs(output, exist_ok=True)
        frames = ",".join(str(f) for f in range(300))
        cmd = (f'CUDA_VISIBLE_DEVICES={gpu} {BLENDER} --background '
               f'--python {script} -- '
               f'--meshes export/flood/meshes_ss --meta export/flood/meta.json '
               f'--frame {frames} --output {output} --samples 128 --gpus 1')
        log = os.path.join(output, "log_gpu0.txt")
        with open(log, "w") as logf:
            p = subprocess.Popen(cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((p, output, vname))
        print(f"    GPU {gpu}: {vname} (PID {p.pid})", flush=True)

    # Wait and compile videos
    for p, output, vname in procs:
        p.wait()
        n = len(glob.glob(os.path.join(output, "frame_*.png")))
        print(f"    {vname}: {n}/300 frames rendered", flush=True)
        compile_video(output, f"videos/{vname}.mp4")

    # ---- 6b. V2 fixup: re-render any missing frames (large meshes re-meshed) ----
    n_v2 = len(glob.glob("renders_v02/frame_*.png"))
    if n_v2 < 290:
        print(f"\n[6b] V2 fixup: only {n_v2}/300, re-rendering missing frames...", flush=True)
        render_4gpu("export/flood/meshes_ss", "renders_v02",
                    "export/flood/meta.json", "blender_render.py")
        compile_video("renders_v02", "videos/v02_splashsurf.mp4")

    # ---- 9. Run FLIP variants + Warp PBF in PARALLEL (1 GPU each) ----
    print("\n[9] Launching 4 simulations in parallel...", flush=True)
    sim_jobs = [
        (0, "v07_cfl_safe", f'{PYTHON} run_variants.py --variant v07_cfl_safe'),
        (1, "v06_dense", f'{PYTHON} run_variants.py --variant v06_dense'),
        (2, "v13_fast_flood", f'{PYTHON} run_variants.py --variant v13_fast_flood'),
        (3, "warp_pbf", f'{PYTHON} warp_pbf.py'),
    ]
    sim_procs = []
    for gpu, name, cmd in sim_jobs:
        if name == "warp_pbf":
            vdir = "export/flood_warp"
        else:
            vdir = f"export/{name}"
        os.makedirs(vdir, exist_ok=True)

        # Skip if already done
        if os.path.exists(os.path.join(vdir, "water_000299.ply")):
            print(f"    {name}: sim already done", flush=True)
            sim_procs.append(None)
            continue

        full_cmd = f'CUDA_VISIBLE_DEVICES={gpu} {cmd}'
        log = os.path.join(vdir, "sim_log.txt")
        with open(log, "w") as logf:
            p = subprocess.Popen(full_cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT)
        sim_procs.append(p)
        print(f"    GPU {gpu}: {name} (PID {p.pid})", flush=True)

    # Wait for all sims
    for p in sim_procs:
        if p is not None:
            p.wait()
    print("    All simulations complete!", flush=True)

    # ---- 10. Mesh + render each variant ----
    all_variants = [
        ("v07_cfl_safe", "export/v07_cfl_safe"),
        ("v06_dense", "export/v06_dense"),
        ("v13_fast_flood", "export/v13_fast_flood"),
        ("warp_pbf", "export/flood_warp"),
    ]
    for name, vdir in all_variants:
        if not os.path.exists(os.path.join(vdir, "water_000000.ply")):
            print(f"    SKIP {name} — no sim data", flush=True)
            continue

        # Mesh with SplashSurf
        mesh_dir = os.path.join(vdir, "meshes_ss")
        print(f"    Meshing {name}...", flush=True)
        splashsurf_mesh(vdir, mesh_dir)

        # Bake foam
        print(f"    Baking foam {name}...", flush=True)
        bake_foam_dir(vdir, mesh_dir)

        # Render
        meta = os.path.join(vdir, "meta.json")
        if not os.path.exists(meta):
            meta = "export/flood/meta.json"
        video_name = f"v08_warp_pbf" if name == "warp_pbf" else name
        rdir = f"renders_{name}"
        print(f"    Rendering {name}...", flush=True)
        render_4gpu(mesh_dir, rdir, meta, "blender_render.py")
        compile_video(rdir, f"videos/{video_name}.mp4")

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 60, flush=True)
    print(f"Pipeline complete! Total time: {elapsed/3600:.1f} hours", flush=True)
    print("Videos produced:", flush=True)
    for f in sorted(glob.glob("videos/*.mp4")):
        sz = os.path.getsize(f) // (1024 * 1024)
        print(f"  {os.path.basename(f)} ({sz}MB)", flush=True)
    n_videos = len(glob.glob("videos/*.mp4"))
    print(f"\nTotal: {n_videos} videos", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
