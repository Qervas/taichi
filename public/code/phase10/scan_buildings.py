"""Scan all building OBJs and catalog their dimensions."""
import trimesh
import os
import json
import numpy as np

models_dir = os.path.expanduser("~/Downloads/models")
objs = sorted([f for f in os.listdir(models_dir) if f.endswith(".obj")])

results = []
for fname in objs:
    path = os.path.join(models_dir, fname)
    try:
        scene = trimesh.load(path, force="scene")
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]
        n_v = len(mesh.vertices)
        n_f = len(mesh.faces)

        results.append(dict(
            file=fname,
            dims=dims.tolist(),
            n_verts=n_v,
            n_faces=n_f,
            bounds_min=bounds[0].tolist(),
            bounds_max=bounds[1].tolist(),
            center=((bounds[0] + bounds[1]) / 2).tolist(),
        ))
        print(f"  {fname}: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f}m, "
              f"{n_v:,}v {n_f:,}f")
    except Exception as e:
        print(f"  {fname}: ERROR - {e}")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "building_catalog.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

dims_arr = np.array([r["dims"] for r in results])
total_f = sum(r["n_faces"] for r in results)
print(f"\nSaved {len(results)} buildings to {out_path}")
print(f"  X (width):  {dims_arr[:,0].min():.1f} - {dims_arr[:,0].max():.1f}m")
print(f"  Y (depth):  {dims_arr[:,1].min():.1f} - {dims_arr[:,1].max():.1f}m")
print(f"  Z (height): {dims_arr[:,2].min():.1f} - {dims_arr[:,2].max():.1f}m")
print(f"  Total tris: {total_f:,}")
