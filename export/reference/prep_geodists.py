"""Geodesic-distance pickle (6 GB, float64) -> float32 npy (3 GB) for eval_predictions.py. Needs ~10 GB of RAM.
usage: python export/reference/prep_geodists.py [OUT.npy]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

src = Path.home() / ".torch/iopath_cache/densepose/meshes/geodists/geodists_smpl_27554.pkl"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "out/eval/geodists_smpl_27554.npy"
with open(src, "rb") as f:
    d = pickle.load(f)
d = np.asarray(d)
print(d.shape, d.dtype, f"max {d.max():.3f}")
np.save(out, d.astype(np.float32))
print("wrote", out)
