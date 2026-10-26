"""Geodesic-distance pickle (6 GB, float64) -> float32 npy (3 GB) for eval_predictions.py. Needs ~10 GB of RAM.
Downloads the pickle on first use, as densepose does (into ~/.torch/iopath_cache).
usage: python export/reference/prep_geodists.py [OUT.npy]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

for _old, _new in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"), ("ANTIALIAS", "LANCZOS")):  # aliases Pillow 10 dropped, detectron2 still uses
    if not hasattr(Image, _old):
        setattr(Image, _old, getattr(Image, _new))

from densepose.data.meshes.catalog import MeshCatalog  # noqa: E402
from detectron2.utils.file_io import PathManager  # noqa: E402

src = PathManager.get_local_path(MeshCatalog["smpl_27554"].geodists)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "out/eval/geodists_smpl_27554.npy"
out.parent.mkdir(parents=True, exist_ok=True)
with open(src, "rb") as f:
    d = pickle.load(f)
d = np.asarray(d)
print(d.shape, d.dtype, f"max {d.max():.3f}")
np.save(out, d.astype(np.float32))
print("wrote", out)
