"""Export the static graphs of a Mobile-Parsing-RCNN checkpoint to LiteRT (.tflite).

usage: python export_litert.py configs/cse.yaml [--only graph_a ...]
"""
import argparse
import time

import numpy as np
import torch

import litert_torch

from fnl_export.checkpoint import load_model
from fnl_export.config import ExportConfig
from fnl_export.graphs import build_graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tol", type=float, default=1e-3, help="largest allowed |tflite - torch| on the exporter's inputs")
    args = ap.parse_args()

    cfg = ExportConfig.load(args.config)
    out = cfg.resolve(cfg.out_dir) / "litert" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(str(cfg.resolve(cfg.checkpoint)), cfg.head, cfg.chart_confidence)
    if cfg.head == "cse":   # SMPL vertex embeddings (27554, 16): colours the overlay of engine/tools
        np.save(out / "vertex_embeddings.npy", model.roi_heads.embedder().detach().numpy())

    for name, (module, inputs) in build_graphs(model, cfg).items():
        if args.only and name not in args.only:
            continue
        t = time.time()
        with torch.no_grad():
            ref = module(*inputs)
        ref = ref if isinstance(ref, tuple) else (ref,)
        edge = litert_torch.convert(module, inputs)
        path = out / f"{name}.tflite"
        edge.export(str(path))
        res = edge(*inputs)
        res = res if isinstance(res, (tuple, list)) else (res,)
        if len(res) != len(ref):
            raise SystemExit(f"{name}: {len(res)} outputs, the torch module has {len(ref)}")
        diff = max(float(np.abs(np.asarray(a, dtype=np.float64) - b.numpy().astype(np.float64)).max()) for a, b in zip(res, ref))
        if diff > args.tol:
            raise SystemExit(f"{name}: max|tflite-torch| = {diff:.2e} exceeds --tol {args.tol:g}")
        print(f"{name}: {path.stat().st_size / 1e6:.2f} MB, {time.time() - t:.0f}s, max|tflite-torch| = {diff:.2e}")


if __name__ == "__main__":
    main()
