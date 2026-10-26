"""Box AP, segm AP, GPS and GPSm (the README table's columns) of saved predictions, with detectron2/DensePose's own evaluators
on densepose_coco_2014_minival_cse. Inputs come from export/eval_pipeline.py and eval_detectron2.py.

A run is LABEL=FILE.npz[@MATCHING[@FILTER]]
  MATCHING  stock          detectron2's: embedding resized bilinearly to the box, exact nearest vertex at each annotated point
            engine         the engine's output: exact nearest vertex on the 64x64 person pixels, ids resized nearest to the box
            engine-approx  same with the engine's approximate search (GraphCOptimized)
  FILTER    all (every detection > 0.05) | device (top K2 with score >= S per image, --device K2 S)

Run from the repo root in the reference env. GPS/GPSm need the geodesic distances as npy (prep_geodists.py); --no-gps skips them.
usage: python export/reference/eval_predictions.py "d2=d2_default.npz" "ours=ours_cse.npz@engine-approx@device"
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

for _alias, _type in (("int", int), ("float", float), ("bool", bool)):   # numpy aliases removed in 1.24 that densepose's evaluator still uses
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)

for _old, _new in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"), ("ANTIALIAS", "LANCZOS")):  # aliases Pillow 10 dropped, detectron2 still uses
    if not hasattr(Image, _old):
        setattr(Image, _old, getattr(Image, _new))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "export"))

from detectron2.structures import Boxes, BoxMode, Instances  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402
from densepose.data.build import _add_category_id_to_contiguous_id_maps_to_metadata, _merge_categories  # noqa: E402
from densepose.data.datasets.coco import CocoDatasetInfo, register_dataset  # noqa: E402
from densepose.evaluation import densepose_coco_evaluation as dce  # noqa: E402
from densepose.evaluation.d2_evaluator_adapter import Detectron2COCOEvaluatorAdapter  # noqa: E402
from densepose.evaluation.evaluator import DensePoseCOCOEvaluator, _derive_results_from_coco_eval, _get_densepose_metrics  # noqa: E402
from densepose.modeling.cse.utils import squared_euclidean_distance_matrix  # noqa: E402
from densepose.structures.mesh import create_mesh  # noqa: E402
from fnl_export.graphs import GraphCOptimized  # noqa: E402

_orig_computeOgps = dce.DensePoseCocoEval.computeOgps


def _computeOgps(self, imgId, catId):
    """IOU mode never reads the pointwise GPS matrix (it needs embeddings and geodesic distances), only the box IoUs; GPS/GPSM run detectron2's code."""
    if self._dpEvalMode != dce.DensePoseEvalMode.IOU:
        return _orig_computeOgps(self, imgId, catId)
    p = self.params
    g = self._gts[imgId, catId]
    d = sorted(self._dts[imgId, catId], key=lambda x: -x["score"])[: p.maxDets[-1]]
    if len(g) == 0 or len(d) == 0:
        return []
    ious_bb = dce.maskUtils.iou([x["bbox"] for x in d], [x["bbox"] for x in g], [int(o.get("iscrowd", 0)) for o in g])
    return np.zeros((len(d), len(g))), ious_bb


def _loadGEval(self):
    """The CSE metric reads only these two tables of the UV evaluator's setup (the rest is 1.5 GB of downloads)."""
    self.Mean_Distances = np.array([0, 0.351, 0.107, 0.126, 0.237, 0.173, 0.142, 0.128, 0.150])
    self.CoarseParts = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8])


dce.DensePoseCocoEval.computeOgps = _computeOgps
dce.DensePoseCocoEval._loadGEval = _loadGEval


class Matcher:
    """Pixel embeddings (n,16) -> closest vertex id (n,)."""

    def __init__(self, vertices: torch.Tensor, approx: bool):
        self.v, self.approx = vertices, approx
        if approx:
            stub = SimpleNamespace(roi_heads=SimpleNamespace(embedder=lambda: vertices))
            self.c = GraphCOptimized(stub, SimpleNamespace(cse_blocks=4096, cse_group=4)).eval()

    def __call__(self, pix: torch.Tensor) -> torch.Tensor:
        out = []
        with torch.no_grad():
            for i in range(0, len(pix), 4096):
                chunk = pix[i:i + 4096]
                if self.approx:
                    out.append(self.c(F.pad(chunk, (0, 0, 0, 4096 - len(chunk))))[:len(chunk)].long())
                else:
                    out.append(squared_euclidean_distance_matrix(chunk, self.v).argmin(1))
        return torch.cat(out)


def patch_cse_pairs(vertices, matching):
    """computeOgps_single_pair_cse with the matching swapped, on the CPU (the original is hard-wired to cuda), and per-detection resampling cached across an image's GT objects."""
    matcher = Matcher(vertices, matching == "engine-approx") if matching != "stock" else None
    cache = {"img": None, "dt": {}}

    def pair(self, dt, gt, py, px, pt_mask, coarse_segm, embedding, bbox):
        if cache["img"] != dt["image_id"]:
            cache["img"], cache["dt"] = dt["image_id"], {}
        ent = cache["dt"].get(id(dt))
        h, w = max(int(bbox[3]), 1), max(int(bbox[2]), 1)
        if ent is None:
            emb = torch.from_numpy(embedding).float() if isinstance(embedding, np.ndarray) else embedding
            labels = dce.resample_coarse_segm_tensor_to_bbox(coarse_segm.unsqueeze(0), bbox).squeeze(0)
            if matching == "stock":
                grid = F.interpolate(emb.unsqueeze(0), (h, w), mode="bilinear", align_corners=False).squeeze(0)
            else:
                person = coarse_segm.argmax(0) > 0
                ids = torch.full(person.shape, -1, dtype=torch.long)
                if person.any():
                    ids[person] = matcher(emb[:, person].t())
                grid = F.interpolate(ids[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
            ent = cache["dt"][id(dt)] = (labels, grid)
        labels, grid = ent
        cVertsGT = torch.as_tensor(gt["dp_vertex"], dtype=torch.int64)
        valid = pt_mask > -1
        py_pt, px_pt = torch.from_numpy(py[valid]), torch.from_numpy(px[valid])
        cVerts = torch.ones_like(cVertsGT) * -1
        if matching == "stock":
            v = squared_euclidean_distance_matrix(grid[:, py_pt, px_pt].t(), vertices).argmin(1)
        else:
            v = grid[py_pt, px_pt].clone()
        v[labels[py_pt, px_pt] <= 0] = -1
        cVerts[valid] = v
        dist = self.getDistancesCse(cVertsGT, cVerts, gt["ref_model"])
        if gt["ref_model"] == "smpl_27554" and "dp_I" in gt:
            norm = self.Mean_Distances[self.CoarseParts[np.array(gt["dp_I"], dtype=int)]]
        else:
            norm = 0.255
        return dist, norm

    dce.DensePoseCocoEval.computeOgps_single_pair_cse = pair


def load(path):
    z = np.load(path)
    boxes, scores, margin = z["boxes"], z["scores"], z["margin"]   # NpzFile re-reads an array on every access: take each once
    emb = z["embedding"] if "embedding" in z.files else None
    off = np.concatenate([[0], np.cumsum(z["counts"])])
    return [(int(i), boxes[off[k]:off[k + 1]], scores[off[k]:off[k + 1]], margin[off[k]:off[k + 1]],
             None if emb is None else emb[off[k]:off[k + 1]]) for k, i in enumerate(z["image_ids"])]


def device_filter(preds, k2, thresh):
    out = []
    for i, boxes, scores, margin, emb in preds:
        keep = [j for j in np.argsort(-scores, kind="stable") if scores[j] >= thresh][:k2]
        out.append((i, boxes[keep], scores[keep], margin[keep], None if emb is None else emb[keep]))
    return out


def evaluate(preds, dataset, sizes, gps):
    img_ids = [x[0] for x in preds] if len(preds) < len(sizes) else None   # a subset run is scored on its own images only
    box_eval = Detectron2COCOEvaluatorAdapter(dataset, distributed=False)
    box_eval.reset()
    seg_eval = DensePoseCOCOEvaluator(dataset, distributed=False, evaluator_type="cse", min_iou_threshold=0.5)
    seg_dets = []
    for img_id, boxes, scores, margin, emb in preds:
        h, w = sizes[img_id]
        inst = Instances((h, w))
        inst.pred_boxes = Boxes(torch.from_numpy(boxes).reshape(-1, 4))
        inst.scores = torch.from_numpy(scores)
        inst.pred_classes = torch.zeros(len(scores), dtype=torch.int64)
        with contextlib.redirect_stdout(io.StringIO()):
            box_eval.process([{"image_id": img_id}], [{"instances": inst}])
        xywh = BoxMode.convert(torch.from_numpy(boxes).reshape(-1, 4), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()
        m = torch.from_numpy(margin.astype(np.float32))
        for k in range(len(scores)):   # margin as 2-channel logits (argmax > 0 iff margin > 0; resizing is linear)
            d = {"image_id": img_id, "category_id": 0, "bbox": xywh[k], "score": float(scores[k]),
                 "coarse_segm": torch.stack([-m[k] / 2, m[k] / 2])}
            if gps:
                d["embedding"] = emb[k]
            seg_dets.append(d)
    with contextlib.redirect_stdout(io.StringIO()):
        bbox = box_eval.evaluate(img_ids)["bbox"]
    coco_dt = seg_eval._coco_api.loadRes(seg_dets) if seg_dets else None
    metrics = _get_densepose_metrics(0.5)
    out = {"box AP": bbox["AP"], "box AP50": bbox["AP50"]}
    for name, mode in (("segm", "IOU"), ("GPS", "GPS"), ("GPSm", "GPSM")):
        if name != "segm" and not gps:
            continue
        ev = dce.DensePoseCocoEval(seg_eval._coco_api, coco_dt, "densepose", None, None, dpEvalMode=getattr(dce.DensePoseEvalMode, mode))
        with contextlib.redirect_stdout(io.StringIO()):
            r = _derive_results_from_coco_eval(ev, mode, metrics, seg_eval._metadata.get("thing_classes"), 0.5, img_ids)
        out[f"{name} AP"], out[f"{name} AP50"] = r["AP"], r["AP50"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=FILE.npz[@stock|engine|engine-approx[@all|device]]")
    ap.add_argument("--device", nargs=2, type=float, metavar=("K2", "SCORE"), default=(4, 0.5), help="the 'device' filter (default 4 0.5)")
    ap.add_argument("--no-gps", action="store_true", help="box AP and segm AP only")
    ap.add_argument("--geodists", default=str(REPO / "export/out/eval/geodists_smpl_27554.npy"), help="float32 npy from prep_geodists.py")
    ap.add_argument("--vertices", default=str(REPO / "export/out/reference_480x640/vertex_embeddings.npy"))
    ap.add_argument("--ann", default="coco/annotations/densepose_minival2014_cse.json",
                    help="annotation json under datasets/ (densepose registers datasets/coco_cse/..., not where the file is here)")
    ap.add_argument("--images", default="coco/val2014", help="image dir, relative to datasets/")
    a = ap.parse_args()
    gps = not a.no_gps

    dataset = "fnl_minival_cse"
    register_dataset(CocoDatasetInfo(name=dataset, images_root=a.images, annotations_fpath=a.ann), "datasets")
    DatasetCatalog.get(dataset)   # sets meta.categories
    _add_category_id_to_contiguous_id_maps_to_metadata(_merge_categories([dataset]))   # id maps, as the dataloader builder does
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(MetadataCatalog.get(dataset).json_file)
    sizes = {i: (im["height"], im["width"]) for i, im in coco.imgs.items()}

    vertices = torch.from_numpy(np.load(a.vertices)).float()
    if gps:
        print("loading geodesic distances ...", flush=True)
        create_mesh("smpl_27554", "cpu")._geodists = torch.from_numpy(np.load(a.geodists))

    rows, cache = [], {}
    for run in a.runs:
        label, spec = run.split("=", 1)
        path, *opt = spec.split("@")
        matching, flt = (opt + ["stock", "all"][len(opt):])[:2]
        if path not in cache:
            cache.clear()
            cache[path] = load(path)
        preds = cache[path]
        if flt == "device":
            preds = device_filter(preds, int(a.device[0]), a.device[1])
        if gps:
            patch_cse_pairs(vertices, matching)
        t0 = time.time()
        r = evaluate(preds, dataset, sizes, gps)
        dets = sum(len(x[2]) for x in preds) / len(preds)
        rows.append((label, matching if gps else "-", flt, dets, r))
        print(f"{label:<44} {matching:<14} {flt:<7} {dets:5.1f} dets/img  " + "  ".join(f"{k} {v:5.1f}" for k, v in r.items() if k.endswith(" AP"))
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    cols = ["box AP", "segm AP"] + (["GPS AP", "GPSm AP"] if gps else [])
    print("\n| model | matching | detections | dets/img | " + " | ".join(c.replace(" AP", "") + " AP" for c in cols) + " |")
    print("|---|---|---|---|" + "---|" * len(cols))
    for label, matching, flt, dets, r in rows:
        print(f"| {label} | {matching} | {'top-%d, score >= %.2f' % (a.device[0], a.device[1]) if flt == 'device' else 'all > 0.05'} | {dets:.1f} | "
              + " | ".join(f"{r[c]:.1f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
