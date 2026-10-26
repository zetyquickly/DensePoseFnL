"""Check fnl_export/glue.py against the tensors dump_detectron2.py captured. Each step is fed detectron2's own input,
so a mismatch points at one function (the box head needs 1000 proposals here, as detectron2 runs it on all of them).
Exits non-zero on a count mismatch or a difference above test_glue.cc's tolerances (1e-2 for boxes and scores,
1e-3 for pooled features).
usage: python check_glue.py out/reference_480x640/*.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import roi_align

from fnl_export import glue

failed, current = [], ""   # "<file>: <check>" of every failed check; the file being checked


def count(name, ours, ref):
    print(f"  {name}: ours {ours}, detectron2 {ref}")
    if ours != ref:
        failed.append(f"{current}: {name}")


def report(name, ours, ref, tol):
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if ours.shape != ref.shape:
        print(f"    {name:<28} SHAPE {ours.shape} vs {ref.shape}  FAIL")
        failed.append(f"{current}: {name}")
        return
    d = np.abs(ours - ref)
    ok = d.size == 0 or d.max() <= tol
    print(f"    {name:<28} {str(ref.shape):<22} max|diff| {d.max(initial=0):.2e}  mean|diff| {d.mean() if d.size else 0:.2e}  "
          f"max|ref| {np.abs(ref).max(initial=0):.2e}  {'ok' if ok else 'FAIL'}")
    if not ok:
        failed.append(f"{current}: {name}")


def check_roi_align(feat):
    """glue.roi_align against torchvision's on boxes of several sizes; a zero-size box pools to zeros there."""
    boxes = np.array([[100, 50, 300, 200], [640, 100, 640, 300], [100, 80, 400, 80], [10, 10, 10.5, 10.5]], np.float32)
    ref = roi_align(torch.from_numpy(feat), [torch.from_numpy(boxes)], 32, 1 / 4, 0, True).numpy()
    print("roi_align vs torchvision: normal, zero width, zero height, tiny")
    for b, ours, r in zip(boxes, glue.roi_align(feat[0], boxes, 32, 1 / 4), ref):
        report(f"box {b.tolist()}", ours, r, 1e-3)


def main(files):
    global current
    if not files:
        raise SystemExit("usage: python check_glue.py out/reference_480x640/*.npz (no dumps given)")
    for f in files:
        current = Path(f).stem
        r = np.load(f)
        h, w = r["image_raw_padded"].shape[2:]
        print(Path(f).name)
        logits = [r[f"rpn_logits_{i}"] for i in range(5)]
        deltas = [r[f"rpn_deltas_{i}"] for i in range(5)]
        props = glue.rpn_proposals(logits, deltas, (h, w), post_nms_topk=1000)
        feats = [r[f"feat_p{i}"] for i in (2, 3, 4, 5)]
        ref_in = r["box_head_in"]
        count("proposals", len(props), len(ref_in))
        n = min(len(props), len(ref_in))
        report("pool_box (box_head_in)", glue.pool_box(feats, props[:n], 7), ref_in[:n], 1e-3)

        probs = np.exp(r["box_cls_logits"] - r["box_cls_logits"].max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        boxes, scores = glue.detections(probs[:n], r["box_deltas"][:n], props[:n], (h, w))
        ref_b, ref_s = r["det_boxes"], r["det_scores"]
        count("detections", len(boxes), len(ref_b))
        m = min(len(boxes), len(ref_b))
        if m:
            report("det_boxes", boxes[:m], ref_b[:m], 1e-2)
            report("det_scores", scores[:m], ref_s[:m], 1e-2)
        if "dp_head_in" in r:
            report("pool_dp (dp_head_in)", glue.pool_dp(feats[0], ref_b, 32), r["dp_head_in"], 1e-3)
    current = "roi_align"
    check_roi_align(np.load(files[0])["feat_p2"])
    if failed:
        raise SystemExit(f"FAILED ({len(failed)}):\n  " + "\n  ".join(failed))
    print("all ok")


if __name__ == "__main__":
    main(sys.argv[1:])
