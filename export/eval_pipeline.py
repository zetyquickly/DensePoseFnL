"""Run the exported graphs + glue over a COCO split; save per detection the box (original-image pixels), score,
coarse-segmentation margin (fg - bg logit, 64x64) and embedding (fp16). Score with reference/eval_predictions.py.

usage: python eval_pipeline.py configs/cse.yaml --images datasets/coco/val2014 --ann datasets/coco/annotations/densepose_minival2014_cse.json
           [--tflite DIR] [--limit N] [--out out/eval/ours.npz]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from compare_with_detectron2 import Tflite
from fnl_export import glue
from fnl_export.config import ExportConfig
from fnl_export.preproc import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True, help="COCO-format json listing the images to run")
    ap.add_argument("--tflite", help="dir with the .tflite graphs (default: <out_dir>/litert/<name>)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    cfg = ExportConfig.load(a.config)
    out = Path(a.out) if a.out else cfg.resolve(cfg.out_dir) / "eval" / f"ours_{cfg.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tfl = Tflite(a.tflite or cfg.resolve(cfg.out_dir) / "litert" / cfg.name, ["graph_a", "graph_b1", "graph_b2"])
    images = sorted(json.load(open(a.ann))["images"], key=lambda im: im["id"])[: a.limit or None]
    hw = (cfg.input_height, cfg.input_width)

    ids, counts, boxes_all, scores_all, margins, embs = [], [], [], [], [], []
    t0 = time.time()
    for n, im in enumerate(images):
        img = Image.open(Path(a.images) / im["file_name"])
        s = min(cfg.input_width / img.width, cfg.input_height / img.height)
        feats = tfl.a(preprocess(img, cfg))
        props = glue.rpn_proposals(feats[4:9], feats[9:14], hw, cfg.k_proposals)
        probs, deltas = tfl.b1(glue.pool_box(feats[:4], props, cfg.box_pool))
        boxes, scores = glue.detections(probs, deltas, props, hw)
        margin, emb = np.zeros((0, 64, 64), np.float16), np.zeros((0, 16, 64, 64), np.float16)
        if len(boxes):
            emb, seg = tfl.b2(glue.pool_dp(feats[0], boxes, cfg.dp_pool))
            margin, emb = (seg[:, 1] - seg[:, 0]).astype(np.float16), emb.astype(np.float16)
        boxes = glue.clip(boxes / s, img.height, img.width)             # back to original-image pixels
        keep = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
        ids.append(im["id"]), counts.append(int(keep.sum()))
        boxes_all.append(boxes[keep]), scores_all.append(scores[keep]), margins.append(margin[keep]), embs.append(emb[keep])
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(images)}  {(time.time() - t0) / (n + 1):.2f} s/img", flush=True)

    np.savez_compressed(out, image_ids=np.asarray(ids), counts=np.asarray(counts), boxes=np.concatenate(boxes_all).astype(np.float32),
                        scores=np.concatenate(scores_all).astype(np.float32), margin=np.concatenate(margins),
                        embedding=np.concatenate(embs))
    print(f"wrote {out}: {len(ids)} images, {sum(counts)} detections")


if __name__ == "__main__":
    main()
