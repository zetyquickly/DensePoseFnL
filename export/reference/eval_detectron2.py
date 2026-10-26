"""Run the original detectron2 CSE model over a COCO split and save what export/eval_pipeline.py saves.

Modes: default (detectron2's own resize, shortest edge 800) and canvas (the export's letterboxed 480x640 input).
usage: python eval_detectron2.py EXPORT_CONFIG D2_CONFIG IMAGES_DIR ANN_JSON OUT.npz --mode default|canvas [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

for _old, _new in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"), ("ANTIALIAS", "LANCZOS")):  # aliases Pillow 10 dropped, detectron2 still uses
    if not hasattr(Image, _old):
        setattr(Image, _old, getattr(Image, _new))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "export"))

import backbone, roi_heads  # noqa: F401,E402  (register BACKBONE / ROI_HEADS)
from config import add_timmnets_config  # noqa: E402
from densepose import add_densepose_config  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from fnl_export.config import ExportConfig  # noqa: E402
from fnl_export.preproc import preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_config")
    ap.add_argument("config")
    ap.add_argument("images")
    ap.add_argument("ann")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["default", "canvas"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ecfg = ExportConfig.load(a.export_config)
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_timmnets_config(cfg)
    cfg.merge_from_file(a.config)
    cfg.MODEL.WEIGHTS = str(ecfg.resolve(ecfg.checkpoint))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.TIMMNETS.PRETRAINED = False
    for spec in cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS.values():
        spec.INIT_FILE = ""
    cfg.freeze()
    pred = DefaultPredictor(cfg)
    model = pred.model.eval()

    images = sorted(json.load(open(a.ann))["images"], key=lambda im: im["id"])[: a.limit or None]
    ids, counts, boxes_all, scores_all, margins, embs = [], [], [], [], [], []
    t0 = time.time()
    for n, im in enumerate(images):
        img = Image.open(Path(a.images) / im["file_name"]).convert("RGB")
        if a.mode == "default":
            inst = pred(np.asarray(img)[:, :, ::-1])["instances"]       # DefaultPredictor takes BGR, resizes to 800 itself
            s = 1.0
        else:
            x = preprocess(img, ecfg)[0]                                 # RGB 0..255 at the canvas size
            with torch.no_grad():
                inst = model([{"image": torch.from_numpy(x.copy()), "height": ecfg.input_height, "width": ecfg.input_width}])[0]["instances"]
            s = min(ecfg.input_width / img.width, ecfg.input_height / img.height)
        boxes = (inst.pred_boxes.tensor / s).numpy()
        boxes[:, 0::2], boxes[:, 1::2] = boxes[:, 0::2].clip(0, img.width), boxes[:, 1::2].clip(0, img.height)
        seg = inst.pred_densepose.coarse_segm.detach().numpy() if len(inst) else np.zeros((0, 2, 64, 64), np.float32)
        margin = (seg[:, 1] - seg[:, 0]).astype(np.float16)
        emb = inst.pred_densepose.embedding.detach().numpy().astype(np.float16) if len(inst) else np.zeros((0, 16, 64, 64), np.float16)
        keep = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
        ids.append(im["id"]), counts.append(int(keep.sum()))
        boxes_all.append(boxes[keep]), scores_all.append(inst.scores.numpy()[keep]), margins.append(margin[keep]), embs.append(emb[keep])
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(images)}  {(time.time() - t0) / (n + 1):.2f} s/img", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, image_ids=np.asarray(ids), counts=np.asarray(counts), boxes=np.concatenate(boxes_all).astype(np.float32),
                        scores=np.concatenate(scores_all).astype(np.float32), margin=np.concatenate(margins),
                        embedding=np.concatenate(embs))
    print(f"wrote {a.out}: {len(ids)} images, {sum(counts)} detections")


if __name__ == "__main__":
    main()
