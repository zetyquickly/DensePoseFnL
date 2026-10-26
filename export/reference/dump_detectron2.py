"""Run the ORIGINAL detectron2/DensePose eager model and dump the tensors around each piece our graphs replace.

Runs in the `reference` env (torch 2.2, timm 0.4.12, detectron2 @ bb96d0b).
The image is preprocessed exactly like the exported graphs expect (size from the export config, see fnl_export/preproc.py)
and fed to the model as-is, so detectron2 and our graphs see identical pixels.
usage: python dump_detectron2.py EXPORT_CONFIG D2_CONFIG OUT_DIR IMG [IMG ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

for _old, _new in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"), ("ANTIALIAS", "LANCZOS")):  # aliases Pillow 10 dropped, detectron2 still uses
    if not hasattr(Image, _old):
        setattr(Image, _old, getattr(Image, _new))

REPO = Path(__file__).resolve().parents[2]  # DensePoseFnL repo root: backbone.py, roi_heads.py, config.py
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "export"))  # fnl_export (config + preproc only, no torch needed)

import backbone, roi_heads  # noqa: F401,E402  (register BACKBONE / ROI_HEADS)
from config import add_timmnets_config  # noqa: E402
from densepose import add_densepose_config  # noqa: E402
from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES, squared_euclidean_distance_matrix  # noqa: E402
from densepose.modeling.densepose_checkpoint import DensePoseCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from fnl_export.config import ExportConfig  # noqa: E402
from fnl_export.preproc import preprocess  # noqa: E402


# densepose_predictor layer -> name of the matching graph B2 output (modules.CHART_OUTPUTS)
CHART_LAYERS = {"ann_index_lowres": "coarse", "index_uv_lowres": "fine", "u_lowres": "u", "v_lowres": "v",
                "sigma_2_lowres": "sigma_2", "kappa_u_lowres": "kappa_u", "kappa_v_lowres": "kappa_v",
                "fine_segm_confidence_lowres": "fine_conf", "coarse_segm_confidence_lowres": "coarse_conf"}


def main(export_config, config, out_dir, images):
    ecfg = ExportConfig.load(export_config)
    ckpt = str(ecfg.resolve(ecfg.checkpoint))
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_timmnets_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.TIMMNETS.PRETRAINED = False   # weights come from the checkpoint
    for spec in cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS.values():
        spec.INIT_FILE = ""                 # vertex features are stored in the checkpoint
    cfg.freeze()

    pred = DefaultPredictor(cfg)            # builds model + loads weights via DensePoseCheckpointer
    model = pred.model.eval()
    print("input format:", cfg.INPUT.FORMAT, "export size:", ecfg.input_height, "x", ecfg.input_width)

    cap = {}

    def hook(name, take_in=False):
        def fn(mod, inp, out):
            cap[name + ("_in" if take_in else "")] = inp[0].detach().clone() if take_in else out
        return fn

    model.backbone.register_forward_hook(hook("backbone_in", True))
    model.backbone.register_forward_hook(hook("backbone_out"))
    model.proposal_generator.rpn_head.register_forward_hook(hook("rpn_out"))
    model.roi_heads.box_head.register_forward_hook(hook("box_head_in", True))
    model.roi_heads.box_predictor.register_forward_hook(hook("box_pred_out"))
    model.roi_heads.densepose_head.register_forward_hook(hook("dp_head_in", True))
    model.roi_heads.densepose_predictor.register_forward_hook(hook("dp_pred_out"))
    if ecfg.head == "chart":   # raw transposed-conv outputs, i.e. what our graph B2 emits (before the segm-confidence scaling)
        for layer, name in CHART_LAYERS.items():
            if hasattr(model.roi_heads.densepose_predictor, layer):
                getattr(model.roi_heads.densepose_predictor, layer).register_forward_hook(hook(f"dp_raw_{name}"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if ecfg.head == "cse":
        V = model.roi_heads.embedder("smpl_27554").detach()
        np.save(out / "vertex_embeddings.npy", V.numpy())

    for p in images:
        cap.clear()
        x = preprocess(Image.open(p), ecfg)[0]   # (3,H,W) RGB 0..255, already at the final size: no resize/pad inside detectron2
        if cfg.INPUT.FORMAT == "BGR":
            x = x[::-1]
        with torch.no_grad():
            inst = model([{"image": torch.from_numpy(x.copy()), "height": ecfg.input_height, "width": ecfg.input_width}])[0]["instances"]
        std = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, 3, 1, 1)
        mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, 3, 1, 1)
        d = {"image_raw_padded": cap["backbone_in_in"] * std + mean}      # undo normalisation -> 0..255 RGB, mean-padded
        d["image_norm_padded"] = cap["backbone_in_in"]
        for k, v in cap["backbone_out"].items():
            d[f"feat_{k}"] = v
        logits, deltas = cap["rpn_out"]
        for i, (a, b) in enumerate(zip(logits, deltas)):
            d[f"rpn_logits_{i}"], d[f"rpn_deltas_{i}"] = a, b
        d["box_head_in"] = cap["box_head_in_in"]
        d["box_cls_logits"], d["box_deltas"] = cap["box_pred_out"]
        n_dp = 0
        if "dp_head_in_in" in cap and ecfg.head == "chart":
            d["dp_head_in"] = cap["dp_head_in_in"]
            for k, v in cap.items():
                if k.startswith("dp_raw_"):
                    d[k] = v
            o = cap["dp_pred_out"]   # final outputs (segm confidences already applied): what the converter resamples
            d["dp_final_coarse"], d["dp_final_fine"], d["dp_u"], d["dp_v"] = o.coarse_segm, o.fine_segm, o.u, o.v
            d["segm_conf_eps"] = torch.tensor(cfg.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE.EPSILON)
            n_dp = len(o.u)
        elif "dp_head_in_in" in cap:
            d["dp_head_in"] = cap["dp_head_in_in"]
            d["dp_embedding"] = cap["dp_pred_out"].embedding
            d["dp_coarse_segm"] = cap["dp_pred_out"].coarse_segm
            # original CSE matching at the native 64x64 resolution (what densepose's converter does)
            ids, masks = [], []
            for e, s in zip(d["dp_embedding"], d["dp_coarse_segm"]):
                cv, m = get_closest_vertices_mask_from_ES(e[None], s[None], e.shape[-2], e.shape[-1], V, torch.device("cpu"))
                ids.append(cv), masks.append(m)
            d["cse_ids"], d["cse_mask"] = torch.stack(ids), torch.stack(masks)
            # same distance function + argmin, but on every pixel (the converter above only fills person pixels)
            allp = d["dp_embedding"].permute(0, 2, 3, 1).reshape(-1, d["dp_embedding"].shape[1])
            d["cse_ids_all"] = torch.cat([
                squared_euclidean_distance_matrix(c, V).argmin(1) for c in allp.split(10000)
            ]).reshape(d["dp_embedding"].shape[0], *d["dp_embedding"].shape[2:])
            n_dp = len(ids)
        d["det_scores"], d["det_boxes"] = inst.scores, inst.pred_boxes.tensor
        np.savez_compressed(out / f"{Path(p).stem}.npz", **{k: v.detach().cpu().numpy() for k, v in d.items()})
        print(f"{Path(p).name}: input {tuple(d['image_norm_padded'].shape)}, {d['box_head_in'].shape[0]} proposals, "
              f"{n_dp} detections, scores {[round(float(s), 2) for s in inst.scores[:6]]}")


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4:])
