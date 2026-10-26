"""Dump inputs and expected outputs of fnl_export/glue.py as raw float32 files, for engine/tests/test_glue.cc.

Inputs are detectron2's own tensors (reference/dump_detectron2.py npz), expected outputs come from the numpy glue
(which check_glue.py has shown equal to detectron2). K1 proposals / K2 DensePose ROIs match the engine's exported graphs.
usage: python engine/tools/make_test_vectors.py OUT_DIR out/reference_480x640/*.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fnl_export import glue  # noqa: E402

K1, K2 = 100, 4


def save(d: Path, name: str, a: np.ndarray, dtype=np.float32):
    np.ascontiguousarray(a, dtype=dtype).tofile(d / name)


def main(out, files):
    for f in files:
        r = np.load(f)
        d = Path(out) / Path(f).stem
        d.mkdir(parents=True, exist_ok=True)
        h, w = r["image_raw_padded"].shape[2:]
        logits = [r[f"rpn_logits_{i}"] for i in range(5)]
        deltas = [r[f"rpn_deltas_{i}"] for i in range(5)]
        feats = [r[f"feat_p{i}"] for i in (2, 3, 4, 5)]
        for i in range(5):
            save(d, f"rpn_logits_{i}.f32", logits[i])
            save(d, f"rpn_deltas_{i}.f32", deltas[i])
        for i, ft in zip((2, 3, 4, 5), feats):
            save(d, f"feat_p{i}.f32", ft)

        props = glue.rpn_proposals(logits, deltas, (h, w), K1)
        save(d, "props.f32", props)
        # non-finite logits (fp16 overflow on the GPU) are not candidates: every 3rd p2 logit NaN, as test_glue.cc injects it
        nan_logits = [lg.copy() for lg in logits]
        nan_logits[0].reshape(-1)[::3] = np.nan
        save(d, "props_nan.f32", glue.rpn_proposals(nan_logits, deltas, (h, w), K1))
        save(d, "pool_box.f32", glue.pool_box(feats, props, 7))

        logit = r["box_cls_logits"][:K1]
        probs = np.exp(logit - logit.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        save(d, "probs.f32", probs)
        save(d, "box_deltas.f32", r["box_deltas"][:K1])
        boxes, scores = glue.detections(probs, r["box_deltas"][:K1], props, (h, w))
        save(d, "dets.f32", np.concatenate([boxes, scores[:, None]], 1))
        save(d, "pool_dp.f32", glue.pool_dp(feats[0], boxes[:K2], 32))

        if "dp_embedding" in r:
            emb, seg = r["dp_embedding"][:K2], r["dp_coarse_segm"][:K2]
            mask, px = glue.person_pixels(emb, seg)
            save(d, "emb.f32", emb)
            save(d, "seg.f32", seg)
            save(d, "mask.u8", mask, np.uint8)
            save(d, "px.f32", px)
        print(f"{d}: {len(props)} proposals, {len(boxes)} detections")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
