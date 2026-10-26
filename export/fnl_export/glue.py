"""Host-side glue between the static graphs, in numpy; engine/src/glue.cc is checked against it.

Mirrors detectron2: anchors, find_top_rpn_proposals, ROIPooler + ROIAlignV2, Box2BoxTransform, fast_rcnn_inference.
"""
import math

import numpy as np

STRIDES = (4, 8, 16, 32, 64)            # p2..p6
ANCHOR_SIZES = (32, 64, 128, 256, 512)  # one per level
ASPECT_RATIOS = (0.5, 1.0, 2.0)
SCALE_CLAMP = math.log(1000.0 / 16)
RPN_PRE_NMS_TOPK, RPN_NMS = 1000, 0.7   # per level
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BOX_SCORE_THRESH, BOX_NMS, BOX_TOPK = 0.05, 0.5, 100


def anchors(level: int, h: int, w: int) -> np.ndarray:
    """(h*w*3, 4) xyxy anchors of one level, ordered (y, x, aspect) like the flattened RPN head outputs. Grid offset is 0."""
    size, stride = ANCHOR_SIZES[level], STRIDES[level]
    cell = []
    for a in ASPECT_RATIOS:
        bw = math.sqrt(size * size / a)
        bh = a * bw
        cell.append([-bw / 2, -bh / 2, bw / 2, bh / 2])
    sy, sx = np.meshgrid(np.arange(h) * stride, np.arange(w) * stride, indexing="ij")
    shifts = np.stack([sx, sy, sx, sy], -1).reshape(-1, 1, 4).astype(np.float32)
    return (shifts + np.asarray(cell, np.float32)[None]).reshape(-1, 4)


def decode(deltas: np.ndarray, boxes: np.ndarray, weights=(1.0, 1.0, 1.0, 1.0)) -> np.ndarray:
    """Box2BoxTransform.apply_deltas: (N,4) deltas on (N,4) xyxy boxes."""
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx, cy = boxes[:, 0] + 0.5 * w, boxes[:, 1] + 0.5 * h
    dx, dy = deltas[:, 0] / weights[0], deltas[:, 1] / weights[1]
    dw, dh = np.minimum(deltas[:, 2] / weights[2], SCALE_CLAMP), np.minimum(deltas[:, 3] / weights[3], SCALE_CLAMP)
    pcx, pcy, pw, ph = dx * w + cx, dy * h + cy, np.exp(dw) * w, np.exp(dh) * h
    return np.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], -1).astype(np.float32)


def clip(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0::2], out[:, 1::2] = out[:, 0::2].clip(0, w), out[:, 1::2].clip(0, h)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    """Greedy NMS; indices of the kept boxes, by decreasing score."""
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(i)
        iw = np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0])
        ih = np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        order = rest[inter / (area[i] + area[rest] - inter) <= thresh]
    return np.asarray(keep, np.int64)


def rpn_proposals(logits, deltas, image_hw, post_nms_topk: int):
    """Graph A's RPN outputs (5 x (1,3,H,W) logits, 5 x (1,12,H,W) deltas) -> top post_nms_topk proposals by objectness.

    Per level: top-1000 by logit, decode, clip, drop empty, NMS 0.7; then merged by score.
    """
    boxes, scores = [], []
    for lv, (lg, dl) in enumerate(zip(logits, deltas)):
        _, a, h, w = lg.shape
        lg = lg[0].transpose(1, 2, 0).reshape(-1)                              # (h*w*a,)
        dl = dl[0].reshape(a, 4, h, w).transpose(2, 3, 0, 1).reshape(-1, 4)    # (h*w*a, 4)
        # a NaN/Inf logit (fp16 overflow on the GPU delegate) is not a candidate; detectron2 drops non-finite scores before NMS
        finite = np.flatnonzero(np.isfinite(lg))
        cand = lg[finite]
        k = min(len(cand), RPN_PRE_NMS_TOPK)
        top = finite[np.argpartition(-cand, k - 1)[:k] if k < len(cand) else np.arange(len(cand))]
        b = clip(decode(dl[top], anchors(lv, h, w)[top]), *image_hw)
        ok = ((b[:, 2] - b[:, 0]) > 0) & ((b[:, 3] - b[:, 1]) > 0)
        b, s = b[ok], lg[top][ok]
        keep = nms(b, s, RPN_NMS)
        boxes.append(b[keep]), scores.append(s[keep])
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    order = np.argsort(-scores, kind="stable")[:post_nms_topk]
    return boxes[order]


def roi_align(feat: np.ndarray, boxes: np.ndarray, out: int, scale: float) -> np.ndarray:
    """torchvision roi_align(aligned=True, sampling_ratio=0), i.e. detectron2's ROIAlignV2: feat (C,H,W), boxes (K,4) xyxy -> (K,C,out,out)."""
    c, h, w = feat.shape
    res = np.zeros((len(boxes), c, out, out), np.float32)

    def taps(start, bin_size, n_bins, grid, size):
        # per sample: (low, high, weight low, weight high); zero weights outside the map
        pos = start + (np.arange(n_bins)[:, None] + (np.arange(grid)[None] + 0.5) / grid) * bin_size
        pos = pos.reshape(-1)
        valid = (pos >= -1.0) & (pos <= size)
        pos = np.maximum(pos, 0.0)
        lo = pos.astype(np.int64)
        hi = np.minimum(lo + 1, size - 1)
        frac = np.where(lo >= size - 1, 0.0, pos - lo)
        lo = np.minimum(lo, size - 1)
        return lo, hi, np.where(valid, 1 - frac, 0.0).astype(np.float32), np.where(valid, frac, 0.0).astype(np.float32)

    for k, (x1, y1, x2, y2) in enumerate(boxes):
        sx, sy = x1 * scale - 0.5, y1 * scale - 0.5
        rw, rh = (x2 - x1) * scale, (y2 - y1) * scale
        gw, gh = math.ceil(rw / out), math.ceil(rh / out)
        if gw == 0 or gh == 0:   # a zero-size box takes no samples in torchvision and pools to zeros
            continue
        ylo, yhi, wy0, wy1 = taps(sy, rh / out, out, gh, h)
        xlo, xhi, wx0, wx1 = taps(sx, rw / out, out, gw, w)
        rows = feat[:, ylo, :] * wy0[None, :, None] + feat[:, yhi, :] * wy1[None, :, None]      # (C, out*gh, W)
        s = rows[:, :, xlo] * wx0[None, None] + rows[:, :, xhi] * wx1[None, None]               # (C, out*gh, out*gw)
        res[k] = s.reshape(c, out, gh, out, gw).mean((2, 4))
    return res


def box_levels(boxes: np.ndarray) -> np.ndarray:
    """ROIPooler level assignment (canonical 224 px on p4): 0..3 for p2..p5."""
    size = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    return (np.clip(np.floor(4 + np.log2(size / 224 + 1e-8)), 2, 5) - 2).astype(np.int64)


def pool_box(feats, boxes: np.ndarray, out: int) -> np.ndarray:
    """p2..p5 (each (1,C,H,W)) + (K,4) boxes -> (K,C,out,out), each box pooled from the level it is assigned to."""
    lv = box_levels(boxes)
    res = np.zeros((len(boxes), feats[0].shape[1], out, out), np.float32)
    for i, f in enumerate(feats):
        sel = lv == i
        if sel.any():
            res[sel] = roi_align(f[0], boxes[sel], out, 1.0 / STRIDES[i])
    return res


def pool_dp(p2: np.ndarray, boxes: np.ndarray, out: int) -> np.ndarray:
    """DensePose ROIs always come from p2 (decoder returns features[0])."""
    return roi_align(p2[0], boxes, out, 1.0 / STRIDES[0])


def detections(probs: np.ndarray, deltas: np.ndarray, proposals: np.ndarray, image_hw):
    """Graph B1 outputs -> (boxes (D,4), scores (D,)), detectron2's fast_rcnn_inference for one class, by decreasing score."""
    boxes = clip(decode(deltas, proposals, BOX_WEIGHTS), *image_hw)
    scores = probs[:, 0]
    sel = np.nonzero(scores > BOX_SCORE_THRESH)[0]
    keep = nms(boxes[sel], scores[sel], BOX_NMS)[:BOX_TOPK]
    return boxes[sel][keep], scores[sel][keep]


def person_pixels(embedding: np.ndarray, coarse: np.ndarray):
    """B2 outputs -> person mask (K,R,R) (coarse segmentation argmax > 0, as densepose) and the embeddings (N,16) of its pixels."""
    mask = coarse.argmax(1) > 0
    return mask, np.ascontiguousarray(embedding.transpose(0, 2, 3, 1)[mask])
