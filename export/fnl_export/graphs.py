"""Static graphs that get exported. Everything dynamic (anchors, decode, NMS, ROIAlign) lives on the host."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExportConfig
from .modules import MobileParsingRCNN


class GraphA(nn.Module):
    """image (1,3,H,W) RGB 0..255 -> p2..p5 features, RPN objectness logits + deltas for p2..p6."""

    def __init__(self, m: MobileParsingRCNN, cfg: ExportConfig):
        super().__init__()
        self.backbone = m.backbone
        self.rpn_head = m.proposal_generator.rpn_head
        self.register_buffer("mean", torch.tensor(cfg.pixel_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.pixel_std).view(1, 3, 1, 1))

    def forward(self, image):
        feats = self.backbone((image - self.mean) / self.std)
        logits, deltas = self.rpn_head(list(feats))
        return (*feats[:4], *logits, *deltas)


class GraphB1(nn.Module):
    """pooled box ROIs (K1,64,7,7) -> class probabilities (K1,2) and box deltas (K1,4)."""

    def __init__(self, m: MobileParsingRCNN):
        super().__init__()
        self.head = m.roi_heads.box_head
        self.pred = m.roi_heads.box_predictor

    def forward(self, rois):
        h = self.head(rois)
        return F.softmax(self.pred.cls_score(h), dim=-1), self.pred.bbox_pred(h)


class GraphB2(nn.Module):
    """pooled DensePose ROIs (K2,64,32,32) -> the predictor's 64x64 outputs.

    cse:   embeddings (K2,16,64,64), coarse segmentation (K2,2,64,64)
    chart: coarse (2), fine part (25), U (25), V (25) and with confidence sigma_2, kappa_u, kappa_v (25 each), fine/coarse
           segm confidence (1 each), all (K2,C,64,64); see modules.CHART_OUTPUTS.
    """

    def __init__(self, m: MobileParsingRCNN):
        super().__init__()
        self.head = m.roi_heads.densepose_head
        self.pred = m.roi_heads.densepose_predictor

    def forward(self, rois):
        h = self.head(rois)
        return self.pred(h)


def kd_blocks(v: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """(n_blocks, b) vertex ids: balanced kd-tree leaves over the embeddings, in tree order (neighbouring leaves are
    close in embedding space), padded to equal size by repeating a member."""
    def split(idx, depth):
        if depth == 0:
            return [idx]
        o = idx[v[idx, v[idx].var(0).argmax()].argsort()]
        return split(o[:len(o) // 2], depth - 1) + split(o[len(o) // 2:], depth - 1)
    leaves = split(torch.arange(len(v)), int(math.log2(n_blocks)))
    b = max(map(len, leaves))
    return torch.stack([torch.cat([x, x[:1].expand(b - len(x))]) for x in leaves])


class GraphC(nn.Module):
    """embeddings (K2,16,R,R) -> closest SMPL vertex id per pixel (K2,R,R).

    Vertex embeddings are unit-norm, so argmin ||e-v||^2 == argmax e.v (same result as densepose's EDM argmin).
    """

    def __init__(self, m: MobileParsingRCNN, cfg: ExportConfig):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("vertices", m.roi_heads.embedder().clone())  # (27554, 16)
        self.res = cfg.cse_match_resolution

    def forward(self, emb):
        if emb.shape[-1] != self.res:
            emb = F.interpolate(emb, size=(self.res, self.res), mode="bilinear", align_corners=False)
        k, d, h, w = emb.shape
        pix = emb.permute(0, 2, 3, 1).reshape(k * h * w, d)
        ids = torch.argmax(pix @ self.vertices.t(), dim=-1)
        return ids.reshape(k, h, w).to(torch.int32)


class GraphCOptimized(nn.Module):
    """GraphC, approximate and flat: pixel embeddings (N,16) -> closest SMPL vertex id per pixel (N,) int32.

    Two-stage search instead of one (N,27554) matmul + argmax. The vertices are cut into L kd-tree blocks; a pixel is scored against
    the L block centroids, one ARG_MAX finds the best block, and the exact argmax runs over the aligned group of
    cse_group neighbouring blocks around it. Only ARG_MAX is used to pick blocks (TOPK_V2 and REDUCE_MAX are
    single-threaded and several times slower in TFLite).
    """

    def __init__(self, m: MobileParsingRCNN, cfg: ExportConfig):
        super().__init__()
        with torch.no_grad():
            v = m.roi_heads.embedder().clone()                        # (27554, 16)
        self.group = cfg.cse_group
        assert cfg.cse_blocks & (cfg.cse_blocks - 1) == 0 and cfg.cse_blocks % self.group == 0
        idx = kd_blocks(v, cfg.cse_blocks)                            # (L, b)
        self.register_buffer("vb", v[idx])                            # (L, b, 16)
        self.register_buffer("cent", v[idx].mean(1))                  # (L, 16)
        self.register_buffer("ids", idx.to(torch.int32))
        self.register_buffer("offs", torch.arange(self.group))

    def forward(self, x):
        n = x.shape[0]
        g = torch.div((x @ self.cent.t()).argmax(-1), self.group, rounding_mode="floor")
        top = g[:, None] * self.group + self.offs                     # (N, T) block ids
        best = torch.bmm(self.vb[top].view(n, -1, 16), x[:, :, None])[..., 0].argmax(-1, keepdim=True)
        return self.ids[top].view(n, -1).gather(1, best)[:, 0]


def build_graphs(m: MobileParsingRCNN, cfg: ExportConfig):
    """name -> (module, example inputs)"""
    g = {
        "graph_a": (GraphA(m, cfg).eval(), (torch.rand(1, 3, cfg.input_height, cfg.input_width) * 255,)),
        "graph_b1": (GraphB1(m).eval(), (torch.rand(cfg.k_proposals, 64, cfg.box_pool, cfg.box_pool),)),
        "graph_b2": (GraphB2(m).eval(), (torch.rand(cfg.k_detections, 64, cfg.dp_pool, cfg.dp_pool),)),
    }
    if cfg.head == "cse" and cfg.cse_match == "in_graph":
        g["graph_c"] = (GraphC(m, cfg).eval(), (torch.rand(cfg.k_detections, 16, 64, 64),))
    if cfg.head == "cse" and cfg.cse_match == "optimized":
        with torch.no_grad():   # vertex embeddings + noise: like real pixels, unlike gaussian noise full of near-ties
            px = m.roi_heads.embedder()[torch.randint(27554, (cfg.cse_pixels,))] + 0.03 * torch.randn(cfg.cse_pixels, 16)
        g["graph_c"] = (GraphCOptimized(m, cfg).eval(), (px,))
    return g
