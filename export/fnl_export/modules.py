"""Detectron2-free re-implementation of Mobile-Parsing-RCNN-B (inference only): CSE head or chart (part + UV) head.

Submodule / parameter names mirror the original checkpoint so that its state dict loads strictly.
"""
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

FPN_LEVELS = ("p2", "p3", "p4", "p5", "p6")


class DepthwiseSeparableConv2d(nn.Sequential):
    def __init__(self, channels, stride=1):
        super().__init__(
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )


class Conv3x3BnReLU(nn.Sequential):
    def __init__(self, channels, stride=1):
        super().__init__(
            DepthwiseSeparableConv2d(channels, stride),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )


class FastNormalizedFusion(nn.Module):
    def __init__(self, in_nodes):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_nodes))
        self.register_buffer("eps", torch.tensor(0.0001))

    def forward(self, xs: List[torch.Tensor]):
        w = F.relu(self.weight)
        out = xs[0] * w[0]
        for i in range(1, len(xs)):
            out = out + xs[i] * w[i]
        return out / (w.sum() + self.eps)


class BiFPN(nn.Module):
    """timm EfficientNet features (p2..p5 with 'expansion' taps) -> BiFPN -> p2..p6."""

    def __init__(self, timm_name="spnasnet_100", out_channels=64):
        super().__init__()
        self.bottom_up = timm.create_model(
            timm_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4),
            feature_location="expansion",
        )
        ch = [f["num_chs"] for f in self.bottom_up.feature_info]
        self.l2 = nn.Conv2d(ch[1], out_channels, 1)
        self.l3 = nn.Conv2d(ch[2], out_channels, 1)
        self.l4 = nn.Conv2d(ch[3], out_channels, 1)
        self.l5 = nn.Conv2d(ch[4], out_channels, 1)
        self.p4_tr = Conv3x3BnReLU(out_channels)
        self.p3_tr = Conv3x3BnReLU(out_channels)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p4_tr = FastNormalizedFusion(2)
        self.fuse_p3_tr = FastNormalizedFusion(2)
        self.down_p2 = Conv3x3BnReLU(out_channels, stride=2)
        self.down_p3 = Conv3x3BnReLU(out_channels, stride=2)
        self.down_p4 = Conv3x3BnReLU(out_channels, stride=2)
        self.fuse_p5_out = FastNormalizedFusion(2)
        self.fuse_p4_out = FastNormalizedFusion(3)
        self.fuse_p3_out = FastNormalizedFusion(3)
        self.fuse_p2_out = FastNormalizedFusion(2)
        self.p5_out = Conv3x3BnReLU(out_channels)
        self.p4_out = Conv3x3BnReLU(out_channels)
        self.p3_out = Conv3x3BnReLU(out_channels)
        self.p2_out = Conv3x3BnReLU(out_channels)

    def forward(self, x):
        p2, p3, p4, p5 = self.bottom_up(x)
        p5, p4, p3, p2 = self.l5(p5), self.l4(p4), self.l3(p3), self.l2(p2)
        p4_tr = self.p4_tr(self.fuse_p4_tr([p4, self.up(p5)]))
        p3_tr = self.p3_tr(self.fuse_p3_tr([p3, self.up(p4_tr)]))
        p2_out = self.p2_out(self.fuse_p2_out([p2, self.up(p3_tr)]))
        p3_out = self.p3_out(self.fuse_p3_out([p3, p3_tr, self.down_p2(p2_out)]))
        p4_out = self.p4_out(self.fuse_p4_out([p4, p4_tr, self.down_p3(p3_out)]))
        p5_out = self.p5_out(self.fuse_p5_out([p5, self.down_p4(p4_out)]))
        p6_out = F.max_pool2d(p5_out, kernel_size=1, stride=2, padding=0)  # detectron2 LastLevelMaxPool
        return p2_out, p3_out, p4_out, p5_out, p6_out


class RPNHead(nn.Module):
    def __init__(self, channels=64, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.objectness_logits = nn.Conv2d(channels, num_anchors, 1)
        self.anchor_deltas = nn.Conv2d(channels, num_anchors * 4, 1)

    def forward(self, features: List[torch.Tensor]):
        logits, deltas = [], []
        for x in features:
            t = F.relu(self.conv(x))
            logits.append(self.objectness_logits(t))
            deltas.append(self.anchor_deltas(t))
        return logits, deltas


class BoxHead(nn.Module):
    """FastRCNNConvFCHead with 2 convs, no fc."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.flatten(F.relu(self.conv2(F.relu(self.conv1(x)))), 1)


class BoxPredictor(nn.Module):
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes + 1)
        self.bbox_pred = nn.Linear(in_features, 4)


class ASPP(nn.Module):
    """Variant used by this repo (no norm, dilations 6/12/18)."""

    def __init__(self, channels=64):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU()),
            *[nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d), nn.ReLU()) for d in (6, 12, 18)],
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.ReLU()),
        ])
        self.project = nn.Sequential(nn.Conv2d(5 * channels, channels, 1), nn.ReLU())

    def forward(self, x):
        h, w = x.shape[-2:]
        branches = [c(x) for c in self.convs[:4]]
        # bilinear upsampling of a 1x1 map is a constant broadcast; nearest resize is the same op
        # but lowers to a plain RESIZE_NEAREST_NEIGHBOR (expand() became a BROADCAST_TO that failed in tflite)
        branches.append(F.interpolate(self.convs[4](x), size=(h, w), mode="nearest"))
        return self.project(torch.cat(branches, dim=1))


class DensePoseDeepLabHead(nn.Module):
    def __init__(self, channels=64, num_convs=4):
        super().__init__()
        self.ASPP = ASPP(channels)
        self.num_convs = num_convs
        for i in range(num_convs):
            self.add_module(f"body_conv_fcn{i + 1}", nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        x = self.ASPP(x)
        for i in range(self.num_convs):
            x = F.relu(getattr(self, f"body_conv_fcn{i + 1}")(x))
        return x


class CSEPredictor(nn.Module):
    def __init__(self, channels=64, embed_size=16, num_coarse=2, kernel=4):
        super().__init__()
        pad = kernel // 2 - 1
        self.coarse_segm_lowres = nn.ConvTranspose2d(channels, num_coarse, kernel, stride=2, padding=pad)
        self.embed_lowres = nn.ConvTranspose2d(channels, embed_size, kernel, stride=2, padding=pad)

    def forward(self, h):
        return self.embed_lowres(h), self.coarse_segm_lowres(h)


CHART_OUTPUTS = ("coarse", "fine", "u", "v", "sigma_2", "kappa_u", "kappa_v", "fine_conf", "coarse_conf")


class ChartPredictor(nn.Module):
    """DensePoseChartWithConfidencePredictor (UP_SCALE 1, indep_aniso UV confidence, segm confidence), raw outputs.

    Returns the transposed-conv outputs in CHART_OUTPUTS order: coarse (2), fine part index (25, 0 = background),
    U, V (25 each); with confidence also sigma_2, kappa_u, kappa_v (25 each) and fine/coarse segm confidence (1 each).
    The softplus + epsilon that detectron2 applies to the segm confidences is left to the host.
    """

    def __init__(self, channels=64, num_coarse=2, num_patches=24, kernel=4, confidence=True):
        super().__init__()
        pad = kernel // 2 - 1
        ct = lambda n: nn.ConvTranspose2d(channels, n, kernel, stride=2, padding=pad)
        self.ann_index_lowres = ct(num_coarse)
        self.index_uv_lowres = ct(num_patches + 1)
        self.u_lowres = ct(num_patches + 1)
        self.v_lowres = ct(num_patches + 1)
        self.confidence = confidence
        if confidence:
            self.sigma_2_lowres = ct(num_patches + 1)
            self.kappa_u_lowres = ct(num_patches + 1)
            self.kappa_v_lowres = ct(num_patches + 1)
            self.fine_segm_confidence_lowres = ct(1)
            self.coarse_segm_confidence_lowres = ct(1)

    def forward(self, h):
        out = [self.ann_index_lowres(h), self.index_uv_lowres(h), self.u_lowres(h), self.v_lowres(h)]
        if self.confidence:
            out += [self.sigma_2_lowres(h), self.kappa_u_lowres(h), self.kappa_v_lowres(h),
                    self.fine_segm_confidence_lowres(h), self.coarse_segm_confidence_lowres(h)]
        return tuple(out)


class VertexFeatureEmbedder(nn.Module):
    def __init__(self, num_vertices=27554, feature_dim=256, embed_dim=16):
        super().__init__()
        self.register_buffer("features", torch.zeros(num_vertices, feature_dim))
        self.embeddings = nn.Parameter(torch.zeros(feature_dim, embed_dim))

    def forward(self):
        e = self.features @ self.embeddings
        return e / torch.clamp(e.norm(dim=1, keepdim=True), min=1e-6)


class Embedder(nn.Module):
    def __init__(self, mesh="smpl_27554", **kw):
        super().__init__()
        self.add_module(f"embedder_{mesh}", VertexFeatureEmbedder(**kw))
        self.mesh = mesh

    def forward(self):
        return getattr(self, f"embedder_{self.mesh}")()


class _Container(nn.Module):
    pass


class MobileParsingRCNN(nn.Module):
    """Holder whose submodule paths equal the detectron2 checkpoint prefixes."""

    def __init__(self, channels=64, timm_name="spnasnet_100", head="cse", chart_confidence=True):
        super().__init__()
        self.backbone = BiFPN(timm_name, channels)
        self.proposal_generator = _Container()
        self.proposal_generator.rpn_head = RPNHead(channels)
        self.roi_heads = _Container()
        self.roi_heads.box_head = BoxHead(channels)
        self.roi_heads.box_predictor = BoxPredictor(channels * 7 * 7)
        self.roi_heads.densepose_head = DensePoseDeepLabHead(channels)
        assert head in ("cse", "chart"), head
        if head == "cse":
            self.roi_heads.densepose_predictor = CSEPredictor(channels)
            self.roi_heads.embedder = Embedder()
        else:
            self.roi_heads.densepose_predictor = ChartPredictor(channels, confidence=chart_confidence)
