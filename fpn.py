from typing import List

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

__all__ = ["BiFPN"]


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class DepthwiseSeparableConv2d(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class Conv3x3BnReLU(nn.Sequential):

    def __init__(self, in_channels, stride=1):
        conv = DepthwiseSeparableConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            bias=False,
            padding=1,
            stride=stride,
        )
        if get_world_size() > 1:
            bn = nn.SyncBatchNorm(in_channels, momentum=0.03)
        else:
            bn = nn.BatchNorm2d(in_channels, momentum=0.03)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class FastNormalizedFusion(nn.Module):

    def __init__(self, in_nodes):
        super().__init__()
        self.in_nodes = in_nodes
        self.weight = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(0.0001))

    def forward(self, x: List[torch.Tensor]):
        if len(x) != self.in_nodes:
            raise RuntimeError(
                "Expected to have {} input nodes, but have {}.".format(self.in_nodes, len(x))
            )

        # where wi ≥ 0 is ensured by applying a relu after each wi (paper)
        weight = F.relu(self.weight)
        x_sum = 0
        for xi, wi in zip(x, weight):
            x_sum = x_sum + xi * wi
        normalized_weighted_x = x_sum / (weight.sum() + self.eps)
        return normalized_weighted_x


class BiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, out_channels, top_block=None):
        super().__init__()

        self.bottom_up = bottom_up
        self.top_block = top_block

        self.l5 = nn.Conv2d(bottom_up.feature_info[4]['num_chs'], out_channels, kernel_size=1)
        self.l4 = nn.Conv2d(bottom_up.feature_info[3]['num_chs'], out_channels, kernel_size=1)
        self.l3 = nn.Conv2d(bottom_up.feature_info[2]['num_chs'], out_channels, kernel_size=1)
        self.l2 = nn.Conv2d(bottom_up.feature_info[1]['num_chs'], out_channels, kernel_size=1)

        self.p4_tr = Conv3x3BnReLU(out_channels)
        self.p3_tr = Conv3x3BnReLU(out_channels)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuse_p4_tr = FastNormalizedFusion(in_nodes=2)
        self.fuse_p3_tr = FastNormalizedFusion(in_nodes=2)

        self.down_p2 = Conv3x3BnReLU(out_channels, stride=2)
        self.down_p3 = Conv3x3BnReLU(out_channels, stride=2)
        self.down_p4 = Conv3x3BnReLU(out_channels, stride=2)

        self.fuse_p5_out = FastNormalizedFusion(in_nodes=2)
        self.fuse_p4_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p3_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p2_out = FastNormalizedFusion(in_nodes=2)

        self.p5_out = Conv3x3BnReLU(out_channels)
        self.p4_out = Conv3x3BnReLU(out_channels)
        self.p3_out = Conv3x3BnReLU(out_channels)
        self.p2_out = Conv3x3BnReLU(out_channels)

        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = 32
        self._out_feature_strides = {}
        for k, name in enumerate(self._out_features):
            self._out_feature_strides[name] = 2 ** (k + 2)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        p2, p3, p4, p5 = self.bottom_up(x)

        if self.training:
            _dummy = sum(x.view(-1)[0] for x in self.bottom_up.parameters()) * 0.0
            p5 = p5 + _dummy

        p5 = self.l5(p5)
        p4 = self.l4(p4)
        p3 = self.l3(p3)
        p2 = self.l2(p2)

        p4_tr = self.p4_tr(self.fuse_p4_tr([p4, self.up(p5)]))
        p3_tr = self.p3_tr(self.fuse_p3_tr([p3, self.up(p4_tr)]))

        p2_out = self.p2_out(self.fuse_p2_out([p2, self.up(p3_tr)]))
        p3_out = self.p3_out(self.fuse_p3_out([p3, p3_tr, self.down_p2(p2_out)]))
        p4_out = self.p4_out(self.fuse_p4_out([p4, p4_tr, self.down_p3(p3_out)]))
        p5_out = self.p5_out(self.fuse_p5_out([p5, self.down_p4(p4_out)]))

        return {"p2": p2_out, "p3": p3_out, "p4": p4_out, "p5": p5_out, "p6": self.top_block(p5_out)[0]}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


if __name__ == "__main__":
    m = timm.create_model('spnasnet_100', pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))
    x = torch.rand(1, 3, 224, 224)
    m2 = BiFPN(bottom_up=m, out_channels=112, top_block=LastLevelMaxPool())
    # torch.jit.trace(m2, x)
    m2 = torch.jit.script(m2)
    print(m2(x))
