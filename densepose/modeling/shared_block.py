import torch
import torch.nn as nn
from detectron2.utils.registry import Registry

from densepose import Interpolate, ROI_DENSEPOSE_HEAD_REGISTRY
from densepose.modeling.layers import non_local_embedded_gaussian

ROI_SHARED_BLOCK_REGISTRY = Registry("ROI_SHARED_BLOCK")
ROI_SHARED_BLOCK_REGISTRY.__doc__ = """
Registry for shared blocks, which computes shared features given per-region features
Mask, keypoint, densepose heads use shared features for predictions

The registered object will be called with `obj(cfg, input_shape)`.
"""

from detectron2.layers import Conv2d, interpolate


def build_shared_block(cfg, in_channels):
    """
    Build a shared block defined by `cfg.MODEL.ROI_SHARED_BLOCK.NAME`.
    """
    name = cfg.MODEL.ROI_SHARED_BLOCK.NAME
    if name == "":
        return None, in_channels
    model = ROI_SHARED_BLOCK_REGISTRY.get(name)(cfg, in_channels)
    return model, model.n_out_channels


@ROI_SHARED_BLOCK_REGISTRY.register()
class ParsingSharedBlock(nn.Module):

    def __init__(self, cfg, in_channels):
        """Creates a ParsingSharedBlock

        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers

        Ref impl: https://github.com/soeaver/Parsing-R-CNN/blob/master/parsingrcnn/modeling/uv_rcnn_heads.py
        Paper: https://arxiv.org/abs/1811.12596
        """
        super(ParsingSharedBlock, self).__init__()

        # fmt: off
        self.has_aspp = cfg.MODEL.ROI_SHARED_BLOCK.ASPP_ON
        aspp_resolution = cfg.MODEL.ROI_SHARED_HEAD.POOLER_RESOLUTION
        aspp_dim = cfg.MODEL.ROI_SHARED_BLOCK.ASPP_DIM
        d1, d2, d3 = cfg.MODEL.ROI_SHARED_BLOCK.ASPP_DILATION

        num_convs_before_aspp_nl = cfg.MODEL.ROI_SHARED_BLOCK.NUM_CONVS_BEFORE_ASPP_NL
        num_convs_after_aspp_nl = cfg.MODEL.ROI_SHARED_BLOCK.NUM_CONVS_AFTER_ASPP_NL
        hidden_dim = cfg.MODEL.ROI_SHARED_BLOCK.CONV_HEAD_DIM
        kernel_size = cfg.MODEL.ROI_SHARED_BLOCK.CONV_HEAD_KERNEL

        self.has_nonlocal = cfg.MODEL.ROI_SHARED_BLOCK.NONLOCAL_ON
        nl_reduction_ratio = cfg.MODEL.ROI_SHARED_BLOCK.NONLOCAL_REDUCTION_RATIO
        nl_use_maxpool = cfg.MODEL.ROI_SHARED_BLOCK.NONLOCAL_USE_MAXPOOL
        nl_use_bn = cfg.MODEL.ROI_SHARED_BLOCK.NONLOCAL_USE_BN

        # fmt: on
        pad_size = kernel_size // 2
        dim_in = in_channels

        self.has_conv_before_aspp_nl = num_convs_before_aspp_nl > 0
        if self.has_conv_before_aspp_nl:
            self.conv_before_aspp_nl = []
            for _ in range(num_convs_before_aspp_nl):
                self.conv_before_aspp_nl.append(Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
                self.conv_before_aspp_nl.append(nn.ReLU(inplace=True))
                dim_in = hidden_dim
            self.conv_before_aspp_nl = nn.Sequential(*self.conv_before_aspp_nl)

        if self.has_aspp:
            self.aspp1 = nn.Sequential(
                Conv2d(dim_in, aspp_dim, 1, 1),
                nn.ReLU(inplace=True))
            self.aspp2 = nn.Sequential(
                Conv2d(dim_in, aspp_dim, 3, 1, d1, dilation=d1),
                nn.ReLU(inplace=True))
            self.aspp3 = nn.Sequential(
                Conv2d(dim_in, aspp_dim, 3, 1, d2, dilation=d2),
                nn.ReLU(inplace=True))
            self.aspp4 = nn.Sequential(
                Conv2d(dim_in, aspp_dim, 3, 1, d3, dilation=d3),
                nn.ReLU(inplace=True))
            self.aspp5 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Conv2d(dim_in, aspp_dim, 1, 1),
                nn.ReLU(inplace=True),
                Interpolate(scale_factor=aspp_resolution))
            self.aspp_agg = nn.Sequential(
                Conv2d(aspp_dim * 5, hidden_dim, 1, 1),
                nn.ReLU(inplace=True))

        if self.has_nonlocal:
            self.non_local = non_local_embedded_gaussian.NONLocalBlock2D(
                hidden_dim,
                int(hidden_dim * nl_reduction_ratio),
                sub_sample=nl_use_maxpool,
                bn_layer=nl_use_bn)

        self.has_conv_after_aspp_nl = num_convs_after_aspp_nl > 0
        if self.has_conv_after_aspp_nl:
            self.conv_after_aspp_nl = []
            for _ in range(num_convs_after_aspp_nl):
                self.conv_after_aspp_nl.append(Conv2d(hidden_dim, hidden_dim, kernel_size, 1, pad_size))
                self.conv_after_aspp_nl.append(nn.ReLU(inplace=True))
            self.conv_after_aspp_nl = nn.Sequential(*self.conv_after_aspp_nl)

        self.n_out_channels = hidden_dim

        if self.has_conv_before_aspp_nl:
            self.conv_before_aspp_nl.apply(self._init_weights)
        if self.has_aspp:
            self.aspp1.apply(self._init_weights)
            self.aspp2.apply(self._init_weights)
            self.aspp3.apply(self._init_weights)
            self.aspp4.apply(self._init_weights)
            self.aspp_agg.apply(self._init_weights)
        if self.has_conv_after_aspp_nl:
            self.conv_after_aspp_nl.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features):
        x = features
        if self.has_conv_before_aspp_nl:
            x = self.conv_before_aspp_nl(x)
        if self.has_aspp:
            x = torch.cat((self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x), self.aspp5(x)), dim=1)
            x = self.aspp_agg(x)
        if self.has_nonlocal:
            x = self.non_local(x)
        if self.has_conv_after_aspp_nl:
            x = self.conv_after_aspp_nl(x)
        return x


class DepthwiseSeparableASPP(nn.Module):

    def __init__(self, in_chs, out_chs, aspp_dilation, aspp_resolution):
        super(DepthwiseSeparableASPP, self).__init__()
        d1, d2, d3 = aspp_dilation

        # self.conv_dw = select_conv2d(in_chs, out_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.aspp1 = nn.Sequential(Conv2d(in_chs, in_chs, 1, 1, groups=in_chs), nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(Conv2d(in_chs, in_chs, 3, 1, d1, dilation=d1, groups=in_chs), nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(Conv2d(in_chs, in_chs, 3, 1, d2, dilation=d2, groups=in_chs), nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(Conv2d(in_chs, in_chs, 3, 1, d3, dilation=d3, groups=in_chs), nn.ReLU(inplace=True))
        self.aspp5 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   Conv2d(in_chs, in_chs, 1, 1, groups=in_chs), nn.ReLU(inplace=True),
                                   Interpolate(scale_factor=aspp_resolution))
        self.aspp_agg = nn.Sequential(
            Conv2d(in_chs * 5, out_chs, 1, 1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        try:
            a1 = self.aspp1(x)
            a2 = self.aspp2(x)
            a3 = self.aspp3(x)
            a4 = self.aspp4(x)
            a5 = self.aspp5(x)
        except RuntimeError:
            print(x.numel())
        x = torch.cat((a1, a2, a3, a4, a5), dim=1)
        # x = torch.cat((self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x), self.aspp5(x)), dim=1)
        x = self.aspp_agg(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chs, out_chs, dw_kernel_size=3):
        super(DepthwiseSeparableConv, self).__init__()

        self.conv_dw = Conv2d(in_chs, in_chs, dw_kernel_size, padding=dw_kernel_size // 2, groups=in_chs)
        self.act1 = nn.ReLU(inplace=True)

        self.conv_pw = Conv2d(in_chs, in_chs, 1, groups=in_chs)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.act1(x)
        x = self.conv_pw(x)
        x = self.act2(x)
        return x


@ROI_SHARED_BLOCK_REGISTRY.register()
class MobileParsingSharedBlock(nn.Module):
    NUM_ANN_INDICES = 15

    def __init__(self, cfg, in_channels):
        super(MobileParsingSharedBlock, self).__init__()

        # fmt: off
        self.has_aspp = cfg.MODEL.ROI_SHARED_BLOCK.ASPP_ON
        aspp_resolution = cfg.MODEL.ROI_SHARED_HEAD.POOLER_RESOLUTION
        d1, d2, d3 = cfg.MODEL.ROI_SHARED_BLOCK.ASPP_DILATION

        num_convs_before_aspp_nl = cfg.MODEL.ROI_SHARED_BLOCK.NUM_CONVS_BEFORE_ASPP_NL
        num_convs_after_aspp_nl = cfg.MODEL.ROI_SHARED_BLOCK.NUM_CONVS_AFTER_ASPP_NL
        hidden_dim = cfg.MODEL.ROI_SHARED_BLOCK.CONV_HEAD_DIM
        kernel_size = cfg.MODEL.ROI_SHARED_BLOCK.CONV_HEAD_KERNEL

        # fmt: on
        pad_size = kernel_size // 2
        dim_in = in_channels

        if self.has_aspp:
            self.aspp = DepthwiseSeparableASPP(dim_in, hidden_dim, (d1, d2, d3), aspp_resolution)

        self.has_conv_after_aspp_nl = num_convs_after_aspp_nl > 0
        if self.has_conv_after_aspp_nl:
            self.conv_after_aspp_nl = []
            for _ in range(num_convs_after_aspp_nl):
                self.conv_after_aspp_nl.append(DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size))
            self.conv_after_aspp_nl = nn.Sequential(*self.conv_after_aspp_nl)

        self.n_out_channels = hidden_dim

        if self.has_aspp:
            self.aspp.apply(self._init_weights)
        if self.has_conv_after_aspp_nl:
            self.conv_after_aspp_nl.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features):
        if self.has_aspp:
            features = self.aspp(features)
        if self.has_conv_after_aspp_nl:
            features = self.conv_after_aspp_nl(features)
        return features


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseConv3Head(nn.Module):
    NUM_ANN_INDICES = 15

    def __init__(self, cfg, input_channels):
        super(DensePoseConv3Head, self).__init__()
        dim_out_ann_index = self.NUM_ANN_INDICES
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        self.ann_index_lowres = Conv2d(input_channels, dim_out_ann_index, kernel_size=3, stride=1, padding=1)
        self.index_uv_lowres = Conv2d(input_channels, dim_out_patches, kernel_size=3, stride=1, padding=1)
        self.u_lowres = Conv2d(input_channels, dim_out_patches, kernel_size=3, stride=1, padding=1)
        self.v_lowres = Conv2d(input_channels, dim_out_patches, kernel_size=3, stride=1, padding=1)
        self.n_out_channels = input_channels

    def forward(self, features):
        ann_index = self.ann_index_lowres(features)
        index_uv = self.index_uv_lowres(features)
        u = self.u_lowres(features)
        v = self.v_lowres(features)
        return (ann_index, index_uv, u, v), (None, None, None, None)


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseNearestConv3Head(nn.Module):
    NUM_ANN_INDICES = 15

    def __init__(self, cfg, input_channels):
        super(DensePoseNearestConv3Head, self).__init__()
        dim_out_ann_index = self.NUM_ANN_INDICES
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        self.ann_index_lowres = Conv2d(input_channels, dim_out_ann_index, kernel_size=3, stride=1, padding=1)
        self.index_uv_lowres = Conv2d(input_channels, dim_out_patches, kernel_size=3, stride=1, padding=1)
        self.u_lowres = Conv2d(input_channels, dim_out_patches, kernel_size=3, stride=1, padding=1)
        self.v_lowres = Conv2d(input_channels, dim_out_patches, kernel_size=3, stride=1, padding=1)
        self.n_out_channels = input_channels

    def forward(self, features):
        features = interpolate(features, scale_factor=2, mode="nearest")
        ann_index = self.ann_index_lowres(features)
        index_uv = self.index_uv_lowres(features)
        u = self.u_lowres(features)
        v = self.v_lowres(features)
        return (ann_index, index_uv, u, v), (None, None, None, None)

#
# if __name__ == "__main__":
#     import torch
#     from detectron2.layers import Conv2d
#
#     x = torch.zeros(size=(0, 20, 100, 100))
#     model = Conv2d(20, 20, 1, 1, groups=20)
#     print(model(x))
