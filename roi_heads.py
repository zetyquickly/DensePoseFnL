from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from densepose import DensePoseROIHeads
from densepose.modeling.roi_heads import DensePoseDeepLabHead
from detectron2.modeling import ROI_HEADS_REGISTRY


@ROI_HEADS_REGISTRY.register()
class MyDensePoseROIHeads(DensePoseROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        if self.use_decoder and cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ISSIMPLE:
            self.decoder = SimpleDecoder()
        if cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM == "" and isinstance(self.densepose_head, DensePoseDeepLabHead):
            # we want to remove group norms, return bias back and change dilation from 56 to 18, make relu inplace
            conv: nn.Conv2d = self.densepose_head.ASPP.convs[0][0]
            self.densepose_head.ASPP.convs[0] = ASPPConv(conv.in_channels, conv.out_channels, 1, 1)
            for i, dilation in zip([1, 2, 3], [6, 12, 18]):
                conv: nn.Conv2d = self.densepose_head.ASPP.convs[i][0]
                self.densepose_head.ASPP.convs[i] = ASPPConv(conv.in_channels, conv.out_channels, dilation)
            conv: nn.Conv2d = self.densepose_head.ASPP.convs[4][1]
            self.densepose_head.ASPP.convs[4] = ASPPPooling(conv.in_channels, conv.out_channels)
            self.densepose_head.ASPP.project = nn.Sequential(
                nn.Conv2d(5 * conv.out_channels, conv.out_channels, (1, 1)),
                nn.ReLU(True)
            )
        self.densepose_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class SimpleDecoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, features: List[torch.Tensor]):
        return features[0]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        if kernel_size > 1:
            padding = dilation
        else:
            padding = 0
            assert dilation == 1

        modules = [
            nn.Conv2d(
                in_channels, out_channels, (kernel_size, kernel_size), padding=padding, dilation=dilation
            ),
            nn.ReLU(True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.ReLU(True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
