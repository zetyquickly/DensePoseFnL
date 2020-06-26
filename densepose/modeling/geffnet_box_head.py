import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import ShapeSpec, BatchNorm2d, NaiveSyncBatchNorm
from detectron2.modeling import ROI_BOX_HEAD_REGISTRY
from geffnet import efficientnet_builder
from geffnet.efficientnet_builder import initialize_weight_default, EfficientNetBuilder, decode_arch_def
from torch import nn
from torch.nn import functional as F

from densepose.modeling.converters import convert_conv2d_to_detectron2_format


class EfficientFastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, **model_kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        block_args = model_kwargs.pop('block_args')

        pool_size = model_kwargs.pop('pool_size', None)
        self.has_pooling = pool_size is not None

        assert norm is None or norm in ["BN", "SyncBN"]
        if norm == "BN":
            model_kwargs['norm_layer'] = BatchNorm2d
        elif norm == "SyncBN":
            model_kwargs['norm_layer'] = NaiveSyncBatchNorm
        builder = EfficientNetBuilder(**model_kwargs)

        self.conv = nn.Sequential(*builder(input_shape.channels, block_args))

        if self.has_pooling:
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self._output_size = (block_args[-1][-1]['out_chs'],) + pool_size
        else:
            stride = np.prod([b[0]['stride'] for b in block_args])
            self._output_size = (block_args[-1][-1]['out_chs'],
                                 input_shape.height // stride,
                                 input_shape.width // stride)
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        initialize_weight_default(self.conv)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        x = self.conv(x)
        if self.has_pooling:
            x = self.pool(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size

