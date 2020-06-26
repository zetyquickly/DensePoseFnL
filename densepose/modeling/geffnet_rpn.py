from typing import List

import torch.nn as nn
from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm, BatchNorm2d
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN_HEAD_REGISTRY
from geffnet import efficientnet_builder
from geffnet.efficientnet_builder import EfficientNetBuilder, decode_arch_def, initialize_weight_default

class EfficientRPNHead(StandardRPNHead):

    def __init__(self, cfg, input_shape: List[ShapeSpec], **model_kwargs):
        super().__init__(cfg, input_shape)

        block_args = model_kwargs.pop('block_args')

        norm = cfg.MODEL.RPN.NORM
        assert norm is None or norm in ["BN", "SyncBN"]
        if norm == "BN":
            model_kwargs['norm_layer'] = BatchNorm2d
        elif norm == "SyncBN":
            model_kwargs['norm_layer'] = NaiveSyncBatchNorm
        builder = EfficientNetBuilder(**model_kwargs)

        in_chs = self.conv.in_channels
        del self.conv
        self.conv = nn.Sequential(*builder(in_chs, block_args))
        initialize_weight_default(self.conv)
        conv_out_channels = block_args[-1][-1]['out_chs']
        self.objectness_logits = nn.Conv2d(conv_out_channels, self.objectness_logits.out_channels, kernel_size=1,
                                           stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            conv_out_channels, self.anchor_deltas.out_channels, kernel_size=1, stride=1
        )

        for l in [self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)