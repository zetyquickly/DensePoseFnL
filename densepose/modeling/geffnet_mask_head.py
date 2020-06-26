from detectron2.layers import ShapeSpec, Conv2d, BatchNorm2d
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from geffnet import efficientnet_builder
from geffnet.efficientnet_builder import EfficientNetBuilder, initialize_weight_default, decode_arch_def
from torch import nn

from densepose import NaiveSyncBatchNorm
from densepose.modeling.converters import convert_conv2d_to_detectron2_format


class EfficientMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, **model_kwargs):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        input_channels = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        block_args = model_kwargs.pop('block_args')

        assert norm is None or norm in ["BN", "SyncBN"]
        if norm == "BN":
            model_kwargs['norm_layer'] = BatchNorm2d
        elif norm == "SyncBN":
            model_kwargs['norm_layer'] = NaiveSyncBatchNorm
        builder = EfficientNetBuilder(**model_kwargs)

        self.conv = nn.Sequential(*builder(input_channels, block_args))

        output_channels = block_args[-1][-1]['out_chs']
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(output_channels, num_mask_classes, kernel_size=1, stride=1, padding=0)

        initialize_weight_default(self.conv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return self.predictor(x)
