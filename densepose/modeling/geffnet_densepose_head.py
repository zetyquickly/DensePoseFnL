import torch.nn as nn
from detectron2.layers.wrappers import _NewEmptyTensorOp
from geffnet import efficientnet_builder
from geffnet.efficientnet_builder import EfficientNetBuilder, initialize_weight_default, decode_arch_def

from densepose import ROI_DENSEPOSE_HEAD_REGISTRY, convert_conv2d_to_detectron2_format

class GroupNorm32(nn.GroupNorm):

    def __init__(self, num_channels):
        super().__init__(num_groups=32, num_channels=num_channels)

    def forward(self, x):
        if x.numel() > 0:
            return super(GroupNorm32, self).forward(x)
        # get output shape
        output_shape = x.shape
        _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return _NewEmptyTensorOp.apply(x, output_shape) + _dummy


class EfficientDenseposeHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_channels, **model_kwargs):
        super().__init__()
        block_args = model_kwargs.pop('block_args')
        model_kwargs['norm_layer'] = GroupNorm32
        builder = EfficientNetBuilder(**model_kwargs)
        self.conv = nn.Sequential(*builder(input_channels, block_args))
        self.n_out_channels = block_args[-1][-1]['out_chs']
        initialize_weight_default(self.conv)

    def forward(self, x):
        # print(x.size())
        return self.conv(x)
