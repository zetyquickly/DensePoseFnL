import timm
from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from geffnet import set_exportable, set_scriptable

from densepose.modeling.converters import convert_norm_eps_momentum_to_tf_defaults, convert_norm_to_detectron2_format, \
    convert_norm_to_detectron2_format_and_init_default
from densepose.modeling.layers.bifpn import BiFPN


# class WrappedEfficientNetFeatures(EfficientNetFeatures):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def forward(self, x):
#         _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
#         return x + _dummy


@BACKBONE_REGISTRY.register()
def build_efficientnetv2_backbone(cfg, input_shape):
    """
    Create a GenEfficientNet instance from config.

    Returns:
        GenEfficientNet: a :class:`GenEfficientNet` instance.
    """

    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    norm = cfg.MODEL.EFFICIENTNETS.NORM
    out_features = cfg.MODEL.EFFICIENTNETS.OUT_FEATURES
    model_name = cfg.MODEL.EFFICIENTNETS.NAME
    pretrained = cfg.MODEL.EFFICIENTNETS.PRETRAINED
    set_exportable(cfg.MODEL.EFFICIENTNETS.EXPORTABLE)
    set_scriptable(cfg.MODEL.EFFICIENTNETS.SCRIPTABLE)

    # GET MODEL BY NAME
    model = timm.create_model('spnasnet_100', pretrained=pretrained, features_only=True, out_indices=out_features)

    # LOAD MODEL AND CONVERT NORM
    # NOTE: why I use if/else: see the strange function _load_from_state_dict in FrozenBatchNorm2d
    assert norm in ["FrozenBN", "SyncBN", "BN"]
    if norm == "FrozenBN":
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    elif pretrained:
        model = convert_norm_to_detectron2_format(model, norm)
    else:
        model = convert_norm_to_detectron2_format_and_init_default(model, norm)

    # USE TENSORFLOW EPS, MOMENTUM defaults if model is tf pretrained
    if "tf" in model_name:
        model = convert_norm_eps_momentum_to_tf_defaults(model)

    # PRUNE REDUNDANT BLOCKS
    # FIXME PRUNE
    # prune(model, is_mobilenetv3)

    max_block_number = int(model.feature_info[1]['name'][7:8])
    for p in model.conv_stem.parameters():
        p.requires_grad = False
    model.bn1 = FrozenBatchNorm2d.convert_frozen_batchnorm(model.bn1)
    for block_number in range(0, max_block_number + 1):
        for p in model.blocks[block_number].parameters():
            p.requires_grad = False
        model.blocks[block_number] = FrozenBatchNorm2d.convert_frozen_batchnorm(model.blocks[block_number])

    # def decorate_forward(cls):
    #     old_forward = cls.forward
    #
    #     def new_forward(self, x):
    #         x = old_forward(self, x)
    #
    #         return x + _dummy
    #
    #     cls.forward = new_forward
    #     return cls
    #
    # model = decorate_forward(model)

    return model


@BACKBONE_REGISTRY.register()
def build_efficientnetv2_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        modeling (Backbone): modeling module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_efficientnetv2_backbone(cfg, input_shape)
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = BiFPN(
        bottom_up=bottom_up,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(),
    )
    return backbone
