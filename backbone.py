from typing import Callable, Union

import timm
import torch
import torch.nn as nn
from detectron2.layers import get_norm, BatchNorm2d, NaiveSyncBatchNorm, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from timm.models.efficientnet_builder import BN_MOMENTUM_TF_DEFAULT, BN_EPS_TF_DEFAULT

from fpn import BiFPN


@BACKBONE_REGISTRY.register()
def build_timm_backbone(cfg, input_shape):
    """
    Create a TimmNet instance from config.

    Returns:
        TimmNet: a :class:`TimmNet` instance.
    """
    norm = cfg.MODEL.TIMMNETS.NORM
    out_features = cfg.MODEL.TIMMNETS.OUT_FEATURES
    model_name = cfg.MODEL.TIMMNETS.NAME
    pretrained = cfg.MODEL.TIMMNETS.PRETRAINED
    scriptable = cfg.MODEL.TIMMNETS.SCRIPTABLE
    exportable = cfg.MODEL.TIMMNETS.EXPORTABLE

    # GET MODEL BY NAME
    model = timm.create_model(model_name, pretrained, features_only=True, out_indices=out_features,
                              scriptable=scriptable, exportable=exportable, feature_location='expansion')

    # LOAD MODEL AND CONVERT NORM
    # NOTE: why I use if/else: see the strange function _load_from_state_dict in FrozenBatchNorm2d
    assert norm in ["FrozenBN", "SyncBN", "BN"]
    if norm == "FrozenBN":
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    elif pretrained:
        model = convert_norm_to_detectron2_format(model, norm)
    else:
        model = convert_norm_to_detectron2_format(model, norm, init_default=True)

    # USE TENSORFLOW EPS, MOMENTUM defaults if model is tf pretrained
    if "tf" in model_name:
        model = convert_norm_eps_momentum_to_tf_defaults(model)

    # FREEZE FIRST 2 LAYERS
    max_block_number = int(model.feature_info[1]['module'][7:8])
    # max_block_number = int(model.feature_info[1]['name'][7:8])
    print(f"Freezing stem and first {max_block_number + 1} backbone blocks")
    for p in model.conv_stem.parameters():
        p.requires_grad = False
    model.bn1 = FrozenBatchNorm2d.convert_frozen_batchnorm(model.bn1)
    for block_number in range(0, max_block_number + 1):
        for p in model.blocks[block_number].parameters():
            p.requires_grad = False
        model.blocks[block_number] = FrozenBatchNorm2d.convert_frozen_batchnorm(model.blocks[block_number])

    return model


@BACKBONE_REGISTRY.register()
def build_timm_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        modeling (Backbone): modeling module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_timm_backbone(cfg, input_shape)
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = BiFPN(
        bottom_up=bottom_up,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(),
    )
    return backbone


def convert_norm_to_detectron2_format(module, norm: Union[str, Callable], init_default: bool = False):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = get_norm(norm, out_channels=module.num_features)
        if init_default:
            module_output.weight.data.fill_(1.0)
            module_output.bias.data.zero_()
        else:
            module_output.load_state_dict(module.state_dict())
    for name, child in module.named_children():
        new_child = convert_norm_to_detectron2_format(child, norm, init_default)
        if new_child is not child:
            module_output.add_module(name, new_child)
    return module_output


def convert_norm_eps_momentum_to_tf_defaults(module):
    module_output = module
    if isinstance(module, (nn.BatchNorm2d, BatchNorm2d, NaiveSyncBatchNorm, nn.SyncBatchNorm)):
        module_output.momentum = BN_MOMENTUM_TF_DEFAULT
        module_output.eps = BN_EPS_TF_DEFAULT
    elif isinstance(module, FrozenBatchNorm2d):
        module_output.eps = BN_EPS_TF_DEFAULT
    for name, child in module.named_children():
        new_child = convert_norm_eps_momentum_to_tf_defaults(child)
        module_output.add_module(name, new_child)
    del module
    return module_output
