import sys
from typing import Union

import torch.nn.functional as F
from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from geffnet import GenEfficientNet, set_exportable, set_scriptable, gen_efficientnet, mobilenetv3, efficientnet_builder
from geffnet.efficientnet_builder import decode_arch_def, resolve_act_layer, \
    resolve_bn_args
from geffnet.helpers import load_pretrained
from geffnet.mobilenetv3 import MobileNetV3

from densepose.modeling.converters import convert_norm_to_detectron2_format_and_init_default, \
    convert_norm_eps_momentum_to_tf_defaults
from densepose.modeling.wrappers import *


class GenEfficientNetWrapper(GenEfficientNet, Backbone):
    """
        A wrapper around :class:`geffnet.GenEfficientNet` to support outputting intermediate features
        NOTE blocks = stages, each stage is a Sequential built upon bunch of blocks
    """

    def __init__(self, *args, **kwargs):
        out_features = kwargs.pop("out_features", None)
        Backbone.__init__(self)
        GenEfficientNet.__init__(self, *args, **kwargs)
        post_init(self, out_features, kwargs)

    @property
    def stage_names(self):
        return ('s' + str(i) for i in range(len(self.blocks)))

    def forward(self, x):
        outputs = {}
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in zip(self.blocks, self.stage_names):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if "linear" in self._out_features:
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.global_pool(x)
            x = x.flatten(1)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.classifier(x)
            outputs["linear"] = x
        return outputs


class MobileNetV3Wrapper(MobileNetV3, Backbone):
    """
        A wrapper around :class:`geffnet.MobileNetV3` to support outputting intermediate features
        NOTE blocks = stages, each stage is a Sequential built upon bunch of blocks
    """

    def __init__(self, *args, **kwargs):
        out_features = kwargs.pop("out_features", None)
        Backbone.__init__(self)
        MobileNetV3.__init__(self, *args, **kwargs)
        post_init(self, out_features, kwargs)

    @property
    def stage_names(self):
        return ('s' + str(i) for i in range(len(self.blocks)))

    def forward(self, x):
        outputs = {}
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in zip(self.blocks, self.stage_names):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if "linear" in self._out_features:
            x = self.global_pool(x)
            x = self.conv_head(x)
            x = self.act2(x)
            x = x.flatten(1)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.classifier(x)
            outputs["linear"] = x
        return outputs


def post_init(cls: Union[GenEfficientNetWrapper, MobileNetV3Wrapper], out_features, kwargs):
    current_stride = 2
    cls._out_feature_strides = {"stem": current_stride}
    cls._out_feature_channels = {"stem": cls.conv_stem.out_channels}

    stage_strides = [b[0]['stride'] for b in kwargs["block_args"]]
    stage_out_channels = [b[-1]['out_chs'] for b in kwargs["block_args"]]
    assert len(cls.blocks) == len(stage_strides) == len(stage_out_channels)

    for i, (stage, name, stride, out_channels) in enumerate(
            zip(cls.blocks, cls.stage_names, stage_strides, stage_out_channels)):
        current_stride = current_stride * stride
        if name in out_features:
            cls._out_feature_strides[name] = current_stride
            cls._out_feature_channels[name] = out_channels

    if out_features is None:
        out_features = ["linear"]
    cls._out_features = out_features
    assert len(cls._out_features)
    for out_feature in cls._out_features:
        assert out_feature in cls.stage_names


def freeze(cls: Union[GenEfficientNetWrapper, MobileNetV3Wrapper], freeze_at, is_mobilenetv3=False):
    assert -1 <= freeze_at <= len(cls.blocks)
    # stem == -1, block == 0...n-1, linear == n
    for i in range(-1, freeze_at + 1):
        if i == -1:
            # freeze stem
            for module in (cls.conv_stem, cls.bn1, cls.act1):
                for p in module.parameters():
                    p.requires_grad = False
            cls.bn1 = FrozenBatchNorm2d.convert_frozen_batchnorm(cls.bn1)
        elif i == len(cls.blocks):
            # freeze linear
            for module in (cls.conv_head, cls.act2, cls.global_pool, cls.classifier):
                for p in module.parameters():
                    p.requires_grad = False

            if not is_mobilenetv3:
                for p in cls.bn2.parameters():
                    p.requires_grad = False
                cls.bn2 = FrozenBatchNorm2d.convert_frozen_batchnorm(cls.bn2)
        else:
            # freeze intermediate block
            for p in cls.blocks[i].parameters():
                p.requires_grad = False
            cls.blocks[i] = FrozenBatchNorm2d.convert_frozen_batchnorm(cls.blocks[i])


def prune(cls: Union[GenEfficientNetWrapper, MobileNetV3Wrapper], is_mobilenetv3=False):
    if cls._out_features[-1] in cls.stage_names:
        # then we can prune the model
        last_idx = int(cls._out_features[-1][1:])
        cls.blocks = torch.nn.Sequential(*list(cls.blocks.children())[:last_idx + 1])
        assert "linear" not in cls.stage_names
        del cls.conv_head, cls.act2, cls.global_pool, cls.drop_rate, cls.classifier
        if not is_mobilenetv3:
            del cls.bn2


@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, input_shape):
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
    is_mobilenetv3 = "mobilenetv3" in model_name
    if is_mobilenetv3:
        mobilenetv3._create_model = lambda model_kwargs, variant, pretrained=False: (
            model_kwargs, mobilenetv3.model_urls[variant])
        model_kwargs, url = getattr(mobilenetv3, model_name)()
        assert not pretrained or url is not None
        model = MobileNetV3Wrapper(**model_kwargs, out_features=out_features)
    else:
        gen_efficientnet._create_model = lambda model_kwargs, variant, pretrained=False: (
            model_kwargs, gen_efficientnet.model_urls[variant])
        model_kwargs, url = getattr(gen_efficientnet, model_name)()
        assert not pretrained or url is not None
        model = GenEfficientNetWrapper(**model_kwargs, out_features=out_features)

    # LOAD MODEL AND CONVERT NORM
    # NOTE: why I use if/else: see the strange function _load_from_state_dict in FrozenBatchNorm2d
    assert norm in ["FrozenBN", "SyncBN", "BN"]
    if norm == "FrozenBN":
        if pretrained:
            load_pretrained(model, url, filter_fn=None, strict=True)
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    else:
        model = convert_norm_to_detectron2_format_and_init_default(model, norm)
        if pretrained:
            load_pretrained(model, url, filter_fn=None, strict=True)

    # USE TENSORFLOW EPS, MOMENTUM defaults if model is tf pretrained
    if "tf" in model_name:
        model = convert_norm_eps_momentum_to_tf_defaults(model)

    # PRUNE REDUNDANT BLOCKS
    prune(model, is_mobilenetv3)

    # FREEZE AT
    if freeze_at >= -1:
        freeze(model, freeze_at, is_mobilenetv3)

    return model


@BACKBONE_REGISTRY.register()
def build_efficientnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        modeling (Backbone): modeling module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_efficientnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
