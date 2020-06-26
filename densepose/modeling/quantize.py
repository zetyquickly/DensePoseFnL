import copy
import itertools
import logging
import math
from typing import List

import torch.nn
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.quantized.functional as F
import torch.nn.intrinsic.qat
import torch.nn.quantized
from detectron2.engine import DefaultTrainer
from detectron2.layers import ShapeSpec, Conv2d, interpolate, NaiveSyncBatchNorm, FrozenBatchNorm2d, Linear
from detectron2.layers.wrappers import _NewEmptyTensorOp, BatchNorm2d
from detectron2.modeling.meta_arch import build_model as bld
from detectron2.utils.events import EventStorage
from geffnet.efficientnet_builder import drop_connect
from torch.quantization import QuantStub, DeQuantStub


# TODO resnets
# TODO semseg head

class ConvBNReLU(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.norm = kwargs.pop("norm", None)
        self.activation = kwargs.pop("activation", None)
        self.conv = nn.Conv2d(*args, **kwargs) if len(args) > 0 or len(kwargs) > 0 else None

    @classmethod
    def from_conv2d(cls, conv2d: Conv2d, activation=None):
        new_instance = cls()
        new_instance.norm = copy.deepcopy(conv2d.norm)
        if conv2d.activation is None:
            new_instance.activation = copy.deepcopy(activation)
        else:
            new_instance.activation = copy.deepcopy(conv2d.activation)
        if new_instance.activation == F.relu:
            new_instance.activation = nn.ReLU(inplace=False)
        # del conv2d.norm, conv2d.activation
        device = next(conv2d.parameters()).device
        new_instance.conv = nn.Conv2d(in_channels=conv2d.in_channels,
                                      out_channels=conv2d.out_channels,
                                      kernel_size=conv2d.kernel_size,
                                      stride=conv2d.stride,
                                      padding=conv2d.padding,
                                      dilation=conv2d.dilation,
                                      groups=conv2d.groups,
                                      bias=conv2d.bias is not None,
                                      padding_mode=conv2d.padding_mode
                                      ).to(device)
        new_instance.conv.load_state_dict(conv2d.state_dict())
        return new_instance

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def fuse(self):
        if self.norm is not None and self.activation is not None:
            torch.quantization.fuse_modules(self, ['conv', 'norm', 'activation'], inplace=True)
        if self.norm is None and self.activation is not None:
            torch.quantization.fuse_modules(self, ['conv', 'activation'], inplace=True)
        if self.norm is not None and self.activation is None:
            torch.quantization.fuse_modules(self, ['conv', 'norm'], inplace=True)


class LinearReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(*args, **kwargs) if len(args) > 0 or len(kwargs) > 0 else None
        self.activation = nn.ReLU(inplace=False)

    @classmethod
    def from_linenar(cls, linear):
        new_instance = cls()
        new_instance.linear = copy.deepcopy(linear)
        return new_instance

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

    def fuse(self):
        torch.quantization.fuse_modules(self, ['linear', 'activation'], inplace=True)


def EfficientNetFeatures_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        # self.quant = QuantStub()

    def new_forward(self, x):
        return old_forward(self, x)


    def fuse(self):
        torch.quantization.fuse_modules(self, ['conv_stem', 'bn1', 'act1'], inplace=True)

    cls.__init__ = new_init
    cls.forward = new_forward
    cls.fuse = fuse
    return cls

def FastNormalizedFusion_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def new_forward(self, *x):
        if len(x) != self.in_nodes:
            raise RuntimeError(
                "Expected to have {} input nodes, but have {}.".format(self.in_nodes, len(x))
            )

        # where wi ≥ 0 is ensured by applying a relu after each wi (paper)   
        weight = F.relu(self.weight)
        weighted_xs = [self.dequant(xi) * wi for xi, wi in zip(x, weight)]
        normalized_weighted_x = sum(weighted_xs) / (weight.sum() + self.eps)
        return self.quant(normalized_weighted_x)

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls

def BiFPN_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.quant_in = QuantStub()
        self.dequant_out = DeQuantStub()
        self.dequant_up = DeQuantStub()
        self.quant_up = QuantStub()

    def new_forward(self, x):
        x = self.quant_in(x)
        p5, p4, p3, p2 = self.bottom_up(x)  # top->down
        # _dummy = sum(x.view(-1)[0] for x in self.bottom_up.parameters()) * 0.0
        # _dummy = torch.tensor(_dummy).repeat(p5.size())
        # p5 = self.add.add(p5, self.quant_in(_dummy))

        p5 = self.l5(p5)
        p4 = self.l4(p4)
        p3 = self.l3(p3)
        p2 = self.l2(p2)

        p4_tr = self.p4_tr(self.fuse_p4_tr(p4, self.quant_up(self.up(self.dequant_up(p5)))))
        p3_tr = self.p3_tr(self.fuse_p3_tr(p3, self.quant_up(self.up(self.dequant_up(p4_tr)))))

        p2_out = self.p2_out(self.fuse_p2_out(p2, self.quant_up(self.up(self.dequant_up(p3_tr)))))
        p3_out = self.p3_out(self.fuse_p3_out(p3, p3_tr, self.down_p2(p2_out)))
        p4_out = self.p4_out(self.fuse_p4_out(p4, p4_tr, self.down_p3(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5, self.down_p4(p4_out)))

        # p4_tr = self.p4_tr(self.fuse_p4_tr(p4, self.up(p5)))
        # p3_tr = self.p3_tr(self.fuse_p3_tr(p3, self.up(p4_tr)))

        # p2_out = self.p2_out(self.fuse_p2_out(p2, self.up(p3_tr)))
        # p3_out = self.p3_out(self.fuse_p3_out(p3, p3_tr, self.down_p2(p2_out)))
        # p4_out = self.p4_out(self.fuse_p4_out(p4, p4_tr, self.down_p3(p3_out)))
        # p5_out = self.p5_out(self.fuse_p5_out(p5, self.down_p4(p4_out)))

        return {"p2": p2_out, "p3": p3_out, "p4": p4_out, "p5": p5_out, 
                "p6": self.quant_up(self.top_block(self.dequant_up(p5_out))[0])}


    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def LastLevelMaxPool_decorate(cls):

    def new_forward(self, x):
        return [torch.nn.quantized.functional.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

    cls.forward = new_forward
    return cls


def Conv3x3BnReLU_decorate(cls):
    def fuse(self):
        torch.quantization.fuse_modules(self, ['0.1', '1', '2'], inplace=True)
    cls.fuse = fuse
    return cls


def GenEfficientNet_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        # self.quant = QuantStub()

    def new_forward(self, x):
        # x = self.quant(x)
        return old_forward(self, x)

    def fuse(self):
        torch.quantization.fuse_modules(self, ['conv_stem', 'bn1', 'act1'], inplace=True)

    cls.__init__ = new_init
    cls.forward = new_forward
    cls.fuse = fuse
    return cls


def FPN_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)

        self.adds = nn.ModuleList([torch.nn.quantized.FloatFunctional() for i in range(len(self.lateral_convs[1:]))])
        self.muls = nn.ModuleList([torch.nn.quantized.FloatFunctional() for i in range(len(self.lateral_convs[1:]))])

        in_strides = [self.bottom_up.out_feature_strides[f] for f in self.in_features]
        for i in list(range(len(self.lateral_convs)))[::-1]:
            lateral_conv = ConvBNReLU.from_conv2d(self.lateral_convs[i])
            output_conv = ConvBNReLU.from_conv2d(self.output_convs[i])
            self.lateral_convs[i] = lateral_conv
            self.output_convs[i] = output_conv
            stage = int(math.log2(in_strides[i]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

    def new_forward(self, x):
        """
            Args:
                input (dict[str: Tensor]): mapping feature map name (e.g., "res5") to
                    feature map tensor for each feature level in high to low resolution order.

            Returns:
                dict[str: Tensor]:
                    mapping from feature map name to FPN feature map tensor
                    in high to low resolution order. Returned feature names follow the FPN
                    paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                    ["p2", "p3", ..., "p6"].
            """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for i, features, lateral_conv, output_conv in zip(itertools.count(), x[1:], self.lateral_convs[1:],
                                                          self.output_convs[1:]):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = self.adds[i].add(lateral_features, top_down_features)
            if self._fuse_type == "avg":
                prev_features = self.muls[i].mul(prev_features, 0.5)
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def DepthwiseSeparableConv_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if self.has_residual:
            self.add = torch.nn.quantized.FloatFunctional()

    def new_forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x = self.add.add(x, residual)
        return x

    def fuse(self):
        torch.quantization.fuse_modules(self, ['conv_dw', 'bn1', 'act1'], inplace=True)
        if isinstance(self.act2, torch.nn.Identity):
            torch.quantization.fuse_modules(self, ['conv_pw', 'bn2'], inplace=True)
        else:
            torch.quantization.fuse_modules(self, ['conv_pw', 'bn2', 'act2'], inplace=True)

    cls.__init__ = new_init
    cls.forward = new_forward
    cls.fuse = fuse
    return cls

def DensePoseNearestConv3Head_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.dequant_in = DeQuantStub()
        self.quant_in = QuantStub()
        self.dequant_out = DeQuantStub()

    def new_forward(self, features):
        features = self.quant_in(
            F.interpolate(self.dequant_in(features), scale_factor=2, mode="nearest")
        )
        ann_index = self.dequant_out(self.ann_index_lowres(features))
        index_uv = self.dequant_out(self.index_uv_lowres(features))
        u = self.dequant_out(self.u_lowres(features))
        v = self.dequant_out(self.v_lowres(features))
        return (ann_index, index_uv, u, v), (None, None, None, None)

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls
    

def ParsingSharedBlock_decorate(cls):
    old_init = cls.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if self.has_aspp:
            self.cat = torch.nn.quantized.FloatFunctional()
        self.aspp5 = nn.Sequential(
            DeQuantStub(),
            self.aspp5[0],
            QuantStub(),
            *self.aspp5[1:3],
            DeQuantStub(),
            self.aspp5[3],
            QuantStub()
        )
    
    def new_forward(self, features):
        x = features
        if x.nelement() == 0:
            return x
        if self.has_conv_before_aspp_nl:
            x = self.conv_before_aspp_nl(x)
        if self.has_aspp:
            content = (self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x), self.aspp5(x),)
            x = self.cat.cat(content, dim=1)
            x = self.aspp_agg(x)
        if self.has_nonlocal:
            x = self.non_local(x)
        if self.has_conv_after_aspp_nl:
            x = self.conv_after_aspp_nl(x)
        return x

    def fuse(self):
        if self.has_aspp:
            torch.quantization.fuse_modules(
                self, 
                [
                    ['aspp1.0', 'aspp1.1'],
                    ['aspp2.0', 'aspp2.1'],
                    ['aspp3.0', 'aspp3.1'],
                    ['aspp4.0', 'aspp4.1'],
                    ['aspp5.3', 'aspp5.4'],
                    ['aspp_agg.0', 'aspp_agg.1'],
                ], 
                inplace=True
            )
        if self.has_conv_after_aspp_nl:
            torch.quantization.fuse_modules(
                self, 
                [
                    ['conv_after_aspp_nl.0', 'conv_after_aspp_nl.1'],
                    ['conv_after_aspp_nl.2', 'conv_after_aspp_nl.3'],
                    ['conv_after_aspp_nl.4', 'conv_after_aspp_nl.5'],
                    ['conv_after_aspp_nl.6', 'conv_after_aspp_nl.7'],
                ], 
                inplace=True
            )
        
    cls.__init__ = new_init
    cls.forward = new_forward
    cls.fuse = fuse
    return cls

def InvertedResidual_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if self.has_residual:
            self.add = torch.nn.quantized.FloatFunctional()

    def new_forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x = self.add.add(x, residual)
        return x

    def fuse(self):
        torch.quantization.fuse_modules(
            self, 
            [
                ['conv_pw', 'bn1', 'act1'],
                ['conv_dw', 'bn2', 'act2'],
                ['conv_pwl', 'bn3']
            ],
            inplace=True
        )
        
    cls.__init__ = new_init
    cls.forward = new_forward
    cls.fuse = fuse
    return cls


def StandardRPNHead_decorate(cls):
    old_init = cls.__init__

    def new_init(self, cfg, input_shape: List[ShapeSpec], **kwargs):
        old_init(self, cfg, input_shape, **kwargs)
        self.has_relu = type(self.conv) == nn.Conv2d
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)
        self.num_levels = len(input_shape)
        self.dequant_objectness_logits = nn.ModuleList([DeQuantStub() for i in range(self.num_levels)])
        self.dequant_anchor_deltas = nn.ModuleList([DeQuantStub() for i in range(self.num_levels)])
        
    def new_forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for i in range(self.num_levels):
            t = self.conv(features[i])
            if self.has_relu:
                t = self.relu(t)
            pred_objectness_logits.append(self.dequant_objectness_logits[i](self.objectness_logits(t)))
            pred_anchor_deltas.append(self.dequant_anchor_deltas[i](self.anchor_deltas(t)))
        return pred_objectness_logits, pred_anchor_deltas

    def fuse(self):
        if self.has_relu:
            torch.quantization.fuse_modules(self, ['conv', 'relu'], inplace=True)

    cls.__init__ = new_init
    cls.forward = new_forward
    cls.fuse = fuse
    return cls


def FastRCNNConvFCHead_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        for k in range(len(self.conv_norm_relus)):
            conv = ConvBNReLU.from_conv2d(self.conv_norm_relus[k])
            self.conv_norm_relus[k] = conv
            self.add_module("conv{}".format(k + 1), conv)
            
        for k in range(len(self.fcs)):
            fc = LinearReLU.from_linenar(self.fcs[k])
            self.fcs[k] = fc
            self.add_module("fc{}".format(k + 1), fc)

    def new_forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        # if len(self.fcs):
        #     if x.dim() > 2:
        #         x = x.flatten(start_dim=1)
        #     for layer in self.fcs:
        #         x = layer(x)
        return x

    # def fuse(self):
    #     # for k in range(len(self.conv_norm_relus)):
    #     #     self.conv_norm_relus[k].fuse()
    #     for k in range(len(self.conv_norm_relus)):
    #         torch.quantization.fuse_modules(
    #             self, 
    #             ["conv{}".format(k + 1), "relu{}".format(k + 1)], 
    #             inplace=True
    #         )

    cls.__init__ = new_init
    cls.forward = new_forward
    # cls.fuse = fuse
    return cls


def EfficientFastRCNNConvFCHead_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        for k in range(len(self.fcs)):
            fc = LinearReLU.from_linenar(self.fcs[k])
            self.fcs[k] = fc
            self.add_module("fc{}".format(k + 1), fc)

    def new_forward(self, x):
        x = self.conv(x)
        if self.has_pooling:
            x = self.pool(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = layer(x)
        return x

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def FastRCNNOutputLayers_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.dequant_in = DeQuantStub()
        self.quant_in = QuantStub()
        self.dequant_scores = DeQuantStub()
        self.dequant_proposal_deltas = DeQuantStub()

    def new_forward(self, x):
        x = self.dequant_scores(x)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        x = self.quant_in(x)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        scores = self.dequant_scores(scores)
        proposal_deltas = self.dequant_proposal_deltas(proposal_deltas)
        return scores, proposal_deltas

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def ROIPooler_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.dequant = nn.ModuleList([DeQuantStub() for i in range(len(self.level_poolers))])
        self.quant = QuantStub()

    def new_forward(self, x, box_lists):
        for i in range(len(self.level_poolers)):
            x[i] = self.dequant[i](x[i])
        output = old_forward(self, x, box_lists)
        if output.nelement() != 0:
            output = self.quant(output)
        return output

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def MaskRCNNConvUpsampleHead_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        for k in range(len(self.conv_norm_relus)):
            conv = ConvBNReLU.from_conv2d(self.conv_norm_relus[k])
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus[k] = conv
        self.predictor = ConvBNReLU.from_conv2d(self.predictor)

    cls.__init__ = new_init
    return cls


def KRCNNConvDeconvUpsampleHead_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        for idx in range(len(self.blocks)):
            conv = ConvBNReLU.from_conv2d(self.blocks[idx], nn.ReLU(inplace=False))
            self.blocks[idx] = conv
            self.add_module("conv_fcn{}".format(idx), conv)

    def new_forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def DensePoseV1ConvXHead_decorate(cls):
    old_init = cls.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            layer = ConvBNReLU.from_conv2d(getattr(self, layer_name), nn.ReLU(inplace=False))
            self.add_module(layer_name, layer)

    def new_forward(self, features):
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            output = x
        return output

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def DensePosePredictor_decorate(cls):
    old_init = cls.__init__
    old_forward = cls.forward

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.dequant_head = DeQuantStub()

    def new_forward(self, head_outputs):
        head_outputs = self.dequant_head(head_outputs)
        return old_forward(self, head_outputs)

    cls.__init__ = new_init
    cls.forward = new_forward
    return cls


def quantize_decorate():
    import densepose
    # densepose.GenEfficientNetWrapper = GenEfficientNet_decorate(densepose.GenEfficientNetWrapper)
    densepose.MobileNetV3Wrapper = GenEfficientNet_decorate(densepose.MobileNetV3Wrapper)
    densepose.BiFPN = BiFPN_decorate(densepose.BiFPN)
    densepose.Conv3x3BnReLU = Conv3x3BnReLU_decorate(densepose.Conv3x3BnReLU)
    densepose.FastNormalizedFusion = FastNormalizedFusion_decorate(densepose.FastNormalizedFusion)
    import geffnet.efficientnet_builder
    geffnet.efficientnet_builder.InvertedResidual = InvertedResidual_decorate(
        geffnet.efficientnet_builder.InvertedResidual)
    import timm.models.efficientnet_blocks
    timm.models.efficientnet_blocks.InvertedResidual = InvertedResidual_decorate(
        timm.models.efficientnet_blocks.InvertedResidual)
    
    geffnet.efficientnet_builder.DepthwiseSeparableConv = DepthwiseSeparableConv_decorate(
        geffnet.efficientnet_builder.DepthwiseSeparableConv)
    timm.models.efficientnet_blocks.DepthwiseSeparableConv = DepthwiseSeparableConv_decorate(
        timm.models.efficientnet_blocks.DepthwiseSeparableConv)
        
    import detectron2.modeling.backbone.fpn
    detectron2.modeling.backbone.fpn.LastLevelMaxPool = LastLevelMaxPool_decorate(detectron2.modeling.backbone.fpn.LastLevelMaxPool)
    import detectron2.modeling
    detectron2.modeling.FPN = FPN_decorate(detectron2.modeling.FPN)
    import detectron2.modeling.proposal_generator.rpn
    detectron2.modeling.proposal_generator.rpn.StandardRPNHead = StandardRPNHead_decorate(
        detectron2.modeling.proposal_generator.rpn.StandardRPNHead)
    densepose.EfficientRPNHead = StandardRPNHead_decorate(densepose.EfficientRPNHead)

    import detectron2.modeling.roi_heads.fast_rcnn
    detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers = FastRCNNOutputLayers_decorate(
        detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers)

    import detectron2.modeling.roi_heads.box_head
    detectron2.modeling.roi_heads.box_head.FastRCNNConvFCHead = FastRCNNConvFCHead_decorate(
        detectron2.modeling.roi_heads.box_head.FastRCNNConvFCHead)
    # densepose.EfficientFastRCNNConvFCHead = EfficientFastRCNNConvFCHead_decorate(densepose.EfficientFastRCNNConvFCHead)

    import detectron2.modeling.poolers
    detectron2.modeling.poolers.ROIPooler = ROIPooler_decorate(detectron2.modeling.poolers.ROIPooler)

    import detectron2.modeling.roi_heads.keypoint_head
    detectron2.modeling.roi_heads.keypoint_head.KRCNNConvDeconvUpsampleHead = KRCNNConvDeconvUpsampleHead_decorate(
        detectron2.modeling.roi_heads.keypoint_head.KRCNNConvDeconvUpsampleHead)

    import densepose.densepose_head
    densepose.densepose_head.DensePoseV1ConvXHead = DensePoseV1ConvXHead_decorate(
        densepose.densepose_head.DensePoseV1ConvXHead)
    
    import densepose.modeling.shared_block
    densepose.modeling.shared_block.ParsingSharedBlock = ParsingSharedBlock_decorate(
        densepose.modeling.shared_block.ParsingSharedBlock
    )
    densepose.modeling.shared_block.DensePoseNearestConv3Head = DensePoseNearestConv3Head_decorate(
        densepose.modeling.shared_block.DensePoseNearestConv3Head
    )

    densepose.densepose_head.DensePosePredictor = DensePosePredictor_decorate(
        densepose.densepose_head.DensePosePredictor)


    from timm.models.efficientnet import EfficientNetFeatures
    EfficientNetFeatures = EfficientNetFeatures_decorate(EfficientNetFeatures)


def quantize_prepare():
    def convert_relu(module):
        module_output = module
        if isinstance(module, nn.ReLU):
            module_output = nn.ReLU(inplace=False)
        for name, child in module.named_children():
            new_child = convert_relu(child)
            if new_child is not child:
                module_output.add_module(name, new_child)
        if not module.training:
            module_output.eval()
        return module_output

    def convert_batchnorm(module, device):
        module_output = module
        if type(module) == BatchNorm2d or type(module) == NaiveSyncBatchNorm:
            module_output = nn.BatchNorm2d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats
            ).to(device)
            module_output.load_state_dict(module.state_dict())
        elif type(module) == FrozenBatchNorm2d:
            module_output = nn.BatchNorm2d(
                num_features=module.num_features,
                eps=module.eps
            ).to(device)
            module_output.load_state_dict(module.state_dict(), strict=False)
        for name, child in module.named_children():
            new_child = convert_batchnorm(child, device)
            if new_child is not child:
                module_output.add_module(name, new_child)
        if not module.training:
            module_output.eval()
        return module_output

    def convert_conv2d(module):
        module_output = module
        if isinstance(module, Conv2d):
            device = next(module.parameters()).device
            conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode
            ).to(device)
            conv.load_state_dict(module.state_dict())
            module_output = conv
        for name, child in module.named_children():
            new_child = convert_conv2d(child)
            if new_child is not child:
                module_output.add_module(name, new_child)
        if not module.training:
            module_output.eval()
        return module_output

    def convert_linear(module):
        module_output = module
        if isinstance(module, Linear):
            device = next(module.parameters()).device
            linear = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=True if module.bias is not None else False
            ).to(device)
            linear.load_state_dict(module.state_dict())
            module_output = linear
        for name, child in module.named_children():
            new_child = convert_linear(child)
            if new_child is not child:
                module_output.add_module(name, new_child)
        if not module.training:
            module_output.eval()
        return module_output

    def assert_no_detectron_conv2d(module):
        if type(module) == Conv2d:
            raise AssertionError
        for name, child in module.named_children():
            try:
                assert_no_detectron_conv2d(child)
            except AssertionError:
                raise AssertionError(f'{name} is not wrapped accurately')

    def fuse_modules(module):
        module_output = module
        if callable(getattr(module_output, "fuse", None)):
            module_output.fuse()
        for name, child in module.named_children():
            new_child = fuse_modules(child)
            if new_child is not child:
                module_output.add_module(name, new_child)
        return module_output

    def DefaultTrainer_decorate(cls):

        def update_model(model, backend):
            # model = bld(cfg)
            logger = logging.getLogger(__name__)
            model = convert_relu(model)
            model = convert_batchnorm(model, next(model.parameters()).device)
            model = convert_conv2d(model)  # must be after init, because there are conv2d and F.relu going separately in some modules
            model = convert_linear(model) 
            assert_no_detectron_conv2d(model)
            
            model = fuse_modules(model)
            model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            torch.backends.quantized.engine = backend
            model = torch.quantization.prepare_qat(model, inplace=False)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            return model

        cls.update_model = update_model
        # cls.train = train
        # cls.__init__ = new_init
        return cls

    import detectron2.engine
    detectron2.engine.DefaultTrainer = DefaultTrainer_decorate(detectron2.engine.DefaultTrainer)