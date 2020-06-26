import types

import torch
from detectron2.layers import get_norm, BatchNorm2d, NaiveSyncBatchNorm, FrozenBatchNorm2d
from geffnet.conv2d_layers import Conv2dSame, Conv2dSameExport, MixedConv2d, CondConv2d
from geffnet.efficientnet_builder import BN_MOMENTUM_TF_DEFAULT, BN_EPS_TF_DEFAULT

from densepose.modeling.wrappers import nn_conv2d_forward_new, conv2d_same_forward_new, conv2d_same_export_forward_new, \
    mixed_conv2d_forward_new, cond_conv2d_forward_new


def convert_norm_to_detectron2_format(module, norm):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = get_norm(norm, out_channels=module.num_features)
        module_output.load_state_dict(module.state_dict())
    for name, child in module.named_children():
        new_child = convert_norm_to_detectron2_format_and_init_default(child, norm)
        if new_child is not child:
            module_output.add_module(name, new_child)
    return module_output


def convert_norm_to_detectron2_format_and_init_default(module, norm):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = get_norm(norm, out_channels=module.num_features)
        module_output.weight.data.fill_(1.0)
        module_output.bias.data.zero_()
    for name, child in module.named_children():
        new_child = convert_norm_to_detectron2_format_and_init_default(child, norm)
        if new_child is not child:
            module_output.add_module(name, new_child)
    return module_output


def convert_norm_eps_momentum_to_tf_defaults(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, BatchNorm2d) or isinstance(module,
                                                                                                 NaiveSyncBatchNorm):
        module_output.momentum = BN_MOMENTUM_TF_DEFAULT
        module_output.eps = BN_EPS_TF_DEFAULT
    elif isinstance(module, FrozenBatchNorm2d):
        module_output.eps = BN_EPS_TF_DEFAULT
    for name, child in module.named_children():
        new_child = convert_norm_eps_momentum_to_tf_defaults(child)
        module_output.add_module(name, new_child)
    del module
    return module_output


def convert_conv2d_to_detectron2_format(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        module_output.forward = types.MethodType(nn_conv2d_forward_new, module_output)
    elif isinstance(module, Conv2dSame):
        module_output.forward = types.MethodType(conv2d_same_forward_new, module_output)
    elif isinstance(module, Conv2dSameExport):
        module_output.forward = types.MethodType(conv2d_same_export_forward_new, module_output)
    elif isinstance(module, MixedConv2d):
        module_output.forward = types.MethodType(mixed_conv2d_forward_new, module_output)
    elif isinstance(module, CondConv2d):
        module_output.forward = types.MethodType(cond_conv2d_forward_new, module_output)
    else:
        for name, child in module.named_children():
            new_child = convert_conv2d_to_detectron2_format(child)
            module_output.add_module(name, new_child)
    return module_output
