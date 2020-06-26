import torch
from detectron2.layers.wrappers import _NewEmptyTensorOp, interpolate
from geffnet.conv2d_layers import MixedConv2d, Conv2dSame, Conv2dSameExport, CondConv2d


# MONKEY PATCHING
# CODE IS BASED ON Conv2d wrapper from DETECTRON 2


def _empty_tensor(self, x, output_shape):
    empty = _NewEmptyTensorOp.apply(x, output_shape)
    if self.training:
        # This is to make DDP happy.
        # DDP expects all workers to have gradient w.r.t the same set of parameters.
        _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return empty + _dummy
    else:
        return empty


def _compute_output_shape(conv, x):
    return [
        (i + 2 * p - (di * (k - 1) + 1)) // s + 1
        for i, p, di, k, s in zip(
            x.shape[-2:], conv.padding, conv.dilation, conv.kernel_size, conv.stride
        )
    ]


def nn_conv2d_forward_new(self, x):
    if x.numel() == 0:
        output_shape = _compute_output_shape(self, x)
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _empty_tensor(self, x, output_shape)

    x = torch.nn.Conv2d.forward(self, x)
    return x


def conv2d_same_forward_new(self, x):
    if x.numel() == 0:
        # When input is empty, we want to return a empty tensor with "correct" shape,
        # So that the following operations will not panic
        # if they check for the shape of the tensor.
        output_shape = [x.shape[0], self.weight.shape[0]] + x.shape[-2:]
        return _empty_tensor(self, x, output_shape)
    x = Conv2dSame.forward(self, x)
    return x


def conv2d_same_export_forward_new(self, x):
    if x.numel() == 0:
        # When input is empty, we want to return a empty tensor with "correct" shape,
        # So that the following operations will not panic
        # if they check for the shape of the tensor.
        output_shape = [x.shape[0], self.weight.shape[0]] + x.shape[-2:]
        return _empty_tensor(self, x, output_shape)

    x = Conv2dSameExport.forward(self, x)
    return x


def mixed_conv2d_forward_new(self, x):
    if x.numel() == 0:
        # When input is empty, we want to return a empty tensor with "correct" shape,
        # So that the following operations will not panic
        # if they check for the shape of the tensor.
        # This computes the height and width of the output tensor
        out_channels = 0
        prev_output_shape = None
        for c in self._modules.values():
            out_channels += c.out_channels
            if isinstance(c, Conv2dSame) or isinstance(c, Conv2dSameExport):
                next_output_shape = x.shape[-2:]
            elif isinstance(c, torch.nn.Conv2d):
                next_output_shape = _compute_output_shape(c, x)
            else:
                raise ValueError("Unexpected Conv2d class in MixedConv2d")
            assert prev_output_shape is None or prev_output_shape == next_output_shape
            prev_output_shape = next_output_shape
        output_shape = [x.shape[0], out_channels, prev_output_shape]
        return _empty_tensor(self, x, output_shape)

    x = MixedConv2d.forward(self, x)
    return x


def cond_conv2d_forward_new(self, x, routing_weights):
    if x.numel() == 0:
        # When input is empty, we want to return a empty tensor with "correct" shape,
        # So that the following operations will not panic
        # if they check for the shape of the tensor.
        # This computes the height and width of the output tensor
        if self.dynamic_padding:
            output_shape = x.shape[-2:]
        else:
            output_shape = _compute_output_shape(self, x)

        output_shape = [x.shape[0], self.out_channels, output_shape]
        return _empty_tensor(self, x, output_shape)

    x = CondConv2d.forward(self, x, routing_weights)
    return x


class Interpolate(torch.nn.Module):

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class MaxPool2d(torch.nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.UpsamplingNearest2d` to support zero-size tensor.
    """

    def forward(self, x):
        if x.numel() > 0:
            return super(MaxPool2d, self).forward(x)
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // s + 1
            for i, p, di, k, s in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], x.shape[1]] + output_shape
        # This is to make DDP happy.
        # DDP expects all workers to have gradient w.r.t the same set of parameters.
        _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return _NewEmptyTensorOp.apply(x, output_shape) + _dummy


class Conv1d(torch.nn.Conv1d):
    """
    A wrapper around :class:`torch.nn.Conv1d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv1d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

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

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
