"""
Implementation is based on: https://github.com/AlexHex7/Non-local_pytorch
"""
import numpy as np
from detectron2.layers import Conv2d, NaiveSyncBatchNorm, BatchNorm2d
from torch import nn, matmul
from torch.nn import functional as F

from densepose import MaxPool2d


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, detach=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.bn_layer = bn_layer
        self.detach = detach

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = Conv2d
            max_pool_layer = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
            bn = BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, inx):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        if self.detach:
            x = inx.detach()
        else:
            x = inx

        batch_size = x.size(0)

        g_x = self.g(x)
        g_x = g_x.view(batch_size, self.inter_channels, int(np.prod(g_x.shape[2:])))
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x)
        theta_x = theta_x.view(batch_size, self.inter_channels, int(np.prod(theta_x.shape[2:])))
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x)
        phi_x = phi_x.view(batch_size, self.inter_channels, int(np.prod(phi_x.shape[2:])))

        f = matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + inx

        if self.training and self.bn_layer:
            _dummy = sum(x.view(-1)[0] for x in self.W[1].parameters()) * 0.0
            z += _dummy

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, detach=False):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, detach=detach)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())
