import torch
from torch import nn as nn
from torchvision.ops._utils import convert_boxes_to_roi_format


class AdaptivePool(nn.Module):
    def __init__(self, output_size, spatial_scale, **kwargs):
        super(AdaptivePool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, input, rois):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        proposed = []
        if not isinstance(rois, torch.Tensor):
            rois = convert_boxes_to_roi_format(rois)
        for roi in rois:
            index, x_left, y_left, x_right, y_right = int(roi[0]), roi[1], roi[2], roi[3], roi[4]

            # floor() and ceil() operations just for sure, because rounding alread
            # was applied in shared_roi_head.py

            x_left = (self.spatial_scale * x_left - 0.5).floor().long()
            y_left = (self.spatial_scale * y_left - 0.5).floor().long()
            x_right = (self.spatial_scale * x_right - 0.5).ceil().long()
            y_right = (self.spatial_scale * y_right - 0.5).ceil().long()

            x_left = torch.clamp(x_left, min=0)
            y_left = torch.clamp(y_left, min=0)

            if x_right == x_left:
                x_right = x_right + 1
            if y_right == y_left:
                y_right = y_right + 1

            proposed.append(self.adaptive_avg_pooling(input[index, :, y_left:y_right, x_left:x_right]))

        result = (
            torch.stack(proposed, dim=0) if len(proposed) > 0 else torch.empty(input.shape[0], input.shape[1],
                                                                               self.output_size[0],
                                                                               self.output_size[1]).to(
                input.device))
        return result

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        return tmpstr
