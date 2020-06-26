import copy

import numpy as np
import torch
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils.events import get_event_storage


class DensePoseDataFilter(object):
    """
    Same as 'densepose.densepose_head.DensePoseDataFilter'.

    The only difference is that we also return selection mask.
    """

    def __init__(self, cfg):
        self.iou_threshold = cfg.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD

    @torch.no_grad()
    def __call__(self, proposals_with_targets):
        """
        Filters proposals with targets to keep only the ones relevant for
        DensePose training
        proposals: list(Instances), each element of the list corresponds to
            various instances (proposals, GT for boxes and densepose) for one
            image
        """
        selection_masks = []  # MY CODE
        proposals_filtered = []
        for proposals_per_image in proposals_with_targets:
            if not hasattr(proposals_per_image, "gt_densepose"):
                continue
            assert hasattr(proposals_per_image, "gt_boxes")
            assert hasattr(proposals_per_image, "proposal_boxes")
            gt_boxes = proposals_per_image.gt_boxes
            est_boxes = proposals_per_image.proposal_boxes
            # apply match threshold for densepose head
            iou = matched_boxlist_iou(gt_boxes, est_boxes)

            iou_select = iou > self.iou_threshold
            proposals_per_image = proposals_per_image[iou_select]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            # filter out any target without densepose annotation
            gt_densepose = proposals_per_image.gt_densepose
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_densepose)
            selected_indices = [
                i for i, dp_target in enumerate(gt_densepose) if dp_target is not None
            ]
            if len(selected_indices) != len(gt_densepose):
                proposals_per_image = proposals_per_image[selected_indices]

            # if proposals_per_image.proposal_boxes.tensor.size(0) == 0:
            if len(proposals_per_image) == 0:
                mask = iou > 9000.  # just big number > 100
                selection_masks.append(mask)
                continue

            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_densepose)
            proposals_filtered.append(proposals_per_image)

            mask = copy.deepcopy(iou_select)
            i = 0
            selected_num = 0
            for j in range(len(iou_select)):
                if iou_select[j]:
                    if i not in selected_indices:
                        mask[j] = False
                    else:
                        selected_num += 1
                    i += 1
            assert selected_num == len(selected_indices)
            selection_masks.append(mask)

        return proposals_filtered, selection_masks


def build_densepose_data_filter(cfg):
    dp_filter = DensePoseDataFilter(cfg)
    return dp_filter


def select_proposals_with_visible_keypoints(proposals):
    """
    Same as 'detectron2.modeling.roi_heads.roi_heads.select_proposals_with_visible_keypoints'.

    The only difference is that we also return selection mask.
    """
    ret = []
    all_num_fg = []
    selection_masks = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
                (xs >= proposal_boxes[:, :, 0])
                & (xs <= proposal_boxes[:, :, 2])
                & (ys >= proposal_boxes[:, :, 1])
                & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])
        selection_masks.append(selection)

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret, selection_masks
