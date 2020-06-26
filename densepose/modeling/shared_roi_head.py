# -*- coding: utf-8 -*-
import math

import detectron2.modeling.poolers
import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, build_mask_head, build_box_head, build_keypoint_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.poolers import assign_boxes_to_levels
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss, keypoint_rcnn_inference
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, ROIHeads

from densepose.densepose_head import build_densepose_head, build_densepose_losses, densepose_inference, \
    DensePosePredictor, ROI_DENSEPOSE_HEAD_REGISTRY
from densepose.modeling.filters import select_proposals_with_visible_keypoints, build_densepose_data_filter
from densepose.modeling.shared_block import build_shared_block


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class Identity(nn.Module):
    def __init__(self, cfg, input_channels):
        super().__init__()
        self.n_out_channels = input_channels

    def forward(self, x):
        return x


def build_densepose_predictor(cfg, input_channels):
    type = cfg.MODEL.ROI_DENSEPOSE_PREDICTOR.NAME
    if type == 'Identity':
        return Identity(cfg, input_channels)
    assert type == 'Default'
    return DensePosePredictor(cfg, input_channels)


@ROI_HEADS_REGISTRY.register()
class SharedROIHeads(ROIHeads):
    """
    ROIHeads which has
    a separate pooler and shared block (optional) for:
    + keypoint head
    + densepose head
    + mask head
    and a separate pooler for:
    + box head

    The motivation: the feature levels upon which we compute predictions
                    for bbox and keypoints/densepose can be different
    """

    def __init__(self, cfg, input_shape):
        super(SharedROIHeads, self).__init__(cfg, input_shape)

        # If SharedROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        self.in_channels = in_channels[0]

        self._init_box_head(cfg)
        self._init_shared_block(cfg)

        # Check that all feature levels are used either for box prediction or other tasks
        assert len(set(self.in_features_box) | set(self.in_features_shared)) == len(self.in_features)

        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)
        self._init_densepose_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        self.in_features_box = cfg.MODEL.ROI_BOX_HEAD.IN_FEATURES if cfg.MODEL.ROI_BOX_HEAD.IN_FEATURES else self.in_features
        box_pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        box_pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.box_pooler_scales = list(1.0 / self.feature_strides[k] for k in self.in_features_box)
        box_sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on

        assert len(set(self.in_features_box)) > 0 and set(self.in_features_box) <= set(self.in_features)

        self.box_pooler = ROIPooler(
            output_size=box_pooler_resolution,
            scales=self.box_pooler_scales,
            sampling_ratio=box_sampling_ratio,
            pooler_type=box_pooler_type
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=self.in_channels, height=box_pooler_resolution, width=box_pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

        # FOR ADAPTIVE POOLING
        self.box_min_level = int(-math.log2(self.box_pooler_scales[0]))
        self.box_max_level = int(-math.log2(self.box_pooler_scales[-1]))
        self.box_canonical_box_size = self.box_pooler.canonical_box_size
        self.box_canonical_level = self.box_pooler.canonical_level

    def _init_shared_block(self, cfg):
        # fmt: off
        self.in_features_shared = cfg.MODEL.ROI_SHARED_HEAD.IN_FEATURES if cfg.MODEL.ROI_SHARED_HEAD.IN_FEATURES else self.in_features
        self.shared_pooler_resolution = cfg.MODEL.ROI_SHARED_HEAD.POOLER_RESOLUTION
        shared_pooler_type = cfg.MODEL.ROI_SHARED_HEAD.POOLER_TYPE
        self.shared_pooler_scales = list(1.0 / self.feature_strides[k] for k in self.in_features_shared)
        shared_sampling_ratio = cfg.MODEL.ROI_SHARED_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on

        assert len(set(self.in_features_shared)) > 0 and set(self.in_features_shared) <= set(self.in_features)

        self.shared_pooler = ROIPooler(
            output_size=self.shared_pooler_resolution,
            scales=self.shared_pooler_scales,
            sampling_ratio=shared_sampling_ratio,
            pooler_type=shared_pooler_type
        )

        self.shared_block, self.shared_out_channels = build_shared_block(cfg, self.in_channels)

        # FOR ADAPTIVE POOLING
        self.shared_min_level = int(-math.log2(self.shared_pooler_scales[0]))
        self.shared_max_level = int(-math.log2(self.shared_pooler_scales[-1]))
        self.shared_canonical_box_size = self.shared_pooler.canonical_box_size
        self.shared_canonical_level = self.shared_pooler.canonical_level

    def _init_mask_head(self, cfg):
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return

        self.mask_head = build_mask_head(
            cfg,
            ShapeSpec(channels=self.shared_out_channels,
                      width=self.shared_pooler_resolution, height=self.shared_pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        self.keypoint_head = build_keypoint_head(
            cfg,
            ShapeSpec(channels=self.shared_out_channels,
                      width=self.shared_pooler_resolution, height=self.shared_pooler_resolution)
        )

    def _init_densepose_head(self, cfg):
        self.densepose_on = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)

        self.densepose_head = build_densepose_head(cfg, self.shared_out_channels)
        self.densepose_predictor = build_densepose_predictor(cfg, self.densepose_head.n_out_channels)
        self.densepose_losses = build_densepose_losses(cfg)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """

        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box([features[f] for f in self.in_features_box], proposals)

            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            foreground_proposals, _ = select_foreground_proposals(proposals, self.num_classes)

            shared_features = self._forward_shared_block([features[f] for f in self.in_features_shared],
                                                         [x.proposal_boxes for x in foreground_proposals])
            losses.update(self._forward_mask(shared_features, foreground_proposals))
            losses.update(self._forward_keypoint(shared_features, foreground_proposals))
            losses.update(self._forward_densepose(shared_features, foreground_proposals))
            return proposals, losses
        else:
            has_adaptive_pool = detectron2.modeling.poolers.ROIAlign.__name__ == "AdaptivePool" or detectron2.modeling.poolers.RoIPool.__name__ == "AdaptivePool"

            if has_adaptive_pool:
                # check how it is done in apply deltas
                # try do it with tensors, avoiding loops
                level_assignments = assign_boxes_to_levels([x.proposal_boxes for x in proposals],
                                                           self.box_min_level,
                                                           self.box_max_level,
                                                           self.box_canonical_box_size,
                                                           self.box_canonical_level)
                inds = []
                for level in range(len(self.box_pooler_scales)):
                    inds.append(torch.nonzero(level_assignments == level).squeeze(1))
                i = 0
                for image_idx in range(len(proposals)):
                    boxes = proposals[image_idx].proposal_boxes.tensor
                    for level, spatial_scale in enumerate(self.box_pooler_scales):
                        inds_ = inds[level][i:i + len(boxes)]
                        boxes[inds_, 0] = ((boxes[inds_, 0] * spatial_scale - 0.5).floor() + 0.5) / spatial_scale
                        boxes[inds_, 1] = ((boxes[inds_, 1] * spatial_scale - 0.5).floor() + 0.5) / spatial_scale
                        boxes[inds_, 2] = ((boxes[inds_, 2] * spatial_scale - 0.5).ceil() + 0.5) / spatial_scale
                        boxes[inds_, 3] = ((boxes[inds_, 3] * spatial_scale - 0.5).ceil() + 0.5) / spatial_scale
                    proposals[image_idx].proposal_boxes.tensor = boxes
                    i += len(boxes)

            pred_instances = self._forward_box([features[f] for f in self.in_features_box], proposals)

            if has_adaptive_pool:
                level_assignments = assign_boxes_to_levels([x.pred_boxes for x in pred_instances],
                                                           self.shared_min_level,
                                                           self.shared_max_level,
                                                           self.shared_canonical_box_size,
                                                           self.shared_canonical_level)
                inds = []
                for level in range(len(self.shared_pooler_scales)):
                    inds.append(torch.nonzero(level_assignments == level).squeeze(1))
                i = 0
                for image_idx in range(len(pred_instances)):
                    boxes = pred_instances[image_idx].pred_boxes.tensor
                    for level, spatial_scale in enumerate(self.shared_pooler_scales):
                        inds_ = inds[level][i:i + len(boxes)]
                        boxes[inds_, 0] = ((boxes[inds_, 0] * spatial_scale - 0.5).floor() + 0.5) / spatial_scale
                        boxes[inds_, 1] = ((boxes[inds_, 1] * spatial_scale - 0.5).floor() + 0.5) / spatial_scale
                        boxes[inds_, 2] = ((boxes[inds_, 2] * spatial_scale - 0.5).ceil() + 0.5) / spatial_scale
                        boxes[inds_, 3] = ((boxes[inds_, 3] * spatial_scale - 0.5).ceil() + 0.5) / spatial_scale
                    pred_instances[image_idx].pred_boxes.tensor = boxes
                    i += len(boxes)

            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            shared_features = self._forward_shared_block([features[f] for f in self.in_features_shared],
                                                         [x.pred_boxes for x in pred_instances])
            pred_instances = self.forward_with_given_boxes(shared_features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        instances = self._forward_densepose(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def _forward_shared_block(self, features, boxes):
        x = self.shared_pooler(features, boxes)
        if self.shared_block is None:
            return x
        return self.shared_block(x)

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # NOTE: instances = foreground proposals
            mask_logits = self.mask_head(features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, instances)}
        else:
            mask_logits = self.mask_head(features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, visible_selection_masks = select_proposals_with_visible_keypoints(instances)
            keypoint_features = features[torch.cat(visible_selection_masks, dim=0)]
            keypoint_logits = self.keypoint_head(keypoint_features)
            num_images = len(instances)
            normalizer = (
                    num_images
                    * self.batch_size_per_image
                    * self.positive_sample_fraction
                    * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            keypoint_logits = self.keypoint_head(features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def _forward_densepose(self, features, instances):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        if self.training:
            proposals_dp, masks = self.densepose_data_filter(instances)

            if len(proposals_dp) > 0:
                features_dp = features[torch.cat(masks, dim=0)]
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _ = self.densepose_predictor(densepose_head_outputs)
                densepose_loss_dict = self.densepose_losses(proposals_dp, densepose_outputs)
                return densepose_loss_dict
            else:
                # TO MAKE DDP HAPPY
                # because shared features could be calculated and then not used at all
                loss = 0
                for f in features:
                    loss += f.sum() * 0.0

                def get_dummy(module):
                    _dummy = sum(x.view(-1)[0] for x in module.parameters()) * 0.0
                    for child in module.children():
                        _dummy += get_dummy(child)
                    return _dummy

                loss += get_dummy(self.densepose_head) + get_dummy(self.densepose_predictor)
                losses = {"loss_densepose_U": loss, "loss_densepose_V": loss, "loss_densepose_I": loss,
                          "loss_densepose_S": loss}
                return losses
        else:
            features_dp = features
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _ = self.densepose_predictor(densepose_head_outputs)
            else:
                # If no detection occurred instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 4)

            densepose_inference(densepose_outputs, instances)
            return instances
