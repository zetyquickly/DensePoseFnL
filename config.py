# -*- coding: utf-8 -*-


from detectron2.config import CfgNode as CN


def add_timmnets_config(cfg):
    """
    Add config for TIMMNETS modeling
    """
    _C = cfg

    _C.MODEL.TIMMNETS = CN()
    _C.MODEL.TIMMNETS.NAME = "spnasnet_100"
    _C.MODEL.TIMMNETS.PRETRAINED = True
    _C.MODEL.TIMMNETS.EXPORTABLE = False
    _C.MODEL.TIMMNETS.SCRIPTABLE = False
    _C.MODEL.TIMMNETS.OUT_FEATURES = []
    _C.MODEL.TIMMNETS.NORM = "FrozenBN"
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ISSIMPLE = False
