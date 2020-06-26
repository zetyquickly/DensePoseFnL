
from . import dataset  # just to register data
from .config import add_densepose_config
from .dataset_mapper import DatasetMapper
from .densepose_head import ROI_DENSEPOSE_HEAD_REGISTRY
from .evaluator import DensePoseCOCOEvaluator
from .modeling.config import add_efficientnet_config, add_roi_shared_config
from .modeling.geffnet_backbone import *
from .modeling.geffnet_backbone_v2 import *
from .modeling.geffnet_box_head import *
from .modeling.geffnet_densepose_head import *
from .modeling.geffnet_mask_head import *
from .modeling.geffnet_rpn import *
from .modeling.shared_roi_head import *
from .modeling.layers.bifpn import BiFPN, Conv3x3BnReLU, FastNormalizedFusion
from .roi_head import DensePoseROIHeads
from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
