"""
DensePose Training Script.

This script is similar to the training script in detectron2/tools.

It is an example of how a user might use detectron2 for a new project.
"""
import os
import logging

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_setup, launch, default_argument_parser
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

from densepose import DatasetMapper, DensePoseCOCOEvaluator, add_densepose_config
from densepose.modeling.config import add_efficientnet_config, add_roi_shared_config
from densepose.modeling.quantize import quantize_decorate, quantize_prepare
# from densepose.modeling.quantize_caffe2 import quantize_decorate, quantize_prepare
from densepose.modeling.layers.adaptive_pool import AdaptivePool


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        if cfg.MODEL.DENSEPOSE_ON:
            evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def setup(args):
    cfg = get_cfg()
    add_densepose_config(cfg)

    add_efficientnet_config(cfg)
    add_roi_shared_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="densepose")
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.adaptive_pool:
        assert args.eval_only, "AdaptivePool is supported only in test mode"
        import detectron2.modeling.poolers
        detectron2.modeling.poolers.RoIPool = AdaptivePool
        detectron2.modeling.poolers.ROIAlign = AdaptivePool
    
    if args.qat:
        quantize_decorate()
        quantize_prepare()
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.model = Trainer.update_model(trainer.model, args.qbackend)
        trainer.checkpointer.model = trainer.model
        return trainer.train()
        
    elif args.eval_only or args.quant_eval:
        if args.quant_eval:
            quantize_decorate()
            quantize_prepare()
            model = Trainer.build_model(cfg)
            model = Trainer.update_model(model, args.qbackend)
            model.eval()
            from fvcore.common.checkpoint import _strip_prefix_if_present
            weights = torch.load(cfg.MODEL.WEIGHTS)['model']
            _strip_prefix_if_present(weights, "module.")
            model.load_state_dict(weights)
            torch.quantization.convert(model, inplace=True)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(
                cfg.MODEL.WEIGHTS
            )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--adaptive-pool", action="store_true", help="replace roialign, roipool to adaptivepool")
    parser.add_argument("--qat", action="store_true", help="do QAT on GPU")
    parser.add_argument("--quant-eval", action="store_true", help="do post quantization evaluation on CPU")
    parser.add_argument("--qbackend", action="store", help="use qnnpack or fbgemm as quntization backend")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
