import os

import torch
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model

from densepose import add_densepose_config
from densepose.modeling.config import add_efficientnet_config, add_roi_shared_config


def setup(args):
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_efficientnet_config(cfg)
    add_roi_shared_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)

    def count_params(model, depth=1, stop_depth=3):
        if depth == stop_depth:
            return
        for n, m in model.named_children():
            params = sum(p.numel() for p in m.parameters()) / 1e6
            print('', end='\t' * depth)
            print(f"{n}: {params:.2f}M parameters")
            count_params(m, depth + 1)

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    print(model)
    print_size_of_model(model)
    print(f"Whole model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    count_params(model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
