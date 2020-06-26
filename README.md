# Making DensePose fast and light

# Original DensePose project Quick Start

See [ Getting Started ](doc/GETTING_STARTED.md)

# Training and Evaluation

0. The project dependencies:
* `detectron2==0.1.0`
* `pytorch >= 1.4.0`

1. You can train a network from scratch using configs in `./configs` folder and `train_net.py` script. 
  * `s0_bv2_bifpn_f64_s3x.yaml` config corresponds to the `Mobile-RCNN (B s3x)` model, 
  * `s0_bv2_bifpn_f64.yaml` config corresponds to the `Mobile-RCNN (B s1x)` model, 
  * `densepose_parsing_rcnn_spnasnet_100_FPN_s3x.yaml` config corresponds to the `Mobile-RCNN (A s1x)` model, 
  *  `densepose_parsing_rcnn_R_50_FPN_s1x.yaml` config corresponds to the  `Parsing RCNN` model
Then evaluate the model with `--eval_only` flag.

2. You can run QAT of the `Mobile-RCNN (B s3x)` using `train_net.py` with `--qat` flag then evaluate it with `--quant-eval` flag.
To use proposed hooks preserving mechanism it is needed to modify PyTorch source code according to files inside `modify_pytorch` directroy
OR
Build PyTorch from source using the following commit https://github.com/pytorch/pytorch/pull/37233/commits/c8de10d2a394484ac58dd131878950b8ab7ac7a9