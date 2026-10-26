# DensePoseFnL

### Making DensePose fast and light

Ruslan Rakhimov\*, Emil Bogomolov\*, Alexandr Notchenko, Fung Mao, Alexey Artemov, Denis Zorin, Evgeny Burnaev

<sub>\* equal contribution · Skolkovo Institute of Science and Technology · Huawei Moscow Research Center · New York University</sub>

*IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021*

<p align="center">
  <b>[ <a href="https://arxiv.org/abs/2006.15190">📄 Paper</a> | <a href="https://huggingface.co/rusrakhimov/DensePoseFnL">🤗 Checkpoints</a> ]</b>
</p>

This repository contains the official implementation of the paper.

## 📖 Abstract

DensePose estimation task is a significant step forward for enhancing user experience computer vision applications ranging
from augmented reality to cloth fitting. Existing neural network models capable of solving this task are heavily parameterized
and a long way from being transferred to an embedded or mobile device. To enable Dense Pose inference on the end device with
current models, one needs to support an expensive server-side infrastructure and have a stable internet connection. To make
things worse, mobile and embedded devices do not always have a powerful GPU inside.
In this work, we target the problem of redesigning the DensePose R-CNN model's architecture so that the final network retains
most of its accuracy but becomes more light-weight and fast. To achieve that, we tested and incorporated many deep learning
innovations from recent years, specifically performing an ablation study on 23 efficient backbone architectures, multiple
two-stage detection pipeline modifications, and custom model quantization methods.
As a result, we achieved 17× model size reduction and 2× latency improvement compared to the baseline model.

## 📝 Citation

```BibTeX
@inproceedings{rakhimov2021making,
  title={Making DensePose fast and light},
  author={Rakhimov, Ruslan and Bogomolov, Emil and Notchenko, Alexandr and Mao, Fung and Artemov, Alexey and Zorin, Denis and Burnaev, Evgeny},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1869--1877},
  year={2021}
}
```

## 📰 Changelog

- `[October 2026]` Checkpoints moved to [Hugging Face](https://huggingface.co/rusrakhimov/DensePoseFnL).
- `[October 2026]` Added [LiteRT export](export/README.md) of CSE models that runs in realtime on Android CPU+GPU, along with the [densepose-android](https://github.com/zetyquickly/densepose-android) app.
- `[January 2022]` Updated to latest Detectron2 and released the weights. **Breaking**: network quantization lost in this version, see [issue](https://github.com/zetyquickly/DensePoseFnL/issues/17).
- `[June 2020]` Initial code release

## ⚙️ Installation

```bash
# Install Detectron2 and DensePose
git clone https://github.com/facebookresearch/detectron2.git && cd detectron2
git checkout bb96d0b01d0605761ca182d0e3fac6ead8d8df6e
pip install -e .
cd projects/DensePose
pip install -e .
```

* `timm==0.4.12`
* `torch==1.10.1`

The `torch==1.10.1` (CUDA 11.3) build has kernels for GPUs up to Ampere (`sm_86`).

## 🚀 Training and Evaluation

```
# Train
python train_net.py --config-file configs/mobile_parsing_rcnn_b_s3x.yaml --num-gpus 8

# Test
python train_net.py --config-file configs/mobile_parsing_rcnn_b_s3x.yaml --eval-only MODEL.WEIGHTS models/mobile_parsing_rcnn_b_s3x.pth
```

## <a name="ModelZoo"></a> 📦 Model Zoo

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: Mobile-Parsing-RCNN-B -->
<tr><td align="left"><a href="configs/mobile_parsing_rcnn_b_s3x.yaml">Mobile-Parsing-RCNN-B</a></td>
<td align="center">s3x</td>
<td align="center">57.1</td>
<td align="center">59.0</td>
<td align="center">50.4</td>
<td align="center">54.4</td>
<td align="center"><a href="https://huggingface.co/rusrakhimov/DensePoseFnL/resolve/main/mobile_parsing_rcnn_b_s3x.pth">model</a></td>
</tr>
<!-- ROW: Mobile-Parsing-RCNN-B-WC2M -->
<tr><td align="left"><a href="configs/mobile_parsing_rcnn_b_wc2m_s3x.yaml">Mobile-Parsing-RCNN-B-WC2M</a></td>
<td align="center">s3x</td>
<td align="center">59.4</td>
<td align="center">63.7</td>
<td align="center">57.3</td>
<td align="center">60.3</td>
<td align="center"><a href="https://huggingface.co/rusrakhimov/DensePoseFnL/resolve/main/mobile_parsing_rcnn_b_wc2m_s3x.pth">model</a></td>
</tr>
<!-- ROW: Mobile-Parsing-RCNN-B-CSE -->
<tr><td align="left"><a href="configs/mobile_parsing_rcnn_b_cse_s3x.yaml">Mobile-Parsing-RCNN-B-CSE</a></td>
<td align="center">s3x</td>
<td align="center">60.2</td>
<td align="center">64.3</td>
<td align="center">59.0</td>
<td align="center">61.2</td>
<td align="center"><a href="https://huggingface.co/rusrakhimov/DensePoseFnL/resolve/main/mobile_parsing_rcnn_b_cse_s3x.pth">model</a></td>
</tr>
</tbody></table>

Download one or all of them with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
hf download rusrakhimov/DensePoseFnL config.yaml mobile_parsing_rcnn_b_cse_s3x.pth --local-dir models
hf download rusrakhimov/DensePoseFnL --local-dir models   # everything
```

`WC2M` corresponds to new training procedure and the model that performs estimation of confidence in regressed UV
coordinates as well as confidences associated with coarse and fine segmentation;
see [Sanakoyeu et al., 2020](https://arxiv.org/pdf/2003.00080.pdf) for details.

`CSE` corresponds to a continuous surface embeddings model for humans;
see [Neverova et al., 2020](https://arxiv.org/abs/2011.12438) for details.

Note: weights for Mobile-Parsing-RCNN-B (s3x) are not the same as presented in the paper but with a similar performance.

## 📚 More instructions

See [ DensePose (Getting Started) ](https://github.com/facebookresearch/detectron2/blob/bb96d0b01d0605761ca182d0e3fac6ead8d8df6e/projects/DensePose/doc/GETTING_STARTED.md)

## ⚖️ License

The code is released under the MIT license, see [LICENSE](LICENSE). The checkpoints are released under Apache-2.0; the
[model card](https://huggingface.co/rusrakhimov/DensePoseFnL) lists the terms of the data they were trained on.
