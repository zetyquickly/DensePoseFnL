# \[WACV 2021\] Making DensePose fast and light

This repository contains Python implementation of the paper [_Making DensePose fast and
light_](http://arxiv.org/abs/2006.15190)

# Changelog

- `[January 2022]` Updated to latest Detectron2 and released the weights. **Breaking**: network quantization lost in this version [issue](https://github.com/zetyquickly/DensePoseFnL/issues/17).
- `[June 2020]` Initial code release

## Installation

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

## Training and Evaluation

```
# Train
python train_net.py --config-file configs/mobile_parsing_rcnn_b_s3x.yaml --num-gpus 8

# Test
python train_net.py --config-file configs/mobile_parsing_rcnn_b_s3x.yaml --eval-only MODEL.WEIGHTS model.pth
```

## <a name="ModelZoo"></a> Model Zoo

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
<tr><td align="left"><a href="../configs/mobile_parsing_rcnn_b_s3x.yaml">Mobile-Parsing-RCNN-B</a></td>
<td align="center">s3x</td>
<td align="center">57.1</td>
<td align="center">59.0</td>
<td align="center">50.4</td>
<td align="center">54.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1yC5QBT0fYmMrI40RhrA3nB5Pa-F7XpFt/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: Mobile-Parsing-RCNN-B-WC2M -->
<tr><td align="left"><a href="../configs/mobile_parsing_rcnn_b_wc2m_s3x.yaml">Mobile-Parsing-RCNN-B-WC2M</a></td>
<td align="center">s3x</td>
<td align="center">59.4</td>
<td align="center">63.7</td>
<td align="center">57.3</td>
<td align="center">60.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1yEBH7ArbadycdSW-Yk1rM5Hl0v3HE7_V/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: Mobile-Parsing-RCNN-B-CSE -->
<tr><td align="left"><a href="../configs/mobile_parsing_rcnn_b_cse_s3x.yaml">Mobile-Parsing-RCNN-B-CSE</a></td>
<td align="center">s3x</td>
<td align="center">60.2</td>
<td align="center">64.3</td>
<td align="center">59.0</td>
<td align="center">61.2</td>
<td align="center"><a href="https://drive.google.com/file/d/1yfZuM8git92NFVEP2PtjBeMyJuXSeLj3/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

`WC2M` corresponds to new training procedure and the model that performs estimation of confidence in regressed UV
coordinates as well as confidences associated with coarse and fine segmentation;
see [Sanakoyeu et al., 2020](https://arxiv.org/pdf/2003.00080.pdf) for details.

`CSE` corresponds to a continuous surface embeddings model for humans;
see [Neverova et al., 2020](https://arxiv.org/abs/2011.12438) for details.

Note: weights for Mobile-Parsing-RCNN-B (s3x) are not the same as presented in the paper but with a similar performance.

## More instructions

See [ DensePose (Getting Started) ](https://github.com/facebookresearch/detectron2/blob/bb96d0b01d0605761ca182d0e3fac6ead8d8df6e/projects/DensePose/doc/GETTING_STARTED.md)

## Citation

If you find our work useful in your research, please consider citing:

```BibTeX
@inproceedings{rakhimov2021making,
  title={Making DensePose fast and light},
  author={Rakhimov, Ruslan and Bogomolov, Emil and Notchenko, Alexandr and Mao, Fung and Artemov, Alexey and Zorin, Denis and Burnaev, Evgeny},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1869--1877},
  year={2021}
}
```

## License

See the [LICENSE](LICENSE) for more details.
