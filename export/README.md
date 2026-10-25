# Running DensePose on a phone (LiteRT / TFLite)

Mobile-Parsing-RCNN-B-CSE exported to four static TFLite graphs (A backbone + RPN, B1 box head, B2 DensePose head, C vertex matching),
glued by a small C++ engine that runs on the phone's GPU (A, B1, B2) and CPU (C). A 480x640 network does about 49 fps back to back on a
Galaxy S25; the demo app below shows it live.

## 1. Checkpoint

Download `Mobile-Parsing-RCNN-B-CSE` from the [Model Zoo](../README.md#ModelZoo) and put it in `models/` at the repo root
(`models/mobile_parsing_rcnn_b_cse_s3x.pth`, the path the configs point at).

## 2. Export the graphs

    cd export
    uv sync
    uv run python export_litert.py configs/cse.yaml

Writes `out/litert/cse/graph_{a,b1,b2,c}.tflite` and `vertex_embeddings.npy` (colours the overlay). The network input size is in the config:

| config | network input (HxW) |
|---|---|
| `cse` / `cse_portrait` | 480x640 / 640x480 |
| `cse_576` | 576x768 |
| `cse_800` / `cse_800_portrait` | 800x1088 / 1088x800 |
| `cse_480_k300` | 480x640, 300 box proposals instead of 100 |
| `wc2m` | chart head (`mobile_parsing_rcnn_b_wc2m_s3x.pth`) |

`benchmark_android.py configs/cse.yaml [--gpu]` times each graph on a connected phone with TFLite's `benchmark_model`.

## 3. Engine (C++)

`engine/` is the on-device pipeline: frame (centre crop) -> A -> RPN glue -> B1 -> NMS glue -> B2 -> C. Stage 1 (GPU graphs + glue) and
stage 2 (C, CPU) run on two threads. `fnl_run` is a command-line runner that prints per-stage timings; `engine/tools/run_on_device.py`
pushes it, the graphs and a clip over adb and renders the overlay. See [engine/README.md](engine/README.md). Both the CLI and the app need a
TFLite build for arm64-v8a / android-31 with the GPU delegate (see `engine/CMakeLists.txt`).

## 4. Android app

[zetyquickly/densepose-android](https://github.com/zetyquickly/densepose-android) wraps the engine in a Kotlin UI (photo, gallery video,
live camera). It builds against a checkout of this repo (`export/engine` and `export/out/litert`); follow the README there.

## Accuracy

Measured on `densepose_coco_2014_minival_cse` (1508 images) with detectron2/DensePose's own evaluators, on the host CPU with the exported
graphs in fp32 (the phone runs fp16). Detectron2 rows use its default input (shortest edge 800) or the same letterboxed canvas as ours.

| model | box AP | segm AP | GPS | GPSm |
|---|---|---|---|---|
| detectron2, Model Zoo | 60.2 | 64.3 | 59.0 | 61.2 |
| detectron2, 800px input, re-evaluated | 60.2 | 64.2 | 59.1 | 61.2 |
| detectron2 @ 480x640 | 54.0 | 61.3 | 53.0 | 56.7 |
| **ours** 480x640, graphs + glue, stock matching | 52.9 | 61.3 | 53.0 | 56.6 |
| **ours** 480x640, engine output | 52.9 | 61.3 | 50.7 | 55.4 |

"Engine output" is what the engine returns: vertex ids matched on the 64x64 person pixels and resized nearest to the box, with the
approximate graph C. "Stock matching" resizes the embedding bilinearly to the box and takes the exact nearest vertex, as detectron2 does.

Input size is what matters. Engine output (approximate C), S25 GPU, A/B1/B2 fp16, back to back:

| config | dets/img | box AP | segm AP | GPS | GPSm | S25 fps | stage 1 (ms) |
|---|---|---|---|---|---|---|---|
| `cse` 480x640, 100 proposals | 8.1 | 52.9 | 61.3 | 50.7 | 55.4 | 49.0 | 20.3 |
| `cse_480_k300` 480x640, 300 proposals | 12.1 | 53.6 | 61.3 | 50.7 | 55.4 | 41.9 | 23.7 |
| `cse_576` 576x768 | 11.8 | 56.0 | 63.0 | 54.3 | 58.2 | 36.4 | 27.3 |
| `cse_800` 800x1088 | 11.6 | 59.5 | 64.5 | 56.7 | 60.4 | 23.5 | 42.2 |

The 800x1088 network is within 0.7 box AP of detectron2's default but cannot hold 25 fps. Going from 100 to 300 proposals adds
0.7 box AP.

These numbers were produced with a scorer built on detectron2's `DensePoseCocoEval`, which is not part of this repo.
