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

## 5. Validation (optional)

Nothing above needs this. It reproduces the accuracy table and checks the glue against detectron2. detectron2 needs old pins (torch 2.2,
numpy < 2, timm 0.4.12) that conflict with the export stack, so it has its own environment in `reference/`. Run everything below from the
repo root:

    uv sync --project export/reference

The lock includes detectron2 and DensePose at `bb96d0b`. detectron2 is built from source without its CUDA ops, since everything here runs
on the CPU. The build settings in `reference/pyproject.toml` need a recent uv (tested with 0.12).

**Accuracy table.** Box AP, segm AP, GPS and GPSm come from detectron2/DensePose's own `DensePoseCocoEval` on
`densepose_coco_2014_minival_cse` (the images and `densepose_minival2014_cse.json` under `datasets/coco/`). Our predictions are saved from
the export env, detectron2's from the reference env, and one scorer reads both:

    ANN=datasets/coco/annotations/densepose_minival2014_cse.json
    # export env: exported graphs + glue over the split
    uv run --project export python export/eval_pipeline.py export/configs/cse.yaml --images datasets/coco/val2014 --ann $ANN --out export/out/eval/ours.npz
    # reference env: the original model (--mode default: shortest edge 800; --mode canvas: our letterboxed input)
    uv run --project export/reference python export/reference/eval_detectron2.py export/configs/cse.yaml configs/mobile_parsing_rcnn_b_cse_s3x.yaml \
        datasets/coco/val2014 $ANN export/out/eval/d2.npz --mode default
    uv run --project export/reference python export/reference/prep_geodists.py     # once: geodesic distances for GPS / GPSm (6 GB download), needs ~10 GB of RAM
    uv run --project export/reference python export/reference/eval_predictions.py "d2=export/out/eval/d2.npz" "ours=export/out/eval/ours.npz@engine-approx@device"

A run is `LABEL=FILE.npz[@MATCHING[@FILTER]]`: `stock` matching is detectron2's (embedding resized bilinearly to the box, exact nearest
vertex), `engine` and `engine-approx` are the engine's (vertex ids on the 64x64 person pixels, exact or with the approximate graph C), and
the `device` filter keeps what the phone sends on (top `K2` with score >= `S`, default 4 and 0.5) where `all` keeps every detection above 0.05.

**Glue tests.** `engine/tests/test_glue.cc` checks the C++ glue against `fnl_export/glue.py`, and `check_glue.py` checks `glue.py` against
detectron2's own tensors, so the C++ is tied to detectron2 through the numpy port. Both start from tensors dumped by detectron2 on a few images
(`test_glue` also runs a NaN-logit case: non-finite RPN logits are dropped before the top-k, as detectron2 drops them before NMS):

    uv run --project export/reference python export/reference/dump_detectron2.py export/configs/cse.yaml configs/mobile_parsing_rcnn_b_cse_s3x.yaml \
        export/out/reference_480x640 datasets/coco/val2014/<image>.jpg ...
    cd export
    uv run python check_glue.py out/reference_480x640/*.npz
    uv run python engine/tools/make_test_vectors.py out/glue_vectors out/reference_480x640/*.npz
    cmake -S engine -B out/engine_host && make -C out/engine_host test_glue && out/engine_host/test_glue out/glue_vectors/*

`compare_with_detectron2.py` compares detectron2's tensors with our graphs run eagerly and as `.tflite`, graph by graph.
