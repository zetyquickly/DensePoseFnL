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
| `cse_800` / `cse_800_portrait` | 800x1088 / 1088x800, 300 box proposals instead of 100 |
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
graphs in fp32 (the phone runs fp16) and the numpy glue `glue.py`; the engine's vertex matching is emulated in PyTorch. Nothing below was run on
the phone. Detectron2 rows use its default input (shortest edge 800) or the same letterboxed canvas as ours. Unless a row says "top 4", every
detection above 0.05 is scored against the full ground truth. Commands to reproduce are in section 5.

| model | box AP | segm AP | GPS | GPSm |
|---|---|---|---|---|
| detectron2, Model Zoo | 60.2 | 64.3 | 59.0 | 61.2 |
| detectron2, 800px input, re-evaluated | 60.2 | 64.2 | 59.1 | 61.2 |
| detectron2 @ 480x640 | 54.0 | 61.3 | 53.0 | 56.7 |
| **ours** `cse` 480x640, detectron2's vertex matching, all detections | 52.9 | 61.3 | 53.0 | 56.6 |
| **ours** `cse` 480x640, engine's vertex matching, all detections | 52.9 | 61.3 | 50.7 | 55.4 |
| **ours** `cse` 480x640, engine's vertex matching, top 4 with score >= 0.5 per image (as the phone keeps) | 42.2 | 60.1 | 50.0 | 54.6 |
| **ours** `cse_portrait` 640x480, engine's vertex matching, all detections | 50.4 | 59.4 | 48.3 | 53.2 |
| **ours** `cse_portrait` 640x480, engine's vertex matching, top 4 with score >= 0.5 per image | 41.1 | 58.4 | 47.8 | 52.6 |
| **ours** `cse_800` 800x1088, engine's vertex matching, all detections | 59.5 | 64.5 | 56.7 | 60.4 |
| **ours** `cse_800` 800x1088, engine's vertex matching, top 4 with score >= 0.5 per image | 44.9 | 63.0 | 55.5 | 59.1 |
| **ours** `cse_800_portrait` 1088x800, engine's vertex matching, all detections | 57.8 | 63.7 | 55.8 | 59.3 |
| **ours** `cse_800_portrait` 1088x800, engine's vertex matching, top 4 with score >= 0.5 per image | 44.2 | 62.5 | 54.9 | 58.3 |

## 5. Validation (optional)

Nothing above needs this. It is for reproducing the accuracy table and for checking that the glue does what detectron2 does.

detectron2 needs old pins (torch 2.2, numpy < 2, timm 0.4.12) that conflict with the export stack, so it lives in its own environment in `reference/`. Set it up once. All commands in this section run from the repo root; "export env" means `uv run --project export`, "reference env" means `uv run --project export/reference`.

    uv sync --project export/reference
    uv pip install --python export/reference/.venv/bin/python --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git@bb96d0b01d0605761ca182d0e3fac6ead8d8df6e" \
        "detectron2-densepose @ git+https://github.com/facebookresearch/detectron2.git@bb96d0b01d0605761ca182d0e3fac6ead8d8df6e#subdirectory=projects/DensePose"

### Reproducing the accuracy table

The images and `densepose_minival2014_cse.json` go under `datasets/coco/`. The scoring is detectron2/DensePose's own `DensePoseCocoEval`. Predictions are saved first (ours from the export env, detectron2's from the reference env), then one scorer reads both.
    ANN=datasets/coco/annotations/densepose_minival2014_cse.json
    OUT=export/out/eval

    # ours: exported graphs + glue.py on the host (needs export/out/litert/cse from section 2)
    uv run --project export python export/eval_pipeline.py export/configs/cse.yaml \
        --images datasets/coco/val2014 --ann $ANN --out $OUT/ours.npz

    # detectron2: the original model, at its own input size and at ours
    for mode in default canvas; do
        uv run --project export/reference python export/reference/eval_detectron2.py \
            export/configs/cse.yaml configs/mobile_parsing_rcnn_b_cse_s3x.yaml \
            datasets/coco/val2014 $ANN $OUT/d2_$mode.npz --mode $mode
    done

    # once: geodesic distances for GPS / GPSm (needs ~10 GB of RAM)
    uv run --project export/reference python export/reference/prep_geodists.py

    # score
    uv run --project export/reference python export/reference/eval_predictions.py \
        --vertices export/out/litert/cse/vertex_embeddings.npy \
        "detectron2, 800px input=$OUT/d2_default.npz" \
        "detectron2 @ 480x640=$OUT/d2_canvas.npz" \
        "ours, detectron2's matching=$OUT/ours.npz" \
        "ours, engine's matching=$OUT/ours.npz@engine-approx" \
        "ours, engine's matching, top 4=$OUT/ours.npz@engine-approx@device"

Each scored run is `LABEL=FILE.npz` plus up to two options after `@`:

- **Matching** decides how a detection's embedding is turned into vertex ids.
  - `stock` (the default) is detectron2's way: resize the embedding to the box, take the exact nearest vertex.
  - `engine` is the engine's way: take the nearest vertex on the 64x64 person pixels, then resize the ids to the box.
  - `engine-approx` is the same, using the approximate graph C that the phone runs.
- **Filter** decides which detections are scored.
  - `all` (the default) scores every detection above 0.05.
  - `device` keeps what the phone keeps: per image, the top 4 with score >= 0.5 (change with `--device K2 SCORE`). It is applied to the saved host predictions; nothing is run on the phone.

The first row of the table, the Model Zoo numbers, is detectron2's published result and is not rerun.

### Checking the glue

`glue.py` and `glue.cc` are two ports of the same detectron2 code, so they are checked in a chain: detectron2 against `glue.py`, then `glue.py` against `glue.cc`. Both checks start from tensors dumped by detectron2 on a few images.

    # dump detectron2's tensors (reference env)
    uv run --project export/reference python export/reference/dump_detectron2.py \
        export/configs/cse.yaml configs/mobile_parsing_rcnn_b_cse_s3x.yaml \
        export/out/reference_480x640 datasets/coco/val2014/<image>.jpg ...

    # detectron2 vs glue.py: prints the largest difference after each step (export env)
    uv run --project export python export/check_glue.py export/out/reference_480x640/*.npz

    # glue.py vs glue.cc: dump test vectors, build the host test, run it
    uv run --project export python export/engine/tools/make_test_vectors.py \
        export/out/glue_vectors export/out/reference_480x640/*.npz
    cmake -S export/engine -B export/out/engine_host
    make -C export/out/engine_host test_glue
    export/out/engine_host/test_glue export/out/glue_vectors/*

`test_glue` also feeds the C++ glue NaN RPN logits. Non-finite logits are dropped before the top-k, as detectron2 drops them before NMS.

To compare detectron2 with our graphs one graph at a time, run `export/compare_with_detectron2.py`. It checks the graphs eagerly in PyTorch and as `.tflite`, each fed detectron2's own input so errors don't pile up.
