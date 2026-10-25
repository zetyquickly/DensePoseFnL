# fnl engine

On-device CSE DensePose pipeline: frame (center crop) -> graph A -> RPN glue -> graph B1 -> NMS glue -> graph B2 -> graph C.
Stage 1 (GPU graphs + glue) and stage 2 (graph C, CPU) run on two threads, so C of frame T-1 overlaps A/B of frame T.

- `src/glue.cc` is a port of `fnl_export/glue.py`.
- `src/engine.cc` is the pipeline (TFLite C++ API, GPU delegate fp16 for A/B1/B2, XNNPACK fp16 for C); `src/main.cc` is the `fnl_run` CLI.

Android build (needs a TFLite build for arm64, see CMakeLists.txt):

    cmake -S engine -B out/engine_android -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
          -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DTFLITE_SRC_DIR=<tensorflow> -DTFLITE_BUILD_DIR=<tflite build>
    make -C out/engine_android fnl_run

Run on the phone and render the overlay:

    ffmpeg -ss 30 -t 10 -i video.mp4 -vf scale=640:360 -pix_fmt rgb24 -f rawvideo out/video/clip10s_640x360.rgb
    uv run python engine/tools/run_on_device.py configs/cse.yaml --binary out/engine_android/fnl_run \
        --frames out/video/clip10s_640x360.rgb --width 640 --height 360 --fps 25 --overlay out/video/overlay.mp4
