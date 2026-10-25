cmake_minimum_required(VERSION 3.22)
project(fnl_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(fnl_glue STATIC src/glue.cc)
target_include_directories(fnl_glue PUBLIC include)

# On-device runtime (engine + CLI). Needs a TFLite build for the same ABI:
#   cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
#         -DTFLITE_SRC_DIR=<tensorflow>/ -DTFLITE_BUILD_DIR=<tflite cmake build dir> ..
if(TFLITE_SRC_DIR AND TFLITE_BUILD_DIR)
  file(GLOB_RECURSE TFLITE_LIBS "${TFLITE_BUILD_DIR}/*.a")
  # TF's CMake build leaves out these sources that the GPU delegate (delegates/gpu/delegate.cc, in libtensorflow-lite) depends on
  set(TFLITE_GPU_EXTRA ${TFLITE_SRC_DIR}/tensorflow/lite/async/backend_async_kernel_interface.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/android_hardware_buffer.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/async_buffers.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/gl/android_sync.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/gl/egl_environment.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/gl/egl_context.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/gl/egl_surface.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/gl/gl_errors.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/gpu/gl/request_gpu_info.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/utils/async_type_helpers.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/utils/sync_fence.cc
    ${TFLITE_SRC_DIR}/tensorflow/lite/delegates/utils/utils.cc)
  add_library(fnl_engine STATIC src/engine.cc ${TFLITE_GPU_EXTRA})
  target_include_directories(fnl_engine PUBLIC include
    ${TFLITE_SRC_DIR}
    ${TFLITE_BUILD_DIR}/flatbuffers/include
    ${TFLITE_BUILD_DIR}/abseil-cpp
    ${TFLITE_BUILD_DIR}/opencl_headers
    ${TFLITE_BUILD_DIR}/vulkan_headers/include
    ${TFLITE_BUILD_DIR}/FP16-source/include)
  target_link_libraries(fnl_engine PUBLIC fnl_glue -Wl,--start-group ${TFLITE_LIBS} -Wl,--end-group log android EGL GLESv3 dl)

  add_executable(fnl_run src/main.cc)
  target_link_libraries(fnl_run fnl_engine)
endif()
