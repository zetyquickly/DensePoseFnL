// CSE DensePose engine: frame -> boxes + per-pixel SMPL vertex ids, on the phone.
//
// Two pipeline stages on two threads, so graph C of frame T-1 overlaps the GPU graphs of frame T:
//   stage 1 (GPU):  center crop + resize -> A -> RPN glue -> pool -> B1 -> NMS glue -> pool -> B2
//   stage 2 (CPU):  pack person pixels -> C (XNNPACK fp16, in chunks) -> per-person 64x64 vertex id maps + person margins
// Results come out in submission order.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fnl/glue.h"

namespace fnl {

struct EngineOptions {
  std::string model_dir;     // graph_a/b1/b2/c.tflite
  bool gpu = true;           // graphs A, B1, B2 on the GPU delegate (fp16); false: XNNPACK on the CPU
  int cpu_threads = 4;       // graph C (and A/B1/B2 when gpu == false)
  int glue_threads = 4;      // ROI pooling in stage 1 (the stage-1 thread counts as one)
  float score_thresh = 0.5f; // detections below this are not sent to the DensePose head
  int queue_depth = 1;       // frames waiting in front of each stage
};

// Time spent in each part of one frame, ms.
struct StageTimes {
  double preprocess = 0, a = 0, rpn = 0, pool_b1 = 0, b1 = 0, detect = 0, pool_dp = 0, b2 = 0;  // stage 1
  double pack = 0, c = 0;                                                                      // stage 2
  double stage1 = 0, stage2 = 0;
  double latency = 0;  // Submit() entered -> result ready
};

struct Person {
  Box box;                         // network-input pixels; frame px = net px / scale + (crop_x, crop_y)
  float score = 0;
  std::vector<float> margin;       // res*res, fg - bg coarse segmentation logit, > 0 = person
  std::vector<int32_t> vertex;     // res*res, closest SMPL vertex id on person pixels, -1 elsewhere
};

struct FrameResult {
  int64_t id = 0;
  std::vector<Person> people;
  int res = 0;                     // side of the per-person maps (64)
  int net_w = 0, net_h = 0;        // network canvas
  // The network sees the largest centered crop of the frame that has its aspect ratio, resized to net_w x net_h.
  float scale = 1;                 // network pixels per frame pixel
  int crop_x = 0, crop_y = 0;      // top-left of the crop in the frame
  int person_pixels = 0;
  StageTimes t;
};

class Engine {
 public:
  explicit Engine(const EngineOptions& opts);  // loads the graphs; throws std::runtime_error
  ~Engine();

  // RGB8 interleaved frame of any size; center-cropped to the network's aspect ratio and resized. Blocks while the pipeline is full.
  void Submit(int64_t id, const uint8_t* rgb, int width, int height);
  // Next finished frame, in submission order. Returns false when closed and drained.
  bool Next(FrameResult* out);
  // No more frames: let the pipeline drain, after which Next() returns false.
  void Close();

  int net_width() const;
  int net_height() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fnl
