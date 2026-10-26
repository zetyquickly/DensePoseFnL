// Host-side glue between the exported graphs: port of fnl_export/glue.py, checked against it by tests/test_glue.cc.
// No TFLite dependency; tensors are NCHW float32.
#pragma once

#include <cstdint>
#include <vector>

namespace fnl {

class ThreadPool;

struct Box {
  float x1, y1, x2, y2;
};

struct Detection {
  Box box;
  float score;
};

constexpr int kLevels = 5;                           // p2..p6
constexpr int kStride[kLevels] = {4, 8, 16, 32, 64};

// One RPN level as graph A emits it: logits (3,h,w), deltas (12,h,w).
struct RpnLevel {
  const float* logits;
  const float* deltas;
  int h, w;
};

// One FPN feature map (c,h,w).
struct Feature {
  const float* data;
  int c, h, w;
};

// Top `post_topk` proposals by objectness: per level top-1000, decode, clip, drop empty, NMS 0.7; levels merged by score.
std::vector<Box> RpnProposals(const RpnLevel (&levels)[kLevels], int img_h, int img_w, int post_topk);

// torchvision roi_align(aligned=True, sampling_ratio=0): `boxes` (n) in image pixels -> dst (n,c,out,out).
// With a pool, the boxes (and channel blocks of them) are spread over its threads.
void RoiAlign(const Feature& f, const Box* boxes, int n, int out, float scale, float* dst, ThreadPool* pool = nullptr);

// Box-head pooling from p2..p5: each box is pooled from the level it is assigned to -> dst (n,c,out,out).
void PoolBox(const Feature (&p2_p5)[4], const Box* boxes, int n, int out, float* dst, ThreadPool* pool = nullptr);

// Graph B1 outputs (probs (n,2), deltas (n,4)) for `n` proposals -> detections by decreasing score
// (score > 0.05, NMS 0.5, top 100).
std::vector<Detection> Detect(const float* probs, const float* deltas, const Box* proposals, int n, int img_h, int img_w);

// Graph B2 outputs of k ROIs (embedding (k,d,r,r), coarse segm (k,2,r,r)) -> `mask` (k*r*r, 1 = person) and `px`
// (count,d): the embeddings of the person pixels in mask raster order. Returns count. `margin` (optional, k*r*r): fg - bg logit.
int PackPersonPixels(const float* emb, const float* seg, int k, int d, int r, std::vector<float>* px,
                     std::vector<uint8_t>* mask, std::vector<float>* margin = nullptr);

}  // namespace fnl
