#include "fnl/glue.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <utility>

#include "fnl/thread_pool.h"

namespace fnl {
namespace {

constexpr float kAnchorSize[kLevels] = {32, 64, 128, 256, 512};
constexpr double kAspect[3] = {0.5, 1.0, 2.0};
constexpr int kRpnPreNmsTopk = 1000;
constexpr float kRpnNms = 0.7f;
constexpr float kBoxWeights[4] = {10.f, 10.f, 5.f, 5.f};
constexpr float kBoxScoreThresh = 0.05f, kBoxNms = 0.5f;
constexpr int kBoxTopk = 100;
const float kScaleClamp = std::log(1000.f / 16);

float Area(const Box& b) { return (b.x2 - b.x1) * (b.y2 - b.y1); }

// detectron2 Box2BoxTransform.apply_deltas for one box.
Box Decode(const float* d, const Box& b, const float* weights) {
  const float w = b.x2 - b.x1, h = b.y2 - b.y1, cx = b.x1 + 0.5f * w, cy = b.y1 + 0.5f * h;
  const float dx = d[0] / weights[0], dy = d[1] / weights[1];
  const float dw = std::min(d[2] / weights[2], kScaleClamp), dh = std::min(d[3] / weights[3], kScaleClamp);
  const float pcx = dx * w + cx, pcy = dy * h + cy, pw = std::exp(dw) * w, ph = std::exp(dh) * h;
  return {pcx - 0.5f * pw, pcy - 0.5f * ph, pcx + 0.5f * pw, pcy + 0.5f * ph};
}

Box Clip(Box b, int h, int w) {
  auto c = [](float v, int hi) { return std::min(std::max(v, 0.f), static_cast<float>(hi)); };
  return {c(b.x1, w), c(b.y1, h), c(b.x2, w), c(b.y2, h)};
}

// Greedy NMS over boxes already sorted by decreasing score; returns the kept indices.
std::vector<int> NmsSorted(const std::vector<Box>& boxes, float thresh) {
  std::vector<int> keep;
  std::vector<char> dead(boxes.size(), 0);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (dead[i]) continue;
    keep.push_back(static_cast<int>(i));
    const Box& a = boxes[i];
    const float area_a = Area(a);
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (dead[j]) continue;
      const Box& b = boxes[j];
      const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
      const float inter = std::max(iw, 0.f) * std::max(ih, 0.f);
      if (inter / (area_a + Area(b) - inter) > thresh) dead[j] = 1;
    }
  }
  return keep;
}

// Whether `b` overlaps any of the (better) boxes `kept` by IoU > thresh; same arithmetic as NmsSorted.
bool Suppressed(const std::vector<Box>& kept, const Box& b, float thresh) {
  const float area_b = Area(b);
  for (const Box& a : kept) {
    const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    const float inter = std::max(iw, 0.f) * std::max(ih, 0.f);
    if (inter / (Area(a) + area_b - inter) > thresh) return true;
  }
  return false;
}

// Bilinear sampling of one ROI axis, folded into one weight vector per output bin. The samples of a bin (grid per bin, positions
// as in torchvision roi_align(aligned=True); samples outside the map weigh zero) reduce to weights over a contiguous run of
// map cells, so ROI align is separable: a pass over rows, then a pass over columns.
struct Axis {
  std::vector<int> lo, n, off;  // per bin: first cell, number of cells, offset into w
  std::vector<float> w;         // weights, already divided by the grid size
  int min = 0, max = -1;        // cell range touched by any bin
  std::vector<int> tap_lo, tap_hi;  // scratch: the samples of one bin
  std::vector<float> tap_w0, tap_w1;
};

void BuildAxis(Axis* a, float start, float bin, int bins, int grid, int size) {
  a->lo.assign(bins, 0), a->n.assign(bins, 0), a->off.assign(bins, 0);
  a->w.clear();
  a->min = size, a->max = -1;
  const float inv = 1.f / grid;
  for (int b = 0; b < bins; ++b) {
    int first = size, last = -1;
    a->tap_lo.assign(grid, -1), a->tap_hi.assign(grid, 0), a->tap_w0.assign(grid, 0.f), a->tap_w1.assign(grid, 0.f);
    int* tap_lo = a->tap_lo.data();
    int* tap_hi = a->tap_hi.data();
    float* tap_w0 = a->tap_w0.data();
    float* tap_w1 = a->tap_w1.data();
    for (int g = 0; g < grid; ++g) {
      float pos = start + (b + (g + 0.5f) / grid) * bin;
      if (!(pos >= -1.f && pos <= size)) continue;
      pos = std::max(pos, 0.f);
      int lo = static_cast<int>(pos);
      const float frac = lo >= size - 1 ? 0.f : pos - lo;
      lo = std::min(lo, size - 1);
      tap_lo[g] = lo, tap_hi[g] = std::min(lo + 1, size - 1);
      tap_w0[g] = (1.f - frac) * inv, tap_w1[g] = frac * inv;
      first = std::min(first, lo), last = std::max(last, tap_hi[g]);
    }
    a->off[b] = static_cast<int>(a->w.size());
    if (last < first) continue;
    a->lo[b] = first, a->n[b] = last - first + 1;
    a->w.resize(a->w.size() + last - first + 1, 0.f);
    float* w = a->w.data() + a->off[b] - first;
    for (int g = 0; g < grid; ++g)
      if (tap_lo[g] >= 0) w[tap_lo[g]] += tap_w0[g], w[tap_hi[g]] += tap_w1[g];
    a->min = std::min(a->min, first), a->max = std::max(a->max, last);
  }
}

// Indices of the k largest of v[0..n) by decreasing value, ties by increasing index. v must be finite: NaN breaks the ordering.
std::vector<int> TopKFinite(const float* v, int n, int k, std::vector<float>* scratch) {
  std::vector<int> idx;
  idx.reserve(k);
  if (k >= n) {
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
  } else {
    scratch->assign(v, v + n);
    std::nth_element(scratch->begin(), scratch->begin() + (k - 1), scratch->end(), std::greater<float>());
    const float kth = (*scratch)[k - 1];
    std::vector<int> ties;
    for (int i = 0; i < n; ++i) {
      if (v[i] > kth)
        idx.push_back(i);
      else if (v[i] == kth)
        ties.push_back(i);
    }
    idx.insert(idx.end(), ties.begin(), ties.begin() + (k - idx.size()));
  }
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] != v[b] ? v[a] > v[b] : a < b; });
  return idx;
}

// TopKFinite over the finite values only. A NaN or Inf logit (fp16 overflow on the GPU delegate) is not a candidate, as detectron2
// drops non-finite scores before NMS (find_top_rpn_proposals); the k best finite ones are kept, so a NaN takes nobody's place.
std::vector<int> TopK(const float* v, int n, int k, std::vector<float>* scratch) {
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += !std::isfinite(v[i]);
  if (bad == 0) return TopKFinite(v, n, k, scratch);
  std::vector<int> at;  // original index of each finite value
  std::vector<float> finite;
  at.reserve(n - bad), finite.reserve(n - bad);
  for (int i = 0; i < n; ++i)
    if (std::isfinite(v[i])) at.push_back(i), finite.push_back(v[i]);
  std::vector<int> idx = TopKFinite(finite.data(), static_cast<int>(finite.size()), std::min(k, static_cast<int>(finite.size())), scratch);
  for (int& i : idx) i = at[i];  // `at` is increasing, so ties stay in increasing original index
  return idx;
}

}  // namespace

// Per level: top-1000 anchors by logit, each decoded and kept unless an already kept box of the level overlaps it more than 0.7
// (greedy NMS). Levels are merged by score as they go and the walk stops at post_topk kept boxes: a kept box only depends on
// better boxes of its own level, so this equals "NMS every level, merge, keep the best post_topk" without the NMS of the rest.
std::vector<Box> RpnProposals(const RpnLevel (&levels)[kLevels], int img_h, int img_w, int post_topk) {
  std::vector<float> flat, scratch;
  std::vector<int> order[kLevels];      // candidate anchors of each level by decreasing logit
  std::vector<float> logit[kLevels];    // logits in flat anchor order
  Box cell[kLevels][3];                 // anchor cells at the origin
  for (int lv = 0; lv < kLevels; ++lv) {
    const RpnLevel& L = levels[lv];
    const int hw = L.h * L.w, n = hw * 3;
    // flat index i = (y*w + x)*3 + a, the order detectron2 flattens the head outputs in
    logit[lv].resize(n);
    for (int a = 0; a < 3; ++a) {
      const float* src = L.logits + static_cast<size_t>(a) * hw;
      float* dst = logit[lv].data() + a;
      for (int p = 0; p < hw; ++p) dst[3 * p] = src[p];
    }
    order[lv] = TopK(logit[lv].data(), n, std::min(n, kRpnPreNmsTopk), &scratch);
    for (int a = 0; a < 3; ++a) {
      const double bw = std::sqrt(static_cast<double>(kAnchorSize[lv]) * kAnchorSize[lv] / kAspect[a]), bh = kAspect[a] * bw;
      cell[lv][a] = {static_cast<float>(-bw / 2), static_cast<float>(-bh / 2), static_cast<float>(bw / 2), static_cast<float>(bh / 2)};
    }
  }

  const float unit[4] = {1.f, 1.f, 1.f, 1.f};
  std::vector<Box> kept[kLevels], out;
  size_t next[kLevels] = {};
  while (static_cast<int>(out.size()) < post_topk) {
    int best = -1;
    float best_logit = 0.f;
    for (int lv = 0; lv < kLevels; ++lv) {
      if (next[lv] == order[lv].size()) continue;
      const float s = logit[lv][order[lv][next[lv]]];
      if (best < 0 || s > best_logit) best = lv, best_logit = s;  // equal scores: the lower level first
    }
    if (best < 0) break;
    const RpnLevel& L = levels[best];
    const int hw = L.h * L.w, i = order[best][next[best]++];
    const int a = i % 3, p = i / 3, y = p / L.w, x = p % L.w;
    const float sx = static_cast<float>(x * kStride[best]), sy = static_cast<float>(y * kStride[best]);
    const Box& c = cell[best][a];
    const Box anchor = {sx + c.x1, sy + c.y1, sx + c.x2, sy + c.y2};
    const float d[4] = {L.deltas[(a * 4 + 0) * hw + p], L.deltas[(a * 4 + 1) * hw + p], L.deltas[(a * 4 + 2) * hw + p],
                        L.deltas[(a * 4 + 3) * hw + p]};
    const Box b = Clip(Decode(d, anchor, unit), img_h, img_w);
    if (!(b.x2 - b.x1 > 0 && b.y2 - b.y1 > 0)) continue;
    if (Suppressed(kept[best], b, kRpnNms)) continue;
    kept[best].push_back(b);
    out.push_back(b);
  }
  return out;
}

namespace {

struct RoiScratch {
  Axis ay, ax;
  std::vector<float> rows;  // (out, columns touched): the row pass of one channel
};

// Channels [c0, c1) of one box -> dst (c1-c0, out, out).
void RoiAlignChannels(const Feature& f, const Box& b, int out, float scale, int c0, int c1, float* dst, RoiScratch* s) {
  const size_t plane = static_cast<size_t>(f.h) * f.w;
  Axis& ay = s->ay;
  Axis& ax = s->ax;
  const float rw = (b.x2 - b.x1) * scale, rh = (b.y2 - b.y1) * scale;
  const int gw = std::max(static_cast<int>(std::ceil(rw / out)), 1), gh = std::max(static_cast<int>(std::ceil(rh / out)), 1);
  BuildAxis(&ay, b.y1 * scale - 0.5f, rh / out, out, gh, f.h);
  BuildAxis(&ax, b.x1 * scale - 0.5f, rw / out, out, gw, f.w);
  if (ay.max < ay.min || ax.max < ax.min) {  // entirely outside the map
    std::fill(dst, dst + static_cast<size_t>(c1 - c0) * out * out, 0.f);
    return;
  }
  const int cols = ax.max - ax.min + 1;
  s->rows.resize(static_cast<size_t>(out) * cols);
  for (int c = c0; c < c1; ++c) {
    const float* p = f.data + c * plane + ax.min;
    for (int py = 0; py < out; ++py) {
      float* __restrict r = s->rows.data() + static_cast<size_t>(py) * cols;
      std::fill(r, r + cols, 0.f);
      const float* wy = ay.w.data() + ay.off[py];
      for (int i = 0; i < ay.n[py]; ++i) {
        const float* __restrict src = p + static_cast<size_t>(ay.lo[py] + i) * f.w;
        const float w = wy[i];
        for (int x = 0; x < cols; ++x) r[x] += w * src[x];
      }
    }
    float* o = dst + static_cast<size_t>(c - c0) * out * out;
    for (int py = 0; py < out; ++py) {
      const float* r = s->rows.data() + static_cast<size_t>(py) * cols;
      for (int px = 0; px < out; ++px) {
        const float* wx = ax.w.data() + ax.off[px];
        const float* src = r + (ax.lo[px] - ax.min);
        float sum = 0.f;
        for (int i = 0; i < ax.n[px]; ++i) sum += wx[i] * src[i];
        o[py * out + px] = sum;
      }
    }
  }
}

}  // namespace

void RoiAlign(const Feature& f, const Box* boxes, int n, int out, float scale, float* dst, ThreadPool* pool) {
  // items: (box, block of channels); several blocks per box keep the workers busy when there are few boxes
  const int blocks = pool && pool->size() > 1 && n < 2 * pool->size() ? std::min(f.c, 4 * pool->size() / std::max(n, 1)) : 1;
  const int per = (f.c + blocks - 1) / blocks;
  auto item = [&](int i) {
    thread_local RoiScratch scratch;
    const int k = i / blocks, c0 = (i % blocks) * per, c1 = std::min(c0 + per, f.c);
    if (c0 < c1) RoiAlignChannels(f, boxes[k], out, scale, c0, c1, dst + (static_cast<size_t>(k) * f.c + c0) * out * out, &scratch);
  };
  if (pool)
    pool->For(n * blocks, item);
  else
    for (int i = 0; i < n * blocks; ++i) item(i);
}

void PoolBox(const Feature (&p2_p5)[4], const Box* boxes, int n, int out, float* dst, ThreadPool* pool) {
  const size_t per_box = static_cast<size_t>(p2_p5[0].c) * out * out;
  auto item = [&](int k) {
    thread_local RoiScratch scratch;
    // ROIPooler level assignment: canonical 224 px box on p4, clamped to p2..p5
    const float size = std::sqrt(Area(boxes[k]));
    const int lv = static_cast<int>(std::min(std::max(std::floor(4 + std::log2(size / 224 + 1e-8f)), 2.f), 5.f)) - 2;
    RoiAlignChannels(p2_p5[lv], boxes[k], out, 1.f / kStride[lv], 0, p2_p5[lv].c, dst + k * per_box, &scratch);
  };
  if (pool)
    pool->For(n, item);
  else
    for (int k = 0; k < n; ++k) item(k);
}

std::vector<Detection> Detect(const float* probs, const float* deltas, const Box* proposals, int n, int img_h, int img_w) {
  std::vector<Detection> cand;
  for (int i = 0; i < n; ++i) {
    if (!(probs[i * 2] > kBoxScoreThresh)) continue;
    cand.push_back({Clip(Decode(deltas + i * 4, proposals[i], kBoxWeights), img_h, img_w), probs[i * 2]});
  }
  std::stable_sort(cand.begin(), cand.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Box> boxes;
  for (const auto& d : cand) boxes.push_back(d.box);
  std::vector<Detection> out;
  for (int i : NmsSorted(boxes, kBoxNms)) {
    if (static_cast<int>(out.size()) == kBoxTopk) break;
    out.push_back(cand[i]);
  }
  return out;
}

int PackPersonPixels(const float* emb, const float* seg, int k, int d, int r, std::vector<float>* px,
                     std::vector<uint8_t>* mask, std::vector<float>* margin) {
  const int rr = r * r;
  mask->assign(static_cast<size_t>(k) * rr, 0);
  if (margin) margin->resize(static_cast<size_t>(k) * rr);
  px->clear();
  int count = 0;
  for (int i = 0; i < k; ++i) {
    const float* bg = seg + static_cast<size_t>(i) * 2 * rr;
    const float* fg = bg + rr;
    const float* e = emb + static_cast<size_t>(i) * d * rr;
    for (int p = 0; p < rr; ++p) {
      if (margin) (*margin)[static_cast<size_t>(i) * rr + p] = fg[p] - bg[p];
      if (!(fg[p] > bg[p])) continue;  // argmax over the 2 channels is 1 only when fg is strictly larger
      (*mask)[static_cast<size_t>(i) * rr + p] = 1;
      for (int c = 0; c < d; ++c) px->push_back(e[c * rr + p]);
      ++count;
    }
  }
  return count;
}

}  // namespace fnl
