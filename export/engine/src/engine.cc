#include "fnl/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "fnl/thread_pool.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace fnl {
namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

std::string ShapeStr(const std::vector<int>& s) {
  std::string r = "(";
  for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
  return r + ")";
}

void ExpectShape(const char* what, const std::vector<int>& got, const std::vector<int>& want) {
  if (got != want)
    throw std::runtime_error(std::string(what) + " has shape " + ShapeStr(got) + ", the engine expects " + ShapeStr(want));
}

template <typename T>
class Queue {
 public:
  explicit Queue(size_t cap) : cap_(cap) {}
  bool Push(T&& v) {  // blocks while full; false if closed
    std::unique_lock<std::mutex> l(m_);
    not_full_.wait(l, [&] { return closed_ || q_.size() < cap_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    not_empty_.notify_one();
    return true;
  }
  bool Pop(T* v) {  // blocks while empty; false when closed and drained
    std::unique_lock<std::mutex> l(m_);
    not_empty_.wait(l, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) return false;
    *v = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }
  void Close() {
    std::lock_guard<std::mutex> l(m_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable not_empty_, not_full_;
  std::deque<T> q_;
  size_t cap_;
  bool closed_ = false;
};

// One .tflite graph on the GPU delegate or on XNNPACK. Must be created, run and destroyed on one thread (GPU contexts are
// thread bound).
class Graph {
 public:
  Graph(const std::string& path, bool gpu, int threads, bool fp16) {
    model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!model_) throw std::runtime_error("cannot load " + path);
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;  // delegates are applied explicitly below
    tflite::InterpreterBuilder(*model_, resolver)(&interp_);
    if (!interp_) throw std::runtime_error("cannot build interpreter for " + path);
    interp_->SetNumThreads(threads);
    if (interp_->AllocateTensors() != kTfLiteOk) throw std::runtime_error("AllocateTensors failed for " + path);
    if (gpu) {
      TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
      o.is_precision_loss_allowed = fp16 ? 1 : 0;
      o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;  // repeated invocations: ~1 ms faster on graph A
      delegate_ = {TfLiteGpuDelegateV2Create(&o), TfLiteGpuDelegateV2Delete};
    } else {
      TfLiteXNNPackDelegateOptions o = TfLiteXNNPackDelegateOptionsDefault();
      o.num_threads = threads;
      if (fp16) o.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
      delegate_ = {TfLiteXNNPackDelegateCreate(&o), TfLiteXNNPackDelegateDelete};
    }
    if (interp_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk)
      throw std::runtime_error(std::string(gpu ? "GPU" : "XNNPACK") + " delegate rejected " + path);
    for (int i : interp_->inputs()) std::memset(interp_->tensor(i)->data.raw, 0, interp_->tensor(i)->bytes);
  }

  float* In(int i = 0) { return interp_->typed_input_tensor<float>(i); }
  const float* Out(int i) const { return interp_->typed_output_tensor<float>(i); }
  const int32_t* OutInt(int i) const { return interp_->typed_output_tensor<int32_t>(i); }
  std::vector<int> InShape(int i = 0) const { return Shape(interp_->input_tensor(i)); }
  std::vector<int> OutShape(int i) const { return Shape(interp_->output_tensor(i)); }
  void Run() {
    if (interp_->Invoke() != kTfLiteOk) throw std::runtime_error("Invoke failed");
  }

 private:
  static std::vector<int> Shape(const TfLiteTensor* t) { return std::vector<int>(t->dims->data, t->dims->data + t->dims->size); }

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_{nullptr, [](TfLiteDelegate*) {}};
  std::unique_ptr<tflite::Interpreter> interp_;  // destroyed before the delegate
};

struct Crop {
  int x0, y0, w, h;  // centered region of the frame with the network's aspect ratio
  float scale;       // network pixels per frame pixel
};

Crop CenterCrop(int w, int h, int net_w, int net_h) {
  int cw = w, ch = h;
  if (static_cast<int64_t>(w) * net_h > static_cast<int64_t>(h) * net_w)
    cw = std::max(static_cast<int>(std::lround(h * static_cast<double>(net_w) / net_h)), 1);  // wider than the network: crop the sides
  else
    ch = std::max(static_cast<int>(std::lround(w * static_cast<double>(net_h) / net_w)), 1);  // taller: crop top and bottom
  return {(w - cw) / 2, (h - ch) / 2, cw, ch, net_w / static_cast<float>(cw)};
}

// RGB8 frame -> planar float image (3,net_h,net_w): the center crop, bilinearly resized (plain bilinear, no antialiasing:
// for a big downscale pre-scale the frames); a crop that already has the network size is copied.
Crop Preprocess(const uint8_t* rgb, int w, int h, int net_w, int net_h, float* dst) {
  const Crop crop = CenterCrop(w, h, net_w, net_h);
  const size_t plane = static_cast<size_t>(net_w) * net_h;
  const size_t stride = static_cast<size_t>(w) * 3;
  const uint8_t* origin = rgb + crop.y0 * stride + static_cast<size_t>(crop.x0) * 3;
  if (crop.w == net_w && crop.h == net_h) {
    for (int y = 0; y < net_h; ++y) {
      const uint8_t* p = origin + y * stride;
      for (int x = 0; x < net_w; ++x)
        for (int c = 0; c < 3; ++c) dst[c * plane + static_cast<size_t>(y) * net_w + x] = p[x * 3 + c];
    }
    return crop;
  }
  std::vector<int> x0(net_w), x1(net_w);
  std::vector<float> fx(net_w);
  for (int x = 0; x < net_w; ++x) {
    const float sx = std::max((x + 0.5f) * crop.w / net_w - 0.5f, 0.f);
    x0[x] = std::min(static_cast<int>(sx), crop.w - 1);
    x1[x] = std::min(x0[x] + 1, crop.w - 1);
    fx[x] = sx - x0[x];
  }
  for (int y = 0; y < net_h; ++y) {
    const float sy = std::max((y + 0.5f) * crop.h / net_h - 0.5f, 0.f);
    const int y0 = std::min(static_cast<int>(sy), crop.h - 1), y1 = std::min(y0 + 1, crop.h - 1);
    const float fy = sy - y0;
    const uint8_t* r0 = origin + y0 * stride;
    const uint8_t* r1 = origin + y1 * stride;
    for (int x = 0; x < net_w; ++x)
      for (int c = 0; c < 3; ++c) {
        const float top = r0[x0[x] * 3 + c] * (1 - fx[x]) + r0[x1[x] * 3 + c] * fx[x];
        const float bot = r1[x0[x] * 3 + c] * (1 - fx[x]) + r1[x1[x] * 3 + c] * fx[x];
        dst[c * plane + static_cast<size_t>(y) * net_w + x] = top * (1 - fy) + bot * fy;
      }
  }
  return crop;
}

struct InFrame {
  int64_t id = 0;
  std::vector<uint8_t> rgb;
  int w = 0, h = 0;
  Clock::time_point submitted;
};

// Stage 1 -> stage 2: detections and the DensePose head outputs of the detected people.
struct MidFrame {
  int64_t id = 0;
  std::vector<Detection> dets;
  std::vector<float> emb, seg;  // (n,16,R,R), (n,2,R,R)
  Crop crop{};
  StageTimes t;
  Clock::time_point submitted;
};

}  // namespace

struct Engine::Impl {
  explicit Impl(const EngineOptions& o) : opts(o), in_q(o.queue_depth), mid_q(o.queue_depth), out_q(1 << 20) {
    std::promise<void> ready1, ready2;
    auto f1 = ready1.get_future(), f2 = ready2.get_future();
    // each thread owns its promise: it outlives this constructor
    t1 = std::thread([this, r = std::move(ready1)]() mutable { Stage1(std::move(r)); });
    t2 = std::thread([this, r = std::move(ready2)]() mutable { Stage2(std::move(r)); });
    try {
      f1.get();
      f2.get();
    } catch (...) {
      Shutdown();
      throw;
    }
  }

  ~Impl() { Shutdown(); }

  void Shutdown() {
    in_q.Close();
    mid_q.Close();
    out_q.Close();
    if (t1.joinable()) t1.join();
    if (t2.joinable()) t2.join();
  }

  void Fail(std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> l(err_m);
      if (!err) err = e;
    }
    in_q.Close();
    mid_q.Close();
    out_q.Close();
  }

  void Stage1(std::promise<void> ready) {
    bool started = false;
    try {
      const std::string dir = opts.model_dir + "/";
      const int cpu = opts.cpu_threads;
      Graph a(dir + "graph_a.tflite", opts.gpu, cpu, true), b1(dir + "graph_b1.tflite", opts.gpu, cpu, true),
          b2(dir + "graph_b2.tflite", opts.gpu, cpu, true);
      ThreadPool pool(opts.glue_threads);
      const auto in = a.InShape();
      net_h = in[2], net_w = in[3];
      k1 = b1.InShape()[0], k2 = b2.InShape()[0];
      res = b2.OutShape(0)[2];
      emb_dim = b2.OutShape(0)[1];
      // the glue pools 7x7 for graph B1 and 32x32 for graph B2; a model set exported with other pool sizes would read wrong data
      const int ch = a.OutShape(0)[1];
      ExpectShape("graph_b1 input", b1.InShape(), {k1, ch, 7, 7});
      ExpectShape("graph_b2 input", b2.InShape(), {k2, ch, 32, 32});
      ready.set_value();
      started = true;

      InFrame f;
      while (in_q.Pop(&f)) {
        MidFrame m;
        m.id = f.id, m.submitted = f.submitted;
        auto t0 = Clock::now(), t = t0;
        auto lap = [&](double* ms) {
          auto n = Clock::now();
          *ms = Ms(t, n);
          t = n;
        };
        m.crop = Preprocess(f.rgb.data(), f.w, f.h, net_w, net_h, a.In());
        lap(&m.t.preprocess);
        a.Run();
        lap(&m.t.a);

        fnl::RpnLevel levels[kLevels];
        for (int i = 0; i < kLevels; ++i) {
          const auto ls = a.OutShape(4 + i);
          levels[i] = {a.Out(4 + i), a.Out(9 + i), ls[2], ls[3]};
        }
        std::vector<Box> props = RpnProposals(levels, net_h, net_w, k1);
        lap(&m.t.rpn);

        Feature feats[4];
        for (int i = 0; i < 4; ++i) {
          const auto s = a.OutShape(i);
          feats[i] = {a.Out(i), s[1], s[2], s[3]};
        }
        const int np = static_cast<int>(props.size());
        PoolBox(feats, props.data(), np, 7, b1.In(), &pool);
        lap(&m.t.pool_b1);
        b1.Run();
        lap(&m.t.b1);

        std::vector<Detection> dets = Detect(b1.Out(0), b1.Out(1), props.data(), np, net_h, net_w);
        for (const auto& d : dets)
          if (d.score >= opts.score_thresh && static_cast<int>(m.dets.size()) < k2) m.dets.push_back(d);
        lap(&m.t.detect);

        const int nd = static_cast<int>(m.dets.size());
        if (nd > 0) {
          std::vector<Box> boxes;
          for (const auto& d : m.dets) boxes.push_back(d.box);
          RoiAlign(feats[0], boxes.data(), nd, 32, 1.f / kStride[0], b2.In(), &pool);
          lap(&m.t.pool_dp);
          b2.Run();
          const size_t rr = static_cast<size_t>(res) * res;
          m.emb.assign(b2.Out(0), b2.Out(0) + nd * emb_dim * rr);
          m.seg.assign(b2.Out(1), b2.Out(1) + nd * 2 * rr);
          lap(&m.t.b2);
        }
        m.t.stage1 = Ms(t0, Clock::now());
        if (!mid_q.Push(std::move(m))) break;
      }
    } catch (...) {
      const auto e = std::current_exception();
      if (!started) ready.set_exception(e);  // the constructor is waiting on it
      else Fail(e);
    }
    mid_q.Close();
  }

  void Stage2(std::promise<void> ready) {
    bool started = false;
    try {
      Graph c(opts.model_dir + "/graph_c.tflite", false, opts.cpu_threads, true);
      if (c.InShape().size() != 2)
        throw std::runtime_error("graph_c input has shape " + ShapeStr(c.InShape()) +
                                 ", the engine expects the optimized graph C (N,16): export with cse_match: optimized");
      const int chunk = c.InShape()[0];
      ready.set_value();
      started = true;

      MidFrame m;
      std::vector<float> px;
      std::vector<uint8_t> mask;
      std::vector<float> margin;
      std::vector<int32_t> ids;
      while (mid_q.Pop(&m)) {
        FrameResult r;
        r.id = m.id, r.t = m.t, r.scale = m.crop.scale, r.crop_x = m.crop.x0, r.crop_y = m.crop.y0, r.res = res, r.net_w = net_w, r.net_h = net_h;
        const int nd = static_cast<int>(m.dets.size());
        const size_t rr = static_cast<size_t>(res) * res;
        auto t0 = Clock::now();
        std::vector<int32_t> vertex(nd * rr, -1);
        if (nd > 0) {
          const int n = PackPersonPixels(m.emb.data(), m.seg.data(), nd, emb_dim, res, &px, &mask, &margin);
          r.person_pixels = n;
          auto t1 = Clock::now();
          r.t.pack = Ms(t0, t1);
          ids.resize(n);
          for (int off = 0; off < n; off += chunk) {
            const int cnt = std::min(chunk, n - off);
            std::memcpy(c.In(), px.data() + static_cast<size_t>(off) * emb_dim, sizeof(float) * cnt * emb_dim);
            if (cnt < chunk) std::memset(c.In() + static_cast<size_t>(cnt) * emb_dim, 0, sizeof(float) * (chunk - cnt) * emb_dim);
            c.Run();
            std::memcpy(ids.data() + off, c.OutInt(0), sizeof(int32_t) * cnt);
          }
          int j = 0;
          for (size_t p = 0; p < mask.size(); ++p)
            if (mask[p]) vertex[p] = ids[j++];
          r.t.c = Ms(t1, Clock::now());
        }
        for (int i = 0; i < nd; ++i) {
          Person p;
          p.box = m.dets[i].box, p.score = m.dets[i].score;
          p.margin.assign(margin.begin() + i * rr, margin.begin() + (i + 1) * rr);
          p.vertex.assign(vertex.begin() + i * rr, vertex.begin() + (i + 1) * rr);
          r.people.push_back(std::move(p));
        }
        auto done = Clock::now();
        r.t.stage2 = Ms(t0, done);
        r.t.latency = Ms(m.submitted, done);
        if (!out_q.Push(std::move(r))) break;
      }
    } catch (...) {
      const auto e = std::current_exception();
      if (!started) ready.set_exception(e);  // the constructor is waiting on it
      else Fail(e);
    }
    out_q.Close();
  }

  EngineOptions opts;
  Queue<InFrame> in_q;
  Queue<MidFrame> mid_q;
  Queue<FrameResult> out_q;
  std::thread t1, t2;
  std::mutex err_m;
  std::exception_ptr err;
  int net_w = 0, net_h = 0, k1 = 0, k2 = 0, res = 0, emb_dim = 0;
};

Engine::Engine(const EngineOptions& opts) : impl_(new Impl(opts)) {}
Engine::~Engine() = default;

void Engine::Submit(int64_t id, const uint8_t* rgb, int width, int height) {
  InFrame f;
  f.id = id, f.w = width, f.h = height, f.submitted = Clock::now();
  f.rgb.assign(rgb, rgb + static_cast<size_t>(width) * height * 3);
  impl_->in_q.Push(std::move(f));
}

bool Engine::Next(FrameResult* out) {
  if (impl_->out_q.Pop(out)) return true;
  std::lock_guard<std::mutex> l(impl_->err_m);
  if (impl_->err) std::rethrow_exception(impl_->err);
  return false;
}

void Engine::Close() { impl_->in_q.Close(); }

int Engine::net_width() const { return impl_->net_w; }
int Engine::net_height() const { return impl_->net_h; }

}  // namespace fnl
