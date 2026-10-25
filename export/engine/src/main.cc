// fnl_run: run the whole CSE pipeline over a raw RGB8 frame stream (e.g. `ffmpeg ... -pix_fmt rgb24 -f rawvideo`).
//
//   fnl_run --models DIR --frames clip.rgb --width 640 --height 360 [--limit N] [--fps F] [--cpu] [--threads T]
//           [--thresh S] [--warmup N] [--save out.fnl] [--glue-threads G]
//
// --fps 0 (default) feeds frames back to back (throughput); --fps 25 paces them like a camera (latency).
// --save writes the results for tools/render_overlay.py: "FNL2", u32 net_w, net_h, res, f32 scale, crop_x, crop_y, then per frame
// i64 id, u32 n, and per person f32 box[4], f32 score, f32 margin[res*res] (fg - bg, > 0 = person), i32 vertex[res*res].
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "fnl/engine.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Stat {
  std::vector<double> v;
  void Print(const char* name) const {
    if (v.empty()) return;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double x : s) sum += x;
    std::printf("  %-12s mean %7.2f  p50 %7.2f  p95 %7.2f  max %7.2f ms\n", name, sum / s.size(), s[s.size() / 2],
                s[static_cast<size_t>(s.size() * 0.95)], s.back());
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) continue;
    k = k.substr(2);
    args[k] = (k == "cpu" || i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) ? "1" : argv[++i];
  }
  auto get = [&](const char* k, const char* d) { return args.count(k) ? args[k] : std::string(d); };
  if (!args.count("models") || !args.count("frames") || !args.count("width") || !args.count("height")) {
    std::fprintf(stderr, "usage: fnl_run --models DIR --frames F.rgb --width W --height H [--limit N] [--fps F] [--cpu] "
                         "[--threads T] [--glue-threads G] [--thresh S] [--warmup N] [--save out.fnl]\n");
    return 2;
  }
  const int w = std::stoi(get("width", "0")), h = std::stoi(get("height", "0"));
  const int limit = std::stoi(get("limit", "0")), warmup = std::stoi(get("warmup", "10"));
  const double fps = std::stod(get("fps", "0"));

  fnl::EngineOptions opts;
  opts.model_dir = get("models", "");
  opts.gpu = !args.count("cpu");
  opts.cpu_threads = std::stoi(get("threads", "4"));
  opts.glue_threads = std::stoi(get("glue-threads", "4"));
  opts.score_thresh = std::stof(get("thresh", "0.5"));

  std::FILE* in = std::fopen(get("frames", "").c_str(), "rb");
  if (!in) {
    std::perror("frames");
    return 1;
  }
  std::FILE* save = args.count("save") ? std::fopen(args["save"].c_str(), "wb") : nullptr;

  try {
    auto t_init = Clock::now();
    fnl::Engine engine(opts);
    std::printf("engine ready in %.0f ms: network %dx%d, %s, graph C on %d CPU threads\n",
                std::chrono::duration<double, std::milli>(Clock::now() - t_init).count(), engine.net_width(),
                engine.net_height(), opts.gpu ? "A/B1/B2 on GPU" : "A/B1/B2 on CPU", opts.cpu_threads);

    std::map<std::string, Stat> stats;
    int frames = 0;
    double person_px = 0;
    Clock::time_point first_done, last_done;
    std::thread consumer([&] {
      fnl::FrameResult r;
      bool header = false;
      while (engine.Next(&r)) {
        last_done = Clock::now();
        if (frames == warmup) first_done = last_done;
        if (save) {
          if (!header) {
            std::fwrite("FNL2", 1, 4, save);
            const uint32_t d[3] = {static_cast<uint32_t>(r.net_w), static_cast<uint32_t>(r.net_h), static_cast<uint32_t>(r.res)};
            std::fwrite(d, 4, 3, save);
            const float crop[3] = {r.scale, static_cast<float>(r.crop_x), static_cast<float>(r.crop_y)};
            std::fwrite(crop, 4, 3, save);
            header = true;
          }
          const uint32_t n = static_cast<uint32_t>(r.people.size());
          std::fwrite(&r.id, 8, 1, save);
          std::fwrite(&n, 4, 1, save);
          for (const auto& p : r.people) {
            std::fwrite(&p.box, 4, 4, save);
            std::fwrite(&p.score, 4, 1, save);
            std::fwrite(p.margin.data(), 4, p.margin.size(), save);
            std::fwrite(p.vertex.data(), 4, p.vertex.size(), save);
          }
        }
        if (frames >= warmup) {
          const auto& t = r.t;
          const std::pair<const char*, double> rows[] = {
              {"preprocess", t.preprocess}, {"A", t.a}, {"rpn glue", t.rpn}, {"pool B1", t.pool_b1}, {"B1", t.b1},
              {"detect glue", t.detect}, {"pool DP", t.pool_dp}, {"B2", t.b2}, {"pack", t.pack}, {"C", t.c},
              {"stage 1", t.stage1}, {"stage 2", t.stage2}, {"latency", t.latency}};
          for (const auto& row : rows) stats[row.first].v.push_back(row.second);
          person_px += r.person_pixels;
        }
        if (++frames % 25 == 0)
          std::printf("frame %5d: %zu people, %5d px, stage1 %.1f ms, stage2 %.1f ms, latency %.1f ms\n", frames, r.people.size(),
                      r.person_pixels, r.t.stage1, r.t.stage2, r.t.latency);
      }
    });

    std::vector<uint8_t> buf(static_cast<size_t>(w) * h * 3);
    auto next_due = Clock::now();
    int64_t id = 0;
    while ((limit == 0 || id < limit) && std::fread(buf.data(), 1, buf.size(), in) == buf.size()) {
      if (fps > 0) {
        next_due += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));
        std::this_thread::sleep_until(next_due);
      }
      engine.Submit(id++, buf.data(), w, h);
    }
    engine.Close();
    consumer.join();

    const int timed = frames - warmup;
    std::printf("\n%d frames (%d timed after %d warm-up), %.0f person pixels/frame\n", frames, timed, warmup,
                timed > 0 ? person_px / timed : 0.0);
    if (timed > 1) {
      const double secs = std::chrono::duration<double>(last_done - first_done).count();
      std::printf("throughput %.1f fps (%.1f ms/frame)%s\n", (timed - 1) / secs, 1000 * secs / (timed - 1),
                  fps > 0 ? "" : "  [back to back: latency includes waiting for the pipeline]");
    }
    for (const char* k : {"preprocess", "A", "rpn glue", "pool B1", "B1", "detect glue", "pool DP", "B2", "pack", "C", "stage 1",
                          "stage 2", "latency"})
      stats[k].Print(k);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  if (save) std::fclose(save);
  std::fclose(in);
  return 0;
}
