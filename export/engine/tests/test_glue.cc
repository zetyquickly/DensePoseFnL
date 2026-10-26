// Checks fnl/glue.h against the numpy reference. Vectors come from tools/make_test_vectors.py.
// usage: test_glue VECTOR_DIR [VECTOR_DIR ...]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "fnl/glue.h"

namespace {

template <typename T>
std::vector<T> Load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
  return v;
}

int g_fail = 0;

void Report(const char* name, size_t n_ours, size_t n_ref, double max_diff, double tol) {
  const bool ok = n_ours == n_ref && max_diff <= tol;
  std::printf("    %-22s n %zu/%zu  max|diff| %.2e  %s\n", name, n_ours, n_ref, max_diff, ok ? "ok" : "FAIL");
  g_fail += !ok;
}

double MaxDiff(const float* a, const float* b, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  constexpr int H = 480, W = 640;
  const int dims[5][2] = {{120, 160}, {60, 80}, {30, 40}, {15, 20}, {8, 10}};
  const int fdims[4][2] = {{120, 160}, {60, 80}, {30, 40}, {15, 20}};
  for (int a = 1; a < argc; ++a) {
    const std::string d = std::string(argv[a]) + "/";
    std::printf("%s\n", argv[a]);
    std::vector<float> logits[5], deltas[5], feat[4];
    fnl::RpnLevel levels[5];
    for (int i = 0; i < 5; ++i) {
      logits[i] = Load<float>(d + "rpn_logits_" + std::to_string(i) + ".f32");
      deltas[i] = Load<float>(d + "rpn_deltas_" + std::to_string(i) + ".f32");
      levels[i] = {logits[i].data(), deltas[i].data(), dims[i][0], dims[i][1]};
    }
    fnl::Feature feats[4];
    for (int i = 0; i < 4; ++i) {
      feat[i] = Load<float>(d + "feat_p" + std::to_string(i + 2) + ".f32");
      feats[i] = {feat[i].data(), 64, fdims[i][0], fdims[i][1]};
    }

    const auto ref_props = Load<float>(d + "props.f32");
    const auto props = fnl::RpnProposals(levels, H, W, 100);
    Report("rpn_proposals", props.size() * 4, ref_props.size(),
           props.size() * 4 == ref_props.size() ? MaxDiff(&props[0].x1, ref_props.data(), ref_props.size()) : 1e9, 1e-2);

    // every 3rd p2 logit NaN (make_test_vectors.py makes the same): dropped, not a crash, and the rest ranks as in the reference
    if (std::ifstream(d + "props_nan.f32")) {
      const auto ref_nan = Load<float>(d + "props_nan.f32");
      std::vector<float> nan_logits = logits[0];
      for (size_t i = 0; i < nan_logits.size(); i += 3) nan_logits[i] = std::nanf("");
      fnl::RpnLevel nan_levels[5];
      std::copy(levels, levels + 5, nan_levels);
      nan_levels[0].logits = nan_logits.data();
      const auto nan_props = fnl::RpnProposals(nan_levels, H, W, 100);
      Report("rpn_proposals_nan", nan_props.size() * 4, ref_nan.size(),
             nan_props.size() * 4 == ref_nan.size() ? MaxDiff(&nan_props[0].x1, ref_nan.data(), ref_nan.size()) : 1e9, 1e-2);
    }

    const auto ref_pool = Load<float>(d + "pool_box.f32");
    std::vector<float> pool(props.size() * 64 * 7 * 7);
    fnl::PoolBox(feats, props.data(), static_cast<int>(props.size()), 7, pool.data());
    Report("pool_box", pool.size(), ref_pool.size(), pool.size() == ref_pool.size() ? MaxDiff(pool.data(), ref_pool.data(), pool.size()) : 1e9, 1e-3);

    const auto probs = Load<float>(d + "probs.f32"), bdel = Load<float>(d + "box_deltas.f32");
    const auto ref_dets = Load<float>(d + "dets.f32");
    const auto dets = fnl::Detect(probs.data(), bdel.data(), ref_props.empty() ? nullptr : reinterpret_cast<const fnl::Box*>(ref_props.data()),
                                  static_cast<int>(probs.size() / 2), H, W);
    double dd = dets.size() * 5 == ref_dets.size() ? 0 : 1e9;
    for (size_t i = 0; dd < 1e9 && i < dets.size(); ++i) {
      dd = std::max(dd, MaxDiff(&dets[i].box.x1, &ref_dets[i * 5], 4));
      dd = std::max(dd, static_cast<double>(std::fabs(dets[i].score - ref_dets[i * 5 + 4])));
    }
    Report("detect", dets.size() * 5, ref_dets.size(), dd, 1e-2);

    const auto ref_dp = Load<float>(d + "pool_dp.f32");
    const int n_dp = static_cast<int>(ref_dp.size() / (64 * 32 * 32));
    std::vector<float> dp(ref_dp.size());
    std::vector<fnl::Box> dp_boxes;
    for (int i = 0; i < n_dp; ++i) dp_boxes.push_back({ref_dets[i * 5], ref_dets[i * 5 + 1], ref_dets[i * 5 + 2], ref_dets[i * 5 + 3]});
    fnl::RoiAlign(feats[0], dp_boxes.data(), n_dp, 32, 0.25f, dp.data());
    Report("pool_dp", dp.size(), ref_dp.size(), MaxDiff(dp.data(), ref_dp.data(), dp.size()), 1e-3);

    std::ifstream has_emb(d + "emb.f32");
    if (has_emb) {
      const auto emb = Load<float>(d + "emb.f32"), seg = Load<float>(d + "seg.f32"), ref_px = Load<float>(d + "px.f32");
      const auto ref_mask = Load<uint8_t>(d + "mask.u8");
      const int k = static_cast<int>(emb.size() / (16 * 64 * 64));
      std::vector<float> px;
      std::vector<uint8_t> mask;
      fnl::PackPersonPixels(emb.data(), seg.data(), k, 16, 64, &px, &mask);
      Report("pack_person_pixels", px.size(), ref_px.size(), px == ref_px && mask == ref_mask ? 0 : 1e9, 0);
    }
  }
  std::printf(g_fail ? "FAILED (%d)\n" : "all ok\n", g_fail);
  return g_fail != 0;
}
