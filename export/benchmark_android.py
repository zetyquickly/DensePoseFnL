"""Latency of each exported .tflite graph on a connected Android device (TFLite benchmark_model over adb).

Sweeps CPU (XNNPACK) thread counts and, with --gpu, the GPU delegate. Random inputs, no host glue: this is the pure
graph time, so the A/B1/B2/C numbers add up to the model part of one frame.
usage: python benchmark_android.py configs/cse.yaml [--threads 1 4 8] [--gpu] [--op-profile] [--graphs graph_a ...]
"""
import argparse
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from fnl_export.config import ExportConfig

BINARY_URL = "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
REMOTE = "/data/local/tmp/fnl"
STATS = re.compile(r"count=(\d+) first=(\d+) curr=\d+ min=(\d+) max=(\d+) avg=([\d.]+) std=([\d.]+) p5=(\d+) median=(\d+) p95=(\d+)")


class Adb:
    def __init__(self, adb, serial):
        self.cmd = [adb] + (["-s", serial] if serial else [])

    def run(self, *args, timeout=600):
        return subprocess.run([*self.cmd, *args], capture_output=True, text=True, timeout=timeout)

    def shell(self, line, timeout=600):
        return self.run("shell", line, timeout=timeout)


def benchmark(adb, graph, flags, runs, warmup):
    r = adb.shell(f"cd {REMOTE} && ./benchmark_model --graph={graph}.tflite --num_runs={runs} --warmup_runs={warmup} "
                  f"--enable_op_profiling={'true' if 'op_profiling' in flags else 'false'} "
                  + " ".join(f"--{k}={v}" for k, v in flags.items() if k != "op_profiling") + " 2>&1")
    m = STATS.findall(r.stdout)
    if not m:
        return None, r.stdout[-1500:]
    count, first, mn, mx, avg, std, p5, med, p95 = m[-1]
    return {"count": int(count), "first_ms": int(first) / 1e3, "min_ms": int(mn) / 1e3, "avg_ms": float(avg) / 1e3,
            "std_ms": float(std) / 1e3, "median_ms": int(med) / 1e3, "p95_ms": int(p95) / 1e3}, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tflite", help="dir with the .tflite graphs (default: <out_dir>/litert/<name>)")
    ap.add_argument("--graphs", nargs="*", help="default: every .tflite in the dir")
    ap.add_argument("--threads", nargs="*", type=int, default=[1, 2, 4, 8], help="CPU thread counts to sweep")
    ap.add_argument("--gpu", action="store_true", help="also run the GPU delegate (fp32 and fp16)")
    ap.add_argument("--op-profile", action="store_true", help="print the per-op profile of the best CPU run of each graph")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cooldown", type=float, default=3.0, help="seconds between runs, against thermal throttling")
    ap.add_argument("--adb", default=shutil.which("adb") or "/opt/homebrew/bin/adb")
    ap.add_argument("--serial")
    ap.add_argument("--binary", help="android aarch64 benchmark_model (default: downloaded to <out_dir>/android_bin)")
    ap.add_argument("--out", help="json with all results (default: <out_dir>/android/<name>.json)")
    a = ap.parse_args()

    cfg = ExportConfig.load(a.config)
    out_dir = cfg.resolve(cfg.out_dir)
    src = Path(a.tflite) if a.tflite else out_dir / "litert" / cfg.name
    graphs = a.graphs or sorted(p.stem for p in src.glob("*.tflite"))
    binary = Path(a.binary) if a.binary else out_dir / "android_bin" / "benchmark_model"
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BINARY_URL}")
        urllib.request.urlretrieve(BINARY_URL, binary)

    adb = Adb(a.adb, a.serial)
    model = adb.shell("getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release").stdout.split()
    print(f"device: {' / '.join(model)}")
    adb.shell(f"mkdir -p {REMOTE}")
    r = adb.run("push", str(binary), *[str(src / f"{g}.tflite") for g in graphs], REMOTE)
    assert r.returncode == 0, r.stderr
    adb.shell(f"chmod +x {REMOTE}/benchmark_model")

    backends = [(f"cpu x{t}", {"num_threads": t, "use_xnnpack": "true"}) for t in a.threads]
    if a.gpu:
        backends += [("gpu fp32", {"use_gpu": "true", "gpu_precision_loss_allowed": "false"}),
                     ("gpu fp16", {"use_gpu": "true", "gpu_precision_loss_allowed": "true"})]

    failed = []
    results = {"device": model, "runs": a.runs, "graphs": {}}
    for g in graphs:
        results["graphs"][g] = {}
        for label, flags in backends:
            time.sleep(a.cooldown)
            res, log = benchmark(adb, g, flags, a.runs, a.warmup)
            if res is None:
                print(f"[FAIL] {g} {label}:\n{log}")
                failed.append(f"{g} {label}")
            results["graphs"][g][label] = res
            if res:
                print(f"{g:<10} {label:<9} median {res['median_ms']:7.2f} ms  avg {res['avg_ms']:7.2f}  p95 {res['p95_ms']:7.2f}  "
                      f"min {res['min_ms']:7.2f}  first {res['first_ms']:7.2f}")

    print("\nmedian ms" + "".join(f"{label:>10}" for label, _ in backends))
    for g in graphs:
        print(f"{g:<9} " + "".join(f"{(results['graphs'][g][label] or {}).get('median_ms', float('nan')):>10.2f}" for label, _ in backends))
    print("\nbest per graph (sum = model time of one frame, host glue excluded):")
    total = 0.0
    missing = []
    for g in graphs:
        ok = {k: v for k, v in results["graphs"][g].items() if v}
        if ok:
            best = min(ok, key=lambda k: ok[k]["median_ms"])
            total += ok[best]["median_ms"]
            print(f"  {g:<10} {ok[best]['median_ms']:7.2f} ms  ({best})")
        else:
            missing.append(g)
    print(f"  {'sum':<10} {total:7.2f} ms" + (f"  (incomplete: no result for {', '.join(missing)})" if missing else ""))

    if a.op_profile:
        for g in graphs:
            ok = {k: v for k, v in results["graphs"][g].items() if v and k.startswith("cpu")}
            if not ok:
                continue
            best = min(ok, key=lambda k: ok[k]["median_ms"])
            flags = dict(next(f for l, f in backends if l == best), op_profiling="true")
            _, log = benchmark(adb, g, flags, a.runs, a.warmup)
            print(f"\n=== op profile {g} ({best}) ===")
            print(log[log.find("Operator-wise Profiling") - 10:][:6000] if "Operator-wise Profiling" in log else log[-3000:])

    path = Path(a.out) if a.out else out_dir / "android" / f"{cfg.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
