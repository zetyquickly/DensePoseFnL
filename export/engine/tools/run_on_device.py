"""Push fnl_run, the graphs and a raw frame clip to the phone over adb, run the pipeline, pull the results back.

usage: python engine/tools/run_on_device.py configs/cse.yaml --binary BUILD/fnl_run --frames clip.rgb --width 640 --height 360
           [--limit N] [--fps 25] [--cpu] [--overlay out.mp4]  [extra fnl_run flags after --]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fnl_export.config import ExportConfig  # noqa: E402

REMOTE = "/data/local/tmp/fnl_engine"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--binary", required=True, help="android aarch64 fnl_run")
    ap.add_argument("--frames", required=True, help="raw rgb24 frames")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--overlay", help="render the results over the frames into this mp4")
    ap.add_argument("--adb", default=shutil.which("adb") or "/opt/homebrew/bin/adb")
    ap.add_argument("--serial")
    ap.add_argument("extra", nargs="*", help="more fnl_run flags")
    a = ap.parse_args()

    cfg = ExportConfig.load(a.config)
    graphs = cfg.resolve(cfg.out_dir) / "litert" / cfg.name
    adb = [a.adb] + (["-s", a.serial] if a.serial else [])
    sh = lambda line: subprocess.run([*adb, "shell", line], text=True)
    push = lambda src, dst: subprocess.run([*adb, "push", str(src), dst], check=True, capture_output=True)

    sh(f"mkdir -p {REMOTE}/models")
    push(a.binary, f"{REMOTE}/fnl_run")
    for g in ("graph_a", "graph_b1", "graph_b2", "graph_c"):
        push(graphs / f"{g}.tflite", f"{REMOTE}/models/{g}.tflite")
    sh(f"chmod +x {REMOTE}/fnl_run")
    remote = f"{REMOTE}/{Path(a.frames).name}"
    size = subprocess.run([*adb, "shell", f"stat -c %s {remote}"], text=True, capture_output=True).stdout.strip()
    if size != str(Path(a.frames).stat().st_size):
        print("pushing frames ...")
        push(a.frames, remote)

    flags = f"--models {REMOTE}/models --frames {remote} --width {a.width} --height {a.height} --save {REMOTE}/out.fnl"
    flags += f" --limit {a.limit} --fps {a.fps}" + (" --cpu" if a.cpu else "") + " " + " ".join(a.extra)
    sh(f"cd {REMOTE} && ./fnl_run {flags}")

    out = cfg.resolve(cfg.out_dir) / "engine" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([*adb, "pull", f"{REMOTE}/out.fnl", str(out / "results.fnl")], check=True, capture_output=True)
    print(f"results: {out / 'results.fnl'}")
    if a.overlay:
        emb = cfg.resolve(cfg.out_dir) / "litert" / cfg.name / "vertex_embeddings.npy"
        subprocess.run([sys.executable, str(Path(__file__).parent / "render_overlay.py"), str(out / "results.fnl"), a.frames,
                        str(a.width), str(a.height), str(emb), a.overlay, "--fps", "25"], check=True)


if __name__ == "__main__":
    main()
