"""Render the results of `fnl_run --save` over the frames it ran on: vertex ids coloured by the PCA of the vertex embeddings
(neighbouring body surface = similar colour), person boxes with scores. Writes an mp4. Same drawing as the app's Composite():
person where the bilinearly upsampled fg - bg margin is positive, colour = bilinear blend of the neighbouring cells' colours.

usage: python engine/tools/render_overlay.py RESULTS.fnl FRAMES.rgb WIDTH HEIGHT VERTEX_EMBEDDINGS.npy OUT.mp4 [--fps 25]
"""
import argparse
import struct
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def vertex_colors(V):
    """PCA of the (smooth) vertex embeddings -> RGB, so neighbouring body surface gets similar colours."""
    x = V - V.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:3].T
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    return (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def read_results(path):
    """-> (net_w, net_h, res, scale, crop_x, crop_y), iterator of (frame id, [(box, score, margin (res,res) float32, vertex (res,res) int32)])"""
    f = open(path, "rb")
    assert f.read(4) == b"FNL2", "not a fnl_run results file"
    net_w, net_h, res = struct.unpack("<3I", f.read(12))
    scale, crop_x, crop_y = struct.unpack("<3f", f.read(12))

    def frames():
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            fid, n = struct.unpack("<qI", head)
            people = []
            for _ in range(n):
                box = np.frombuffer(f.read(16), np.float32)
                (score,) = struct.unpack("<f", f.read(4))
                margin = np.frombuffer(f.read(4 * res * res), np.float32).reshape(res, res)
                vertex = np.frombuffer(f.read(4 * res * res), np.int32).reshape(res, res)
                people.append((box, score, margin, vertex))
            yield fid, people

    return (net_w, net_h, res, scale, crop_x, crop_y), frames()


def draw_person(canvas, box, margin, vertex, colors, alpha):
    """Blend one person into canvas (h,w,3 float, in place); box in frame pixels, margin / vertex the (res,res) maps."""
    res = margin.shape[0]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 1 or bh < 1:
        return
    px0, px1 = max(int(np.floor(x1)), 0), min(int(np.ceil(x2)), canvas.shape[1])
    py0, py1 = max(int(np.floor(y1)), 0), min(int(np.ceil(y2)), canvas.shape[0])
    if px1 <= px0 or py1 <= py0:
        return

    def axis(lo, size, a0, a1):  # frame pixels a0..a1 -> (cell index, next cell index, weight of the next)
        t = np.clip((np.arange(a0, a1) + 0.5 - lo) / size * res - 0.5, 0, res - 1)
        i0 = np.minimum(t.astype(int), res - 1)
        return i0, np.minimum(i0 + 1, res - 1), (t - i0).astype(np.float32)

    c0, c1, fx = axis(x1, bw, px0, px1)
    r0, r1, fy = axis(y1, bh, py0, py1)
    fx, fy = fx[None, :], fy[:, None]
    w = ((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx)
    cells = ((r0[:, None], c0[None, :]), (r0[:, None], c1[None, :]), (r1[:, None], c0[None, :]), (r1[:, None], c1[None, :]))
    person = sum(wi * margin[ix] for wi, ix in zip(w, cells)) > 0
    ok = (vertex >= 0) & (vertex < len(colors))
    rgb = colors[np.where(ok, vertex, 0)].astype(np.float32)          # (res,res,3); cells without a vertex id carry no weight
    k = [wi * ok[ix] for wi, ix in zip(w, cells)]
    ks = sum(k)
    person &= ks > 1e-6
    col = sum(ki[..., None] * rgb[ix] for ki, ix in zip(k, cells)) / np.maximum(ks, 1e-6)[..., None]
    region = canvas[py0:py1, px0:px1]
    region[person] = (1 - alpha) * region[person] + alpha * col[person]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("frames")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("vertex_embeddings")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=25)
    a = ap.parse_args()

    colors = vertex_colors(np.load(a.vertex_embeddings))
    (_, _, _, scale, crop_x, crop_y), frames = read_results(a.results)
    size = a.width * a.height * 3
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
                            "-r", str(a.fps), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", a.out],
                           stdin=subprocess.PIPE)
    with open(a.frames, "rb") as src:
        for fid, people in frames:
            src.seek(fid * size)
            canvas = np.frombuffer(src.read(size), np.uint8).reshape(a.height, a.width, 3).astype(np.float32)
            boxes = []
            for box, score, margin, vertex in people:
                fb = box / scale + np.array([crop_x, crop_y, crop_x, crop_y], np.float32)
                draw_person(canvas, fb, margin, vertex, colors, 0.65)
                boxes.append((fb, score))
            img = Image.fromarray(canvas.astype(np.uint8))
            d = ImageDraw.Draw(img)
            for fb, score in boxes:
                d.rectangle([*fb], outline=(255, 255, 0), width=2)
                d.text((fb[0] + 3, fb[1] + 3), f"{score:.2f}", fill=(255, 255, 0))
            d.text((6, 4), f"frame {fid}", fill=(255, 255, 255))
            enc.stdin.write(np.asarray(img).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
