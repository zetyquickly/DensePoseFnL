"""Compare detectron2 (eager, dumped by reference/dump_detectron2.py) against our graphs run eagerly in PyTorch and
against the exported .tflite graphs.

Images are preprocessed with the export config (fnl_export/preproc.py); that is the input of every runner, and it is
checked against the input detectron2 saw. Each graph is fed the *original's* input at that point, so errors don't
accumulate across stages (the sheets chain B2 -> C instead, i.e. what the app would produce).
usage: python compare_with_detectron2.py configs/cse.yaml out/reference/*.npz --images DIR [--tflite DIR] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageDraw

from fnl_export.checkpoint import load_model
from fnl_export.config import ExportConfig
from fnl_export.graphs import GraphCOptimized, build_graphs
from fnl_export.modules import CHART_OUTPUTS
from fnl_export.preproc import preprocess


class Eager:
    """Our graphs as PyTorch modules (dynamic batch)."""
    name = "eager graphs"

    def __init__(self, model, graphs):
        self.model, self.g = model, graphs

    def _run(self, name, *xs):
        with torch.no_grad():
            out = self.g[name](*[torch.from_numpy(x) for x in xs])
        return [o.numpy() for o in out] if isinstance(out, tuple) else out.numpy()

    def a(self, x):
        return self._run("graph_a", x)

    def b1(self, x):
        return self._run("graph_b1", x)

    def b2(self, x):
        return self._run("graph_b2", x)

    def c(self, e):
        if not isinstance(self.g["graph_c"], GraphCOptimized):
            return self._run("graph_c", e).astype(np.int64)
        k, d, h, w = e.shape
        pix = np.ascontiguousarray(e.transpose(0, 2, 3, 1).reshape(-1, d))
        return np.concatenate([self._run("graph_c", pix[i:i + 16384]) for i in range(0, len(pix), 16384)]).astype(np.int64).reshape(k, h, w)

    def p6(self, x):
        with torch.no_grad():
            return self.model.backbone((torch.from_numpy(x) - self.g["graph_a"].mean) / self.g["graph_a"].std)[4].numpy()


class Tflite:
    """The exported .tflite files. Static shapes: ROI batches are chunked and zero-padded to the exported K."""
    name = "tflite"

    def __init__(self, directory, names):
        self.it = {}
        for n in names:
            path = Path(directory) / f"{n}.tflite"
            assert path.exists(), f"{path} not found: run export_litert.py first"
            self.it[n] = Interpreter(model_path=str(path))
            self.it[n].allocate_tensors()

    def _run(self, name, x):
        it = self.it[name]
        (inp,), outs = it.get_input_details(), it.get_output_details()
        assert tuple(inp["shape"][1:]) == x.shape[1:], f"{name}: exported for {tuple(inp['shape'])}, got {x.shape}"
        it.set_tensor(inp["index"], x.astype(np.float32))
        it.invoke()
        return [it.get_tensor(o["index"]) for o in outs]

    def _chunked(self, name, x):
        k = self.it[name].get_input_details()[0]["shape"][0]
        parts = []
        for i in range(0, len(x), k):
            chunk = x[i:i + k]
            pad = np.zeros((k - len(chunk), *chunk.shape[1:]), np.float32)
            parts.append([o[:len(chunk)] for o in self._run(name, np.concatenate([chunk, pad]))])
        return [np.concatenate(o) for o in zip(*parts)]

    def a(self, x):
        return self._run("graph_a", x)

    def b1(self, x):
        return self._chunked("graph_b1", x)

    def b2(self, x):
        return self._chunked("graph_b2", x)

    def c(self, e):
        if len(self.it["graph_c"].get_input_details()[0]["shape"]) == 4:
            return self._chunked("graph_c", e)[0].astype(np.int64)
        k, d, h, w = e.shape
        return self._chunked("graph_c", e.transpose(0, 2, 3, 1).reshape(-1, d))[0].astype(np.int64).reshape(k, h, w)


def stat(name, ours, ref):
    ours, ref = np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    assert ours.shape == ref.shape, f"{name}: shape {ours.shape} vs {ref.shape}"
    d = np.abs(ours - ref)
    scale = np.abs(ref).max() + 1e-12
    print(f"    {name:<22} shape {str(tuple(ref.shape)):<22} max|diff| {d.max():.2e}  mean|diff| {d.mean():.2e}  "
          f"max|ref| {scale:.2e}  rel {d.max() / scale:.1e}")
    return d.max() / scale


def vertex_colors(V):
    """PCA of the (smooth) vertex embeddings -> RGB, so neighbouring body surface gets similar colours."""
    x = V - V.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:3].T
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    return (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def paste_back(box, seg, ids64):
    """Resize one ROI's 64x64 outputs to its box (as densepose does): person mask + vertex ids at box resolution."""
    w, h = max(int(box[2] - box[0]), 1), max(int(box[3] - box[1]), 1)
    mask = F.interpolate(torch.from_numpy(seg)[None], size=(h, w), mode="bilinear")[0].argmax(0) > 0
    ids = F.interpolate(torch.from_numpy(ids64)[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
    return mask.numpy(), ids.numpy()


def blend(canvas, box, mask, rgb, alpha):
    x0, y0 = int(box[0]), int(box[1])
    ys, xs = np.nonzero(mask)
    Y, X = ys + y0, xs + x0
    ok = (Y >= 0) & (Y < canvas.shape[0]) & (X >= 0) & (X < canvas.shape[1])
    canvas[Y[ok], X[ok]] = (1 - alpha) * canvas[Y[ok], X[ok]] + alpha * rgb[ok]


def draw_boxes(img, boxes, scores):
    d = ImageDraw.Draw(img)
    for b, sc in zip(boxes, scores):
        d.rectangle([*b], outline=(255, 255, 0), width=2)
        d.text((b[0] + 3, b[1] + 3), f"{sc:.2f}", fill=(255, 255, 0))
    return img


def assemble(panels, titles):
    w, h = panels[0].size
    sheet = Image.new("RGB", (w * len(panels), h + 18), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for k, (pn, t) in enumerate(zip(panels, titles)):
        sheet.paste(pn, (k * w, 18))
        dr.text((k * w + 6, 3), t, fill=(255, 255, 255))
    return sheet


def densepose_sheet(img, boxes, scores, ref, results, paint, differs, metric=None):
    """detectron2 | one panel per runner | diff of the last runner (red = different vertex/part, blue = different person mask).

    ref: [(mask, payload)] per detection; results: [(label, [(mask, payload)])]. paint(mask, payload) -> (n,3) colours of the
    mask pixels; differs(payload_ref, payload_ours) -> bool array; metric(ref, ours) -> extra text for the panel title.
    """
    base = np.asarray(img, dtype=np.float32)
    ref_canvas = base.copy()
    for box, (m, p) in zip(boxes, ref):
        blend(ref_canvas, box, m, paint(m, p), 0.65)
    canvases, titles = [ref_canvas], ["detectron2 (original eager)"]
    n_px = sum(int(m.sum()) for m, _ in ref)
    for label, items in results:
        canvas, diff = base.copy(), base * 0.45
        n_lab = n_mask = 0
        for box, (m_ref, p_ref), (m_our, p_our) in zip(boxes, ref, items):
            blend(canvas, box, m_our, paint(m_our, p_our), 0.65)
            d_lab, d_mask = (m_ref & m_our) & differs(p_ref, p_our), m_ref != m_our
            n_lab, n_mask = n_lab + int(d_lab.sum()), n_mask + int(d_mask.sum())
            blend(diff, box, d_lab, np.tile([255.0, 0, 0], (int(d_lab.sum()), 1)), 1.0)
            blend(diff, box, d_mask, np.tile([0.0, 90, 255], (int(d_mask.sum()), 1)), 1.0)
        canvases.append(canvas)
        titles.append(f"{label}: {n_lab} label px, {n_mask} mask px differ (of {n_px})" + (metric(ref, items) if metric else ""))
    canvases.append(diff)
    titles.append(f"diff {results[-1][0]} vs detectron2 (red=vertex/part, blue=mask)")
    return assemble([draw_boxes(Image.fromarray(x.astype(np.uint8)), boxes, scores) for x in canvases], titles)


def chart_final(outs, eps):
    """Raw graph B2 outputs -> (coarse, fine, u, v) as detectron2 hands them to its converter (segm confidences applied)."""
    coarse, fine, u, v = outs[:4]
    if len(outs) > 4:
        softplus = lambda t: F.softplus(torch.from_numpy(t)).numpy() + eps
        fine, coarse = fine * softplus(outs[7]), coarse * softplus(outs[8])
    return coarse, fine, u, v


def chart_decode(box, coarse, fine, u, v):
    """detectron2's chart converter for one ROI: resample to the box, person mask = coarse argmax, part = fine argmax, UV of that part."""
    w, h = max(int(box[2] - box[0]), 1), max(int(box[3] - box[1]), 1)
    rs = lambda a: F.interpolate(torch.from_numpy(a)[None], size=(h, w), mode="bilinear", align_corners=False)
    mask = rs(coarse).argmax(1)[0] > 0
    labels = rs(fine).argmax(1)[0] * mask
    pick = lambda a: rs(a).gather(1, labels[None, None])[0, 0].numpy()
    return mask.numpy(), (labels.numpy(), pick(u), pick(v))


def part_paint(mask, payload):
    """Part -> hue, U -> brightness, V -> saturation."""
    labels, u, v = (a[mask] for a in payload)
    hue = ((labels - 1) % 24) / 24.0
    s, val = 0.55 + 0.45 * np.clip(v, 0, 1), 0.55 + 0.45 * np.clip(u, 0, 1)
    k = np.floor(hue * 6).astype(int) % 6
    f = hue * 6 - np.floor(hue * 6)
    p, q, t = val * (1 - s), val * (1 - f * s), val * (1 - (1 - f) * s)
    rgb = [np.select([k == i for i in range(6)], c) for c in ((val, q, p, p, t, val), (t, val, val, q, p, p), (p, p, t, val, val, q))]
    return np.stack(rgb, -1) * 255


def uv_metric(ref, ours):
    du = dv = n = 0
    for (m_r, (l_r, u_r, v_r)), (m_o, (l_o, u_o, v_o)) in zip(ref, ours):
        ok = m_r & m_o & (l_r == l_o)
        du, dv, n = du + np.abs(u_r - u_o)[ok].sum(), dv + np.abs(v_r - v_o)[ok].sum(), n + int(ok.sum())
    return f", mean |dU| {du / max(n, 1):.1e} |dV| {dv / max(n, 1):.1e} on matching parts"


def dp_sheet(cfg, img, r, runs, colors, min_score=0.5):
    keep = [i for i in range(len(r["det_scores"])) if r["det_scores"][i] >= min_score]
    boxes, scores = r["det_boxes"][keep], r["det_scores"][keep]
    if cfg.head == "cse":
        ref = [paste_back(r["det_boxes"][i], r["dp_coarse_segm"][i], r["cse_ids_all"][i]) for i in keep]
        results = [(label, [paste_back(r["det_boxes"][i], seg[i], ids[i]) for i in keep]) for label, (seg, ids) in runs]
        paint, differs, metric = (lambda m, ids: colors[ids[m]].astype(np.float32)), (lambda a, b: a != b), None
    else:
        ref = [chart_decode(r["det_boxes"][i], *(a[i] for a in (r["dp_final_coarse"], r["dp_final_fine"], r["dp_u"], r["dp_v"])))
               for i in keep]
        results = []
        for label, outs in runs:
            fin = chart_final(outs, float(r["segm_conf_eps"]))
            results.append((label, [chart_decode(r["det_boxes"][i], *(a[i] for a in fin)) for i in keep]))
        paint, differs, metric = part_paint, (lambda a, b: a[0] != b[0]), uv_metric
    return densepose_sheet(img, boxes, scores, ref, results, paint, differs, metric)


def rpn_sheet(r, runs):
    """RPN objectness (max over anchors and FPN levels) over the network input: detectron2 | one per runner | |diff| x100 of the last.

    runs: [(label, rpn logits p2..p6)]
    """
    raw = np.clip(r["image_raw_padded"][0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    H, W = raw.shape[:2]

    def heat(levels):
        up = [F.interpolate(torch.sigmoid(torch.from_numpy(np.asarray(lg))).amax(1, keepdim=True), size=(H, W), mode="nearest")
              for lg in levels]
        return torch.stack(up).amax(0)[0, 0].numpy()

    heats = [heat([r[f"rpn_logits_{i}"] for i in range(5)])] + [heat(lg) for _, lg in runs]
    tint = lambda h: np.clip(0.4 * raw + 0.6 * np.stack([h * 255, h * 60, (1 - h) * 120], -1), 0, 255).astype(np.uint8)
    dif = np.clip(np.abs(heats[0] - heats[-1]) * 100, 0, 1)
    panels = [Image.fromarray(tint(h)) for h in heats] + [Image.fromarray(np.stack([dif * 255] * 3, -1).astype(np.uint8))]
    titles = ["detectron2 RPN objectness (max over levels)"] + [f"{label} graph A" for label, _ in runs]
    titles.append(f"|diff {runs[-1][0]} vs detectron2| x100 (max {np.abs(heats[0] - heats[-1]).max():.1e})")
    return assemble(panels, titles)


def main(config, files, images, out=None, tflite=None):
    cfg = ExportConfig.load(config)
    model = load_model(str(cfg.resolve(cfg.checkpoint)), cfg.head, cfg.chart_confidence)
    g = {n: m for n, (m, _) in build_graphs(model, cfg).items()}
    eager = Eager(model, g)
    tfl = Tflite(tflite or cfg.resolve(cfg.out_dir) / "litert" / cfg.name, list(g))
    runners = [eager, tfl]
    out_dir = Path(out) if out else cfg.resolve(cfg.out_dir) / "compare" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = vertex_colors(model.roi_heads.embedder().detach().numpy()) if cfg.head == "cse" else None
    worst = {rn.name: 0.0 for rn in runners}
    id_agree = {rn.name: [] for rn in runners}
    for f in files:
        stem = Path(f).stem
        src = next(iter(sorted(Path(images).glob(stem + ".*"))), None)
        assert src is not None, f"no image named {stem}.* in {images}"
        img = Image.open(src)
        x = preprocess(img, cfg)
        r = np.load(f)
        ref_in = r["image_raw_padded"]
        assert ref_in.shape == x.shape and np.abs(ref_in - x).max() < 1e-2, \
            f"{Path(f).name}: input detectron2 saw {ref_in.shape} differs from preprocess({src.name}) {x.shape}; " \
            f"re-run dump_detectron2.py with this config"
        n_dp = r["dp_head_in"].shape[0] if "dp_head_in" in r else 0
        print(f"{Path(f).name}  (input {x.shape}, {r['box_head_in'].shape[0]} proposals, {n_dp} dp ROIs)")

        sheets = {}
        for rn in runners:
            w = lambda v: worst.__setitem__(rn.name, max(worst[rn.name], v))
            print(f"  [{rn.name}]")
            print("  graph A: image -> p2..p5 + RPN logits/deltas")
            a_out = rn.a(x)
            for i, name in enumerate(["p2", "p3", "p4", "p5"]):
                w(stat(f"feat_{name}", a_out[i], r[f"feat_{name}"]))
            for i in range(5):
                w(stat(f"rpn_logits_{i}", a_out[4 + i], r[f"rpn_logits_{i}"]))
                w(stat(f"rpn_deltas_{i}", a_out[9 + i], r[f"rpn_deltas_{i}"]))
            if rn is eager:
                w(stat("feat_p6", rn.p6(x), r["feat_p6"]))
            sheets[rn.name] = {"rpn": a_out[4:9]}

            print("  graph B1: pooled box ROIs -> class probs + box deltas")
            probs, deltas = rn.b1(r["box_head_in"])
            w(stat("box_probs", probs, F.softmax(torch.from_numpy(r["box_cls_logits"]), -1).numpy()))
            w(stat("box_deltas", deltas, r["box_deltas"]))

            if n_dp and cfg.head == "chart":
                print("  graph B2: pooled DensePose ROIs -> coarse segm, part, U, V" + (" + confidences" if cfg.chart_confidence else ""))
                outs = rn.b2(r["dp_head_in"])
                for name, o in zip(CHART_OUTPUTS, outs):
                    w(stat(name, o, r[f"dp_raw_{name}"]))
                sheets[rn.name]["dp"] = outs
            elif n_dp:
                print("  graph B2: pooled DensePose ROIs -> embedding + coarse segm")
                emb, seg = rn.b2(r["dp_head_in"])
                w(stat("embedding", emb, r["dp_embedding"]))
                w(stat("coarse_segm", seg, r["dp_coarse_segm"]))

                print("  graph C: embedding -> closest vertex (on person pixels, vs densepose EDM argmin)")
                ids = rn.c(r["dp_embedding"])
                m = r["cse_mask"]
                agree = float((ids[m] == r["cse_ids"][m]).mean())
                id_agree[rn.name].append(agree)
                print(f"    vertex id agreement on {int(m.sum())} person pixels: {agree * 100:.4f}%")
                sheets[rn.name]["dp"] = (seg, rn.c(emb))

        pairs = lambda key: [(rn.name, sheets[rn.name][key]) for rn in runners if key in sheets[rn.name]]
        rpn_sheet(r, pairs("rpn")).save(out_dir / f"{stem}_rpn.jpg", quality=92)
        print(f"  wrote {out_dir / (stem + '_rpn.jpg')}")
        if n_dp:
            dp_sheet(cfg, Image.fromarray(x[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)), r, pairs("dp"), colors) \
                .save(out_dir / f"{stem}_densepose.jpg", quality=92)
            print(f"  wrote {out_dir / (stem + '_densepose.jpg')}")
    print()
    for rn in runners:
        line = f"{rn.name:<13} worst relative error over all float outputs: {worst[rn.name]:.1e}"
        if id_agree[rn.name]:
            line += f"   graph C vertex id agreement min {min(id_agree[rn.name]) * 100:.4f}%"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("files", nargs="+", help="npz dumps from reference/dump_detectron2.py")
    ap.add_argument("--images", required=True, help="dir with the original images (matched by file stem)")
    ap.add_argument("--tflite", help="dir with the .tflite graphs (default: <out_dir>/litert/<name>)")
    ap.add_argument("--out", help="where to write the sheets (default: <out_dir>/compare/<name>)")
    a = ap.parse_args()
    main(a.config, a.files, a.images, a.out, a.tflite)
