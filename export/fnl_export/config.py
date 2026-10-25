from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExportConfig:
    checkpoint: str
    name: str = "cse"
    # DensePose head: cse (embeddings + vertex matching) | chart (part index + UV, optionally with confidences)
    head: str = "cse"
    chart_confidence: bool = True
    # image fed to graph A: RGB, 0..255, NCHW float, must be divisible by 32
    input_height: int = 480
    input_width: int = 640
    pixel_mean: tuple = (123.675, 116.280, 103.530)
    pixel_std: tuple = (58.395, 57.120, 57.375)
    # number of ROIs per static graph
    k_proposals: int = 100   # graph B1 (box head), pooled 7x7 from p2..p5
    k_detections: int = 4    # graph B2 (DensePose head), pooled 32x32 from p2
    box_pool: int = 7
    dp_pool: int = 32
    # CSE vertex matching (graph C): in_graph (exact, GraphC) | optimized (approximate, GraphCOptimized) | off
    cse_match: str = "in_graph"
    cse_match_resolution: int = 64   # in_graph: embeddings are resized to this before matching (native: 64)
    # optimized
    cse_pixels: int = 4096           # pixels per call: the host packs person-pixel embeddings (N,16), padding the last call
    cse_blocks: int = 4096           # L: vertex blocks of the approximate search (power of 2)
    cse_group: int = 4               # T: neighbouring blocks searched exactly around the best one
    out_dir: str = "out"
    root: Path = field(default=Path("."), repr=False)

    @classmethod
    def load(cls, path) -> "ExportConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        flat = {}
        for k, v in raw.items():
            flat.update(v if isinstance(v, dict) else {k: v})
        cfg = cls(**flat)
        cfg.root = path.parent.resolve()
        assert cfg.input_height % 32 == 0 and cfg.input_width % 32 == 0
        return cfg

    def resolve(self, p: str) -> Path:
        return (self.root / p).resolve()
