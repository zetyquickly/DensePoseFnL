"""Image -> network input, driven by the export config. numpy + PIL only, so the detectron2 reference env can import it too."""
import numpy as np
from PIL import Image

from .config import ExportConfig


def preprocess(img: Image.Image, cfg: ExportConfig) -> np.ndarray:
    """PIL image -> (1,3,H,W) float32 RGB 0..255 at the exported size.

    Scaled to fit input_width x input_height keeping the aspect ratio, then padded bottom/right with the pixel mean
    (so the padding is 0 after normalisation, like detectron2's own batching).
    """
    img = img.convert("RGB")
    s = min(cfg.input_width / img.width, cfg.input_height / img.height)
    w, h = max(round(img.width * s), 1), max(round(img.height * s), 1)
    canvas = np.empty((cfg.input_height, cfg.input_width, 3), np.float32)
    canvas[:] = np.asarray(cfg.pixel_mean, np.float32)
    canvas[:h, :w] = np.asarray(img.resize((w, h), Image.BILINEAR), np.float32)
    return canvas.transpose(2, 0, 1)[None].copy()
