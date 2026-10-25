import torch

from .modules import MobileParsingRCNN


def load_model(checkpoint: str, head: str = "cse", chart_confidence: bool = True) -> MobileParsingRCNN:
    """Build the model and load a detectron2 checkpoint strictly (BN counters are the only keys synthesized)."""
    model = MobileParsingRCNN(head=head, chart_confidence=chart_confidence)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    for k in model.state_dict():
        if k.endswith("num_batches_tracked") and k not in state:
            state[k] = torch.zeros((), dtype=torch.long)
    model.load_state_dict(state, strict=True)
    return model.eval()
