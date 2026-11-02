import time
import torch
import numpy as np

def fast_nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, max_output: int = 100) -> torch.Tensor:
    """
    Pure PyTorch GPU/CPU vectorized NMS.
    boxes: [N, 4] (x1, y1, x2, y2)
    scores: [N]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0 and len(keep) < max_output:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / torch.clamp(union, min=1e-7)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

print("Fast PyTorch NMS defined successfully!")
