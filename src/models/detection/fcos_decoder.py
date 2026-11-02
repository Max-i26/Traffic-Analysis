"""
FCOS Prediction Decoder & Multi-Scale GPU NMS.
High-Speed Vectorized Decoder with:
- Per-Level Top-K Pre-filtering directly on GPU
- Stride-normalized coordinate decoding
- Centerness-weighted ranking (S = S_cls * S_cnt^0.5)
- Standard Hard-NMS and Soft-NMS options.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple

try:
    import torchvision
    HAS_TORCHVISION_NMS = hasattr(torchvision.ops, 'nms')
except ImportError:
    HAS_TORCHVISION_NMS = False

def fast_nms_torch(boxes: torch.Tensor,
                    scores: torch.Tensor,
                    iou_threshold: float = 0.50,
                    max_output: int = 100,
                    soft_nms: bool = False,
                    soft_sigma: float = 0.5) -> torch.Tensor:
    """
    Vectorized NMS running natively on GPU/CPU tensors.
    Supports standard hard-NMS and Gaussian soft-NMS.
    """
    if not soft_nms and HAS_TORCHVISION_NMS and boxes.is_cuda:
        return torchvision.ops.nms(boxes, scores, iou_threshold)[:max_output]

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    scores_copy = scores.clone()
    keep = []

    for _ in range(min(max_output, len(scores_copy))):
        max_idx = torch.argmax(scores_copy)
        if scores_copy[max_idx] <= 1e-4:
            break
        keep.append(max_idx.item())

        i = max_idx
        xx1 = torch.maximum(x1[i], x1)
        yy1 = torch.maximum(y1[i], y1)
        xx2 = torch.minimum(x2[i], x2)
        yy2 = torch.minimum(y2[i], y2)

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        union = areas[i] + areas - inter
        iou = inter / torch.clamp(union, min=1e-7)

        if soft_nms:
            decay = torch.exp(-(iou ** 2) / soft_sigma)
            scores_copy = scores_copy * decay
            scores_copy[i] = 0.0
        else:
            suppress = iou > iou_threshold
            scores_copy[suppress] = 0.0

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

class FCOSDecoder:
    def __init__(self,
                 strides: List[int] = [4, 8, 16],
                 score_thresh: float = 0.05,
                 pre_nms_top_k: int = 500,
                 post_nms_top_k: int = 100,
                 nms_iou_thresh: float = 0.50,
                 normalize_by_stride: bool = True,
                 centerness_weighted: bool = False,
                 soft_nms: bool = False,
                 img_size: Tuple[int, int] = (512, 288)):
        self.strides = strides
        self.score_thresh = score_thresh
        self.pre_nms_top_k = pre_nms_top_k
        self.post_nms_top_k = post_nms_top_k
        self.nms_iou_thresh = nms_iou_thresh
        self.normalize_by_stride = normalize_by_stride
        self.centerness_weighted = centerness_weighted
        self.soft_nms = soft_nms
        self.img_w, self.img_h = img_size

    def decode_predictions(self,
                           preds: Dict[str, List[torch.Tensor]],
                           score_thresh: float = None,
                           pre_nms_top_k: int = None) -> List[Dict[str, np.ndarray]]:
        thresh = score_thresh if score_thresh is not None else self.score_thresh
        top_k = pre_nms_top_k if pre_nms_top_k is not None else self.pre_nms_top_k

        cls_logits = preds['cls_logits']
        bbox_preds = preds['bbox_preds']
        cnt_logits = preds['cnt_logits']

        B = cls_logits[0].size(0)
        num_classes = cls_logits[0].size(1)
        device = cls_logits[0].device

        batch_detections = []

        for b in range(B):
            lvl_boxes = []
            lvl_scores = []
            lvl_labels = []

            for level_idx, stride in enumerate(self.strides):
                H, W = cls_logits[level_idx].shape[2:]

                shift_x = (torch.arange(0, W, device=device) + 0.5) * stride
                shift_y = (torch.arange(0, H, device=device) + 0.5) * stride
                y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
                px_all = x.reshape(-1)
                py_all = y.reshape(-1)

                lvl_cls = torch.sigmoid(cls_logits[level_idx][b]).permute(1, 2, 0).reshape(-1, num_classes)
                lvl_cnt = torch.sigmoid(cnt_logits[level_idx][b]).reshape(-1)
                lvl_reg = bbox_preds[level_idx][b].permute(1, 2, 0).reshape(-1, 4)

                if self.centerness_weighted:
                    scores_all = lvl_cls * torch.sqrt(lvl_cnt.unsqueeze(1))
                else:
                    scores_all = torch.sqrt(lvl_cls * lvl_cnt.unsqueeze(1))

                max_scores, max_classes = scores_all.max(dim=1)
                valid_mask = max_scores > thresh

                if not valid_mask.any():
                    continue

                valid_indices = torch.where(valid_mask)[0]
                if len(valid_indices) > top_k:
                    _, topk_sub = torch.topk(max_scores[valid_indices], k=top_k)
                    valid_indices = valid_indices[topk_sub]

                sel_scores = max_scores[valid_indices]
                sel_labels = max_classes[valid_indices]
                px = px_all[valid_indices]
                py = py_all[valid_indices]
                reg = lvl_reg[valid_indices]

                # Convert normalized offset to pixels if normalized
                scale_mul = float(stride) if self.normalize_by_stride else 1.0
                x1 = torch.clamp(px - reg[:, 0] * scale_mul, min=0.0, max=float(self.img_w))
                y1 = torch.clamp(py - reg[:, 1] * scale_mul, min=0.0, max=float(self.img_h))
                x2 = torch.clamp(px + reg[:, 2] * scale_mul, min=0.0, max=float(self.img_w))
                y2 = torch.clamp(py + reg[:, 3] * scale_mul, min=0.0, max=float(self.img_h))

                lvl_boxes.append(torch.stack([x1, y1, x2, y2], dim=-1))
                lvl_scores.append(sel_scores)
                lvl_labels.append(sel_labels)

            if len(lvl_boxes) == 0:
                batch_detections.append({
                    'boxes': np.empty((0, 4), dtype=np.float32),
                    'scores': np.empty((0,), dtype=np.float32),
                    'labels': np.empty((0,), dtype=np.int64)
                })
                continue

            all_boxes = torch.cat(lvl_boxes, dim=0)
            all_scores = torch.cat(lvl_scores, dim=0)
            all_labels = torch.cat(lvl_labels, dim=0)

            final_boxes = []
            final_scores = []
            final_labels = []

            for c in range(num_classes):
                c_mask = (all_labels == c)
                if not c_mask.any():
                    continue

                c_boxes = all_boxes[c_mask]
                c_scores = all_scores[c_mask]

                keep = fast_nms_torch(
                    c_boxes, c_scores,
                    iou_threshold=self.nms_iou_thresh,
                    max_output=self.post_nms_top_k,
                    soft_nms=self.soft_nms
                )
                if len(keep) > 0:
                    final_boxes.append(c_boxes[keep].cpu().numpy())
                    final_scores.append(c_scores[keep].cpu().numpy())
                    final_labels.append(all_labels[c_mask][keep].cpu().numpy())

            if len(final_boxes) > 0:
                batch_detections.append({
                    'boxes': np.concatenate(final_boxes, axis=0),
                    'scores': np.concatenate(final_scores, axis=0),
                    'labels': np.concatenate(final_labels, axis=0)
                })
            else:
                batch_detections.append({
                    'boxes': np.empty((0, 4), dtype=np.float32),
                    'scores': np.empty((0,), dtype=np.float32),
                    'labels': np.empty((0,), dtype=np.int64)
                })

        return batch_detections
