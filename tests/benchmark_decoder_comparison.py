import time
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.detection.fcos_decoder import FCOSDecoder

# Try importing torchvision nms
try:
    import torchvision
    HAS_TORCHVISION = hasattr(torchvision.ops, 'nms')
except ImportError:
    HAS_TORCHVISION = False

def fast_nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, max_output: int = 100) -> torch.Tensor:
    if HAS_TORCHVISION:
        return torchvision.ops.nms(boxes, scores, iou_threshold)[:max_output]

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

class FastFCOSDecoder:
    """
    High-Speed GPU Vectorized FCOS Decoder with Per-Level Top-K Pre-filtering.
    Reduces candidate count from ~48,000 down to <= 500 before NMS.
    """
    def __init__(self,
                 strides = [4, 8, 16],
                 score_thresh: float = 0.05,
                 pre_nms_top_k: int = 500,
                 post_nms_top_k: int = 100,
                 nms_iou_thresh: float = 0.50,
                 img_size = (512, 288)):
        self.strides = strides
        self.score_thresh = score_thresh
        self.pre_nms_top_k = pre_nms_top_k
        self.post_nms_top_k = post_nms_top_k
        self.nms_iou_thresh = nms_iou_thresh
        self.img_w, self.img_h = img_size

    def decode_predictions(self, preds, score_thresh: float = None):
        thresh = score_thresh if score_thresh is not None else self.score_thresh
        cls_logits = preds['cls_logits']
        bbox_preds = preds['bbox_preds']
        cnt_logits = preds['cnt_logits']

        B = cls_logits[0].size(0)
        num_classes = cls_logits[0].size(1)
        device = cls_logits[0].device

        batch_detections = []

        for b in range(B):
            lvl_boxes_list = []
            lvl_scores_list = []
            lvl_labels_list = []

            for level_idx, stride in enumerate(self.strides):
                H, W = cls_logits[level_idx].shape[2:]

                shift_x = (torch.arange(0, W, device=device) + 0.5) * stride
                shift_y = (torch.arange(0, H, device=device) + 0.5) * stride
                y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
                points_x = x.reshape(-1)
                points_y = y.reshape(-1)

                lvl_cls = torch.sigmoid(cls_logits[level_idx][b]).permute(1, 2, 0).reshape(-1, num_classes)
                lvl_cnt = torch.sigmoid(cnt_logits[level_idx][b]).reshape(-1)
                lvl_reg = bbox_preds[level_idx][b].permute(1, 2, 0).reshape(-1, 4)

                scores_all = torch.sqrt(lvl_cls * lvl_cnt.unsqueeze(1)) # [H*W, C]

                # Per-level top-K pre-filtering directly on tensor
                max_scores, max_classes = scores_all.max(dim=1)
                valid_mask = max_scores > thresh

                if not valid_mask.any():
                    continue

                valid_indices = torch.where(valid_mask)[0]
                if len(valid_indices) > self.pre_nms_top_k:
                    topk_scores, topk_sub = torch.topk(max_scores[valid_indices], k=self.pre_nms_top_k)
                    valid_indices = valid_indices[topk_sub]

                sel_scores = max_scores[valid_indices]
                sel_labels = max_classes[valid_indices]
                px = points_x[valid_indices]
                py = points_y[valid_indices]
                reg = lvl_reg[valid_indices]

                x1 = torch.clamp(px - reg[:, 0], min=0.0, max=float(self.img_w))
                y1 = torch.clamp(py - reg[:, 1], min=0.0, max=float(self.img_h))
                x2 = torch.clamp(px + reg[:, 2], min=0.0, max=float(self.img_w))
                y2 = torch.clamp(py + reg[:, 3], min=0.0, max=float(self.img_h))
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                lvl_boxes_list.append(boxes)
                lvl_scores_list.append(sel_scores)
                lvl_labels_list.append(sel_labels)

            if len(lvl_boxes_list) == 0:
                batch_detections.append({
                    'boxes': np.empty((0, 4), dtype=np.float32),
                    'scores': np.empty((0,), dtype=np.float32),
                    'labels': np.empty((0,), dtype=np.int64)
                })
                continue

            all_boxes = torch.cat(lvl_boxes_list, dim=0)
            all_scores = torch.cat(lvl_scores_list, dim=0)
            all_labels = torch.cat(lvl_labels_list, dim=0)

            # Class-wise GPU NMS
            final_boxes = []
            final_scores = []
            final_labels = []

            for c in range(num_classes):
                c_mask = (all_labels == c)
                if not c_mask.any():
                    continue

                c_boxes = all_boxes[c_mask]
                c_scores = all_scores[c_mask]

                keep = fast_nms_torch(c_boxes, c_scores, iou_threshold=self.nms_iou_thresh, max_output=self.post_nms_top_k)
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

if __name__ == '__main__':
    # Benchmark synthetic candidate maps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking Decoders on: {device}")

    # Generate synthetic FCOS outputs
    preds = {
        'cls_logits': [
            torch.randn(1, 4, 72, 128, device=device) - 2.0, # P2
            torch.randn(1, 4, 36, 64, device=device) - 2.0,  # P3
            torch.randn(1, 4, 18, 32, device=device) - 2.0   # P4
        ],
        'bbox_preds': [
            torch.rand(1, 4, 72, 128, device=device) * 50.0,
            torch.rand(1, 4, 36, 64, device=device) * 100.0,
            torch.rand(1, 4, 18, 32, device=device) * 200.0
        ],
        'cnt_logits': [
            torch.randn(1, 1, 72, 128, device=device),
            torch.randn(1, 1, 36, 64, device=device),
            torch.randn(1, 1, 18, 32, device=device)
        ]
    }

    # Test old decoder
    old_dec = FCOSDecoder(score_thresh=0.05)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = old_dec.decode_predictions(preds, score_thresh=0.05)
    t_old = (time.perf_counter() - t0) / 10.0 * 1000.0
    print(f"Old CPU-NMS Decoder Latency: {t_old:.2f} ms")

    # Test new fast decoder
    new_dec = FastFCOSDecoder(score_thresh=0.05, pre_nms_top_k=500)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = new_dec.decode_predictions(preds, score_thresh=0.05)
    t_new = (time.perf_counter() - t0) / 10.0 * 1000.0
    print(f"New Fast GPU/Vectorized Decoder Latency: {t_new:.2f} ms")
    print(f"Speedup: {t_old / max(1e-4, t_new):.1f}x faster!")
