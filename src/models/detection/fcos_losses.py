"""
FCOS Loss Functions:
1. Focal Loss (gamma=2.0, alpha=0.25) for dense classification.
2. Centerness BCE loss on positive locations.
3. Generalized IoU (GIoU) loss on positive locations, weighted by centerness targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       alpha = 0.25,
                       gamma: float = 2.0,
                       reduction: str = 'sum') -> torch.Tensor:
    """
    inputs: [N, num_classes] raw logits
    targets: [N, num_classes] one-hot binary targets
    alpha: float or Tensor [num_classes] for class-specific weighting
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if isinstance(alpha, (list, tuple, torch.Tensor)):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=inputs.dtype, device=inputs.device)
        else:
            alpha = alpha.to(inputs.device)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    elif alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """
    pred_boxes: [N, 4] (x1, y1, x2, y2)
    target_boxes: [N, 4] (x1, y1, x2, y2)
    weight: [N] centerness weights
    """
    if pred_boxes.size(0) == 0:
        return pred_boxes.sum() * 0.0

    x1_p, y1_p, x2_p, y2_p = pred_boxes.unbind(dim=-1)
    x1_g, y1_g, x2_g, y2_g = target_boxes.unbind(dim=-1)

    area_p = torch.clamp(x2_p - x1_p, min=0) * torch.clamp(y2_p - y1_p, min=0)
    area_g = torch.clamp(x2_g - x1_g, min=0) * torch.clamp(y2_g - y1_g, min=0)

    # Intersection
    x1_i = torch.maximum(x1_p, x1_g)
    y1_i = torch.maximum(y1_p, y1_g)
    x2_i = torch.minimum(x2_p, x2_g)
    y2_i = torch.minimum(y2_p, y2_g)
    inter = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)

    # Union
    union = area_p + area_g - inter + 1e-7
    iou = inter / union

    # Enclosing convex box
    x1_c = torch.minimum(x1_p, x1_g)
    y1_c = torch.minimum(y1_p, y1_g)
    x2_c = torch.maximum(x2_p, x2_g)
    y2_c = torch.maximum(y2_p, y2_g)
    enclosing_area = torch.clamp(x2_c - x1_c, min=0) * torch.clamp(y2_c - y1_c, min=0) + 1e-7

    giou = iou - (enclosing_area - union) / enclosing_area
    loss = 1.0 - giou

    if weight is not None:
        loss = loss * weight
        return loss.sum() / torch.clamp(weight.sum(), min=1e-6)

    return loss.mean()

class FCOSLoss(nn.Module):
    def __init__(self, num_classes: int = 4, alpha = None, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        # Default class-specific alpha: higher emphasis on minority classes
        # [car, bus, truck, motorcycle]
        if alpha is None:
            self.alpha = [0.25, 0.50, 0.50, 0.75]
        else:
            self.alpha = alpha
        self.gamma = gamma

    def forward(self,
                cls_logits_flat: torch.Tensor,
                bbox_preds_flat: torch.Tensor,
                cnt_logits_flat: torch.Tensor,
                points_flat: torch.Tensor,
                cls_targets: torch.Tensor,
                reg_targets: torch.Tensor,
                cnt_targets: torch.Tensor,
                pos_mask: torch.Tensor,
                strides_flat: torch.Tensor = None) -> dict:
        """
        All tensors flattened across batch and feature points:
        cls_logits_flat: [Total_Points, num_classes]
        bbox_preds_flat: [Total_Points, 4] (l, t, r, b)
        cnt_logits_flat: [Total_Points]
        points_flat:     [Total_Points, 2] (x, y)
        cls_targets:     [Total_Points] (-1 for bg, 0..C-1 for class)
        reg_targets:     [Total_Points, 4]
        cnt_targets:     [Total_Points]
        pos_mask:        [Total_Points] (bool)
        strides_flat:    [Total_Points] (optional, for stride normalization)
        """
        num_pos = torch.clamp(pos_mask.sum(), min=1.0)

        # 1. Classification Loss (Focal Loss)
        one_hot = torch.zeros_like(cls_logits_flat)
        valid_pos = pos_mask & (cls_targets >= 0)
        if valid_pos.any():
            one_hot[valid_pos, cls_targets[valid_pos]] = 1.0

        loss_cls = sigmoid_focal_loss(
            cls_logits_flat,
            one_hot,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='sum'
        ) / num_pos

        if not valid_pos.any():
            loss_reg = bbox_preds_flat.sum() * 0.0
            loss_cnt = cnt_logits_flat.sum() * 0.0
            total_loss = loss_cls
            return {
                'total_loss': total_loss,
                'loss_cls': loss_cls,
                'loss_reg': loss_reg,
                'loss_cnt': loss_cnt,
                'num_pos': 0
            }

        # 2. Regression Loss (GIoU) on positive locations
        pos_points = points_flat[valid_pos]
        pos_pred_ltrb = bbox_preds_flat[valid_pos]
        pos_gt_ltrb = reg_targets[valid_pos]
        pos_cnt_target = cnt_targets[valid_pos]

        if strides_flat is not None:
            s_mul = strides_flat[valid_pos].unsqueeze(1)
            pos_pred_ltrb = pos_pred_ltrb * s_mul
            pos_gt_ltrb = pos_gt_ltrb * s_mul

        # Convert LTRB to (x1, y1, x2, y2)
        pos_pred_x1y1x2y2 = torch.stack([
            pos_points[:, 0] - pos_pred_ltrb[:, 0],
            pos_points[:, 1] - pos_pred_ltrb[:, 1],
            pos_points[:, 0] + pos_pred_ltrb[:, 2],
            pos_points[:, 1] + pos_pred_ltrb[:, 3]
        ], dim=-1)

        pos_gt_x1y1x2y2 = torch.stack([
            pos_points[:, 0] - pos_gt_ltrb[:, 0],
            pos_points[:, 1] - pos_gt_ltrb[:, 1],
            pos_points[:, 0] + pos_gt_ltrb[:, 2],
            pos_points[:, 1] + pos_gt_ltrb[:, 3]
        ], dim=-1)

        loss_reg = giou_loss(pos_pred_x1y1x2y2, pos_gt_x1y1x2y2, weight=pos_cnt_target)

        # 3. Centerness Loss
        pos_cnt_logits = cnt_logits_flat[valid_pos]
        loss_cnt = F.binary_cross_entropy_with_logits(
            pos_cnt_logits,
            pos_cnt_target,
            reduction='mean'
        )

        total_loss = loss_cls + 1.0 * loss_reg + 1.0 * loss_cnt

        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_cnt': loss_cnt,
            'num_pos': int(num_pos.item())
        }
