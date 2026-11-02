"""
Unit test for Stage 3 FCOS Detection Pipeline:
Verifies model architecture, output shapes at all 3 FPN levels, target assigner,
losses, and backward pass on physical hardware.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_assigner import FCOSAssigner
from src.models.detection.fcos_losses import FCOSLoss
from src.models.detection.fcos_decoder import FCOSDecoder

def test_fcos_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running FCOS Pipeline Test on device: {device}")

    # 1. Model Forward Pass Test
    B, C, H, W = 2, 3, 288, 512
    x = torch.randn(B, C, H, W, device=device)
    model = CustomFCOSDetector(num_classes=4).to(device)
    preds = model(x)

    cls_logits = preds['cls_logits']
    bbox_preds = preds['bbox_preds']
    cnt_logits = preds['cnt_logits']

    print("Model Output Level Shapes:")
    expected_shapes = [
        (72, 128), # P2: 288/4 x 512/4
        (36, 64),  # P3: 288/8 x 512/8
        (18, 32)   # P4: 288/16 x 512/16
    ]

    for i, (eh, ew) in enumerate(expected_shapes):
        assert cls_logits[i].shape == (B, 4, eh, ew), f"L{i} cls shape mismatch: {cls_logits[i].shape}"
        assert bbox_preds[i].shape == (B, 4, eh, ew), f"L{i} bbox shape mismatch: {bbox_preds[i].shape}"
        assert cnt_logits[i].shape == (B, 1, eh, ew), f"L{i} cnt shape mismatch: {cnt_logits[i].shape}"
        print(f"  Level P{i+2} (stride {model.fpn_strides[i]}): {cls_logits[i].shape} -> OK")

    # 2. Target Assigner Test
    assigner = FCOSAssigner(strides=[4, 8, 16], scale_ranges=[(0, 64), (64, 128), (128, 1e7)])
    feat_shapes = [(cls.size(2), cls.size(3)) for cls in cls_logits]
    points_per_level = assigner.generate_points(feat_shapes, device=device)

    # Dummy GT boxes: 1 small car (P2), 1 medium car (P3), 1 large truck (P4)
    dummy_gt_boxes = torch.tensor([
        [100.0, 50.0, 130.0, 80.0],    # w=30, h=30 (max=30 -> P2)
        [200.0, 100.0, 280.0, 190.0],  # w=80, h=90 (max=90 -> P3)
        [50.0, 40.0, 250.0, 240.0]     # w=200, h=200 (max=200 -> P4)
    ], device=device)
    dummy_gt_labels = torch.tensor([0, 0, 2], device=device)

    cls_t, reg_t, cnt_t, pos_m = assigner.assign_targets_single_image(
        points_per_level, dummy_gt_boxes, dummy_gt_labels
    )

    print(f"\nTarget Assigner Test:")
    print(f"  Total points across levels: {cls_t.size(0)}")
    print(f"  Positive assigned points: {pos_m.sum().item()} (Foreground cells active)")
    assert pos_m.sum().item() > 0, "No positive points assigned!"

    # 3. Loss & Backpropagation Test
    criterion = FCOSLoss(num_classes=4)

    # Flatten predictions for batch
    all_points_flat = torch.cat(points_per_level, dim=0).repeat(B, 1)

    cls_flat = torch.cat([cls.permute(0, 2, 3, 1).reshape(B, -1, 4) for cls in cls_logits], dim=1).reshape(-1, 4)
    reg_flat = torch.cat([reg.permute(0, 2, 3, 1).reshape(B, -1, 4) for reg in bbox_preds], dim=1).reshape(-1, 4)
    cnt_flat = torch.cat([cnt.reshape(B, -1) for cnt in cnt_logits], dim=1).reshape(-1)

    # Batch targets (repeat dummy for B images)
    cls_targets_batch = cls_t.repeat(B)
    reg_targets_batch = reg_t.repeat(B, 1)
    cnt_targets_batch = cnt_t.repeat(B)
    pos_mask_batch = pos_m.repeat(B)

    loss_dict = criterion(
        cls_flat, reg_flat, cnt_flat, all_points_flat,
        cls_targets_batch, reg_targets_batch, cnt_targets_batch, pos_mask_batch
    )

    print(f"\nLoss Calculation Test:")
    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Cls Loss:   {loss_dict['loss_cls'].item():.4f}")
    print(f"  Reg Loss:   {loss_dict['loss_reg'].item():.4f} (GIoU)")
    print(f"  Cnt Loss:   {loss_dict['loss_cnt'].item():.4f}")

    assert torch.isfinite(loss_dict['total_loss']), "Loss is NaN or Inf!"

    # Backward pass check
    model.zero_grad()
    loss_dict['total_loss'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient Norm Check: {grad_norm:.4f} -> Backpropagation Successful!")
    assert grad_norm > 0, "Zero gradients detected!"

    # 4. Decoder Test
    decoder = FCOSDecoder()
    with torch.no_grad():
        detections = decoder.decode_predictions(preds, score_thresh=0.01)
    print(f"\nDecoder Test:")
    print(f"  Decoded detections for image 0: {len(detections[0]['boxes'])} boxes")

    print("\nALL FCOS PIPELINE TESTS PASSED!")

if __name__ == '__main__':
    test_fcos_pipeline()
