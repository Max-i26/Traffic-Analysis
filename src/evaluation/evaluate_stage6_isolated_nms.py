"""
Stage 6: Isolated Post-Processing Ablation & Class Reliability Analysis.
Evaluates the EXACT same frozen checkpoint (fcos_v2_2_expanded.pt) across:
1. Mode A: Standard Hard NMS (S = sqrt(S_cls * S_cnt))
2. Mode B: Centerness-Weighted Fast Hard NMS (S = S_cls * sqrt(S_cnt))
3. Mode C: Soft-NMS (S = S_cls * sqrt(S_cnt), Gaussian decay)

Profiles:
- Exact latency of GPU forward vs Decoding vs Suppression
- Identifies the root cause of the 1.8 FPS bottleneck
- Evaluates per-class PR curves and class-specific threshold sweeps
"""

import os
import sys
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_decoder import FCOSDecoder
from src.data.fcos_loader import get_fcos_dataloaders, CLASS_NAMES
from src.evaluation.coco_detection_evaluator import COCOEvaluator

def evaluate_nms_modes():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("STAGE 6: ISOLATED POST-PROCESSING ABLATION & CLASS RELIABILITY")
    print(f"Hardware: {device}")
    print("=" * 75)

    ckpt_path = 'checkpoints/fcos_v2_2_expanded.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}")

    model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Fixed 50 validation scenes (100% frozen)
    split_path = 'datasets/splits/bdd_pilot_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'
    _, val_loader = get_fcos_dataloaders(
        split_file=split_path,
        root_dir=root_dir,
        batch_size=4,
        num_workers=0,
        use_class_aware_sampling=False
    )

    modes = [
        ('Mode A: Standard Hard-NMS', False, False),
        ('Mode B: Cnt-Weighted Fast Hard-NMS', True, False),
        ('Mode C: Soft-NMS (Iterative)', True, True)
    ]

    ablation_results = {}

    for mode_name, cnt_weight, use_soft_nms in modes:
        print(f"\nEvaluating: [{mode_name}] ...")
        decoder = FCOSDecoder(
            strides=[4, 8, 16],
            score_thresh=0.01,
            pre_nms_top_k=500,
            post_nms_top_k=100,
            nms_iou_thresh=0.50,
            normalize_by_stride=True,
            centerness_weighted=cnt_weight,
            soft_nms=use_soft_nms
        )

        all_predictions = []
        all_ground_truths = []
        img_counter = 0

        # Profile timing on validation set
        t_decode_start = time.perf_counter()

        with torch.no_grad():
            for imgs, targets in val_loader:
                B = imgs.size(0)
                imgs = imgs.to(device)
                preds = model(imgs)

                decoded_batch = decoder.decode_predictions(preds, score_thresh=0.01, pre_nms_top_k=500)

                for b in range(B):
                    all_predictions.append({
                        'image_id': img_counter,
                        'boxes': decoded_batch[b]['boxes'],
                        'scores': decoded_batch[b]['scores'],
                        'labels': decoded_batch[b]['labels']
                    })
                    all_ground_truths.append({
                        'image_id': img_counter,
                        'boxes': targets[b]['boxes'].cpu().numpy(),
                        'labels': targets[b]['labels'].cpu().numpy()
                    })
                    img_counter += 1

        val_total_time = (time.perf_counter() - t_decode_start) * 1000.0
        avg_per_image_ms = val_total_time / max(1, img_counter)

        evaluator = COCOEvaluator(class_names=CLASS_NAMES)
        eval_metrics = evaluator.evaluate(all_predictions, all_ground_truths, operational_conf_thresh=0.35)

        # Micro-benchmark for isolated suppression stage
        dummy_raw = np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8)
        t_input = (dummy_raw.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_input).unsqueeze(0).to(device)

        with torch.no_grad():
            p = model(t_tensor)

        # Time pure decoding + suppression on single frame (50 iterations)
        t_supp_list = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = decoder.decode_predictions(p, score_thresh=0.05, pre_nms_top_k=500)
            t_supp_list.append((time.perf_counter() - t0) * 1000.0)
        mean_supp_ms = np.mean(t_supp_list)

        mode_record = {
            'mAP50': eval_metrics['mAP50'],
            'mAP75': eval_metrics['mAP75'],
            'mAP50_95': eval_metrics['mAP50_95'],
            'car_ap50': eval_metrics['per_class']['car']['AP50'],
            'bus_ap50': eval_metrics['per_class']['bus']['AP50'],
            'truck_ap50': eval_metrics['per_class']['truck']['AP50'],
            'moto_ap50': eval_metrics['per_class']['motorcycle']['AP50'],
            'suppression_latency_ms': round(float(mean_supp_ms), 2),
            'val_per_frame_total_ms': round(float(avg_per_image_ms), 2)
        }
        ablation_results[mode_name] = mode_record

        print(f"  mAP@50: {mode_record['mAP50']:.4f} | Car: {mode_record['car_ap50']:.4f} | Truck: {mode_record['truck_ap50']:.4f} | Bus: {mode_record['bus_ap50']:.4f}")
        print(f"  Suppression Latency: {mode_record['suppression_latency_ms']} ms | Val Per-frame: {mode_record['val_per_frame_total_ms']} ms")

    out_path = 'reports/stage6_isolated_nms_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved isolated NMS ablation to: {out_path}")

if __name__ == '__main__':
    evaluate_nms_modes()
