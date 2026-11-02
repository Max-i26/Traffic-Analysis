"""
Generates the Unified Stage 5 Controlled Ablation Benchmark Table.
Reads evaluation metrics for:
- v2 Baseline (fcos_v2_fast_baseline.pt)
- v2.1 (Stride-Normalized Regression)
- v2.2 (Expanded Minority Data)
- v2.3 (Centerness-Aware Soft-NMS)
Outputs:
- reports/stage5_ablation_summary.json
- docs/stage5_ablation_report.md
"""

import os
import sys
import json

def generate_report():
    print("Generating Stage 5 Unified Ablation Summary...")

    runs = [
        ('v2 Baseline', 'Original', 'Original (200)', 'Hard-NMS', 'reports/fcos_detector_evaluation.json'),
        ('v2.1', 'Stride-Norm', 'Original (200)', 'Hard-NMS', 'reports/stage5_v2_1_results.json'),
        ('v2.2', 'Stride-Norm', 'Expanded (450)', 'Hard-NMS', 'reports/stage5_v2_2_results.json'),
        ('v2.3', 'Stride-Norm', 'Expanded (450)', 'Cnt-Soft-NMS', 'reports/stage5_v2_3_results.json')
    ]

    records = []
    for name, reg, data, supp, r_path in runs:
        if os.path.exists(r_path):
            with open(r_path, 'r') as f:
                d = json.load(f)
            records.append({
                'name': name,
                'reg': reg,
                'data': data,
                'supp': supp,
                'mAP50': d.get('mAP50', 0.0),
                'mAP75': d.get('mAP75', 0.0),
                'mAP50_95': d.get('mAP50_95', 0.0),
                'car_ap50': d.get('per_class', {}).get('car', {}).get('AP50', 0.0),
                'bus_ap50': d.get('per_class', {}).get('bus', {}).get('AP50', 0.0),
                'truck_ap50': d.get('per_class', {}).get('truck', {}).get('AP50', 0.0),
                'moto_ap50': d.get('per_class', {}).get('motorcycle', {}).get('AP50', 0.0),
                'op_prec': d.get('overall_operational_precision', 0.0),
                'op_rec': d.get('overall_operational_recall', 0.0),
                'fps': d.get('benchmark', {}).get('e2e_fps', d.get('complete_pipeline_benchmark', {}).get('complete_pipeline_fps', 0.0)),
                'vram': d.get('benchmark', {}).get('peak_vram_mb', d.get('complete_pipeline_benchmark', {}).get('peak_gpu_vram_mb', 0.0))
            })

    # Render Markdown table
    md = "# Stage 5 Controlled Detection Ablation Benchmark\n\n"
    md += "Evaluation conducted across the exact same 100% frozen held-out validation set (50 scenes, 546 ground-truth vehicles).\n\n"
    md += "| Version | Regression | Training Data | Suppression | mAP@50 | mAP@75 | mAP@50:95 | Car AP50 | Bus AP50 | Truck AP50 | Moto AP50 | Prec@.35 | Rec@.35 | Pipeline FPS | Peak VRAM |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

    for r in records:
        md += f"| **{r['name']}** | {r['reg']} | {r['data']} | {r['supp']} | "
        md += f"**{r['mAP50']:.4f}** | {r['mAP75']:.4f} | {r['mAP50_95']:.4f} | "
        md += f"{r['car_ap50']:.4f} | {r['bus_ap50']:.4f} | {r['truck_ap50']:.4f} | {r['moto_ap50']:.4f} | "
        md += f"{r['op_prec']*100:.1f}% | {r['op_rec']*100:.1f}% | {r['fps']} FPS | {r['vram']} MB |\n"

    os.makedirs('docs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    with open('docs/stage5_ablation_report.md', 'w') as f:
        f.write(md)

    with open('reports/stage5_ablation_summary.json', 'w') as f:
        json.dump(records, f, indent=2)

    print("Stage 5 Ablation Report generated successfully!")
    print(md)

if __name__ == '__main__':
    generate_report()
