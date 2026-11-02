"""
Phase 6 Video Tracking Pipeline:
Ingests continuous 20-frame highway video clip (20 FPS),
runs Custom FCOS Detector (fcos_v2_2_expanded.pt) with ByteTrack two-stage association,
updates trajectory trails, assigns lane corridors, and renders annotated video & GIF with HUD.
"""

import os
import sys
import time
import json
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_decoder import FCOSDecoder
from src.tracking.kalman_filter import KalmanBoxTracker
from src.tracking.tracker import ByteTracker
from src.tracking.trajectory import TrajectoryManager
from src.data.fcos_loader import CLASS_NAMES

def run_video_tracking():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("PHASE 6: MULTI-OBJECT VEHICLE TRACKING PIPELINE (MODULE D)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 75)

    # 1. Load Promoted Research Baseline Detector
    ckpt_path = 'checkpoints/fcos_v2_2_expanded.pt'
    print(f"Loading frozen detector engine: {ckpt_path}")
    model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Decoder configured to retain down to tau_low (0.10) for ByteTrack two-stage matching
    decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.10,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=True
    )

    # 2. Initialize ByteTracker and Trajectory Manager
    tracker = ByteTracker(
        tau_high=0.35,
        tau_low=0.10,
        match_thresh_stage1=0.50,
        match_thresh_stage2=0.40,
        max_age=15,
        min_hits=2
    )
    traj_manager = TrajectoryManager(max_history=25, fps=20.0)

    # 3. Ingest Continuous Clip Frames
    clip_dir = 'datasets/extracted/tracking_clip_demo'
    frame_files = sorted(
        [os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    print(f"Ingesting {len(frame_files)} sequential continuous frames from: {clip_dir}")

    # Color palette for tracks (persistent by track ID % palette length)
    color_palette = [
        (0, 255, 0),    # Bright Green
        (255, 200, 0),  # Cyan-Yellow
        (0, 200, 255),  # Orange-Yellow
        (255, 0, 180),  # Magenta
        (0, 165, 255),  # Orange
        (180, 255, 50), # Lime
        (255, 100, 100) # Soft Blue
    ]

    annotated_frames = []
    tracking_log = []
    fps_list = []

    img_w, img_h = 512, 288

    for frame_idx, f_path in enumerate(frame_files, start=1):
        t0 = time.perf_counter()
        img_raw = cv2.imread(f_path)
        orig_h, orig_w = img_raw.shape[:2]

        # Preprocessing to 512x288
        img_resized = cv2.resize(img_raw, (img_w, img_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        # Detector Forward Pass
        with torch.no_grad():
            preds = model(img_tensor)
            decoded = decoder.decode_predictions(preds, score_thresh=0.10, pre_nms_top_k=500)[0]

        # ByteTrack Update
        active_tracks = tracker.update(
            det_boxes=decoded['boxes'],
            det_scores=decoded['scores'],
            det_labels=decoded['labels']
        )

        # Trajectory & Lane Corridor Mapping
        traj_manager.update(active_tracks, img_width=img_w, img_height=img_h)
        lane_counts = traj_manager.get_lane_occupancy_counts(active_tracks)

        dt = time.perf_counter() - t0
        frame_fps = 1.0 / max(1e-4, dt)
        fps_list.append(frame_fps)

        # Record log for verification
        frame_log = {
            'frame': frame_idx,
            'fps': round(float(frame_fps), 1),
            'detected_candidates': len(decoded['boxes']),
            'active_tracks_count': len(active_tracks),
            'tracks': []
        }

        # Render Visual HUD on Image
        vis = img_resized.copy()

        # Semi-transparent HUD Header Banner
        hud_overlay = vis.copy()
        cv2.rectangle(hud_overlay, (0, 0), (img_w, 42), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.75, vis, 0.25, 0, vis)

        # HUD Text
        hud_text1 = f"FRAME {frame_idx:02d}/20 | TRACKS: {len(active_tracks)} | FPS: {frame_fps:.1f} | RTX 2050"
        cv2.putText(vis, hud_text1, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        hud_text2 = f"LANES: Host: {lane_counts['Host Lane']} | Left (L1): {lane_counts['L1 (Left Lane)']} | Right (R1): {lane_counts['R1 (Right Lane)']}"
        cv2.putText(vis, hud_text2, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1, cv2.LINE_AA)

        # Render Trajectories and Vehicle Bounding Boxes
        for t in active_tracks:
            tid = t['track_id']
            color = color_palette[tid % len(color_palette)]
            box = t['bbox']
            x1, y1, x2, y2 = map(int, box)
            cat_name = CLASS_NAMES[t['class_id']] if t['class_id'] < len(CLASS_NAMES) else 'veh'

            # 1. Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 2. Trajectory history trail
            pts = t.get('trajectory', [])
            if len(pts) > 1:
                for p_i in range(1, len(pts)):
                    pt_a = (int(pts[p_i - 1][0]), int(pts[p_i - 1][1]))
                    pt_b = (int(pts[p_i][0]), int(pts[p_i][1]))
                    thickness = max(1, int(2 * (p_i / len(pts))))
                    cv2.line(vis, pt_a, pt_b, color, thickness)

            # 3. Label tag
            vx, vy = t.get('v_pixel', (0, 0))
            tag = f"ID:{tid} {cat_name} {t['score']:.2f} [{t['lane'].split()[0]}]"
            tag_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0]
            cv2.rectangle(vis, (x1, max(42, y1 - 16)), (x1 + tag_size[0] + 4, max(56, y1)), color, -1)
            cv2.putText(vis, tag, (x1 + 2, max(53, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)

            frame_log['tracks'].append({
                'track_id': tid,
                'class': cat_name,
                'score': round(float(t['score']), 3),
                'bbox': [x1, y1, x2, y2],
                'lane': t['lane'],
                'v_pixel': [vx, vy]
            })

        tracking_log.append(frame_log)
        annotated_frames.append(vis)
        print(f"Processed Frame [{frame_idx:02d}/20] -> Tracks: {len(active_tracks):<2} | Lane Counts: Host={lane_counts['Host Lane']} L1={lane_counts['L1 (Left Lane)']} | FPS: {frame_fps:.1f}")

    # 4. Save Outputs
    out_dir = 'reports/visual_samples'
    os.makedirs(out_dir, exist_ok=True)

    # Save Video (MP4)
    video_path = os.path.join(out_dir, 'tracking_demo.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, 10.0, (img_w, img_h))
    for f in annotated_frames:
        out_video.write(f)
    out_video.release()
    print(f"\nSaved tracking video to: {video_path}")

    # Save Animated GIF for instant preview
    gif_path = os.path.join(out_dir, 'tracking_demo.gif')
    try:
        from PIL import Image
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in annotated_frames]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            optimize=True,
            duration=100,  # 10 fps playback
            loop=0
        )
        print(f"Saved animated preview GIF to: {gif_path}")
    except Exception as e:
        print(f"GIF generation skipped: {e}")

    # Save Tracking Audit Log
    avg_fps = np.mean(fps_list)
    summary_report = {
        'total_frames': len(frame_files),
        'detector_checkpoint': ckpt_path,
        'average_pipeline_fps': round(float(avg_fps), 1),
        'total_unique_track_ids': KalmanBoxTracker.count,
        'frames': tracking_log
    }

    log_path = 'reports/tracking_evaluation_summary.json'
    with open(log_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved tracking log summary to: {log_path}")

if __name__ == '__main__':
    run_video_tracking()
