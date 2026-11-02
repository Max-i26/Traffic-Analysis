"""
Download a manageable subset of UA-DETRAC benchmark sequence (MVI_39501)
for official identity-annotated tracking evaluation (MOTA, IDF1, HOTA, IDSW, Frag).
"""

import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://huggingface.co/datasets/kalyan1729/trafficmanagementdataset/resolve/main/DETRAC/DETRAC-Images/DETRAC-Images/MVI_39501"
OUTPUT_DIR = "datasets/detrac_benchmark/MVI_39501"

def download_frame(frame_idx: int):
    fname = f"img{frame_idx:05d}.jpg"
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return frame_idx, True
    url = f"{BASE_URL}/{fname}"
    try:
        urllib.request.urlretrieve(url, out_path)
        return frame_idx, True
    except Exception as e:
        print(f"Error downloading frame {frame_idx}: {e}")
        return frame_idx, False

def download_sequence(num_frames: int = 100):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading {num_frames} frames of MVI_39501 to {OUTPUT_DIR}...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(download_frame, range(1, num_frames + 1)))
    success = sum(1 for _, ok in results if ok)
    print(f"Successfully downloaded {success}/{num_frames} frames.")

if __name__ == '__main__':
    download_sequence(num_frames=100)
