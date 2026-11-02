"""
Annotation parser for TuSimple lane detection and BDD100K object detection.
Preserves raw annotation integrity and converts coordinate sequences.
"""

import json
from typing import List, Tuple, Dict, Any, Optional

def parse_tusimple_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single TuSimple JSON record.
    TuSimple format:
        raw_file: path to keyframe image (e.g. "clips/0313-1/6040/20.jpg")
        lanes: list of lists, horizontal coordinates for each lane line
        h_samples: list of vertical sample heights
        Note: x == -2 indicates missing or occluded lane coordinate.

    Returns:
        Dict containing:
            raw_file (str): relative path to keyframe
            h_samples (List[int]): vertical sample coordinates
            lane_points (List[List[Tuple[int, int]]]): list of valid (x, y) point sequences
            num_lanes (int): count of active lanes with at least 2 points
    """
    raw_file = record.get('raw_file', '')
    h_samples = record.get('h_samples', [])
    raw_lanes = record.get('lanes', [])

    valid_lanes: List[List[Tuple[int, int]]] = []

    for lane in raw_lanes:
        pts: List[Tuple[int, int]] = []
        for x, y in zip(lane, h_samples):
            if x >= 0: # Valid coordinate, -2 means missing/occluded
                pts.append((int(x), int(y)))
        if len(pts) >= 2:
            valid_lanes.append(pts)

    return {
        'raw_file': raw_file,
        'h_samples': h_samples,
        'lane_points': valid_lanes,
        'num_lanes': len(valid_lanes)
    }

def parse_bdd_record(record: Dict[str, Any], allowed_categories: Optional[set] = None) -> Dict[str, Any]:
    """
    Parses a single BDD100K standardized frame record.
    Extracts bounding boxes [x1, y1, x2, y2] for selected categories.
    """
    img_name = record.get('name', '')
    width = record.get('width', 1280)
    height = record.get('height', 720)
    labels = record.get('labels', [])

    boxes = []
    classes = []

    for lbl in labels:
        cat = lbl.get('category', '')
        if allowed_categories is not None and cat not in allowed_categories:
            continue

        box = lbl.get('box2d', {})
        x1 = max(0.0, float(box.get('x1', 0.0)))
        y1 = max(0.0, float(box.get('y1', 0.0)))
        x2 = min(float(width), float(box.get('x2', width)))
        y2 = min(float(height), float(box.get('y2', height)))

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            classes.append(cat)

    return {
        'name': img_name,
        'width': width,
        'height': height,
        'boxes': boxes,
        'classes': classes
    }
