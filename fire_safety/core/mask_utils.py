from typing import List

import cv2
import numpy as np

from .types import Detection


def mask_to_detections(mask: np.ndarray, label: str, min_area_px: float,
                       close_kernel: int, source: str = "rule") -> List[Detection]:
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Detection] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        fill = float(np.count_nonzero(m[y:y + h, x:x + w])) / float(w * h)
        out.append(Detection(label, (x, y, w, h), round(fill, 2), source))
    return out
