from typing import List, Optional

import cv2
import numpy as np

from ..core.types import FIRE, SMOKE, Detection

COLORS = {
    FIRE: (0, 0, 230),       # Do ruc (BGR)
    SMOKE: (0, 150, 255),    # Cam / Ho phach (BGR)
}


def draw_detections(frame: np.ndarray, dets: List[Detection],
                    frame_idx: int, fps: float,
                    video_id: Optional[str] = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    th = max(2, w // 600)
    fs = max(0.45, w / 1600.0)

    for d in dets:
        x, y, bw, bh = d.bbox
        color = COLORS.get(d.label, (0, 255, 0))

        # Ve vien bounding box
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, th)

        # Ve header chu nhat cho text nhan
        text = f"{d.label} {d.score:.2f}"
        (tw, tH), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        top = max(0, y - tH - 8)
        cv2.rectangle(out, (x, top), (x + tw + 8, top + tH + 8), color, -1)
        cv2.putText(out, text, (x + 4, top + tH + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), th, cv2.LINE_AA)

    # Banner canh bao (Banner goc tren ben trai)
    labels = {d.label for d in dets}
    if FIRE in labels and SMOKE in labels:
        banner = "FIRE & SMOKE ALERT"
        banner_color = (0, 0, 220)
    elif FIRE in labels:
        banner = "FIRE ALERT"
        banner_color = (0, 0, 220)
    elif SMOKE in labels:
        banner = "SMOKE ALERT"
        banner_color = (0, 140, 255)
    else:
        banner = ""
        banner_color = (0, 0, 0)

    if banner:
        banner_w = int(w * 0.36)
        banner_h = int(36 * fs * 1.8)
        cv2.rectangle(out, (0, 0), (banner_w, banner_h), banner_color, -1)
        cv2.putText(out, banner, (12, int(26 * fs * 1.8)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 1.15, (255, 255, 255), th + 1, cv2.LINE_AA)

    # Thong tin thoi gian va ID video (goc duoi, chu thich ro y nghia t va f)
    time_sec = frame_idx / fps if fps > 0 else 0.0
    info_text = f"t (Thoi gian): {time_sec:6.2f}s | f (Frame): {frame_idx}"
    if video_id:
        info_text = f"[{video_id}]  " + info_text

    # Nen den mo nhe cho info text de luon de doc
    (itw, itH), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.75, 1)
    cv2.rectangle(out, (8, h - itH - 16), (16 + itw, h - 4), (0, 0, 0), -1)
    cv2.putText(out, info_text, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    return out
