"""Xuat du lieu: events.csv va dataset YOLO tu dong (pseudo-label)."""
import csv
import os
from typing import List, Optional

import cv2
import numpy as np

from ..config import ExportConfig
from .types import CLASS_IDS, Detection


class EventLogger:
    def __init__(self, path: str, video_id: Optional[str] = None):
        self.video_id = video_id
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if self.video_id:
            self._w.writerow(["video_id", "frame", "time_s", "label", "x", "y", "w", "h", "score", "source"])
        else:
            self._w.writerow(["frame", "time_s", "label", "x", "y", "w", "h", "score", "source"])

    def log(self, idx: int, fps: float, dets: List[Detection]):
        for d in dets:
            if self.video_id:
                self._w.writerow([self.video_id, idx, round(idx / fps, 3), d.label, *d.bbox, d.score, d.source])
            else:
                self._w.writerow([idx, round(idx / fps, 3), d.label, *d.bbox, d.score, d.source])

    def close(self):
        self._f.close()


class DatasetExporter:
    """Luu frame goc + file label YOLO (class cx cy w h, da chuan hoa)."""

    def __init__(self, cfg: ExportConfig, video_name: str, base_dir: str):
        root = cfg.dataset_dir if os.path.isabs(cfg.dataset_dir) else os.path.join(base_dir, cfg.dataset_dir)
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
        self.every = max(1, cfg.dataset_every_n)
        self.name = video_name
        self.count = 0
        with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("fire\nsmoke\n")

    def maybe_save(self, idx: int, raw_frame: np.ndarray, dets: List[Detection]):
        if not dets or idx % self.every:
            return
        h, w = raw_frame.shape[:2]
        stem = f"{self.name}_{idx:06d}"
        cv2.imwrite(os.path.join(self.img_dir, stem + ".jpg"), raw_frame)
        with open(os.path.join(self.lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            for d in dets:
                x, y, bw, bh = d.bbox
                f.write(f"{CLASS_IDS[d.label]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                        f"{bw / w:.6f} {bh / h:.6f}\n")
        self.count += 1
