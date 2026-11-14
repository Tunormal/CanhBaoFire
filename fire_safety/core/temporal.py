from typing import List, Tuple
import numpy as np

from ..config import TemporalConfig
from .types import FIRE, Detection


class TemporalMaskFilter:
    """Loc theo thoi gian cho mask: pixel phai xuat hien lap lai qua nhieu frame moi duoc xac nhan.
    Giam nhieu 1-frame (den xe, flash, phan chieu)."""

    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self._acc = None

    def reset(self):
        self._acc = None

    def update(self, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.float32)
        if self._acc is None or self._acc.shape != m.shape:
            self._acc = np.zeros_like(m)
        self._acc = self._acc * self.cfg.decay + m
        return ((self._acc >= self.cfg.threshold) * 255).astype(np.uint8)


class TrackedBox:
    """Luu vet 1 bounding box qua cac frame lien tiep de lam muot toa do va triet tieu flicker."""

    def __init__(self, label: str, bbox: Tuple[int, int, int, int], score: float, source: str):
        self.label = label
        self.bbox = list(bbox)  # [x, y, w, h]
        self.score = float(score)
        self.source = source
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.model_gap = 0

    def update(self, bbox: Tuple[int, int, int, int], score: float,
               alpha: float = 0.65, source: str = ""):
        self.age += 1
        self.time_since_update = 0
        if source == "model_track":
            # Giu vi tri on dinh qua nhip lua tan ngan, khong tang hits hay reset model_gap (predict() da +1 model_gap)
            self.score = round(max(0.20, self.score * 0.98), 2)
            return

        # Exponential moving average cho toa do box khi co phat hien that tu model/rule
        self.bbox = [
            int(round(alpha * n + (1.0 - alpha) * o))
            for n, o in zip(bbox, self.bbox)
        ]
        self.score = round(alpha * float(score) + (1.0 - alpha) * self.score, 2)
        if source:
            self.source = source
        self.hits += 1
        self.model_gap = 0

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        self.model_gap += 1


class BBoxTemporalFilter:
    """Bo loc bounding box theo thoi gian su dung IoU tracking:
    - Triet tieu nhap nhay 1 frame (chong bao gia tu cac vat the di ngang qua nhanh).
    - Giu on dinh khung hinh khi ngon lua/khoi dao dong lien tuc.
    - Duy tri khung hinh khi bi occluded / drop 1-2 frame."""

    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.tracks: List[TrackedBox] = []

    def reset(self):
        self.tracks.clear()

    @staticmethod
    def _iou(b1: List[int], b2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def recent_small_fire_boxes(self, max_area: int = 150000, max_age: int = 12) -> Tuple[Tuple[int, int, int, int], ...]:
        """Tra ve cac box lua dang hoat dong gan day (uu tien hits cao) de zoom kiem tra lai khi miss."""
        cands = [
            t for t in self.tracks
            if t.label == FIRE and t.hits >= 2
            and t.time_since_update <= max_age and t.model_gap <= 26
            and (t.bbox[2] * t.bbox[3] <= max_area)
        ]
        cands.sort(key=lambda t: t.hits, reverse=True)
        return tuple((int(t.bbox[0]), int(t.bbox[1]), int(t.bbox[2]), int(t.bbox[3])) for t in cands)

    def confirmed_fire_boxes(self, min_hits: int = 5, max_gap: int = 26) -> Tuple[Tuple[int, int, int, int], ...]:
        """Tra ve cac box lua da xac nhan chac chan (hits >= 5) dang trong nhip tan ngan."""
        cands = [
            t for t in self.tracks
            if t.label == FIRE and t.hits >= min_hits
            and t.time_since_update <= 6 and t.model_gap <= max_gap
        ]
        cands.sort(key=lambda t: t.hits, reverse=True)
        return tuple((int(t.bbox[0]), int(t.bbox[1]), int(t.bbox[2]), int(t.bbox[3])) for t in cands)

    def has_active_label(self, label: str, max_age: int = 3) -> bool:
        """Kiem tra xem nhan (vd SMOKE) co track xac nhan dang hoat dong gan day khong."""
        return any(
            t.label == label and t.hits >= self.cfg.box_min_hits and t.time_since_update <= max_age
            for t in self.tracks
        )

    def update(self, dets: List[Detection], scene_cut: bool = False) -> List[Detection]:
        if scene_cut:
            self.tracks.clear()

        if not self.cfg.smooth_boxes:
            return dets

        for t in self.tracks:
            t.predict()

        matched_tracks = set()
        matched_dets = set()

        for d_idx, d in enumerate(dets):
            best_iou = 0.0
            best_t_idx = -1
            for t_idx, t in enumerate(self.tracks):
                if t_idx in matched_tracks or t.label != d.label:
                    continue
                iou = self._iou(t.bbox, d.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_iou >= self.cfg.box_iou_thresh and best_t_idx >= 0:
                self.tracks[best_t_idx].update(d.bbox, d.score, source=d.source)
                matched_tracks.add(best_t_idx)
                matched_dets.add(d_idx)

        # Cac phat hien moi chua match (bo qua synthetic model_track neu khong co track goc)
        for d_idx, d in enumerate(dets):
            if d_idx not in matched_dets and d.source != "model_track":
                self.tracks.append(TrackedBox(d.label, d.bbox, d.score, d.source))

        # Giu cac track on dinh lau hon (<= 12 frames) de track-guided zoom co co hoi bat lai khi flicker
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= (max(self.cfg.box_max_age, 12) if t.hits >= 3 else self.cfg.box_max_age)
        ]

        # Xac nhan cac track hop le de xuat ra
        out: List[Detection] = []
        for t in self.tracks:
            # Hien thi ngay neu full-frame conf cao (>= 0.45), con model_zoom phai dat du hits de chong nhieu 1-frame
            is_confirmed = (t.hits >= self.cfg.box_min_hits) or (
                t.score >= 0.45 and t.source != "model_zoom"
            )
            # Cho phep giu track on dinh qua khoang flicker ngan neu da confirm chac chan
            allowed_gap = 4 if t.hits >= 4 else 1
            is_active = (t.time_since_update == 0) or (
                t.hits >= self.cfg.box_min_hits + 1 and t.time_since_update <= allowed_gap
            )

            if is_confirmed and is_active:
                out.append(Detection(t.label, tuple(t.bbox), round(float(t.score), 2), t.source))
        return out
