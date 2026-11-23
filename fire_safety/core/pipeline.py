from typing import List, Optional

import cv2
import numpy as np

from ..config import AppConfig
from ..detectors import (BaseDetector, FireRuleDetector, SmokeRuleDetector,
                         YoloOnnxDetector)
from .motion import MotionAnalyzer
from .temporal import BBoxTemporalFilter
from .types import SMOKE, Detection, FrameContext


class DetectionPipeline:
    """Dieu phoi: resize -> motion -> chay cac detector -> NMS -> Temporal Filter."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.motion = MotionAnalyzer(cfg.motion)
        self.temporal_filter = BBoxTemporalFilter(cfg.temporal)
        self._prev_gray: Optional[np.ndarray] = None
        self.detectors: List[BaseDetector] = self._build_detectors()

    def _build_detectors(self) -> List[BaseDetector]:
        cfg, ds = self.cfg, []
        if cfg.mode in ("rule", "both"):
            if cfg.fire.enabled:
                ds.append(FireRuleDetector(cfg.fire, cfg.temporal))
            if cfg.smoke.enabled:
                ds.append(SmokeRuleDetector(cfg.smoke, cfg.temporal))
        if cfg.mode in ("model", "both"):
            ds.append(YoloOnnxDetector(cfg.model))
        if not ds:
            raise ValueError("Chua bat detector nao (kiem tra mode / Fire / Smoke).")
        return ds

    def reset(self):
        self.motion.reset()
        self.temporal_filter.reset()
        self._prev_gray = None
        for d in self.detectors:
            d.reset()

    def process(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        dets: List[Detection] = []

        scale = 1.0
        small = frame
        if w > self.cfg.detect_width:
            scale = w / float(self.cfg.detect_width)
            small = cv2.resize(frame, (self.cfg.detect_width, int(round(h / scale))))

        motion_mask = self.motion.update(small)
        curr_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        frame_diff: Optional[np.ndarray] = None
        scene_cut = False
        if self._prev_gray is not None and self._prev_gray.shape == curr_gray.shape:
            frame_diff = cv2.absdiff(curr_gray, self._prev_gray)
            scene_cut = float(np.percentile(frame_diff[::4, ::4], 50)) >= 22.0
        self._prev_gray = curr_gray

        ctx = FrameContext(
            motion_mask=motion_mask,
            frame_diff=frame_diff,
            scale=scale,
            recent_fire_tracks=() if scene_cut else self.temporal_filter.recent_small_fire_boxes(),
            confirmed_fire_tracks=() if scene_cut else self.temporal_filter.confirmed_fire_boxes(),
            recent_smoke_tracks=() if scene_cut else self.temporal_filter.recent_small_smoke_boxes(),
            has_active_smoke=False if scene_cut else self.temporal_filter.has_active_label(SMOKE, max_age=3),
            scene_cut=scene_cut,
        )

        for d in self.detectors:
            if isinstance(d, YoloOnnxDetector):
                dets.extend(d.detect(frame, ctx))
            else:
                rule_dets = d.detect(small, ctx)
                if scale != 1.0:
                    rule_dets = [
                        Detection(rd.label, tuple(int(round(v * scale)) for v in rd.bbox), rd.score, rd.source)
                        for rd in rule_dets
                    ]
                dets.extend(rule_dets)

        dets = self._nms(dets)
        dets = self.temporal_filter.update(dets, scene_cut=scene_cut)
        return dets

    @staticmethod
    def _nms(dets: List[Detection], iou: float = 0.5) -> List[Detection]:
        out: List[Detection] = []
        for label in {d.label for d in dets}:
            group = [d for d in dets if d.label == label]
            idx = cv2.dnn.NMSBoxes([list(d.bbox) for d in group],
                                   [d.score for d in group], 0.0, iou)
            out.extend(group[i] for i in np.array(idx).flatten())
        return out
