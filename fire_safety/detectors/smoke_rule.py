from typing import List

import cv2
import numpy as np

from ..config import SmokeRuleConfig, TemporalConfig
from ..core.mask_utils import mask_to_detections
from ..core.temporal import TemporalMaskFilter
from ..core.types import SMOKE, Detection, FrameContext
from .base import BaseDetector


class SmokeRuleDetector(BaseDetector):
    """Khoi = vung xam (it bao hoa) + dang chuyen dong cham + ton tai qua nhieu frame.
    Rule nay de bao dong gia (suong, hoi nuoc, ao xam) -> chi la baseline, nen thay bang model."""
    name = "smoke_rule"

    def __init__(self, cfg: SmokeRuleConfig, temporal_cfg: TemporalConfig):
        self.cfg = cfg
        self.temporal = TemporalMaskFilter(temporal_cfg)

    def reset(self):
        self.temporal.reset()

    def detect(self, frame: np.ndarray, ctx: FrameContext) -> List[Detection]:
        c = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0, 0, c.v_min)), np.array((179, c.s_max, c.v_max)))

        if c.require_motion and ctx.motion_mask is not None:
            mask = cv2.bitwise_and(mask, ctx.motion_mask)

        mask = self.temporal.update(mask)
        min_area = c.min_area_ratio * frame.shape[0] * frame.shape[1]
        return mask_to_detections(mask, SMOKE, min_area, c.close_kernel)
