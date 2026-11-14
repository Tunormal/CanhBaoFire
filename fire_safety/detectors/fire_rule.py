from typing import List

import cv2
import numpy as np

from ..config import FireRuleConfig, TemporalConfig
from ..core.mask_utils import mask_to_detections
from ..core.temporal import TemporalMaskFilter
from ..core.types import FIRE, Detection, FrameContext
from .base import BaseDetector


class FireRuleDetector(BaseDetector):
    """Lua = mau cam/vang/do (HSV) + luat R>G>B + dang chuyen dong + ton tai qua nhieu frame."""
    name = "fire_rule"

    def __init__(self, cfg: FireRuleConfig, temporal_cfg: TemporalConfig):
        self.cfg = cfg
        self.temporal = TemporalMaskFilter(temporal_cfg)

    def reset(self):
        self.temporal.reset()

    def detect(self, frame: np.ndarray, ctx: FrameContext) -> List[Detection]:
        c = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(c.hsv_lower), np.array(c.hsv_upper))

        b, g, r = cv2.split(frame)
        rgb_rule = ((r > g) & (g > b) & (r > c.min_red)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, rgb_rule)

        if c.require_motion and ctx.motion_mask is not None:
            mask = cv2.bitwise_and(mask, ctx.motion_mask)

        mask = self.temporal.update(mask)
        min_area = c.min_area_ratio * frame.shape[0] * frame.shape[1]
        return mask_to_detections(mask, FIRE, min_area, c.close_kernel)
