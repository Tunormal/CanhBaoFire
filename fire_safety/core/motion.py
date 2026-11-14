import cv2
import numpy as np

from ..config import MotionConfig


class MotionAnalyzer:
    """Tach vung chuyen dong bang MOG2. Lua/khoi deu 'dong', tuong/den/ao do dung yen thi khong."""

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._mog = None
        self.reset()

    def reset(self):
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history, varThreshold=self.cfg.var_threshold, detectShadows=False)

    def update(self, frame: np.ndarray) -> np.ndarray:
        fg = self._mog.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.dilate_kernel,) * 2)
        return cv2.dilate(fg, k)
