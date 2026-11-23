from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

FIRE = "FIRE"
SMOKE = "SMOKE"
CLASS_IDS = {FIRE: 0, SMOKE: 1}    # thu tu class khi xuat dataset YOLO


@dataclass
class Detection:
    label: str                      # FIRE | SMOKE
    bbox: Tuple[int, int, int, int]  # x, y, w, h (pixel)
    score: float
    source: str = "rule"            # rule | model


@dataclass
class FrameContext:
    """Thong tin dung chung giua cac detector trong 1 frame."""
    motion_mask: Optional[np.ndarray] = None   # uint8 0/255, da dilate
    frame_diff: Optional[np.ndarray] = None    # uint8 2D chenh lech cuong do sang giua 2 frame lien tiep
    scale: float = 1.0                         # Ti le giua kich thuoc frame goc va detect_width (motion_mask)
    recent_fire_tracks: Tuple[Tuple[int, int, int, int], ...] = ()  # Toa do lua dang theo doi gan day de zoom phuc hoi
    confirmed_fire_tracks: Tuple[Tuple[int, int, int, int], ...] = ()  # Cac track lua da confirm manh (>= 5 hits)
    recent_smoke_tracks: Tuple[Tuple[int, int, int, int], ...] = () # Toa do khoi dang theo doi gan day de zoom phuc hoi
    has_active_smoke: bool = False             # Co track khoi dang hoat dong o frame gan nhat
    scene_cut: bool = False                    # Phat hien chuyen canh dot ngot giua 2 frame
