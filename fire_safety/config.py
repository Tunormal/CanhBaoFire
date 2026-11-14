"""Toan bo tham so cua he thong nam o day. UI/CLI chi viec sua cac dataclass nay."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MotionConfig:
    history: int = 300
    var_threshold: float = 25.0
    dilate_kernel: int = 21          # mo rong vung chuyen dong de bao phu canh lua/khoi


@dataclass
class TemporalConfig:
    decay: float = 0.8               # 0..1, cang lon cang "nho lau" (cho rule mask)
    threshold: float = 1.5           # ~ can xuat hien o >= 2 frame lien tiep
    smooth_boxes: bool = True        # Loc lam muot va on dinh bounding box
    box_min_hits: int = 2            # So frame can thay lien tiep de xac nhan (triet tieu flicker)
    box_max_age: int = 6             # So frame luu giu khi detector drop tam thoi
    box_iou_thresh: float = 0.25     # Nguong IoU tracking qua cac frame


@dataclass
class FireRuleConfig:
    enabled: bool = True
    hsv_lower: Tuple[int, int, int] = (0, 110, 170)
    hsv_upper: Tuple[int, int, int] = (25, 255, 255)  # Tap trung lua do / cam chuan, bo qua vang nhat
    min_red: int = 160               # luat RGB: R > G > B va R > min_red
    min_area_ratio: float = 0.0005   # ti le dien tich frame toi thieu cua 1 vung lua
    require_motion: bool = True
    close_kernel: int = 15


@dataclass
class SmokeRuleConfig:
    enabled: bool = True
    s_max: int = 70                  # khoi: it bao hoa mau
    v_min: int = 90
    v_max: int = 225
    min_area_ratio: float = 0.004
    require_motion: bool = True
    close_kernel: int = 31


@dataclass
class ModelConfig:
    model_path: str = "models/fire_smoke_yolov8.onnx"
    dark_model_path: str = "models/cctv_fire_smoke.onnx"  # Model phu cho canh toi / giai doan tan
    input_size: int = 640
    conf: float = 0.35               # nguong chung
    conf_fire: float = 0.38          # nguong rieng cho LUA DO: triet tieu tuyet doi mat nguoi, ao quan, trung chien
    conf_smoke: float = 0.25         # nguong rieng cho KHOI
    zoom_conf_fire: float = 0.16     # nguong LUA khi zoom can canh vung chuyen dong (Motion-ROI Zoom)
    dark_conf_fire: float = 0.28     # nguong LUA cho canh toi / giai doan tan
    iou: float = 0.45
    class_names: Tuple[str, ...] = ("fire", "smoke")
    use_letterbox: bool = True


@dataclass
class ExportConfig:
    save_events_csv: bool = True
    export_dataset: bool = False     # xuat frame + nhan YOLO tu dong de ve sau review/train
    dataset_dir: str = "dataset_auto"
    dataset_every_n: int = 15


@dataclass
class AppConfig:
    mode: str = "model"              # "model" (chinh xac cao) | "rule" | "both"
    detect_width: int = 640          # resize truoc khi detect cho rule/motion
    preview_every_n: int = 3
    output_dir: str = "video output" # Thu muc mac dinh chua video ket qua
    motion: MotionConfig = field(default_factory=MotionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fire: FireRuleConfig = field(default_factory=FireRuleConfig)
    smoke: SmokeRuleConfig = field(default_factory=SmokeRuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
