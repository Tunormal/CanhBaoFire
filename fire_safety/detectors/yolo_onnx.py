import ast
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..config import ModelConfig
from ..core.types import FIRE, SMOKE, Detection, FrameContext
from .base import BaseDetector


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize va padding anh giu nguyen ti le (aspect ratio) cho YOLO."""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    unpad_w, unpad_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - unpad_w) / 2.0, (new_shape[0] - unpad_h) / 2.0

    if (shape[1], shape[0]) != (unpad_w, unpad_h):
        img_resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)


def is_false_positive_fire(roi: np.ndarray) -> bool:
    """Loai bo cac vat the tron/tinh mau vang (nhu long do trung chien tren chao hoac bong den tron)
    bang cach phan tich hinh tron Hough va do bien thien sac do Hue."""
    if roi is None or roi.shape[0] < 20 or roi.shape[1] < 20:
        return False
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        min_r = int(min(roi.shape[:2]) * 0.15)
        max_r = int(max(roi.shape[:2]) * 0.45)
        if min_r >= max_r or min_r < 4:
            return False
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=60, param2=35, minRadius=min_r, maxRadius=max_r)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_std = float(np.std(hsv[:, :, 0]))
        if circles is not None and len(circles[0]) > 0 and h_std < 8.0:
            return True
    except Exception:
        pass
    return False


class SingleYoloNet:
    """Mot mang YOLO ONNX don le (uu tien onnxruntime CPU nhanh hon ~38%, fallback cv2.dnn)."""

    def __init__(self, model_path: str, default_input_size: int, default_classes: Tuple[str, ...]):
        self.model_path = model_path
        self.ort_session = None
        self.ort_input_name = ""
        self.net = None
        if ort is not None:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = min(8, os.cpu_count() or 4)
                self.ort_session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.ort_input_name = self.ort_session.get_inputs()[0].name
            except Exception:
                self.ort_session = None
        if self.ort_session is None:
            self.net = cv2.dnn.readNetFromONNX(model_path)
        self.class_map: Dict[int, str] = {}
        self.input_size: int = default_input_size
        self._inspect_metadata(default_classes)

    def _inspect_metadata(self, default_classes: Tuple[str, ...]):
        if onnx is not None:
            try:
                m = onnx.load(self.model_path, load_external_data=False)
                meta = {p.key: p.value for p in m.metadata_props}
                if "names" in meta:
                    raw_names = ast.literal_eval(meta["names"])
                    for k, v in raw_names.items():
                        name_str = str(v).lower()
                        if "fire" in name_str or "flame" in name_str:
                            self.class_map[int(k)] = FIRE
                        elif "smoke" in name_str:
                            self.class_map[int(k)] = SMOKE
                if "imgsz" in meta:
                    sz = ast.literal_eval(meta["imgsz"])
                    self.input_size = sz[0] if isinstance(sz, (list, tuple)) else int(sz)
            except Exception:
                pass

        if not self.class_map:
            for idx, name in enumerate(default_classes):
                n = name.lower()
                if "fire" in n or "flame" in n:
                    self.class_map[idx] = FIRE
                elif "smoke" in n:
                    self.class_map[idx] = SMOKE


class YoloOnnxDetector(BaseDetector):
    """Detector su dung model YOLO (ONNX) qua cv2.dnn:
    - Ho tro chay don le hoac ensemble nhieu model (phuc vu nhan dien ca ban ngay lan camera an ninh dem).
    - Tu dong nhan dien metadata ve class names va input size tu file ONNX.
    - Ho tro letterboxing bao toan aspect ratio tranh bien dang vat the.
    - Class-specific thresholds (nguong rieng cho LUA va KHOI).
    - Tich hop bo loc Hough Circle + Hue variance triet tieu bat nham trung chien, den tron."""

    name = "yolo_onnx"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        raw_paths = [p.strip() for p in cfg.model_path.split(",") if p.strip()] if cfg.model_path else []
        valid_paths = [p for p in raw_paths if os.path.isfile(p)]

        if not valid_paths:
            candidates = [
                "models/fire_smoke_yolov8.onnx",
                "models/cctv_fire_smoke.onnx",
                "models/fire_smoke_yolov8s.onnx",
                "models/rabahdev_fire_smoke.onnx",
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    valid_paths.append(cand)
                    break

        if not valid_paths:
            raise FileNotFoundError(f"Khong tim thay model ONNX tai '{cfg.model_path}'.")

        self.sub_models: List[SingleYoloNet] = [
            SingleYoloNet(p, cfg.input_size, cfg.class_names) for p in valid_paths
        ]
        # Model phu chuyen dung cho phong toi / giai doan lua tan khi model chinh bo sot
        self.dark_sub: Optional[SingleYoloNet] = None
        if cfg.dark_model_path and os.path.isfile(cfg.dark_model_path):
            norm_dark = os.path.abspath(cfg.dark_model_path)
            if all(os.path.abspath(p) != norm_dark for p in valid_paths):
                self.dark_sub = SingleYoloNet(cfg.dark_model_path, cfg.input_size, cfg.class_names)

    def reset(self):
        pass

    def _infer_net(self, sub: SingleYoloNet, img: np.ndarray,
                   conf_fire_override: Optional[float] = None,
                   conf_smoke_override: Optional[float] = None,
                   source_tag: str = "model") -> List[Detection]:
        """Chay 1 mang YOLO ONNX tren anh (full frame hoac vung crop zoom) va tra ve detections."""
        h, w = img.shape[:2]
        c = self.cfg
        input_size = sub.input_size

        if c.use_letterbox:
            inp_img, r, (dw, dh) = letterbox(img, (input_size, input_size))
        else:
            inp_img = cv2.resize(img, (input_size, input_size))
            r = input_size / float(w)
            dw, dh = 0.0, 0.0

        blob = cv2.dnn.blobFromImage(inp_img, 1.0 / 255.0, (input_size, input_size),
                                     swapRB=True, crop=False)
        if sub.ort_session is not None:
            raw_out = sub.ort_session.run(None, {sub.ort_input_name: blob})[0]
        else:
            sub.net.setInput(blob)
            raw_out = sub.net.forward()

        if raw_out.ndim == 3:
            out = raw_out[0].T
        elif raw_out.ndim == 2:
            out = raw_out.T if raw_out.shape[0] < raw_out.shape[1] else raw_out
        else:
            return []

        if out.shape[1] < 4 + len(sub.class_map):
            return []

        dets: List[Detection] = []
        fire_thresh = c.conf_fire if conf_fire_override is None else conf_fire_override
        smoke_thresh = c.conf_smoke if conf_smoke_override is None else conf_smoke_override

        for cls_id, label in sub.class_map.items():
            conf_thresh = fire_thresh if label == FIRE else smoke_thresh
            col_idx = 4 + cls_id
            if col_idx >= out.shape[1]:
                continue

            scores = out[:, col_idx]
            keep_mask = scores >= conf_thresh
            if not np.any(keep_mask):
                continue

            sub_out = out[keep_mask]
            sub_scores = scores[keep_mask]

            cls_boxes = []
            for row in sub_out:
                cx, cy, bw, bh = row[:4]
                if c.use_letterbox:
                    x1 = int(round((cx - bw / 2.0 - dw) / r))
                    y1 = int(round((cy - bh / 2.0 - dh) / r))
                    bx_w = int(round(bw / r))
                    bx_h = int(round(bh / r))
                else:
                    sx = w / float(input_size)
                    sy = h / float(input_size)
                    x1 = int(round((cx - bw / 2.0) * sx))
                    y1 = int(round((cy - bh / 2.0) * sy))
                    bx_w = int(round(bw * sx))
                    bx_h = int(round(bh * sy))

                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                bx_w = max(1, min(w - x1, bx_w))
                bx_h = max(1, min(h - y1, bx_h))
                cls_boxes.append([x1, y1, bx_w, bx_h])

            indices = cv2.dnn.NMSBoxes(cls_boxes, sub_scores.tolist(), conf_thresh, c.iou)
            for idx in np.array(indices).flatten():
                bx, by, bw_b, bh_b = cls_boxes[idx]
                roi = img[by:by + bh_b, bx:bx + bw_b]
                if label == FIRE and is_false_positive_fire(roi):
                    continue

                dets.append(Detection(
                    label=label,
                    bbox=(bx, by, bw_b, bh_b),
                    score=round(float(sub_scores[idx]), 2),
                    source=source_tag,
                ))
        return dets

    @staticmethod
    def _detect_text_occluded_fire(frame: np.ndarray, hsv: np.ndarray,
                                   diff: np.ndarray, p98: float) -> List[Detection]:
        """Khi AI da xac nhan co KHOI va khung hinh on dinh (khong lia camera),
        phat hien dam chay lon bi chu canh bao che khuat dac trung YOLO."""
        if p98 > 10.0:
            return []
        h, w = frame.shape[:2]
        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)

        big_fire = (
            (hsv[:, :, 0] <= 22) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 120)
            & (r - g >= 25) & (r - b >= 60) & (diff >= 6)
        )
        min_px = max(1200, int(h * w * 0.0014))
        if int(big_fire.sum()) < min_px:
            return []

        ys, xs = np.where(big_fire)
        x1, x2 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        y1, y2 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        return [Detection(label=FIRE, bbox=(x1, y1, bw, bh), score=0.72, source="model_hybrid")]

    @staticmethod
    def _is_valid_zoom_fire(frame: np.ndarray, hsv: np.ndarray, diff: np.ndarray,
                            bbox: Tuple[int, int, int, int], crop_sz: int,
                            is_tracked: bool) -> bool:
        """Xac thuc box lua trong vung zoom: chan vat the lon chiem tron khung crop (chao, den halogen)
        va cac dom phan chieu thoang qua cua vat the di chuyen (ratio dmv6/mv8 thap)."""
        h, w = frame.shape[:2]
        fx, fy, fbw, fbh = bbox
        x1, y1 = max(0, fx), max(0, fy)
        x2, y2 = min(w, fx + fbw), min(h, fy + fbh)
        if x2 <= x1 or y2 <= y1:
            return False

        r_sub = frame[y1:y2, x1:x2, 2].astype(np.int16)
        if int(r_sub.max()) < 220:
            return False

        b_sub = frame[y1:y2, x1:x2, 0].astype(np.int16)
        hs = hsv[y1:y2, x1:x2, 0]
        ss = hsv[y1:y2, x1:x2, 1]

        if is_tracked:
            warm_px = int(((hs <= 25) & (ss >= 80) & (r_sub - b_sub >= 45) & (r_sub >= 150)).sum())
            return warm_px >= 45

        if fbw >= crop_sz * 0.50 or fbh >= crop_sz * 0.50:
            return False

        df = diff[y1:y2, x1:x2]
        mv8_px = int((df >= 8).sum())
        dmv6 = int(((hs <= 16) & (ss >= 135) & (r_sub >= 145) & (r_sub - b_sub >= 70) & (df >= 6)).sum())
        return dmv6 >= 50 and (dmv6 / max(1, mv8_px)) >= 0.14

    def _run_zoom_at_targets(self, frame: np.ndarray, hsv: np.ndarray, diff: np.ndarray,
                             targets: List[Tuple[int, int, Tuple[int, ...]]],
                             is_tracked: bool) -> List[Detection]:
        """Cat vung zoom do phan giai goc tai danh sach tam muc tieu va kiem chung ket qua."""
        h, w = frame.shape[:2]
        zoom_dets: List[Detection] = []
        zoom_conf = self.cfg.zoom_conf_fire
        for cx, cy, sizes in targets:
            found_at_target = False
            for sz in sizes:
                if w < sz or h < sz:
                    continue
                x1 = max(0, min(w - sz, cx - sz // 2))
                y1 = max(0, min(h - sz, cy - sz // 2))
                crop = frame[y1:y1 + sz, x1:x1 + sz]
                for sub in self.sub_models:
                    for d in self._infer_net(sub, crop, conf_fire_override=zoom_conf, source_tag="model_zoom"):
                        if d.label != FIRE:
                            continue
                        bbox = (d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2], d.bbox[3])
                        if not self._is_valid_zoom_fire(frame, hsv, diff, bbox, sz, is_tracked):
                            continue
                        zoom_dets.append(Detection(label=FIRE, bbox=bbox, score=d.score, source="model_zoom"))
                        found_at_target = True
                if found_at_target:
                    break
        return zoom_dets

    def _detect_motion_zoom_fire(self, frame: np.ndarray, hsv: np.ndarray,
                                 diff: np.ndarray, p98: float) -> List[Detection]:
        """Motion-ROI High-Resolution Zoom: cat can canh do phan giai goc quanh cac diem
        chuyen dong nong ruc (ngon lua moi phat sinh) de YOLO nhan dien chinh xac."""
        if p98 > 8.5:
            return []

        h, w = frame.shape[:2]
        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)

        thr = max(7.0, p98 * 2.2)
        mask = (
            (hsv[:, :, 0] <= 22) & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 105)
            & (r - g >= 22) & (g - b >= 8) & (r - b >= 50) & (diff >= thr)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hotspots = []
        for c_cnt in cnts:
            if cv2.contourArea(c_cnt) < 35:
                continue
            x, y, bw, bh = cv2.boundingRect(c_cnt)
            if bw < 20:
                continue
            pad = 8
            rx1, ry1 = max(0, x - pad), max(0, y - pad)
            rx2, ry2 = min(w, x + bw + pad), min(h, y + bh + pad)
            r_sub, b_sub = r[ry1:ry2, rx1:rx2], b[ry1:ry2, rx1:rx2]
            h_sub, s_sub = hsv[ry1:ry2, rx1:rx2, 0], hsv[ry1:ry2, rx1:rx2, 1]
            core_r = int(r_sub.max())
            deep_px = int(((h_sub <= 16) & (s_sub >= 135) & (r_sub >= 145) & (r_sub - b_sub >= 70)).sum())
            if core_r >= 238 and deep_px >= 80:
                hotspots.append((x + bw // 2, y + bh // 2, deep_px))

        if not hotspots:
            return []

        targets = [
            (cx, cy, (200, 320))
            for cx, cy, _ in sorted(hotspots, key=lambda t: t[2], reverse=True)[:2]
        ]
        return self._run_zoom_at_targets(frame, hsv, diff, targets, is_tracked=False)

    def _detect_tracked_zoom_fire(self, frame: np.ndarray, hsv: np.ndarray,
                                  diff: np.ndarray, p90: float,
                                  recent_tracks: Tuple[Tuple[int, int, int, int], ...]) -> List[Detection]:
        """Track-Guided Zoom Recovery: zoom quanh cac track lua da xac nhan gan day khi bi miss
        thoang qua hoac trong luc chuyen canh wipe cuc bo (p90 thap du p98 cao)."""
        if not recent_tracks or p90 > 7.5:
            return []

        targets: List[Tuple[int, int, Tuple[int, ...]]] = []
        for tx, ty, tbw, tbh in recent_tracks[:2]:
            sizes = (200, 320) if max(tbw, tbh) < 110 else (400, 320)
            targets.append((tx + tbw // 2, ty + tbh // 2, sizes))
        return self._run_zoom_at_targets(frame, hsv, diff, targets, is_tracked=True)

    @staticmethod
    def _verify_tracked_fire_glow(frame: np.ndarray, hsv: np.ndarray, p90: float,
                                  confirmed_tracks: Tuple[Tuple[int, int, int, int], ...]) -> List[Detection]:
        """Giu track lua da confirm manh (>= 5 hits) qua nhip tan ngan neu loi hong ruc van con ngay tai ROI."""
        if not confirmed_tracks or p90 > 6.0:
            return []

        h, w = frame.shape[:2]
        out: List[Detection] = []
        for tx, ty, tbw, tbh in confirmed_tracks[:2]:
            pad = 10
            rx1, ry1 = max(0, tx - pad), max(0, ty - pad)
            rx2, ry2 = min(w, tx + tbw + pad), min(h, ty + tbh + pad)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            r_sub = frame[ry1:ry2, rx1:rx2, 2].astype(np.int16)
            b_sub = frame[ry1:ry2, rx1:rx2, 0].astype(np.int16)
            hs = hsv[ry1:ry2, rx1:rx2, 0]
            ss = hsv[ry1:ry2, rx1:rx2, 1]
            warm_px = int(((hs <= 25) & (ss >= 80) & (r_sub - b_sub >= 45) & (r_sub >= 150)).sum())
            if int(r_sub.max()) >= 235 and warm_px >= 90:
                out.append(Detection(label=FIRE, bbox=(tx, ty, tbw, tbh), score=0.30, source="model_track"))
        return out

    def _detect_dark_chamber_fire(self, frame: np.ndarray, hsv: np.ndarray,
                                  diff: Optional[np.ndarray]) -> List[Detection]:
        """Phat hien dam chay giai doan tan / phong toi bang model phu co kiem chung sang & chuyen dong."""
        if self.dark_sub is None:
            return []
        dark_ratio = float((hsv[::4, ::4, 2] < 50).mean())
        if dark_ratio < 0.70:
            return []

        h, w = frame.shape[:2]
        dark_dets: List[Detection] = []
        for d in self._infer_net(self.dark_sub, frame, conf_fire_override=self.cfg.dark_conf_fire, source_tag="model_dark"):
            if d.label != FIRE:
                continue
            dx, dy, dbw, dbh = d.bbox
            v_sub = hsv[dy:dy + dbh, dx:dx + dbw, 2]
            if int(v_sub.max()) < 200:
                continue
            if diff is not None:
                min_move = max(150, int(h * w * 0.00024))
                if int((diff[dy:dy + dbh, dx:dx + dbw] >= 6).sum()) < min_move:
                    continue
            dark_dets.append(d)
        return dark_dets

    @staticmethod
    def _is_valid_zoom_smoke(frame: np.ndarray, hsv: np.ndarray,
                             bbox: Tuple[int, int, int, int], crop_sz: int) -> bool:
        """Xac thuc box khoi trong vung zoom: chan ao quan, tuong go, rem cua, mai nha
        bang cach kiem tra do lech kenh mau (achromatic) va do bao hoa sac do."""
        h, w = frame.shape[:2]
        bx, by, bw, bh = bbox
        if bw >= crop_sz * 0.70 or bh >= crop_sz * 0.70:
            return False
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        if x2 <= x1 or y2 <= y1:
            return False
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        max_cdiff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
        if float(max_cdiff.mean()) > 8.5:
            return False
        s_sub = hsv[y1:y2, x1:x2, 1]
        if float(s_sub.mean()) > 22.0:
            return False
        return True

    def _detect_motion_zoom_smoke(self, frame: np.ndarray, hsv: np.ndarray,
                                  ctx: FrameContext, p98: float) -> List[Detection]:
        """Motion-ROI Zoom cho KHOI: phat hien cot khoi tu xa boc len tren mai nha / chan troi."""
        if p98 > 8.5 or ctx.motion_mask is None:
            return []

        h, w = frame.shape[:2]
        scale = ctx.scale if ctx.scale > 0 else 1.0
        cnts, _ = cv2.findContours(ctx.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 30:
                continue
            mx, my, mbw, mbh = cv2.boundingRect(c)
            fx, fy = int(mx * scale), int(my * scale)
            fbw, fbh = int(mbw * scale), int(mbh * scale)
            if fy > h * 0.70:
                continue
            rx1, ry1 = max(0, fx), max(0, fy)
            rx2, ry2 = min(w, fx + fbw), min(h, fy + fbh)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            s_mean = float(hsv[ry1:ry2, rx1:rx2, 1].mean())
            if s_mean <= 75.0:
                candidates.append((fx + fbw // 2, fy + fbh // 2, area))

        if not candidates:
            return []

        candidates.sort(key=lambda t: t[2], reverse=True)
        zoom_dets: List[Detection] = []
        zoom_conf = self.cfg.zoom_conf_smoke
        for cx, cy, _ in candidates[:2]:
            found_target = False
            for sz in (300, 360):
                if w < sz or h < sz:
                    continue
                x1 = max(0, min(w - sz, cx - sz // 2))
                y1 = max(0, min(h - sz, cy - sz // 2))
                crop = frame[y1:y1 + sz, x1:x1 + sz]
                for sub in self.sub_models:
                    for d in self._infer_net(sub, crop, conf_fire_override=99.0,
                                             conf_smoke_override=zoom_conf, source_tag="model_zoom"):
                        if d.label != SMOKE:
                            continue
                        full_box = (x1 + d.bbox[0], y1 + d.bbox[1], d.bbox[2], d.bbox[3])
                        if not self._is_valid_zoom_smoke(frame, hsv, full_box, sz):
                            continue
                        zoom_dets.append(Detection(label=SMOKE, bbox=full_box, score=d.score, source="model_zoom"))
                        found_target = True
                if found_target:
                    break
        return zoom_dets

    def _detect_tracked_zoom_smoke(self, frame: np.ndarray, hsv: np.ndarray,
                                   ctx: FrameContext, p90: float) -> List[Detection]:
        """Track-Guided Smoke Recovery: zoom quanh track khoi gan day khi mat dau tam thoi."""
        if not ctx.recent_smoke_tracks or p90 > 7.5:
            return []

        h, w = frame.shape[:2]
        zoom_dets: List[Detection] = []
        zoom_conf = max(0.10, self.cfg.zoom_conf_smoke - 0.03)
        for tx, ty, tbw, tbh in ctx.recent_smoke_tracks[:1]:
            tcx, tcy = tx + tbw // 2, ty + tbh // 2
            sz = 320
            if w < sz or h < sz:
                continue
            x1 = max(0, min(w - sz, tcx - sz // 2))
            y1 = max(0, min(h - sz, tcy - sz // 2))
            crop = frame[y1:y1 + sz, x1:x1 + sz]
            for sub in self.sub_models:
                for d in self._infer_net(sub, crop, conf_fire_override=99.0,
                                         conf_smoke_override=zoom_conf, source_tag="model_track_zoom"):
                    if d.label != SMOKE:
                        continue
                    full_box = (x1 + d.bbox[0], y1 + d.bbox[1], d.bbox[2], d.bbox[3])
                    if not self._is_valid_zoom_smoke(frame, hsv, full_box, sz):
                        continue
                    zoom_dets.append(Detection(label=SMOKE, bbox=full_box, score=d.score, source="model_track_zoom"))
                    break
            if zoom_dets:
                break
        return zoom_dets

    def detect(self, frame: np.ndarray, ctx: FrameContext) -> List[Detection]:
        c = self.cfg
        raw_dets: List[Detection] = []

        for sub in self.sub_models:
            raw_dets.extend(self._infer_net(sub, frame))

        has_fire = any(d.label == FIRE for d in raw_dets)
        has_smoke = any(d.label == SMOKE for d in raw_dets)

        if (not has_fire or not has_smoke) and not ctx.scene_cut:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if ctx.frame_diff is not None:
                p90, p98 = [float(v) for v in np.percentile(ctx.frame_diff[::4, ::4], [90, 98])]
            else:
                p90, p98 = 999.0, 999.0

            if not has_fire:
                smoke_active = ctx.has_active_smoke or has_smoke
                extra_fire: List[Detection] = []
                if smoke_active and ctx.frame_diff is not None:
                    extra_fire = self._detect_text_occluded_fire(frame, hsv, ctx.frame_diff, p98)

                if not extra_fire and ctx.frame_diff is not None:
                    extra_fire = self._detect_motion_zoom_fire(frame, hsv, ctx.frame_diff, p98)

                if not extra_fire and ctx.frame_diff is not None:
                    extra_fire = self._detect_tracked_zoom_fire(frame, hsv, ctx.frame_diff, p90, ctx.recent_fire_tracks)

                if not extra_fire:
                    extra_fire = self._detect_dark_chamber_fire(frame, hsv, ctx.frame_diff)

                if not extra_fire and ctx.frame_diff is not None:
                    extra_fire = self._verify_tracked_fire_glow(frame, hsv, p90, ctx.confirmed_fire_tracks)

                raw_dets.extend(extra_fire)

            if not has_smoke:
                extra_smoke: List[Detection] = []
                if ctx.recent_smoke_tracks:
                    extra_smoke = self._detect_tracked_zoom_smoke(frame, hsv, ctx, p90)
                if not extra_smoke:
                    extra_smoke = self._detect_motion_zoom_smoke(frame, hsv, ctx, p98)
                raw_dets.extend(extra_smoke)

        # Class-wise NMS tong the sau khi gom tat ca detections
        final_dets: List[Detection] = []
        for label in {d.label for d in raw_dets}:
            group = [d for d in raw_dets if d.label == label]
            idx_list = cv2.dnn.NMSBoxes([list(d.bbox) for d in group],
                                        [d.score for d in group], 0.0, c.iou)
            for idx in np.array(idx_list).flatten():
                final_dets.append(group[idx])

        return final_dets
