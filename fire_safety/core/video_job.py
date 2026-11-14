import hashlib
import os
import queue
import re
import threading
import time
from typing import Optional

from ..config import AppConfig
from ..video.draw import draw_detections
from ..video.io import VideoReader, VideoWriter
from .exporter import DatasetExporter, EventLogger
from .pipeline import DetectionPipeline


def generate_video_id(video_path: str, index: int = 1) -> str:
    """Sinh ID va ten gon de doc tu ten file goc (chu Latin), khong dung ma hash ngau nhien:
    Dinh dang: {index:02d}_{clean_latin_slug}
    Vi du: 01_dorm_room_fire, 02_cctv_tv"""
    base = os.path.splitext(os.path.basename(video_path))[0]
    latin_words = re.findall(r"[a-zA-Z0-9]+", base)
    filtered = []
    for w in latin_words:
        wl = w.lower()
        if wl in ("1080p", "720p", "480p", "h264", "x264", "mp4"):
            continue
        filtered.append(wl)

    slug_words = []
    curr_len = 0
    for w in filtered:
        if curr_len + len(w) + 1 > 26:
            break
        slug_words.append(w)
        curr_len += len(w) + 1

    slug = "_".join(slug_words) if slug_words else "video"
    return f"{index:02d}_{slug}"


class VideoJob(threading.Thread):
    """Xu ly 1 video o thread rieng. Giao tiep voi UI/CLI qua queue su kien:
       ("frame", idx, total, annotated_bgr) | ("done", summary_dict) | ("error", msg)"""

    def __init__(self, cfg: AppConfig, input_path: str, output_path: str,
                 events: "queue.Queue", index: int = 1):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.inp = input_path
        self.video_id = generate_video_id(input_path, index=index)

        if not output_path:
            out_dir = cfg.output_dir if cfg.output_dir else "video output"
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"{self.video_id}_analyzed.mp4")

        self.out = output_path
        self.events = events
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.events.put(("error", f"{type(e).__name__}: {e}"))

    def _run(self):
        cfg = self.cfg
        pipeline = DetectionPipeline(cfg)
        reader = VideoReader(self.inp)
        os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
        writer = VideoWriter(self.out, reader.fps, (reader.width, reader.height))

        csv_path = os.path.join(os.path.dirname(os.path.abspath(self.out)), f"{self.video_id}_events.csv")
        logger = EventLogger(csv_path, video_id=self.video_id) if cfg.export.save_events_csv else None
        exporter: Optional[DatasetExporter] = None
        if cfg.export.export_dataset:
            exporter = DatasetExporter(cfg.export, self.video_id,
                                       os.path.dirname(os.path.abspath(self.out)))

        stats = {
            "video_id": self.video_id,
            "frames": 0,
            "fire_frames": 0,
            "smoke_frames": 0,
            "first_alert_s": None
        }
        t0 = time.time()
        try:
            for idx, frame in enumerate(reader.frames()):
                if self._stop_evt.is_set():
                    break
                dets = pipeline.process(frame)
                annotated = draw_detections(frame, dets, idx, reader.fps, video_id=self.video_id)
                writer.write(annotated)

                labels = {d.label for d in dets}
                stats["frames"] += 1
                stats["fire_frames"] += "FIRE" in labels
                stats["smoke_frames"] += "SMOKE" in labels
                if labels and stats["first_alert_s"] is None:
                    stats["first_alert_s"] = round(idx / reader.fps, 2)
                if logger:
                    logger.log(idx, reader.fps, dets)
                if exporter:
                    exporter.maybe_save(idx, frame, dets)

                if idx % max(1, cfg.preview_every_n) == 0:
                    try:
                        self.events.put_nowait(("frame", idx, reader.total, annotated))
                    except queue.Full:
                        pass
        finally:
            reader.release()
            writer.release()
            if logger:
                logger.close()

        stats["seconds"] = round(time.time() - t0, 1)
        stats["output"] = self.out
        stats["dataset_images"] = exporter.count if exporter else 0
        stats["stopped"] = self._stop_evt.is_set()
        self.events.put(("done", stats))
