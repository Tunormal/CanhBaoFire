import argparse
import os
import queue

from .config import AppConfig
from .core.video_job import VideoJob, generate_video_id


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fire & Smoke detector (headless)")
    p.add_argument("--input", required=True, help="Duong dan video vao")
    p.add_argument("--output", default="", help="Duong dan video ra (mac dinh luu vao video output/<video_id>_analyzed.mp4)")
    p.add_argument("--mode", choices=["model", "rule", "both"], default="model", help="Che do phat hien (mac dinh: model)")
    p.add_argument("--model", default="models/fire_smoke_yolov8.onnx", help="Duong dan file model .onnx")
    p.add_argument("--conf", type=float, default=0.35, help="Nguong tin cay tong quat")
    p.add_argument("--conf-fire", type=float, default=0.38, help="Nguong tin cay lua")
    p.add_argument("--conf-smoke", type=float, default=0.25, help="Nguong tin cay khoi")
    p.add_argument("--export-dataset", action="store_true")
    p.add_argument("--index", type=int, default=1, help="So thu tu video (vd: 1 -> '01_', 2 -> '02_')")
    return p


def run_cli(args) -> int:
    cfg = AppConfig(mode=args.mode)
    cfg.model.model_path = args.model
    if args.conf:
        cfg.model.conf = args.conf
    if args.conf_fire:
        cfg.model.conf_fire = args.conf_fire
    if args.conf_smoke:
        cfg.model.conf_smoke = args.conf_smoke
    cfg.export.export_dataset = args.export_dataset

    vid_id = generate_video_id(args.input, index=args.index)
    if args.output:
        out = args.output
    else:
        out_dir = cfg.output_dir if cfg.output_dir else "video output"
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{vid_id}_analyzed.mp4")

    print(f"[*] Bat dau phan tich video:")
    print(f"    - Input:    {args.input}")
    print(f"    - Video ID: {vid_id}")
    print(f"    - Output:   {out}")
    print(f"    - Mode:     {args.mode} (Model: {cfg.model.model_path})")

    q: "queue.Queue" = queue.Queue(maxsize=2)
    job = VideoJob(cfg, args.input, out, q, index=args.index)
    job.start()
    while True:
        ev = q.get()
        if ev[0] == "frame":
            pct = (ev[1] / ev[2] * 100) if ev[2] > 0 else 0
            print(f"\r[Tien do] Frame {ev[1]}/{ev[2]} ({pct:.1f}%)", end="", flush=True)
        elif ev[0] == "error":
            print("\n[LOI]", ev[1])
            return 1
        else:
            print("\n[HOAN TAT]", ev[1])
            return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_cli(build_parser().parse_args()))
