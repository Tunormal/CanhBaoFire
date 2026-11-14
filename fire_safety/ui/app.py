import base64
import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ..config import AppConfig
from ..core.video_job import VideoJob, generate_video_id

PREVIEW_W = 760


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fire & Smoke Detector - An toan lao dong (AI Vision)")
        self.geometry("820x910")
        self.events: "queue.Queue" = queue.Queue(maxsize=2)
        self.job = None
        self._photo = None
        self._build()
        self.after(50, self._poll)

    # ---------- build UI ----------
    def _build(self):
        pad = {"padx": 8, "pady": 4}
        self.v_in, self.v_out, self.v_model = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.v_mode = tk.StringVar(value="model")  # Mac dinh su dung model AI

        default_model = "models/fire_smoke_yolov8.onnx"
        if os.path.isfile(default_model):
            self.v_model.set(default_model)

        self.v_fire, self.v_smoke = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.v_motion = tk.BooleanVar(value=True)
        self.v_dataset = tk.BooleanVar(value=False)
        self.v_conf_fire = tk.DoubleVar(value=0.38)  # Nguong AI lua do (chong bat nham mat nguoi, trung)
        self.v_conf_smoke = tk.DoubleVar(value=0.25) # Nguong AI khoi
        self.v_fire_area = tk.DoubleVar(value=0.5)    # per-mille dien tich frame (cho rule)
        self.v_smoke_area = tk.DoubleVar(value=4.0)
        self.v_width = tk.IntVar(value=640)

        f = ttk.LabelFrame(self, text="Video")
        f.pack(fill="x", **pad)
        for r, (text, var, cmd) in enumerate([("Video vao:", self.v_in, self._pick_in),
                                              ("Video ra:", self.v_out, self._pick_out)]):
            ttk.Label(f, text=text).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(f, textvariable=var).grid(row=r, column=1, sticky="ew", **pad)
            ttk.Button(f, text="Chon...", command=cmd).grid(row=r, column=2, **pad)
        f.columnconfigure(1, weight=1)

        m = ttk.LabelFrame(self, text="Phat hien (AI / Rule)")
        m.pack(fill="x", **pad)
        top = ttk.Frame(m)
        top.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(top, text="Che do:").pack(side="left", **pad)
        ttk.Combobox(top, textvariable=self.v_mode, values=["model", "rule", "both"],
                     width=8, state="readonly").pack(side="left")
        ttk.Checkbutton(top, text="Lua", variable=self.v_fire).pack(side="left", **pad)
        ttk.Checkbutton(top, text="Khoi", variable=self.v_smoke).pack(side="left", **pad)
        ttk.Checkbutton(top, text="Yeu cau chuyen dong (Rule)", variable=self.v_motion).pack(side="left", **pad)

        ttk.Label(m, text="Model .onnx:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(m, textvariable=self.v_model).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(m, text="Chon...", command=self._pick_model).grid(row=1, column=2, **pad)
        m.columnconfigure(1, weight=1)

        for r, (text, var, lo, hi, fmt) in enumerate([
                ("Nguong LUA AI (conf):", self.v_conf_fire, 0.05, 0.8, "{:.2f}"),
                ("Nguong KHOI AI (conf):", self.v_conf_smoke, 0.05, 0.8, "{:.2f}"),
                ("Min vung lua rule (‰ frame):", self.v_fire_area, 0.1, 5, "{:.1f}"),
                ("Min vung khoi rule (‰ frame):", self.v_smoke_area, 0.5, 20, "{:.1f}"),
                ("Detect width px (Rule/Motion):", self.v_width, 320, 1280, "{:.0f}")], start=2):
            ttk.Label(m, text=text).grid(row=r, column=0, sticky="w", **pad)
            val_lbl = ttk.Label(m, text=fmt.format(var.get()), width=8)
            def make_cmd(v=var, l=val_lbl, f=fmt):
                return lambda val: l.configure(text=f.format(float(val)))
            scale = ttk.Scale(m, from_=lo, to=hi, variable=var, command=make_cmd())
            scale.grid(row=r, column=1, sticky="ew", **pad)
            val_lbl.grid(row=r, column=2)
        ttk.Checkbutton(m, text="Xuat dataset tu dong (frame + nhan YOLO) de train sau",
                         variable=self.v_dataset).grid(row=7, column=0, columnspan=3, sticky="w", **pad)

        b = ttk.Frame(self)
        b.pack(fill="x", **pad)
        self.btn_start = ttk.Button(b, text="▶ Bat dau phan tich", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(b, text="■ Dung", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(b, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)

        self.preview = ttk.Label(self, text="Preview", anchor="center", relief="sunken")
        self.preview.pack(fill="both", expand=True, **pad)
        ttk.Label(
            self,
            text="Chu thich tren video:  t = Thoi gian phat (giay)  |  f = So thu tu khung hinh (frame)",
            foreground="#444444",
        ).pack(anchor="w", padx=10, pady=(0, 2))
        self.log = tk.Text(self, height=6, state="disabled")
        self.log.pack(fill="x", **pad)

    # ---------- actions ----------
    def _pick_in(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")])
        if p:
            self.v_in.set(p)
            vid_id = generate_video_id(p)
            out_dir = os.path.join(os.getcwd(), "video output")
            os.makedirs(out_dir, exist_ok=True)
            self.v_out.set(os.path.join(out_dir, f"{vid_id}_analyzed.mp4"))

    def _pick_out(self):
        initial_dir = os.path.join(os.getcwd(), "video output")
        os.makedirs(initial_dir, exist_ok=True)
        p = filedialog.asksaveasfilename(initialdir=initial_dir, defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p:
            self.v_out.set(p)

    def _pick_model(self):
        p = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), "models"), filetypes=[("ONNX", "*.onnx")])
        if p:
            self.v_model.set(p)

    def _build_cfg(self) -> AppConfig:
        cfg = AppConfig(mode=self.v_mode.get())
        cfg.fire.enabled, cfg.smoke.enabled = self.v_fire.get(), self.v_smoke.get()
        cfg.fire.require_motion = cfg.smoke.require_motion = self.v_motion.get()
        cfg.fire.min_area_ratio = self.v_fire_area.get() / 1000.0
        cfg.smoke.min_area_ratio = self.v_smoke_area.get() / 1000.0
        cfg.detect_width = int(self.v_width.get())
        cfg.model.conf_fire = float(self.v_conf_fire.get())
        cfg.model.conf_smoke = float(self.v_conf_smoke.get())
        cfg.model.conf = min(cfg.model.conf_fire, cfg.model.conf_smoke)
        if self.v_model.get():
            cfg.model.model_path = self.v_model.get()
        cfg.export.export_dataset = self.v_dataset.get()
        return cfg

    def _start(self):
        if not os.path.isfile(self.v_in.get()):
            messagebox.showwarning("Thieu video", "Hay chon video dau vao.")
            return

        if not self.v_out.get():
            vid_id = generate_video_id(self.v_in.get())
            out_dir = os.path.join(os.getcwd(), "video output")
            os.makedirs(out_dir, exist_ok=True)
            self.v_out.set(os.path.join(out_dir, f"{vid_id}_analyzed.mp4"))

        vid_id = generate_video_id(self.v_in.get())
        self.job = VideoJob(self._build_cfg(), self.v_in.get(), self.v_out.get(), self.events)
        self.btn_start["state"], self.btn_stop["state"] = "disabled", "normal"
        self.progress["value"] = 0
        self._log(f"Bat dau xu ly [ID: {vid_id}] -> {self.v_out.get()} ...")
        self.job.start()

    def _stop(self):
        if self.job:
            self.job.stop()
            self._log("Dang dung ...")

    # ---------- event loop ----------
    def _poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "frame":
                    _, idx, total, img = ev
                    if total:
                        self.progress["value"] = idx * 100.0 / total
                    self._show(img)
                elif ev[0] == "done":
                    self._finish()
                    s = ev[1]
                    self._log(f"HOAN TAT [{s.get('video_id', '')}]: {s['frames']} frame / {s['seconds']}s | "
                              f"LUA: {s['fire_frames']} frame | KHOI: {s['smoke_frames']} frame | "
                              f"Canh bao dau tien: {s['first_alert_s']}s")
                    self._log(f"File video: {s['output']}")
                    if s["dataset_images"]:
                        self._log(f"Dataset: {s['dataset_images']} anh da xuat")
                    self.progress["value"] = 100
                elif ev[0] == "error":
                    self._finish()
                    self._log("LOI: " + ev[1])
                    messagebox.showerror("Loi", ev[1])
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _finish(self):
        self.btn_start["state"], self.btn_stop["state"] = "normal", "disabled"

    def _show(self, bgr):
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (PREVIEW_W, int(h * PREVIEW_W / w)))
        ok, png = cv2.imencode(".png", bgr)
        if ok:
            self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.preview.configure(image=self._photo, text="")

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def launch_ui():
    App().mainloop()
