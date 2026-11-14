import cv2


class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Khong mo duoc video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        self.cap.release()


class VideoWriter:
    """Ghi mp4 bang codec mp4v (co san trong opencv-python, khong can ffmpeg)."""

    def __init__(self, path: str, fps: float, size):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise IOError(f"Khong tao duoc file output: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
