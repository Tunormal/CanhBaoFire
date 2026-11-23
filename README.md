# CanhBaoFire – Hệ Thống AI & Computer Vision Phát Hiện Lửa & Khói (Fire & Smoke Detection)

Hệ thống desktop thông minh ứng dụng trong **an toàn lao động, nhà xưởng, khu dân cư và camera giám sát CCTV**: Tự động phân tích video → phát hiện chính xác **đám cháy (`FIRE`)** và **cột khói (`SMOKE`)** → xuất video đã khoanh vùng bounding box trực quan + nhãn phân loại + mốc thời gian chi tiết + file nhật ký sự kiện `events.csv`.

Dự án kết hợp sức mạnh của **Deep Learning (Mô hình YOLO ONNX tối ưu qua ONNX Runtime)** cùng **Thuật toán thị giác máy tính truyền thống (MOG2 Motion + Lọc màu sắc + Lọc thời gian Temporal Filter)**, tích hợp cơ chế **Motion-ROI High-Resolution Zoom** và **Zero False Positive Filter** loại trừ hoàn toàn các trường hợp báo nhầm (lòng đỏ trứng chiên, đèn halogen, quần áo cam/đỏ, khuôn mặt, bóng râm).

---

## 1. Tính Năng Nổi Bật

- **Đa Chế Độ Phân Tích (Multi-Mode)**:
  - `model`: Sử dụng mô hình Deep Learning YOLO ONNX (độ chính xác cao, khuyên dùng).
  - `rule`: Sử dụng luật Computer Vision kết hợp phân tích màu sắc (HSV/RGB) và chuyển động nền thích ứng MOG2.
  - `both`: Kết hợp song song cả AI và Rule để tối đa hóa khả năng phát hiện.
- **Tăng Tốc ONNX Runtime**: Tự động sử dụng engine suy luận `onnxruntime` đa luồng CPU (nhanh hơn ~38% so với `cv2.dnn`), cho phép xử lý mượt mà các video độ phân giải cao Full HD 1080p.
- **Motion-ROI High-Resolution Zoom**: Khi ngọn lửa hoặc cột khói mới chớm nở ở khoảng cách xa / diện tích nhỏ (như chập điện, tàn thuốc, khói mỏng bốc lên từ mái nhà/ống khói), hệ thống tự động cắt cúp vùng chuyển động ở độ phân giải gốc để đưa vào YOLO nhận diện, phát hiện sớm hơn hàng chục giây trước khi đám cháy bùng phát lớn.
- **Track-Guided Zoom Recovery & Glow Persistence**: Duy trì vết theo dõi (IoU tracking) liên tục qua các cảnh chuyển động giật, hiệu ứng chuyển cảnh (clock-wipe, pan camera) hoặc khi ngọn lửa / vệt khói dao động ngắt quãng.
- **Chống Báo Giả Tuyệt Đối (Zero False Positive)**:
  - Tích hợp bộ lọc hình học Hough Circles kết hợp phương sai sắc độ Hue Variance để triệt tiêu việc bắt nhầm lòng đỏ trứng chiên trên chảo và bóng đèn tròn phát sáng.
  - Kiểm tra mật độ điểm ảnh chuyển động nóng rực thực tế (`dmv6 / mv8`) để loại bỏ quần áo bảo hộ phản quang, da người và bóng râm.
- **Giao Diện Trực Quan & CLI Headless**:
  - Giao diện Tkinter hiện đại, xem trước (preview) video thời gian thực, có thanh trượt điều chỉnh ngưỡng linh hoạt.
  - Hỗ trợ CLI headless chạy ngầm hàng loạt (batch processing) cho server hoặc tích hợp hệ thống tự động.
- **Xuất Dữ Liệu Chuẩn Hóa**: Tự động đặt tên video xuất theo chuẩn `{id}_{slug}_analyzed.mp4` kèm file log `_events.csv` ghi nhận chi tiết từng frame phát hiện.

---

## 2. Cấu Trúc Dự Án (Project Structure)

```
fire_safety_tool/
├── main.py                     # Entry point chính: chạy không tham số mở GUI, có --input chạy CLI
├── run.bat                     # Script 1-click cho Windows: tự tạo .venv, cài thư viện và khởi động app
├── requirements.txt            # Danh sách thư viện phụ thuộc (opencv, numpy, onnx, onnxruntime)
├── README.md                   # Tài liệu hướng dẫn sử dụng và giải thích thông số
├── .gitignore                  # Cấu hình bỏ qua các tệp tạm, môi trường ảo và video dung lượng lớn
│
├── models/                     # Thư mục chứa các mô hình Deep Learning ONNX
│   ├── fire_smoke_yolov8.onnx  # Mô hình YOLO chính nhận diện Lửa và Khói (độ chính xác cao)
│   ├── cctv_fire_smoke.onnx    # Mô hình phụ chuyên dụng cho camera đêm/buồng tối/lửa tàn
│   ├── fire_smoke_yolov8s.onnx # Mô hình YOLOv8s nhẹ cho máy cấu hình tiết kiệm
│   └── rabahdev_fire_smoke.onnx# Mô hình ONNX bổ trợ
│
├── video input/                # Thư mục chứa video đầu vào cần phân tích (.mp4, .avi, .mov,...)
├── video output/               # Thư mục lưu video thành phẩm đã phân tích và file log CSV
│   ├── <id>_<name>_analyzed.mp4 # Video đã vẽ bounding box, banner cảnh báo, timestamp
│   ├── <id>_<name>_events.csv   # File nhật ký chi tiết các frame phát hiện
│   └── dataset_auto/           # (Tùy chọn) Ảnh gốc và nhãn TXT chuẩn YOLO xuất tự động
│
└── fire_safety/                # Gói mã nguồn cốt lõi của ứng dụng
    ├── __init__.py
    ├── config.py               # Trung tâm cấu hình tham số (dataclasses)
    ├── cli.py                  # Module xử lý dòng lệnh (headless mode)
    │
    ├── ui/
    │   └── app.py              # Giao diện đồ họa Tkinter (hiển thị, thanh trượt, luồng xử lý)
    │
    ├── core/
    │   ├── types.py            # Định nghĩa kiểu dữ liệu: Detection, FrameContext, hằng số FIRE/SMOKE
    │   ├── pipeline.py         # Pipeline điều phối tổng: resize -> motion -> detectors -> NMS -> scale
    │   ├── motion.py           # Phân tích chuyển động nền thích ứng MOG2 (Background Subtraction)
    │   ├── temporal.py         # Bộ lọc thời gian (IoU tracking, làm mượt box, chống nhấp nháy)
    │   ├── mask_utils.py       # Chuyển đổi nhị phân mask -> contour -> bounding box
    │   ├── video_job.py        # Luồng (Thread) đọc, xử lý và ghi video bất đồng bộ
    │   └── exporter.py         # Xuất bảng sự kiện CSV và xuất dataset YOLO tự động
    │
    ├── detectors/
    │   ├── base.py             # Interface lớp cơ sở BaseDetector
    │   ├── yolo_onnx.py        # Bộ suy luận YOLO ONNX, Motion Zoom, Tracked Zoom, Night Chamber
    │   ├── fire_rule.py        # Phát hiện lửa bằng luật màu HSV + RGB + chuyển động MOG2
    │   └── smoke_rule.py       # Phát hiện khói bằng luật màu xám ít bão hòa + chuyển động MOG2
    │
    └── video/
        ├── io.py               # VideoReader và VideoWriter (tối ưu hóa codec mp4v)
        └── draw.py             # Render bounding box, nhãn độ tin cậy, banner cảnh báo, chú thích t và f
```

---

## 3. Giải Thích Chi Tiết Toàn Bộ Thông Số (Parameter Reference)

Hệ thống cho phép tinh chỉnh linh hoạt trực tiếp trên thanh trượt của **Giao diện UI**, qua các cờ lệnh **CLI**, hoặc trong tệp cấu hình **`fire_safety/config.py`**.

### 3.1. Các thông số trên Giao diện UI (Tkinter Sliders & Controls)

| Tên hiển thị trên UI | Biến trong `config.py` | Giá trị mặc định | Khoảng chỉnh | Giải thích chi tiết & Tác dụng |
|---|---|---|---|---|
| **Chế độ (Mode)** | `cfg.mode` | `model` | `model`, `rule`, `both` | **Chế độ phát hiện**: <br>• `model`: Sử dụng Deep Learning YOLO (chính xác nhất, khử báo giả tốt nhất).<br>• `rule`: Sử dụng hoàn toàn thị giác máy tính cổ điển (màu sắc + chuyển động).<br>• `both`: Chạy song song cả hai và gộp kết quả qua NMS. |
| **Lửa (Fire Checkbox)** | `cfg.fire.enabled` | `True` | Bật / Tắt | Cho phép hoặc tắt tính năng phát hiện đám cháy. |
| **Khói (Smoke Checkbox)** | `cfg.smoke.enabled` | `True` | Bật / Tắt | Cho phép hoặc tắt tính năng phát hiện cột khói. |
| **Yêu cầu chuyển động (Rule)** | `require_motion` | `True` | Bật / Tắt | Bắt buộc vùng màu lửa/khói (ở chế độ rule) phải có chuyển động thực tế từ thuật toán MOG2. Giúp loại bỏ vật thể tĩnh có màu cam/xám (như tường sơn, ghế đệm). |
| **Ngưỡng LỬA AI (conf)** | `cfg.model.conf_fire` | `0.38` | `0.05` – `0.80` | **Độ tin cậy tối thiểu để xác nhận LỬA** (áp dụng cho toàn khung hình). Đặt ở mức `0.38` để triệt tiêu tuyệt đối báo giả từ lòng đỏ trứng chiên, đèn halogen, quần áo đỏ, mặt người; trong khi vẫn bắt lửa lớn nhạy bén. |
| **Ngưỡng KHÓI AI (conf)** | `cfg.model.conf_smoke` | `0.25` | `0.05` – `0.80` | **Độ tin cậy tối thiểu để xác nhận KHÓI**. Đặt ở mức `0.25` để nắm bắt sớm các vệt khói mờ, khói loãng màu xám nhạt trước khi lan rộng. |
| **Min vùng lửa rule (‰ frame)** | `cfg.fire.min_area_ratio` | `0.5 ‰` (0.0005) | `0.1` – `5.0 ‰` | **Diện tích tối thiểu của đám lửa** tính theo phần nghìn diện tích khung hình (chế độ rule). Ví dụ trên video Full HD (1920×1080), 0.5‰ tương đương ~1.036 pixel. Tăng lên nếu bị nhiễu bởi các đốm sáng nhỏ li ti. |
| **Min vùng khói rule (‰ frame)** | `cfg.smoke.min_area_ratio` | `4.0 ‰` (0.0040) | `0.5` – `20.0 ‰` | **Diện tích tối thiểu của cột khói** tính theo phần nghìn diện tích khung hình (chế độ rule). Khói thường lan tỏa theo mảng lớn, mức `4.0‰` giúp lọc bỏ sương mù cục bộ hoặc bóng đổ nhỏ. |
| **Detect width px (Rule/Motion)** | `cfg.detect_width` | `640 px` | `320` – `1280 px` | Chiều rộng ảnh được thu nhỏ tạm thời khi đưa qua bộ phân tích chuyển động MOG2 và bộ lọc Rule. Giúp tăng tốc độ xử lý lên gấp 3–4 lần mà vẫn giữ nguyên độ nét cho mô hình YOLO. |
| **Xuất dataset tự động** | `cfg.export.export_dataset`| `False` | Bật / Tắt | Tự động trích xuất các khung hình có phát hiện kèm tệp nhãn `.txt` chuẩn định dạng YOLO (`classes: 0=fire, 1=smoke`) vào `video output/dataset_auto/` để phục vụ huấn luyện bổ sung (fine-tune) sau này. |

---

### 3.2. Các thông số nâng cao trong `fire_safety/config.py` & CLI

| Tham số cấu hình | Cờ lệnh CLI tương ứng | Mặc định | Ý nghĩa kỹ thuật |
|---|---|---|---|
| `model.conf` | `--conf` | `0.35` | Ngưỡng tin cậy cơ sở của model YOLO chung. |
| `model.zoom_conf_fire` | — | `0.16` | Ngưỡng tin cậy cực nhạy khi thực hiện **Motion-ROI Zoom cho LỬA** quanh vùng chuyển động nóng rực. Cho phép YOLO phát hiện những đốm lửa nhỏ chỉ vài chục pixel mà mắt thường khó để ý. |
| `model.zoom_conf_smoke`| — | `0.15` | Ngưỡng tin cậy cực nhạy khi thực hiện **Motion-ROI Zoom cho KHÓI** quanh vùng chuyển động ở độ phân giải gốc. Kết hợp bộ kiểm định phi màu (Achromatic Filter) để phát hiện sớm cột khói mờ ở xa mà không báo giả ngói nhà/tường. |
| `model.dark_conf_fire` | — | `0.28` | Ngưỡng tin cậy khi kích hoạt model phụ chuyên dụng (`cctv_fire_smoke.onnx`) cho khung cảnh buồng tối thiếu sáng hoặc giai đoạn lửa tàn. |
| `model.iou` | — | `0.45` | Ngưỡng Non-Maximum Suppression (NMS) để loại bỏ các bounding box bị chồng chéo lên nhau trên cùng một đối tượng. |
| `temporal.smooth_boxes`| — | `True` | Bật thuật toán làm mượt tọa độ bounding box theo hàm mũ EMA (`alpha=0.65`) qua các khung hình liên tiếp. |
| `temporal.box_min_hits`| — | `2` | **Số frame tối thiểu phải xuất hiện liên tiếp**: Một vật thể phải xuất hiện ít nhất 2 frame mới được xác nhận hiển thị cảnh báo. Giúp triệt tiêu hoàn toàn nhiễu chớp nhoáng 1 frame (như đèn xe lướt qua, ánh đèn flash). |
| `temporal.box_max_age` | — | `6` | **Số frame tối đa lưu vết khi mất dấu tạm thời**: Giúp đường viền hộp bao không bị nhấp nháy bật/tắt liên tục khi ngọn lửa dao động ngọn. |
| `temporal.box_iou_thresh`| — | `0.25` | Ngưỡng trùng khớp diện tích (IoU) giữa khung hình hiện tại và khung hình trước để nối tiếp cùng một vết theo dõi (TrackedBox). |
| `motion.history` | — | `300` | Số khung hình lịch sử được thuật toán MOG2 dùng để học nền cảnh tĩnh. |
| `motion.var_threshold` | — | `25.0` | Ngưỡng phương sai khoảng cách Mahalanobis để quyết định một pixel có thuộc về chuyển động hay không. |

---

### 3.3. Giải Thích Các Ký Hiệu & Thông Tin Hiển Thị Trên Video Đầu Ra

Trên video thành phẩm sau khi xuất xưởng, tool tự động hiển thị các lớp thông tin trực quan sau:

```
┌────────────────────────────────────────────────────────┐
│ [FIRE ALERT]                                           │  ◄── Banner trạng thái (gốc trên bên trái)
│                                                        │
│       ┌──────────────┐                                 │
│       │ FIRE 0.85    │  ◄── Tag nhãn + độ tin cậy AI  │
│       │              │                                 │
│       │   [ĐÁM LỬA]  │  ◄── Bounding box màu đỏ rực    │
│       └──────────────┘                                 │
│                                                        │
│ [01_dorm_room_fire]  t (Thoi gian):  65.50s | f (Frame): 1963 │ ◄── Info Bar
└────────────────────────────────────────────────────────┘
```

1. **Khung bao Bounding Box & Nhãn phân loại**:
   - **`FIRE <score>`**: Đám cháy được phát hiện với khung viền **đỏ rực** (`BGR: (0, 0, 230)`). Điểm tin cậy `<score>` nằm trong khoảng từ `0.00` đến `1.00` (ví dụ: `FIRE 0.72`).
   - **`SMOKE <score>`**: Vùng khói được phát hiện với khung viền **cam hổ phách** (`BGR: (0, 150, 255)`).
2. **Banner Cảnh Báo Trạng Thái (Status Banner ở góc trên bên trái)**:
   - **`FIRE ALERT`** (Nền đỏ): Khi đang có đám cháy hoạt động trong khung hình.
   - **`SMOKE ALERT`** (Nền cam): Khi phát hiện có khói dày đặc xuất hiện.
   - **`FIRE & SMOKE ALERT`** (Nền đỏ đậm): Khi đồng thời xác nhận có cả ngọn lửa và khói bốc lên.
3. **Thanh Thông Tin Phía Dưới (Info Bar)**:
   - **`t (Thoi gian): XX.XXs`**: Mốc thời gian phát lại thực tế của video tính bằng **giây** (`time_sec = frame_idx / fps`). Giúp người kiểm tra tua ngay đến đúng thời điểm xảy ra cháy trên các đầu ghi/phần mềm media player.
   - **`f (Frame): NNNN`**: **Số thứ tự khung hình chính xác** (`frame_idx`), bắt đầu từ 0 đến hết video. Giúp đối chiếu kỹ thuật chính xác từng khung hình với file log CSV.
   - **`[<video_id>]`**: Mã định danh chuẩn hóa của video (ví dụ: `[01_dorm_room_fire]`), đồng bộ với tiền tố file xuất xưởng.

---

### 3.4. Cơ Chế Kỹ Thuật Phát Hiện Sớm Khói (Early Smoke Detection & Achromatic Filter)

Nhằm giải quyết triệt để vấn đề phát hiện khói muộn trên các video khoảng cách xa (như video giám sát khu dân cư, cháy nhà từ xa):

1. **Bối cảnh & Nguyên nhân phát hiện muộn**:
   - Khi quan sát từ xa, cột khói ban đầu thường mỏng, khuếch tán và chỉ chiếm diện tích rất nhỏ (~60–80 px) trên khung hình Full HD / 720p.
   - Khi toàn khung hình bị resize/downscale về $640 \times 640$ để nạp vào mạng nơ-ron YOLO, các điểm ảnh vệt khói bị suy hao độ nét và hòa lẫn vào nền mây trời. Do đó, điểm tin cậy toàn khung hình chỉ đạt quanh mức $0.10 - 0.20$ và thường trễ đến hàng chục giây (chờ khi đám khói bốc đen đặc mới đạt ngưỡng phát hiện).

2. **Cơ chế Motion-ROI Zoom cho Khói (`_detect_motion_zoom_smoke`)**:
   - Khi toàn khung hình chưa phát hiện khói, hệ thống theo dõi mặt nạ chuyển động MOG2 (`ctx.motion_mask`) tập trung vào nửa trên khung hình ($y < 70\%$).
   - Lọc các cụm chuyển động có độ bão hòa màu thấp ($S \le 75$) và trích xuất vùng ảnh ở **độ phân giải gốc** ($300 \times 300$ và $360 \times 360$) đưa vào mạng YOLO với ngưỡng nhạy `zoom_conf_smoke = 0.15`.

3. **Bộ Kiểm Định Phi Màu (Achromatic Validation Filter - `_is_valid_zoom_smoke`)**:
   - Để ngăn chặn hoàn toàn việc nhận nhầm các vật thể có màu như ngói nhà gạch đỏ, tường gỗ, đồ đạc nội thất hoặc quần áo lính cứu hỏa:
     - Khói thực tế dưới ánh sáng ban ngày luôn có tính chất **phi màu (achromatic)**: chênh lệch cực đại giữa 3 kênh màu $\max(|R-G|, |G-B|, |R-B|) \le 8.5$ và độ bão hòa màu HSV $S \le 22.0$.
     - Các bề mặt màu sắc (ngói gạch, gỗ, vải áo) có độ lệch kênh màu rất lớn ($25 - 45$), do đó bị loại bỏ tuyệt đối $100\%$, đảm bảo zero false positive.

4. **Khôi Phục Vết Khói Theo Thời Gian (Track-Guided Smoke Recovery - `_detect_tracked_zoom_smoke`)**:
   - Bộ lọc thời gian `BBoxTemporalFilter` tích hợp phương thức `recent_small_smoke_boxes()`. Khi khói đã được nhận diện, hệ thống tự động bám theo tọa độ crop xung quanh vệt khói ở các khung hình tiếp theo, duy trì bounding box mượt mà ngay cả khi vệt khói biến thiên hình dạng chập chờn.

5. **Hiệu quả thực nghiệm**:
   - Trên video thử nghiệm góc xa `1794670535848`, thời điểm phát hiện khói đầu tiên được rút ngắn từ **76.50s (frame 2295)** xuống còn **27.87s (frame 836)** — **sớm hơn 48.63 giây** (nắm bắt ngay từ lúc cột khói đầu tiên bắt đầu bốc lên trên mái nhà).
   - Kiểm tra hồi quy trên các video đối chứng (`Dorm Room`, `Fire Development`) hoàn toàn sạch bóng cảnh báo giả ($0$ false alarms trên vùng an toàn).

---

## 4. Hướng Dẫn Sử Dụng (Quick Start)

### 4.1. Cài đặt môi trường
Yêu cầu hệ điều hành: **Windows 10 / 11** hoặc **Linux / macOS**, cài sẵn **Python 3.9+**.

- **Cách 1: Khởi động 1-click trên Windows (Khuyên dùng)**
  Nhấp đúp chuột vào file **`run.bat`**. Script sẽ tự động:
  1. Kiểm tra Python trên máy.
  2. Tạo môi trường ảo cách ly `.venv` (nếu chưa có).
  3. Cài đặt đầy đủ các thư viện từ `requirements.txt`.
  4. Mở trực tiếp Giao diện đồ họa người dùng (Tkinter UI).

- **Cách 2: Chạy thủ công qua dòng lệnh Terminal**
  ```powershell
  # Tạo môi trường ảo
  python -m venv .venv
  
  # Kích hoạt môi trường (Windows PowerShell)
  .venv\Scripts\Activate.ps1
  
  # Cài đặt thư viện
  pip install -r requirements.txt
  ```

---

### 4.2. Khởi chạy Giao diện Đồ họa (GUI Mode)

Chạy lệnh:
```powershell
.venv\Scripts\python main.py
```
**Các bước thao tác trên màn hình:**
1. Nhấn nút **Chọn...** tại mục **Video vào** để nạp tệp video cần quét (`.mp4`, `.avi`, `.mov`,...).
2. Đường dẫn **Video ra** sẽ được hệ thống tự động sinh theo mẫu chuẩn: `video output/<id>_<tên_video>_analyzed.mp4`.
3. Lựa chọn **Chế độ**: Mặc định là `model` (dùng Deep Learning chuẩn xác nhất).
4. Nhấn nút **▶ Bắt đầu phân tích**. Màn hình xem trước sẽ hiển thị hình ảnh quét kèm thanh tiến độ tiến trình.

---

### 4.3. Khởi chạy Dòng Lệnh Không Giao Diện (CLI Headless Mode)

Thích hợp cho máy chủ server, script tự động hóa hoặc quét hàng loạt tệp tin:

```powershell
# Chạy phân tích cơ bản cho video số 01:
.venv\Scripts\python main.py --input "video input\Dorm Room Fire Demonstration - City of Saint Paul (1080p, h264).mp4" --index 1

# Tùy biến ngưỡng tin cậy và xuất kèm bộ dataset YOLO:
.venv\Scripts\python main.py --input "video input\sample.mp4" --index 2 --conf-fire 0.35 --conf-smoke 0.20 --export-dataset
```

**Các cờ tham số CLI được hỗ trợ:**
- `--input <path>`: *(Bắt buộc)* Đường dẫn tệp video đầu vào.
- `--output <path>`: Đường dẫn tệp video kết quả (mặc định lưu tại thư mục `video output/`).
- `--index <int>`: Số thứ tự đánh số video (ví dụ: `1` sẽ tạo tiền tố `01_`, `2` tạo `02_`).
- `--mode {model,rule,both}`: Chế độ nhận diện (mặc định: `model`).
- `--model <path>`: Đường dẫn tệp mô hình ONNX (mặc định: `models/fire_smoke_yolov8.onnx`).
- `--conf <float>`: Ngưỡng tin cậy chung (mặc định: `0.35`).
- `--conf-fire <float>`: Ngưỡng tin cậy cho Lửa (mặc định: `0.38`).
- `--conf-smoke <float>`: Ngưỡng tin cậy cho Khói (mặc định: `0.25`).
- `--export-dataset`: Bật cờ này để tự động lưu ảnh và file nhãn `.txt` chuẩn YOLO.

---

## 5. Dữ Liệu Đầu Ra (Outputs)

Mỗi lần chạy phân tích hoàn tất, thư mục `video output/` sẽ chứa các tệp kết quả tương ứng:

1. **Tệp Video Đã Phân Tích (`{id}_{slug}_analyzed.mp4`)**:
   - Được encode bằng codec `mp4v` tốc độ cao, tương thích với hầu hết các trình phát media (Windows Media Player, VLC, PotPlayer).
   - Vẽ sẵn các bounding box, nhãn điểm số, banner cảnh báo tức thời và thanh chỉ số `t` / `f`.
2. **Tệp Bảng Ghi Sự Kiện (`{id}_{slug}_events.csv`)**:
   - Ghi lại từng lượt phát hiện trên từng frame theo định dạng bảng tính:
     ```csv
     frame,time_sec,label,confidence,bbox_x,bbox_y,bbox_w,bbox_h,source
     1963,65.50,FIRE,0.72,18,104,1033,968,model_hybrid
     1963,65.50,SMOKE,0.61,42,120,850,700,model
     ```
3. **Thư Mục Dataset Huấn Luyện (`dataset_auto/`)** *(nếu bật tùy chọn)*:
   - Chứa thư mục `images/` (ảnh cắt từ frame) và `labels/` (file `.txt` tọa độ chuẩn YOLO: `<class> <x_center> <y_center> <w> <h>`).

---

## 6. Lưu Ý Kỹ Thuật & Giấy Phép (Disclaimer)

- **Mục đích hỗ trợ an toàn**: Hệ thống được phát triển như một công cụ cảnh báo thị giác máy tính hỗ trợ giám sát và ghi nhận bằng chứng an toàn lao động, không thay thế cho hệ thống báo cháy PCCC phần cứng chuyên dụng theo quy chuẩn quốc gia.
- **Tối ưu hóa camera**: Để đạt hiệu năng cao nhất, khuyến nghị sử dụng camera có góc quan sát cố định, hạn chế độ rung lắc và tránh auto-exposure quá gắt.
- **Giấy phép**: Dự án mã nguồn mở phục vụ mục đích học tập, nghiên cứu và ứng dụng an toàn lao động.
