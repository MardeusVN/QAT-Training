# Báo cáo Quantization-Aware Training (QAT) — VITS2+BigVGAN

**Ngày:** 2026-08-27 (cập nhật 2026-08-28)
**Mục tiêu:** Chuyển checkpoint FP32 sang INT8 cho triển khai CPU edge (ràng buộc 1 core / 2 threads, ví dụ Intel N150), dùng ONNX Runtime.

Có 2 checkpoint FP32 được tìm thấy trong dự án và cả hai đều đã được xử lý:

| | BanhmiTTS_v1 | Piper_baseline |
|---|---|---|
| Nguồn FP32 | `best-epoch=1079-val_loss_mel=19.2943.ckpt` | `best-epoch=773-val_loss_mel=20.2158.ckpt` |
| Trạng thái | ✅ Hoàn tất | ✅ Hoàn tất |
| WER (n=500) | **7.17%** | 7.84% |
| UTMOS (n=500) | **3.29** | 3.01 |
| RTF | 0.111 | **0.093** |
| Size | 46.4MB | **39.9MB** |

→ BanhmiTTS_v1 thắng về chất lượng (WER, UTMOS); Piper_baseline thắng về tốc độ/dung lượng. Chi tiết đầy đủ ở mục 2 và 3.

---

## 1. Phát hiện quan trọng: 2 checkpoint có kiến trúc khác nhau

Ban đầu tưởng đây là 2 checkpoint của cùng một model (chỉ khác epoch), nhưng kiểm tra graph ONNX xuất ra cho thấy **chúng là hai kiến trúc khác nhau**:

| | BanhmiTTS_v1 (epoch1079) | Piper_baseline (epoch773) |
|---|---|---|
| Tổng lớp Conv/ConvTranspose | 160 | 132 |
| enc_p (text encoder) | 37 | 37 |
| dp (duration predictor) | 80 | 32 |
| dec (BigVGAN decoder) | 23 | 23 |
| flow (normalizing flow) | ~64 | 40 |
| f0_predictor | có (3 lớp) | **không có** |

→ Piper_baseline có flow nông hơn nhiều và **thiếu hẳn module F0 predictor**. Đây gần như chắc chắn là một thử nghiệm kiến trúc cũ/khác, không phải cùng dòng training với epoch1079.

Hệ quả quan trọng nhất: **layer nào "nhạy cảm" khi quantize hoá ra khác nhau hoàn toàn giữa hai kiến trúc** — không thể copy nguyên công thức loại trừ từ model này sang model kia (xem mục 3).

---

## 2. BanhmiTTS_v1 (epoch1079) — deliverable chính

### Layer bị loại trừ khi quantize (không quantize)
- **`flow`** (normalizing flow, ~64 lớp Conv, 32.9% tổng tham số của model_g)
- **`dec.conv_post`** (lớp Conv1d cuối cùng trước tanh, quyết định trực tiếp giá trị sample của waveform)

### Lý do
- **flow**: là invertible transform gồm 4 coupling layer nối tiếp — sai số lượng tử hóa dồn tích qua từng layer (error compounding). Đây là thứ **gây sụt UTMOS nhiều nhất** trong mọi thử nghiệm loại trừ đã làm cho kiến trúc này.
- **conv_post**: layer nhỏ (1 lớp) nhưng theo tài liệu mixed-precision quantization, layer đầu/cuối luôn nhạy cảm bất thường. Loại trừ nó tốn rất ít dung lượng (1 layer) nhưng cứu được một phần chất lượng đáng kể.

### Layer được quantize
`enc_p` (37) + `dp` (80) + `dec` trừ conv_post (22) + `f0_predictor` (3) = **95/160 lớp Conv/ConvTranspose** (59%).

### Quy trình QAT
- Wrap 143 lớp (Conv1d/ConvTranspose1d/Linear, tính cả Linear nên nhiều hơn con số Conv/ConvTranspose ở trên) tại `enc_p`, `dp`, `dec` (trừ `conv_post`), `f0_predictor`.
- LR: 5e-6 → 5e-7 (giảm dần 10x qua 150-211 epoch), discriminator warmup 1 epoch đầu.
- Train qua 3 lần chạy nối tiếp: run1 → run2 → run3 → run3_ext (resume liên tục, tổng ~150 epoch = ~60,000 bước, gần đạt mốc 10% của 844,560 bước training gốc theo khuyến nghị NVIDIA).
- **Best checkpoint: epoch 62** (val_loss_mel=20.7220). Từ epoch 62 → 150 (88 epoch thêm) không có cải thiện nào — training đã bão hòa, dừng ở epoch 150.

### Kết quả kiểm chứng (n=500 câu thật, Whisper WER + UTMOSv2)

| | WER | CER | UTMOS | RTF | Size |
|---|---|---|---|---|---|
| FP32 gốc | 7.68% | — | 3.85 | 0.110 | 70.6MB |
| PTQ thuần (không QAT) | 7.63% | 4.31% | 3.18 | 0.110 | 46.4MB |
| **QAT (epoch 62)** | **7.17%** | **4.24%** | **3.29** | 0.111 | 46.4MB |

→ QAT thắng PTQ thuần trên cả WER lẫn UTMOS, tốc độ không đổi.

**File giao:**
- `qat_transfer/voice.int8.qat_final.onnx` (INT8, deliverable chính)
- `qat_transfer/BanhmiTTS_v1_qat_best_checkpoint.ckpt` (checkpoint PyTorch gốc, epoch 62)

---

## 3. Piper_baseline (epoch773) — quy trình phát hiện & sửa lỗi

### Bước 1 — áp nguyên công thức của epoch1079 → THẤT BẠI

Thử loại trừ `flow` + `conv_post` (giống hệt BanhmiTTS_v1), quantize phần còn lại (`enc_p`+`dp`+`dec`+giả sử có f0_predictor):

- Train QAT 92 epoch với scope này → checkpoint best val_loss_mel=21.3995 (số liệu training bình thường, không có dấu hiệu bất ổn).
- Nhưng khi export INT8 và đo thật: **WER = 91.35%, UTMOS = 1.74** — audio gần như hỏng hoàn toàn.
- Kiểm tra chéo: PTQ thuần (không train QAT gì cả) với cùng cấu hình loại trừ flow+conv_post trên chính FP32 gốc cũng cho **WER = 97.3%, UTMOS = 1.68** — chứng minh đây **không phải lỗi do quá trình QAT training**, mà lỗi nằm ở việc **loại trừ sai layer** cho kiến trúc này.
- FP32 gốc (chưa quantize gì) hoàn toàn bình thường: WER=10.0%, UTMOS=3.36 — xác nhận checkpoint gốc không hề hỏng.

### Bước 2 — khảo sát thực nghiệm tìm layer nhạy cảm thật sự

Chạy PTQ với 12 tổ hợp loại trừ khác nhau (mỗi lần chỉ mất ~5-10 phút vì không cần train), đo WER trên 10-30 câu:

| Cấu hình loại trừ | Số lớp loại trừ | WER |
|---|---|---|
| Không loại trừ gì | 0/132 | 107% ❌ |
| Chỉ `flow` | 40/132 | 99% ❌ |
| `flow` + `conv_post` (copy epoch1079) | 41/132 | 93% ❌ |
| Chỉ `dp` | 32/132 | 96% ❌ |
| Chỉ `dec` | 23/132 | 105% ❌ |
| Chỉ `enc_p` | 37/132 | 25% ⚠️ |
| `flow` + `enc_p` | 77/132 | 26% ⚠️ |
| `flow` + `dp` | 72/132 | 99% ❌ |
| `flow` + `dec` | 63/132 | 100% ❌ |
| `flow` + `enc_p` + `dp` | 109/132 | 4.0% ✅ |
| **`enc_p` + `dp`** | 69/132 | 3.8% ✅ |
| **`enc_p` + `dp` + `conv_post`** | 70/132 | 3.8% ✅ |

→ Kết luận: với kiến trúc epoch773, layer nhạy cảm là **`enc_p` (text encoder) và `dp` (duration predictor)** — **ngược hoàn toàn** với epoch1079 (nơi `flow` mới là thủ phạm). `flow` và `dec` ở kiến trúc này quantize hoàn toàn bình thường, không gây hỏng.

### Layer bị loại trừ khi quantize (cấu hình cuối, đã chọn)
- **`enc_p`** (text encoder, 37 lớp)
- **`dp`** (stochastic duration predictor, 32 lớp)
- **`dec.conv_post`** (giữ nguyên lý do như epoch1079: layer cuối trước tanh)

### Lý do (khác epoch1079)
Không rõ nguyên nhân sâu xa tuyệt đối (không có thời gian phân tích activation distribution chi tiết), nhưng có thể liên quan tới việc kiến trúc epoch773 **thiếu f0_predictor** — có thể enc_p/dp trong biến thể này gánh vai trò biểu diễn thông tin nhạy cảm hơn (không có nhánh F0 riêng để giảm tải), khiến chúng dễ vỡ hơn khi lượng tử hóa. Đây là **kết luận rút ra từ thực nghiệm (empirical)**, không phải suy luận lý thuyết trước.

### Layer được quantize
`flow` (40) + `dec` trừ conv_post (22) = **62/132 lớp** (47%).

### Kết quả PTQ (n=30, đã kiểm chứng)

| | WER | CER | UTMOS | RTF | Size |
|---|---|---|---|---|---|
| FP32 gốc | 10.0% | 5.9% | 3.36 | 0.093 | 63.6MB |
| **PTQ (cấu hình đúng)** | **8.53%** | **4.79%** | **2.80** | 0.093 | 39.9MB |

### QAT training (hoàn tất)
Sau khi tìm được cấu hình đúng, khởi động lại QAT training với scope khớp chính xác cấu hình export:
- Wrap **62 lớp** tại `flow` (40) + `dec` trừ conv_post (22) — **không đụng tới `enc_p`/`dp`** vì chúng luôn ở FP32 khi export.
- Cùng LR schedule/discriminator warmup như BanhmiTTS_v1, chạy đủ 150 epoch (~10 giờ, nhanh hơn ~35% so với BanhmiTTS_v1 vì chỉ wrap 62 lớp thay vì 143).
- **Best checkpoint: epoch 134** (val_loss_mel=21.3403). Từ epoch 134→149 không cải thiện thêm — cùng pattern bão hòa như BanhmiTTS_v1.

### Kết quả cuối cùng (n=500, so sánh công bằng với BanhmiTTS_v1)

| | WER | CER | UTMOS | RTF | Size |
|---|---|---|---|---|---|
| FP32 gốc | 10.0% | 5.9% | 3.36 | 0.093 | 63.6MB |
| PTQ thuần (cấu hình đúng, chưa QAT) | 8.53% | 4.79% | 2.80 | 0.093 | 39.9MB |
| **QAT (epoch 134)** | **7.84%** | **4.44%** | **3.01** | **0.093** | 39.9MB |

→ QAT cải thiện đáng kể so với PTQ thuần (UTMOS 2.80→3.01, WER 8.53%→7.84%), đúng hướng kỳ vọng — nhưng **không vượt qua được BanhmiTTS_v1** (UTMOS 3.29, WER 7.17%), vì FP32 gốc epoch773 vốn đã kém hơn epoch1079 ngay từ đầu. Đổi lại, Piper_baseline **nhanh hơn ~16%** (RTF 0.093 vs 0.111) và **nhẹ hơn ~14%** (39.9MB vs 46.4MB) — có thể là lựa chọn hợp lý nếu ưu tiên tốc độ/dung lượng hơn chất lượng tuyệt đối.

**File giao (bản QAT cuối cùng):**
- `qat_transfer/voice.int8.piper_baseline_final.onnx` (INT8, deliverable chính)
- `qat_transfer/data_preprocessed/qat_finetune_piper_baseline/lightning_logs/version_0/checkpoints/qat-best-epoch=134-val_loss_mel=21.3403.ckpt` (checkpoint PyTorch gốc)

---

## 4. Bài học rút ra

1. **"Layer nhạy cảm với quantization" không phải thuộc tính cố định của kiến trúc VITS nói chung — nó riêng cho từng model cụ thể.** Không thể giả định flow luôn là thủ phạm; phải đo thực nghiệm cho từng checkpoint/kiến trúc.
2. **val_loss_mel không phản ánh chất lượng audio thực tế sau quantize.** Cả hai lần thất bại (epoch773 với cấu hình sai) đều có val_loss_mel hoàn toàn bình thường trong lúc training — chỉ WER/UTMOS đo trên audio thật mới lộ ra vấn đề.
3. **QAT không sửa được lỗi chọn sai layer để loại trừ.** 92 epoch QAT training trên cấu hình sai (wrap enc_p+dp+dec, loại trừ flow) cho kết quả tệ y hệt PTQ thuần cùng cấu hình — chứng tỏ vấn đề nằm ở *chọn sai layer*, không phải *thiếu training*.
4. Khảo sát PTQ (không cần train) là cách rẻ và nhanh nhất để tìm cấu hình loại trừ đúng trước khi đầu tư nhiều giờ GPU vào QAT training.
