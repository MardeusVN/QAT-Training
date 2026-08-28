# Báo cáo Quantization-Aware Training (QAT) giữa Piper và BanhmiTTS_v1 (VITS2+BigVGAN+F0)

**Ngày:** 2026-08-27 (cập nhật 2026-08-29)

**Mục tiêu:** Chuyển best checkpoint của từng model từ **FP32** sang **INT8** để triển khai trên **CPU Edge** (giới hạn **1 core 2 threads**, ví dụ Intel N150), dùng **ONNX Runtime**.

**Note:**
- Các cấu hình chọn lọc đều được chạy qua PTQ để kiểm tra xem cấu hình đó có thật sự ổn để quantize hay không, trước khi đưa vào QAT — vì không phải layer nào cũng quantize được, chỉ một phần của model chịu được.
- Cả 2 quá trình QAT đều train **150 epoch/model**, mỗi epoch **400 steps**.

## Tổng quan so sánh trên 500 samples test

| | **BanhmiTTS_v1** | **Piper (Baseline)** |
|---|---|---|
| WER (Word Error Rate) | **7.17%** | 7.84% |
| UTMOS (Naturalness) | **3.29** | 3.01 |
| RTF (Real-time Factor) | 0.111 | **0.093** |
| Size | 46.4MB | **39.9MB** |

- **BanhmiTTS_v1** tốt hơn về **WER** (tỉ lệ sai chữ) và **UTMOS** (độ tự nhiên).
- **Piper (Baseline)** tốt hơn về **RTF** (nhanh hơn ~16%) và dung lượng nhỏ hơn (~14%) — một phần vì FP32 gốc của nó vốn đã nhỏ/nhẹ hơn.

---

## 1. Sự khác nhau về kiến trúc

Ban đầu tưởng đây là 2 checkpoint của cùng một model (chỉ khác epoch), nhưng kiểm tra graph ONNX xuất ra cho thấy **chúng là hai kiến trúc khác nhau**:

| Model | **BanhmiTTS_v1** | **Piper (Baseline)** |
|---|---|---|
| **Tổng lớp Conv/ConvTranspose (đo trực tiếp từ ONNX graph)** | **160** | **132** |
| flow (normalizing flow) | ~64 | 40 |
| dec (BigVGAN decoder, gồm conv_post) | 24 | 23 |
| f0_predictor | có (3 lớp) | **không có** |
| *enc_p + dp (phần còn lại, quantize được)* | *~72* | *69 (enc_p 37 + dp 32)* |

*Ghi chú: với BanhmiTTS_v1, số liệu enc_p/dp riêng lẻ không đo trực tiếp qua ONNX graph — chỉ có số lớp "wrap" lúc QAT training (tính cả Linear, xem mục 2.3) là enc_p=37, dp=80, nên không cộng thẳng vào cột Conv/ConvTranspose ở đây để tránh nhầm hai loại đơn vị khác nhau.*

→ Piper (Baseline) có **flow nông hơn nhiều** (40 vs 64 lớp) và **thiếu hẳn module F0 predictor**. Đây gần như chắc chắn là một thử nghiệm kiến trúc cũ/khác, không phải cùng dòng training với BanhmiTTS_v1.

Hệ quả quan trọng nhất: **layer nào "nhạy cảm" khi quantize hoá ra khác nhau hoàn toàn giữa hai kiến trúc** — không thể copy nguyên công thức loại trừ từ model này sang model kia (xem mục 3).

---

## 2. BanhmiTTS_v1

### 2.1. Các layer không quantize
- **`flow`**: invertible transform gồm 4 coupling layer nối tiếp — lượng tử hóa làm sai số **cộng dồn** qua từng layer, gây sụt UTMOS nhiều nhất trong mọi thử nghiệm loại trừ đã làm cho kiến trúc này.
- **`dec.conv_post`**: lớp Conv1d cuối cùng trước tanh, quyết định trực tiếp giá trị sample của waveform. Layer nhỏ (1 lớp) nhưng nhạy cảm bất thường theo tài liệu mixed-precision quantization — loại trừ nó tốn rất ít dung lượng nhưng cứu được chất lượng đáng kể.

### 2.2. Các layer quantize
`enc_p` + `dp` + `dec` (trừ `conv_post`) + `f0_predictor` = **95/160 lớp Conv/ConvTranspose** (59%).

### 2.3. Quy trình QAT
- Wrap **143 lớp** (Conv1d/ConvTranspose1d/Linear — tính cả Linear nên nhiều hơn con số Conv/ConvTranspose ở trên) tại `enc_p` (37), `dp` (80), `dec` trừ `conv_post` (23), `f0_predictor` (3).
- LR: 5e-6 → 5e-7 (giảm dần 10x qua 150 epoch), discriminator warmup 1 epoch đầu.
- Train qua nhiều lần chạy nối tiếp (resume liên tục), tổng ~150 epoch = ~60,000 bước, gần đạt mốc 10% của 844,560 bước training gốc theo khuyến nghị NVIDIA.
- **Best checkpoint: epoch 62** (val_loss_mel=20.7220). Từ epoch 62 → 150 (88 epoch thêm) không có cải thiện nào — training đã bão hòa.

### 2.4. Kết quả kiểm chứng (n=500 câu thật, Whisper WER + UTMOSv2)

| | WER | CER | UTMOS | RTF | Size |
|---|---|---|---|---|---|
| FP32 | 7.68% | — | **3.85** | **0.110** | 70.6MB |
| INT8 (PTQ) | 7.63% | 4.31% | 3.18 | 0.110 | 46.4MB |
| **INT8 (QAT)** | **7.17%** | **4.24%** | 3.29 | 0.111 | **46.4MB** |

→ QAT thắng PTQ thuần trên cả WER lẫn UTMOS, tốc độ không đổi.

**File giao:**
- `qat_transfer/voice.int8.qat_final.onnx` (INT8, deliverable chính)
- `qat_transfer/BanhmiTTS_v1_qat_best_checkpoint.ckpt` (checkpoint PyTorch gốc, epoch 62)

---

## 3. Piper (Baseline)

### 3.1. Bước 1: Áp nguyên công thức của BanhmiTTS_v1 → THẤT BẠI

Thử loại trừ `flow` + `conv_post` (giống hệt BanhmiTTS_v1), quantize phần còn lại (`enc_p` + `dp` + `dec`):

- Train QAT 92 epoch với scope này → checkpoint best val_loss_mel=21.3995 (số liệu training bình thường, không có dấu hiệu bất ổn).
- Nhưng khi export INT8 và đo thật: **WER = 91.35%, UTMOS = 1.74** — audio gần như hỏng hoàn toàn.
- **Kiểm tra chéo**: PTQ thuần (không train QAT gì cả) với cùng cấu hình loại trừ flow+conv_post trên chính FP32 gốc cũng cho **WER = 97.3%, UTMOS = 1.68** → chứng minh đây **không phải lỗi do quá trình QAT training**, mà lỗi nằm ở việc **loại trừ sai layer** cho kiến trúc này.
- FP32 gốc (chưa quantize gì) hoàn toàn bình thường: **WER=10.0%, UTMOS=3.36** → xác nhận checkpoint gốc không hề hỏng.

### 3.2. Bước 2: khảo sát thực nghiệm tìm layer nhạy cảm thật sự

Chạy PTQ với 12 tổ hợp loại trừ khác nhau trên tổng **132 lớp Conv/ConvTranspose** (mỗi lần chỉ mất ~5-10 phút vì không cần train), đo WER trên 10-30 câu:

| Cấu hình loại trừ | Số lớp loại trừ | WER |
|---|---|---|
| Không loại trừ gì | 0/132 | 107% ❌ |
| Chỉ `flow` | 40/132 | 99% ❌ |
| `flow` + `conv_post` (copy BanhmiTTS_v1) | 41/132 | 93% ❌ |
| Chỉ `dp` | 32/132 | 96% ❌ |
| Chỉ `dec` | 23/132 | 105% ❌ |
| Chỉ `enc_p` | 37/132 | 25% ⚠️ |
| `flow` + `enc_p` | 77/132 | 26% ⚠️ |
| `flow` + `dp` | 72/132 | 99% ❌ |
| `flow` + `dec` | 63/132 | 100% ❌ |
| `flow` + `enc_p` + `dp` | 109/132 | 4.0% ✅ |
| **`enc_p` + `dp`** | 69/132 | 3.8% ✅ |
| **`enc_p` + `dp` + `conv_post`** | 70/132 | 3.8% ✅ |

**→ Kết luận:** với kiến trúc Piper, layer nhạy cảm là **`enc_p` (text encoder) và `dp` (duration predictor)** — **ngược hoàn toàn** với BanhmiTTS_v1 (nơi `flow` mới là thủ phạm). `flow` và `dec` ở kiến trúc này quantize hoàn toàn bình thường, không gây hỏng.

Một giả thuyết khả dĩ: kiến trúc Piper **thiếu f0_predictor**, nên có thể `enc_p`/`dp` phải gánh vai trò biểu diễn thông tin nhạy cảm hơn (không có nhánh F0 riêng để giảm tải), khiến chúng dễ vỡ hơn khi lượng tử hóa. Cũng có thể liên quan tới self-attention bên trong các module này phản ứng xấu với nhiễu lượng tử hóa. Đây là **giả thuyết dựa trên quan sát thực nghiệm**, chưa được xác nhận qua phân tích activation distribution chi tiết.

### 3.3. Layer không quantize (cấu hình cuối, đã chọn)
- **`enc_p`** (text encoder, 37 lớp)
- **`dp`** (stochastic duration predictor, 32 lớp)
- **`dec.conv_post`** (giữ nguyên lý do như BanhmiTTS_v1: layer cuối trước tanh)

### 3.4. Layer được quantize
`flow` (40) + `dec` trừ `conv_post` (22) = **62/132 lớp** (47%).

### 3.5. QAT training (hoàn tất)

Sau khi tìm được cấu hình đúng, khởi động lại QAT training với scope khớp chính xác cấu hình export:
- Wrap **62 lớp** tại `flow` (40) + `dec` trừ `conv_post` (22) — **không đụng tới `enc_p`/`dp`** vì chúng luôn ở FP32 khi export.
- Cùng LR schedule/discriminator warmup như BanhmiTTS_v1, chạy đủ 150 epoch (~10 giờ, nhanh hơn ~35% so với BanhmiTTS_v1 vì chỉ wrap 62 lớp thay vì 143).
- **Best checkpoint: epoch 134** (val_loss_mel=21.3403). Từ epoch 134 → 149 không cải thiện thêm — cùng pattern bão hòa như BanhmiTTS_v1.

### 3.6. Kết quả cuối cùng (n=500, so sánh công bằng với BanhmiTTS_v1)

| | WER | CER | UTMOS | RTF | Size |
|---|---|---|---|---|---|
| FP32 | 10.0% | 5.9% | **3.36** | 0.093 | 63.6MB |
| INT8 (PTQ) | 8.53% | 4.79% | 2.80 | 0.093 | 39.9MB |
| **INT8 (QAT)** | **7.84%** | **4.44%** | 3.01 | **0.093** | **39.9MB** |

→ QAT cải thiện đáng kể so với PTQ thuần (UTMOS 2.80→3.01, WER 8.53%→7.84%), đúng hướng kỳ vọng — nhưng **không vượt qua được BanhmiTTS_v1** (UTMOS 3.29, WER 7.17%), vì FP32 gốc Piper vốn đã kém hơn BanhmiTTS_v1 ngay từ đầu. Đổi lại, Piper (Baseline) **nhanh hơn ~16%** (RTF 0.093 vs 0.111) và **nhẹ hơn ~14%** (39.9MB vs 46.4MB) — có thể là lựa chọn hợp lý nếu ưu tiên tốc độ/dung lượng hơn chất lượng tuyệt đối.

**File giao:**
- `qat_transfer/voice.int8.piper_baseline_final.onnx` (INT8, deliverable chính)
- `.../qat_finetune_piper_baseline/lightning_logs/version_0/checkpoints/qat-best-epoch=134-val_loss_mel=21.3403.ckpt` (checkpoint PyTorch gốc)

---

## 4. Bài học rút ra

1. **"Layer nhạy cảm với quantization" không phải thuộc tính cố định của kiến trúc VITS nói chung — nó riêng cho từng model cụ thể.** Không thể giả định flow luôn là thủ phạm; phải đo thực nghiệm cho từng checkpoint/kiến trúc.
2. **val_loss_mel không phản ánh chất lượng audio thực tế sau quantize.** Cả hai lần thất bại (Piper với cấu hình sai) đều có val_loss_mel hoàn toàn bình thường trong lúc training — chỉ WER/UTMOS đo trên audio thật mới lộ ra vấn đề.
3. **QAT không sửa được lỗi chọn sai layer để loại trừ.** 92 epoch QAT training trên cấu hình sai (wrap enc_p+dp+dec, loại trừ flow) cho kết quả tệ y hệt PTQ thuần cùng cấu hình — chứng tỏ vấn đề nằm ở *chọn sai layer*, không phải *thiếu training*.
4. Khảo sát PTQ (không cần train) là cách rẻ và nhanh nhất để tìm cấu hình loại trừ đúng trước khi đầu tư nhiều giờ GPU vào QAT training.
