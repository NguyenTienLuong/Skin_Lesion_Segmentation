# W1-04 — Phân tích yêu cầu thay ResNet34 bằng MobileNetV2



> **Mục đích:** Phân biệt rõ **"thay encoder"** với **"xây một mô hình hoàn toàn khác"**. So sánh công bằng yêu cầu toàn bộ phần còn lại của pipeline giống A càng nhiều càng tốt.
>


---

## Tóm tắt nhanh

| Quyết định | Nội dung |
|---|---|
| Số mức feature | 5 mức (stride 2, 4, 8, 16, 32), giữ nguyên như A |
| Điểm lấy feature MobileNetV2 | `features[1, 3, 6, 13, 17]` |
| Channel encoder | `(16, 24, 32, 96, 320)` |
| Channel ResNet34 (A) | `(64, 64, 128, 256, 512)` |
| Cách thích nghi decoder | **Phương án 1:** `in_channels` động theo `encoder_channels` |
| Phần thay đổi duy nhất | Encoder + vị trí lấy feature + channel đầu vào decoder |
| Phần còn lại | Đóng băng giống A (data, loss, optimizer, metric, threshold, Optuna) |

---

## Câu 1. U-Net cần feature map nhiều mức phân giải để tạo skip connection

### 1.1 Nguyên lý cốt lõi

#### A. Diễn biến bên trong encoder

Qua mỗi stage (stride = 2), encoder làm đồng thời hai việc ngược chiều nhau:

1. **Giảm độ phân giải không gian (H × W)**
   - Ảnh 256 × 256 thu nhỏ dần: 128 → 64 → 32 → 16 → 8.
   - Mục đích: bỏ bớt chi tiết vụn, giúp mô hình nắm "bức tranh toàn cảnh" thay vì bị chi phối bởi từng pixel.
2. **Tăng số kênh đặc trưng (C)**
   - Từ 3 kênh RGB ban đầu, mô hình học ra hàng chục đến hàng trăm kênh đặc trưng (ví dụ 16 → 32 → 64 → 128 → 320).
   - Mỗi kênh là một đặc trưng học được (cạnh, kết cấu, màu, hình dạng...). Các kênh không nhất thiết tương ứng một khái niệm y khoa rõ ràng.
   - Mục đích: bù lại việc mất diện tích không gian và biểu diễn được các khái niệm phức tạp hơn.

```
[Ảnh RGB 3×256×256] ──► (ENCODER) ──► [Bottleneck C×8×8]
                        Spatial: 256 → 8
                        Channels: 3 → C (512 với ResNet34, 320 với MobileNetV2)
```

#### B. Sự đánh đổi giữa feature nông và feature sâu

| Loại feature | Vị trí | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Nông** (độ phân giải cao) | Các stage đầu | Giữ biên, góc cạnh, vị trí không gian sắc nét | Ngữ nghĩa yếu: biết "có đường viền" nhưng chưa biết đó là viền của tổn thương hay nếp da |
| **Sâu** (độ phân giải thấp) | Bottleneck | Ngữ nghĩa mạnh: nhận biết được vùng tổn thương | Mất chi tiết biên do bị nén xuống 8 × 8 |

#### C. Vai trò của skip connection trong bài toán da liễu

- **Skip connection** là "cây cầu" sao chép trực tiếp feature map ở từng mức phân giải của encoder sang decoder.
- Ranh giới giữa da lành và vùng tổn thương thường **mờ, nhấp nhô, bất định**. Nếu decoder chỉ dựa vào feature nhòe ở bottleneck, mask dự đoán sẽ thô và méo.
- Skip connection đưa feature nông (sắc nét) sang để decoder khôi phục đường viền chính xác đến từng pixel.

### 1.2 Feature map của U-Net34 (ResNet34 baseline), input 256 × 256

| Mức | Stride | Kích thước | Channel | Vị trí trích xuất trong ResNet34 | Đặc trưng thường thấy (minh họa) |
|---|---|---|---|---|---|
| Skip 1 | 1/2 | 128 × 128 | 64 | `conv1 → bn1 → relu` | Nét viền mảnh, góc cạnh |
| Skip 2 | 1/4 | 64 × 64 | 64 | `layer1` (sau maxpool) | Hình dạng cơ bản của đốm da |
| Skip 3 | 1/8 | 32 × 32 | 128 | `layer2` | Kết cấu bề mặt, độ tương phản |
| Skip 4 | 1/16 | 16 × 16 | 256 | `layer3` | Hình dáng tổng thể của khối tổn thương |
| Bottleneck | 1/32 | 8 × 8 | 512 | `layer4` | Ngữ cảnh sâu nhất (đáy chữ U) |

**"Vị trí trích xuất" = "điểm bấm cắt" trong code.** Dữ liệu chảy liên tục qua encoder. Tại vị trí đã chọn, ta copy feature vào một biến `skip` để gửi sang decoder, còn luồng chính vẫn tiếp tục xuống các lớp sâu hơn.

Ví dụ cho Skip 1:

```
[Ảnh RGB 256×256]
      │
   [conv1]  ──► 128×128
   [bn1]
   [relu]
      ├────► COPY ──► skip1 = [64, 128, 128] ──► gửi sang decoder
   [maxpool] ──► 64×64 ──► layer1 ──► ...
```

**Cơ chế khôi phục ở decoder:** đi ngược từ bottleneck lên: 8 → 16 → 32 → 64 → 128 → 256. Ở mỗi nấc upsample ×2 (trừ nấc cuối 128 → 256), decoder `concat` với skip có cùng kích thước.

### 1.3 Kết luận cho nhiệm vụ thay ResNet34 bằng MobileNetV2

1. **Khóa cố định 5 mức feature** (stride 2, 4, 8, 16, 32), tương ứng kích thước 128, 64, 32, 16, 8. Logic upsample ×2 và cấu trúc 5 nấc phải **giữ nguyên 100% như A**.
2. **Ràng buộc kiểm tra shape:** H, W của skip phải khớp tuyệt đối với H, W của decoder tại bước upsample đó. Lệch 1 pixel là `torch.cat()` báo lỗi hoặc ghép sai vị trí không gian.
3. **Thích nghi số kênh:** kích thước H × W cố định, nhưng số kênh do encoder quyết định. MobileNetV2 trả về `(16, 24, 32, 96, 320)`, nhỏ hơn nhiều so với ResNet34 `(64, 64, 128, 256, 512)`. B chỉ cần chỉnh `in_channels` ở các khối decoder.
4. **Bản chất của việc thay encoder:** không xây mô hình mới từ đầu, mà thay **"bộ cung cấp 5 feature map"** bằng một bộ nhẹ hơn. Logic skip, upsample, pipeline huấn luyện, loss và metric giữ nguyên.

---

## Câu 2. Stage của MobileNetV2 tương ứng với các mức decoder

### 2.1 Yêu cầu giao diện

Decoder yêu cầu encoder cung cấp **5 feature map** giảm dần theo hệ số 2. Với input 256 × 256:

```
128×128  →  64×64  →  32×32  →  16×16  →  8×8
```

> Input phải có H, W chia hết cho 32 (128 và 256 đều thỏa) thì 5 mức này khớp mà không cần resize/interpolate thêm.

### 2.2 Cấu trúc `torchvision.models.mobilenet_v2().features`

`features` gồm 19 khối (index 0 đến 18):

| Index | Loại khối | Channel ra | Stride tích lũy |
|---|---|---|---|
| 0 | Conv 3×3, stride 2 | 32 | 1/2 |
| 1 | InvertedResidual | **16** | 1/2 |
| 2–3 | InvertedResidual (khối 2 stride 2) | **24** | 1/4 |
| 4–6 | InvertedResidual (khối 4 stride 2) | **32** | 1/8 |
| 7–10 | InvertedResidual (khối 7 stride 2) | 64 | 1/16 |
| 11–13 | InvertedResidual | **96** | 1/16 |
| 14–16 | InvertedResidual (khối 14 stride 2) | 160 | 1/32 |
| 17 | InvertedResidual | **320** | 1/32 |
| 18 | Conv 1×1 | 1280 | 1/32 |

### 2.3 Điểm trích xuất được chọn

| Mức decoder | Stride | Kích thước | Khối trích xuất | Channel | Ghi chú |
|---|---|---|---|---|---|
| Skip 1 | 1/2 | 128 × 128 | `features[1]` | **16** | Khối cuối ở độ phân giải 1/2 |
| Skip 2 | 1/4 | 64 × 64 | `features[3]` | **24** | Khối cuối ở 1/4 |
| Skip 3 | 1/8 | 32 × 32 | `features[6]` | **32** | Khối cuối ở 1/8 |
| Skip 4 | 1/16 | 16 × 16 | `features[13]` | **96** | Khối cuối ở 1/16 |
| Bottleneck | 1/32 | 8 × 8 | `features[17]` | **320** | Khối InvertedResidual cuối |

> **Lưu ý về bottleneck:** `features[17]` cho 320 kênh, còn `features[18]` (conv 1×1 cuối) cho **1280 kênh**. Tài liệu này chọn `features[17]` để khớp dàn kênh `(16, 24, 32, 96, 320)` và giữ encoder gọn. Quyết định này cần ghi nhất quán ở `specs/mobilenetv2_unet_spec.md` (W1-13).

### 2.4 Vì sao chọn các khối này: nguyên tắc "cắt trước khi nén"

- Trong MobileNetV2, khối **đầu tiên** của stage mới dùng stride = 2 để giảm độ phân giải.
- Do đó skip phải lấy ở **khối cuối cùng của stage hiện tại**: nơi thông tin ở độ phân giải đó đã được xử lý đầy đủ nhất và ngay trước khi bị nén tiếp.
- Ví dụ: `features[3]` là khối cuối còn giữ 64 × 64. `features[4]` có stride 2 nên nén xuống 32 × 32, vì vậy phải cắt tại `features[3]`.

### 2.5 Sơ đồ kết nối encoder và decoder

```
[Encoder: MobileNetV2]                          [Decoder: U-Net]
features[1]   (16 ch, 128×128) ── Skip 1 ───►  UpBlock 4 (concat 16 ch)
features[3]   (24 ch,  64×64 ) ── Skip 2 ───►  UpBlock 3 (concat 24 ch)
features[6]   (32 ch,  32×32 ) ── Skip 3 ───►  UpBlock 2 (concat 32 ch)
features[13]  (96 ch,  16×16 ) ── Skip 4 ───►  UpBlock 1 (concat 96 ch)
features[17]  (320 ch,  8×8  ) ── Bottleneck ► UpBlock 1 (input 320 ch)
```

### 2.6 Kết luận

1. Spatial resolution khớp hoàn toàn với 5 mức decoder yêu cầu, không cần thêm lớp resize/interpolate.
2. B khai báo `encoder_channels = (16, 24, 32, 96, 320)` để decoder tự thiết lập số kênh các lớp conv tương ứng.

---

## Câu 3. Channel dự kiến và thiết kế adapter cho decoder

### 3.1 So sánh số kênh

| Mức | Stride | Kích thước | ResNet34 (A) | MobileNetV2 (B) | Mức giảm |
|---|---|---|---|---|---|
| Skip 1 | 1/2 | 128 × 128 | 64 | 16 | 4.0× |
| Skip 2 | 1/4 | 64 × 64 | 64 | 24 | 2.7× |
| Skip 3 | 1/8 | 32 × 32 | 128 | 32 | 4.0× |
| Skip 4 | 1/16 | 16 × 16 | 256 | 96 | 2.7× |
| Bottleneck | 1/32 | 8 × 8 | 512 | 320 | 1.6× |

### 3.2 Tính số kênh đầu vào của từng decoder block

Công thức:

```
C_in_decoder = C_upsampled_from_below + C_skip_from_encoder
```

Upsample ×2 không đổi số kênh. Các giá trị kênh **đầu ra** của decoder (256, 128, 64, 32) dưới đây là **giá trị giả định**, cần thay bằng giá trị thực trong code của A.

| Block | Độ phân giải | Từ dưới lên | Skip | Sau concat | Conv block biến đổi |
|---|---|---|---|---|---|
| 1 | 8 → 16 | 320 | 96 (`features[13]`) | **416** | 416 → 256 |
| 2 | 16 → 32 | 256 | 32 (`features[6]`) | **288** | 288 → 128 |
| 3 | 32 → 64 | 128 | 24 (`features[3]`) | **152** | 152 → 64 |
| 4 | 64 → 128 | 64 | 16 (`features[1]`) | **80** | 80 → 32 |

Sau block 4, decoder còn một nấc upsample 128 → 256 **không có skip**, rồi đến output head (conv 1×1 → 1 channel logits). Số kênh của nấc này cũng lấy theo A.

### 3.3 Hai phương án thích nghi

#### Phương án 1 — Dynamic channel adaptation (khuyến nghị)

- `DecoderBlock` nhận `in_channels` động, tính từ `encoder_channels = (16, 24, 32, 96, 320)`.
- Số kênh sau concat tự tính ra `(416, 288, 152, 80)`.
- Conv 3×3 đầu tiên của mỗi block nén số kênh này về kích thước mong muốn. Không cần thêm lớp phụ.
- **Ưu điểm:** gọn, ít tham số hơn, tận dụng lợi thế nhẹ của MobileNetV2.
- **Lưu ý:** channel đầu ra của decoder phải **giữ như A**, chỉ channel đầu vào đổi.

#### Phương án 2 — Conv 1×1 adapter (projection layer)

Giữ nguyên decoder của A (thiết kế cho `64, 64, 128, 256, 512`), thêm conv 1×1 trên mỗi đường skip để chiếu về số kênh cũ:

| Vị trí | Chiếu kênh |
|---|---|
| Skip 1 | 16 → 64 |
| Skip 2 | 24 → 64 |
| Skip 3 | 32 → 128 |
| Skip 4 | 96 → 256 |
| Bottleneck | 320 → 512 |

- **Ưu điểm:** không phải sửa code decoder của A.
- **Nhược điểm:** thêm tham số/FLOPs ở các lớp adapter và đưa thêm một yếu tố mới vào so sánh.

### 3.4 Kết luận

1. **Số kênh khai báo:** `(16, 24, 32, 96, 320)`.
2. **Khuyến nghị:** dùng **Phương án 1**, vì mô hình gọn hơn và không phát sinh các lớp adapter làm nhiễu so sánh.
3. **Lưu ý khi báo cáo "nhẹ hơn":** mức giảm tham số chủ yếu đến từ encoder. Decoder vẫn có thể chiếm tỉ lệ lớn (ví dụ conv 416 → 256 ở block 1). Nên **đếm tham số thực tế** (encoder, decoder, tổng) thay vì suy đoán.

---

## Câu 4. Phần phải giữ giống A (experimental controls)

B **chỉ được thay kiến trúc mạng** (encoder MobileNetV2 và cấu hình channel decoder). Mọi thứ xung quanh phải đóng băng giống hệt A.

| # | Hạng mục | Quy chuẩn giữ giống A | Lý do |
|---|---|---|---|
| 1 | Input / Output | Input `[B, 3, H, W]` RGB; output `[B, 1, H, W]` logits. H = W = 256 (và 128 theo W1-11) | Đúng bài toán; mask phải trùng kích thước ảnh để tính metric |
| 2 | Preprocessing & augmentation | ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`); cùng resize/crop; cùng augmentation (rotate, flip...) | Khác đi sẽ làm thay đổi bản chất dữ liệu, sai lệch so sánh |
| 3 | Dataset split | Cùng danh sách file train/val/test, cùng seed | Hai mô hình học và được đánh giá trên cùng dữ liệu |
| 4 | Loss | Cùng công thức và trọng số *(điền theo A: Dice / BCE / Dice+BCE)* | Loss định hướng tối ưu; khác loss là so sánh "táo với cam" |
| 5 | Optimizer baseline | Cùng thuật toán, LR mặc định, scheduler *(điền theo A)* | Cùng điều kiện cập nhật trọng số ban đầu |
| 6 | Metric | Cùng định nghĩa và cách tính *(theo A: Dice, IoU/Jaccard, Precision, Recall)* | Chuẩn đo độ chính xác của mask |
| 7 | Threshold | Cùng ngưỡng nhị phân *(theo A, dự kiến 0.5)* sau sigmoid | Giữ cố định để không làm lệch Dice/IoU |
| 8 | Ngân sách Optuna | Cùng số trial *(điền theo A)*; cùng search space (LR, weight decay, batch size, decoder dropout 0/0.1/0.2) | Hai mô hình có cơ hội tìm siêu tham số ngang nhau |

> Các mục có chữ *(điền theo A)* là ví dụ trong bản nháp. Cần thay bằng giá trị thực tế của A trước khi chốt.

### Ý nghĩa phương pháp luận

Đây là một thí nghiệm có **một biến độc lập duy nhất: kiến trúc encoder**. Nếu đổi thêm bất kỳ yếu tố nào khác, kết quả không còn quy được về encoder.

### Khi triển khai code

1. Dùng lại 100% `DataModule`, `Loss`, `Metrics`, `Trainer` và script Optuna của A.
2. Chỉ viết thêm file mô hình mới `MobileNetV2_UNet`.
3. Đưa mô hình vào chung pipeline, chạy thực nghiệm và xuất bảng so sánh.

---

## Câu 5. Phần bắt buộc khác do encoder

1. **Vị trí lấy feature:** `features[1, 3, 6, 13, 17]` thay cho `conv1, layer1…layer4` của ResNet34.
2. **Channel đầu vào của các khối decoder:** `(416, 288, 152, 80)` thay cho giá trị tương ứng với `(64, 64, 128, 256, 512)` của A (hoặc dùng adapter nếu chọn Phương án 2).
3. **Pretrained weights:** MobileNetV2 ImageNet thay cho ResNet34 ImageNet (cùng nguồn ImageNet nên vẫn cùng điều kiện khởi tạo).

---

## Câu 6. Bảng "Giữ nguyên / Thay đổi"

| Yếu tố | U-Net34 (A) | MobileNetV2-UNet (B) | Trạng thái |
|---|---|---|---|
| Encoder | ResNet34 | MobileNetV2 | **Thay đổi** |
| Pretrained | ImageNet | ImageNet | Giữ nguyên |
| Số mức skip | 4 skip + bottleneck | 4 skip + bottleneck | Giữ nguyên |
| Độ phân giải mỗi mức | 128/64/32/16/8 | 128/64/32/16/8 | Giữ nguyên |
| Vị trí lấy feature | `conv1, layer1–layer4` | `features[1,3,6,13,17]` | **Thay đổi** |
| Channel skip/bottleneck | 64/64/128/256/512 | 16/24/32/96/320 | **Thay đổi** |
| Channel **vào** decoder block | theo ResNet34 | 416/288/152/80 (Phương án 1) | **Thay đổi** |
| Channel **ra** decoder block | theo A | theo A | Giữ nguyên |
| Output head | 1 channel logits | 1 channel logits | Giữ nguyên |
| Decoder dropout | cấu hình được (0/0.1/0.2) | cấu hình được (0/0.1/0.2) | Giữ nguyên |
| Input / preprocessing | RGB, ImageNet norm | RGB, ImageNet norm | Giữ nguyên |
| Dataset split + seed | cố định | giống A | Giữ nguyên |
| Loss | theo A | theo A | Giữ nguyên |
| Optimizer baseline | theo A | theo A | Giữ nguyên |
| Metric / threshold | theo A | theo A | Giữ nguyên |
| Ngân sách Optuna | theo A | theo A | Giữ nguyên |

**Quy tắc:** chỉ các dòng **Thay đổi** được khác A. Muốn đổi thêm thứ gì khác thì phải ghi riêng và giải thích lý do.