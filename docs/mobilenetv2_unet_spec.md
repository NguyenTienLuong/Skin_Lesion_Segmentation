# W1-13 — Đặc tả MobileNetV2-UNet

 
**Mục đích:** khóa cách lấy feature từ MobileNetV2 *trước khi code*, để decoder không bị thiết kế bằng thử-sai.

**Cách đọc tài liệu này:** mỗi mục gồm phần **đặc tả** (quyết định cuối, người code làm theo) và phần **Giải thích dễ hiểu** (vì sao lại quyết định như vậy).

**Quy ước đánh dấu:**
- `[THEO A]`: giá trị phải lấy từ code U-Net34 của A.

## Mục lục
 
0. [Bức tranh tổng thể](#0-bức-tranh-tổng-thể)
1. [Encoder](#1-encoder)
2. [Điểm lấy feature (skip) và shape dự kiến](#2-điểm-lấy-feature-skip-và-shape-dự-kiến)
3. [Khớp độ phân giải giữa skip và decoder](#3-khớp-độ-phân-giải-giữa-skip-và-decoder)
4. [Decoder: channel cần đổi](#4-decoder-channel-cần-đổi)
5. [Output head và dropout](#5-output-head-và-dropout)
6. [Giao diện lớp](#6-giao-diện-lớp)
7. [Giữ giống U-Net34 và phần thay đổi](#7-giữ-giống-u-net34-và-phần-thay-đổi)
8. [Smoke-test](#8-smoke-test)
   - [8.1 Kiểm tra feature của encoder](#81-kiểm-tra-feature-của-encoder-xác-nhận-mục-2)
   - [8.2 Kiểm tra toàn mô hình](#82-kiểm-tra-toàn-mô-hình-tương-tự-w1-11)
   - [8.3 Tiêu chí đạt](#83-tiêu-chí-đạt)



---

## 0. Bức tranh tổng thể

```
Ảnh RGB [B,3,H,W]
      │
┌─────▼───────────────────────┐
│  ENCODER = MobileNetV2      │  ◄── phần DUY NHẤT thay đổi so với A
│  (pretrained ImageNet)      │
└─┬───┬───┬────┬────┬─────────┘
  │   │   │    │    │   5 feature map: skip1, skip2, skip3, skip4, bottleneck
  ▼   ▼   ▼    ▼    ▼
┌─────────────────────────────┐
│  DECODER (kiểu U-Net)       │  ◄── cấu trúc giữ như A, chỉ chỉnh số kênh đầu vào
│  upsample ×2 + concat skip  │
└─────────────┬───────────────┘
              ▼
   Output head (conv 1×1) ──► logits [B,1,H,W]
```

> **Giải thích dễ hiểu**
> Hãy coi U-Net như một **bộ khung có 5 ổ cắm**. Mỗi ổ cắm cần một feature map với kích thước nhất định. ResNet34 là một "bộ nguồn" cắm vừa 5 ổ này. MobileNetV2 là một bộ nguồn khác, nhẹ hơn, nhưng cũng cắm vừa 5 ổ đó. Điểm khác duy nhất là "điện áp" (số kênh) của mỗi ổ khác nhau, nên các dây nối ở phía decoder phải chỉnh lại cho khớp.
> Vì vậy đây là **thay encoder**, không phải **xây mô hình mới**.

---

## 1. Encoder

| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | MobileNetV2 (`torchvision.models.mobilenet_v2`) |
| Pretrained | ImageNet (`weights="IMAGENET1K_V1"`) |
| Phần dùng | Chỉ `.features` (index 0–17). **Bỏ** `features[18]` và `classifier` |
| Input | `[B, 3, H, W]`, RGB, ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) |
| Điều kiện kích thước | H, W chia hết cho 32 (hỗ trợ 128 và 256) |
| Fine-tune | Toàn bộ encoder được huấn luyện, không đóng băng `[THEO A]` |

> **Giải thích dễ hiểu**
> - **Pretrained ImageNet** nghĩa là mạng đã được học sẵn từ hàng triệu ảnh, đã biết nhận ra cạnh, góc, kết cấu. Ta không bắt nó học từ con số 0 trên tập ảnh da, nên học nhanh và ổn định hơn. ResNet34 của A cũng pretrained ImageNet, nên hai bên **xuất phát cùng điều kiện**.
> - **Chỉ dùng `.features`:** MobileNetV2 gốc được thiết kế để *phân loại ảnh* (ra 1000 lớp). Phần `classifier` ở cuối chỉ phục vụ việc đó, còn bài toán của ta là phân đoạn nên không cần.
> - **Vì sao bỏ `features[18]`:** khối này là conv 1×1 nở số kênh lên **1280** để phục vụ phân loại. Ta dừng ở `features[17]` (**320** kênh) để encoder gọn và dàn kênh gọn.
> - **Vì sao H, W phải chia hết cho 32:** encoder giảm kích thước 5 lần, mỗi lần chia đôi (2⁵ = 32). Nếu không chia hết, các lần chia đôi sẽ làm tròn và kích thước skip lệch so với decoder, khiến phép ghép bị lỗi.

---

## 2. Điểm lấy feature (skip) và shape dự kiến

| Tên | Stride | Index `features[...]` | Channel | Shape với input 256 | Shape với input 128 |
|---|---|---|---|---|---|
| `skip1` | 1/2 | 1 | 16 | `[B, 16, 128, 128]` | `[B, 16, 64, 64]` |
| `skip2` | 1/4 | 3 | 24 | `[B, 24, 64, 64]` | `[B, 24, 32, 32]` |
| `skip3` | 1/8 | 6 | 32 | `[B, 32, 32, 32]` | `[B, 32, 16, 16]` |
| `skip4` | 1/16 | 13 | 96 | `[B, 96, 16, 16]` | `[B, 96, 8, 8]` |
| `bottleneck` | 1/32 | 17 | 320 | `[B, 320, 8, 8]` | `[B, 320, 4, 4]` |

```python
ENCODER_OUT_INDICES = (1, 3, 6, 13, 17)
ENCODER_CHANNELS    = (16, 24, 32, 96, 320)
```

**Nguyên tắc chọn điểm cắt:** lấy ở **khối cuối cùng của mỗi mức phân giải**, ngay trước khối stride 2 kế tiếp.

**Bảng cấu trúc `features` (để đối chiếu):**

| Index | Channel ra | Stride tích lũy | Ghi chú |
|---|---|---|---|
| 0 | 32 | 1/2 | Conv 3×3 stride 2 |
| **1** | **16** | 1/2 | ← `skip1` |
| 2 | 24 | 1/4 | Khối đầu stage, stride 2 |
| **3** | **24** | 1/4 | ← `skip2` |
| 4 | 32 | 1/8 | Khối đầu stage, stride 2 |
| 5 | 32 | 1/8 | |
| **6** | **32** | 1/8 | ← `skip3` |
| 7–10 | 64 | 1/16 | Khối 7 stride 2 |
| 11, 12 | 96 | 1/16 | |
| **13** | **96** | 1/16 | ← `skip4` |
| 14–16 | 160 | 1/32 | Khối 14 stride 2 |
| **17** | **320** | 1/32 | ← `bottleneck` |
| 18 | 1280 | 1/32 | **Không dùng** (conv 1×1 cho phân loại) |

> **Giải thích dễ hiểu: "cắt trước khi nén"**
> Mạng MobileNetV2 giống một dây chuyền chạy liên tục. Cứ vài khối thì có một khối **nén ảnh nhỏ đi một nửa** (stride 2). Ta muốn lấy feature ở độ phân giải đó khi nó **đã được xử lý đủ kỹ nhất**, tức là khối cuối trước khi bị nén.
>
> Ví dụ:
> ```
> features[2]  → 64×64 (vừa nén từ 128)
> features[3]  → 64×64  ◄── lấy ở đây (khối cuối còn giữ 64×64)
> features[4]  → 32×32 (bị nén tiếp)   ← lấy ở đây là quá muộn
> ```
> Nếu lấy `features[2]` thì thông tin chưa được xử lý hết. Nếu lấy `features[4]` thì đã sang độ phân giải khác. `features[3]` là điểm duy nhất vừa đúng.
>
> Lưu ý số kênh của MobileNetV2 **nhỏ hơn nhiều** so với ResNet34 (16/24/32/96/320 so với 64/64/128/256/512). Đây là lý do mạng nhẹ hơn, và cũng là lý do decoder phải chỉnh số kênh đầu vào (mục 4).

---

## 3. Khớp độ phân giải giữa skip và decoder

| Decoder block | Upsample | Skip dùng | Kích thước (input 256) |
|---|---|---|---|
| `dec1` | 8 → 16 | `skip4` | 16 × 16 |
| `dec2` | 16 → 32 | `skip3` | 32 × 32 |
| `dec3` | 32 → 64 | `skip2` | 64 × 64 |
| `dec4` | 64 → 128 | `skip1` | 128 × 128 |
| `dec5` | 128 → 256 | không có | 256 × 256 |

**Ràng buộc:** H, W của tensor sau upsample phải **bằng đúng** H, W của skip. Không dùng resize/crop để "chữa" lệch. Nếu lệch thì coi là lỗi cấu hình và dừng.

> **Giải thích dễ hiểu**
> Decoder làm việc ngược với encoder: encoder thu nhỏ ảnh dần, decoder **phóng to lại từng bậc ×2**. Sau mỗi lần phóng to, nó **dán** thêm một skip có cùng kích thước vào, giống như đặt hai tấm ảnh chồng lên nhau.
>
> Hai tấm ảnh phải **trùng khít** thì mới dán được. Lệch dù 1 pixel thì phép ghép `torch.cat()` báo lỗi hoặc ghép sai vị trí. Vì vậy ta kiểm tra kích thước trước khi code, không để đến lúc chạy mới phát hiện.
>
> Bậc cuối (`dec5`: 128 → 256) **không có skip**, vì encoder không có feature nào ở độ phân giải gốc 256×256.

---

## 4. Decoder: channel cần đổi

Công thức: `C_concat = C_upsampled + C_skip`

| Block | Từ dưới lên | Skip | `C_concat` | Kênh ra |
|---|---|---|---|---|
| `dec1` | 320 | 96 | **416** | 256 `[THEO A]` |
| `dec2` | 256 | 32 | **288** | 128 `[THEO A]` |
| `dec3` | 128 | 24 | **152** | 64 `[THEO A]` |
| `dec4` | 64 | 16 | **80** | 32 `[THEO A]` |
| `dec5` | 32 | — | 32 | 16 `[THEO A]` |

```python
DECODER_CHANNELS = (256, 128, 64, 32, 16)   # [THEO A] phải khớp channel ra của decoder U-Net34
```

**Phương án đã chọn: dynamic channel adaptation.** Không thêm conv 1×1 adapter. `in_channels` của mỗi decoder block được **tính tự động** từ `ENCODER_CHANNELS` và `DECODER_CHANNELS`.

**Cấu trúc một decoder block** `[THEO A]`: upsample ×2 → concat skip → `Conv3×3 → BN → ReLU` ×2 → dropout. Cách upsample (bilinear / nearest / ConvTranspose) lấy theo A.

> **Giải thích dễ hiểu: "ghép hai chồng giấy"**
> Số kênh giống như **số tờ giấy trong một chồng**. Khi concat, ta đặt chồng này lên chồng kia và đếm tổng số tờ.
>
> Ví dụ `dec1`:
> ```
> chồng từ dưới lên : 320 tờ  (bottleneck sau upsample, vẫn 320 kênh)
> chồng skip4       :  96 tờ
> ─────────────────────────────
> sau khi ghép      : 416 tờ  ──► Conv 3×3 ──► nén còn 256 tờ
> ```
> Conv 3×3 đầu tiên của block có nhiệm vụ **nén** 416 kênh về 256 kênh, nên ta chỉ cần khai báo đúng số kênh *đầu vào* (416). Không cần thêm lớp phụ nào.
>
> **Vì sao không dùng phương án 2 (thêm conv 1×1 để chiếu 16 → 64, 24 → 64, ...)?** Phương án đó phải thêm các lớp mới mà A không có, tức là đưa thêm một yếu tố lạ vào bài so sánh. Phương án dynamic chỉ sửa số kênh đầu vào, là thay đổi tối thiểu bắt buộc.
>
> **Điều quan trọng:** kênh **ra** của decoder phải giống A. Chỉ kênh **vào** được đổi, vì đó là chỗ MobileNetV2 buộc phải khác.

---

## 5. Output head và dropout

| Thành phần | Quy định |
|---|---|
| Output head | `Conv2d(DECODER_CHANNELS[-1], 1, kernel_size=1)` |
| Output | `[B, 1, H, W]`, **logits**, **không sigmoid** |
| Decoder dropout | Tham số `decoder_dropout`, Optuna thử: `0.0`, `0.1`, `0.2` |
| Vị trí dropout | Sau khối conv của mỗi decoder block, trước output head `[THEO A]` |
| Khi `decoder_dropout = 0.0` | Không tạo layer dropout, hoặc dùng `Identity` |

Sigmoid chỉ áp dụng **bên ngoài model**, ở bước tính metric/threshold. Loss dùng dạng nhận logits (ví dụ `BCEWithLogitsLoss`).

> **Giải thích dễ hiểu**
> - **Output head 1 kênh:** bài toán là phân đoạn nhị phân (vùng tổn thương / không phải), nên mỗi pixel chỉ cần **một con số** thể hiện "mức độ tin là tổn thương". Conv 1×1 gom các kênh của decoder thành đúng 1 kênh đó.
> - **Logits là gì:** là con số "thô", có thể là bất kỳ số thực nào (ví dụ -3.2 hoặc +5.1). Sigmoid là bước ép con số đó về khoảng 0–1 để đọc thành xác suất.
> - **Vì sao model không sigmoid:** `BCEWithLogitsLoss` đã **tích hợp sẵn sigmoid** và tính toán ổn định hơn so với tự sigmoid rồi mới tính loss. Nếu model đã sigmoid, rồi loss lại sigmoid lần nữa thì kết quả sai. Vì vậy hai bên A và B đều chốt: model trả logits, ai cần xác suất thì tự sigmoid.
> - **Dropout là gì:** khi huấn luyện, tạm "tắt ngẫu nhiên" một phần nơ-ron để mô hình không học thuộc lòng dữ liệu (chống overfit). Khi đánh giá (`model.eval()`), dropout tự tắt. Ta để `decoder_dropout` là tham số để Optuna tự thử 0, 0.1, 0.2 và chọn mức tốt nhất, thay vì chọn bằng cảm tính.

---

## 6. Giao diện lớp

Tên tham số cần khớp `specs/model_interface.md` (W1-11). Đề xuất:

```python
class MobileNetV2UNet(nn.Module):
    def __init__(
        self,
        encoder_weights: str | None = "IMAGENET1K_V1",   # None khi chạy smoke-test không tải weights
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),  # [THEO A]
        decoder_dropout: float = 0.0,
        in_channels: int = 3,
        num_classes: int = 1,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> logits: [B, 1, H, W]"""
```

Ví dụ gọi model và loss:

```python
model = MobileNetV2UNet(decoder_dropout=0.1)
logits = model(images)                                  # [B, 1, H, W]
loss = nn.BCEWithLogitsLoss()(logits, masks.float())    # masks: [B, 1, H, W]; thay bằng loss của A
```

> **Giải thích dễ hiểu**
> Mục này giống như **"cổng kết nối chuẩn"**. Nếu A và B đặt tên tham số và định dạng input/output giống nhau, thì cùng một DataLoader, một vòng huấn luyện và một đoạn đánh giá chạy được cho cả hai mô hình mà không phải sửa gì. Khi đó chênh lệch kết quả chỉ do encoder, không do hai pipeline khác nhau.

---

## 7. Giữ giống U-Net34 và phần thay đổi

| Yếu tố | U-Net34 (A) | MobileNetV2-UNet (B) | Trạng thái |
|---|---|---|---|
| Encoder | ResNet34 | MobileNetV2 | **Thay đổi** |
| Pretrained | ImageNet | ImageNet | Giữ nguyên |
| Điểm lấy feature | `conv1`, `layer1–4` | `features[1,3,6,13,17]` | **Thay đổi** |
| Channel skip/bottleneck | 64/64/128/256/512 | 16/24/32/96/320 | **Thay đổi** |
| Channel vào decoder block | theo ResNet34 | 416/288/152/80 | **Thay đổi** |
| Số mức skip, độ phân giải mỗi mức | 4 skip + bottleneck | 4 skip + bottleneck | Giữ nguyên |
| Channel ra decoder block | theo A | theo A | Giữ nguyên |
| Cách upsample, cấu trúc conv block | theo A | theo A | Giữ nguyên |
| Output head | 1 channel logits | 1 channel logits | Giữ nguyên |
| Decoder dropout | cấu hình được | cấu hình được | Giữ nguyên |
| Input/output tensor, normalization | RGB, ImageNet | RGB, ImageNet | Giữ nguyên |
| Dataset split và seed | cố định | giống A | Giữ nguyên |
| Loss, optimizer, scheduler | theo A | theo A | Giữ nguyên |
| Metric, threshold | theo A | theo A | Giữ nguyên |
| Ngân sách và search space Optuna | theo A | theo A | Giữ nguyên |

> **Giải thích dễ hiểu: "đổi một thứ mỗi lần"**
> Giống thí nghiệm khoa học: muốn biết một loại phân bón có tốt không, ta chỉ đổi **phân bón**, còn giống cây, đất, lượng nước, ánh sáng đều giữ nguyên. Nếu đổi cả phân bón lẫn lượng nước, khi cây lớn nhanh hơn ta không biết nhờ cái nào.
>
> Ở đây "phân bón" là **encoder**. Mọi thứ khác (dữ liệu, loss, optimizer, metric, ngân sách Optuna) là "đất, nước, ánh sáng" và phải giống A. Chỉ các dòng **Thay đổi** trong bảng được khác. Nếu muốn đổi thêm bất kỳ thứ gì, phải ghi riêng và nêu lý do.

---

## 8. Smoke-test

> **Giải thích dễ hiểu**
> Smoke-test (kiểm tra khói) giống việc **bật máy lên xem có bốc khói không** trước khi dùng thật. Ta đưa vào model một tensor ngẫu nhiên và kiểm tra output có đúng hình dạng không. Test này không đánh giá độ chính xác, chỉ bắt các lỗi lệch shape, sai số kênh, hay nhầm index ngay từ đầu, trước khi tốn nhiều giờ huấn luyện.

### 8.1 Kiểm tra feature của encoder (xác nhận mục 2)

```python
import torch
from torchvision.models import mobilenet_v2

def test_encoder_feature_shapes():
    feats = mobilenet_v2(weights=None).features.eval()
    x = torch.randn(1, 3, 256, 256)
    expected = {
        1:  (1, 16, 128, 128),
        3:  (1, 24, 64, 64),
        6:  (1, 32, 32, 32),
        13: (1, 96, 16, 16),
        17: (1, 320, 8, 8),
    }
    got = {}
    with torch.no_grad():
        for i, layer in enumerate(feats):
            x = layer(x)
            if i in expected:
                got[i] = tuple(x.shape)
            if i == 17:
                break
    assert got == expected, f"Shape không khớp: {got}"
```

### 8.2 Kiểm tra toàn mô hình (tương tự W1-11)

```python
import torch
from models.mobilenetv2_unet import MobileNetV2UNet   # đường dẫn theo cấu trúc repo

def test_model_output_shape():
    for size in (128, 256):
        for p in (0.0, 0.1, 0.2):
            model = MobileNetV2UNet(encoder_weights=None, decoder_dropout=p).eval()
            x = torch.randn(2, 3, size, size)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (2, 1, size, size), (size, p, tuple(y.shape))
            assert y.dtype == torch.float32

def test_backward_with_loss():
    model = MobileNetV2UNet(encoder_weights=None)
    x = torch.randn(2, 3, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss = torch.nn.BCEWithLogitsLoss()(model(x), target)   # thay bằng loss của A
    loss.backward()
```

Việc xác nhận "không có sigmoid trong `forward()`" làm bằng cách **đọc code**, không dựa vào test.

### 8.3 Tiêu chí đạt

- [ ] Shape encoder ở 8.1 khớp bảng mục 2.
- [ ] Output `[B, 1, H, W]` với H = W ∈ {128, 256} và `decoder_dropout` ∈ {0, 0.1, 0.2}.
- [ ] `forward()` trả logits, không có sigmoid (đã đọc code).
- [ ] `loss.backward()` chạy được với loss của A, không lỗi shape/dtype.
- [ ] Đếm và ghi lại số tham số: encoder, decoder, tổng.

---

