# Phân đoạn tổn thương da bằng U-Net và các chiến lược huấn luyện hiệu quả

> **Bài báo gốc:** _Skin lesion segmentation using a U-Net and good training strategies_

> **Tác giả:** Frederico Guth, Teofilo E. de Campos (Đại học Brasília, Brazil)

> **Nguồn:** arXiv:1811.11314v1 [cs.CV], 27/11/2018 (preprint – _work in progress_)

> **Cuộc thi:** ISIC Challenge 2018 – Task 1: Lesion Boundary Segmentation

---

## Mục lục

0. [Tóm tắt nhanh](#0-tóm-tắt-nhanh)
1. [Bối cảnh](#1-bối-cảnh)
2. [Bài toán thực chất là gì?](#2-bài-toán-thực-chất-là-gì)
3. [Mô hình U-Net34](#3-mô-hình-u-net34)
4. [Các chiến lược huấn luyện](#4-các-chiến-lược-huấn-luyện-phần-quan-trọng-nhất)
5. [Các thước đo đánh giá (có ví dụ số)](#5-các-thước-đo-đánh-giá)
6. [Kết quả](#6-kết-quả-đọc-thế-nào-cho-đúng)
7. [Phân tích lỗi](#7-những-chỗ-ai-làm-sai-và-vì-sao)
8. [Nhận xét phản biện](#8-nhận-xét-phản-biện)
9. [Thuật ngữ](#9-bảng-thuật-ngữ)
10. [Kết luận](#10-kết-luận)

---

## 0. Tóm tắt nhanh

- Tác giả lấy mạng **U-Net** (nhìn được cả tổng thể lẫn chi tiết), lắp **ResNet34 đã học sẵn** làm phần "mắt", rồi **dạy rất có phương pháp** để AI khoanh vùng tổn thương da chính xác trên bộ dữ liệu ISIC 2018.

**Điểm cần nhớ:** kiến trúc **không mới**; cái mới là **tổ hợp các cách huấn luyện**:

| #   | Chiến lược                      | Ví von                                         |
| --- | ------------------------------- | ---------------------------------------------- |
| 1   | Fine-tuning từ ImageNet         | Học sinh đã biết đọc, không dạy lại từ chữ cái |
| 2   | Pyramid transfer (128 → 256 px) | Phác hình khối trước, tô chi tiết sau          |
| 3   | LR finder + 1-cycle / STLR      | Chọn và điều chỉnh bước chân khi xuống đồi     |
| 4   | Freeze / Unfreeze               | Đừng làm hỏng phần đã giỏi                     |
| 5   | Data augmentation               | Một ảnh biến thành nhiều ảnh                   |
| 6   | BestDice + Ensemble 3-fold      | Giữ bản giỏi nhất, rồi hỏi ý kiến 3 người      |

**Kết quả chính:** Ensemble đạt **85,39% Jaccard** và **78,43% Threshold Jaccard** (validation tự chia); BestDice đạt **75,5%** trên validation chính thức online. Abstract nêu **77,5%**.

---

## 1. Bối cảnh

| Thông tin                                           | Số liệu (theo bài báo)         |
| --------------------------------------------------- | ------------------------------ |
| Ung thư da không phải melanoma                      | 2–3 triệu ca/năm trên toàn cầu |
| Ung thư da melanoma (hắc tố)                        | ~132.000 ca/năm                |
| Tỷ lệ melanoma trong tổng ca ung thư da             | < 6,5%                         |
| Tỷ lệ tử vong do ung thư da mà melanoma gây ra      | ~75%                           |
| Độ chính xác chẩn đoán bằng mắt thường (chuyên gia) | ~60%                           |
| Độ chính xác khi dùng dermoscopy (có đào tạo)       | 75%–84%                        |

- Melanoma **ít gặp nhưng rất nguy hiểm**; **phát hiện sớm** quyết định khả năng sống sót.
- Hiện nay **quan sát bằng mắt** vẫn là cách chẩn đoán phổ biến nhất.
- CNN đã đạt hoặc vượt con người ở nhiều tác vụ ảnh (phân loại ImageNet, ảnh mô bệnh học, phân loại ung thư da ở mức bác sĩ da liễu – Esteva et al., 2017) → có tiềm năng hỗ trợ chẩn đoán trên quy mô lớn.

**ISIC (International Skin Imaging Collaboration)** có bộ dữ liệu công khai > 20.000 ảnh soi da và tổ chức cuộc thi hằng năm từ 2016, gồm 3 tác vụ:

1. **Segmentation** (phân đoạn tổn thương) ← _bài báo này làm tác vụ này_
2. Dermoscopic feature extraction (trích xuất đặc trưng)
3. Classification (phân loại)

Các tác giả cho rằng đây là **công trình đầu tiên** áp dụng kiến trúc dựa trên U-Net **kết hợp** các chiến lược huấn luyện gần đây cho tác vụ này.

---

## 2. Bài toán thực chất là gì?

Cho một bức ảnh chụp da có một nốt đen. AI phải **tô trắng đúng vùng nốt đen và tô đen mọi thứ còn lại** (da lành, lông, viền kính…).

- **Đầu vào:** ảnh soi da (RGB).
- **Đầu ra:** **mặt nạ (mask)** nhị phân cùng kích thước ảnh: mỗi pixel là **0 = nền** hoặc **255 = tổn thương**.

Có thể xem AI đang trả lời câu hỏi có/không cho **từng pixel**: _"điểm này thuộc tổn thương hay không?"_ Ảnh 256×256 có 65.536 pixel, tức 65.536 câu hỏi.

Việc này gọi là **segmentation (phân đoạn)**. Khác với **classification** ("ảnh này có ung thư không?"), segmentation cần **vẽ chính xác ranh giới**, giúp bác sĩ đo kích thước và theo dõi tổn thương có lớn lên hay không.

**Thách thức:**

- Ảnh chụp bằng nhiều loại dermatoscope, nhiều vị trí cơ thể, nhiều bệnh nhân, nhiều cơ sở.
- Có nhiễu: lông, vết mực bút đánh dấu, viền kính soi.
- Ranh giới nhiều khi mờ; đôi khi khó biết **mô hình sai** hay **nhãn gốc chưa chuẩn**.

---

## 3. Mô hình U-Net34

Tên gọi ghép từ hai phần: **U-Net** (bộ khung) + **ResNet34** (phần "mắt" bên trong).

### 3.1. U-Net – vì sao có hình chữ U?

**Ví dụ xem bản đồ:**

- **Nhìn từ xa (thu nhỏ):** hiểu tổng thể ("đây là thành phố, kia là sông") nhưng **mất chi tiết** (không thấy từng con hẻm).
- **Nhìn từ gần (phóng to):** thấy rõ từng con hẻm nhưng **không hiểu bối cảnh**.

U-Net làm được cả hai:

```
Ảnh vào (256)                                         Mặt nạ ra (256)
   │ ───────────── skip connection ────────────────────► │
   ▼                                                      ▲
 (128) ─────────── skip connection ──────────────────► (128)
   ▼                                                      ▲
  (64) ──────────── skip connection ─────────────────► (64)
   ▼                                                      ▲
  (32) ──────────────► đáy chữ U ─────────────────────► (32)
 ENCODER (thu nhỏ, hiểu ngữ cảnh)        DECODER (phóng to, vẽ chi tiết)
```

- **Nửa trái – Encoder (đi xuống):** thu nhỏ dần ảnh (256 → 128 → 64 → 32), **mỗi bước nhân đôi số kênh đặc trưng**. Càng nhỏ thì mạng càng "hiểu" tổng thể (_"ở giữa có một cục đen lớn"_) nhưng đường biên càng mờ.
- **Nửa phải – Decoder (đi lên):** phóng to dần về kích thước ban đầu để vẽ mặt nạ.
- **Skip connection (đường nối tắt) – điểm hay nhất của U-Net:** ở mỗi mức kích thước, nửa trái **chuyển một bản sao chi tiết** sang nửa phải cùng mức. Nhờ vậy khi phóng to, nửa phải có **cả hai nguồn thông tin**:
  1. Hiểu biết tổng thể từ dưới đi lên ("vùng này là tổn thương").
  2. Chi tiết sắc nét từ bên trái ("đường biên chính xác nằm ở đây").

  Trong bài báo, phép "nối" này là **concatenation**: ghép hai khối đặc trưng chồng lên nhau để lớp kế tiếp dùng cả hai.

Không có skip connection thì ảnh phóng lại sẽ **bị nhòe**, vì chi tiết đã mất lúc thu nhỏ. U-Net cũng từng được dùng cho ảnh vệ tinh.

### 3.2. ResNet – vì sao cần "đường đi tắt" nữa?

Người ta từng nghĩ mạng càng sâu càng giỏi, nhưng thực tế mạng quá sâu lại **học tệ đi** (_degradation problem_).

ResNet giải quyết bằng ý rất đơn giản: thay vì bắt một khối lớp học **toàn bộ** phép biến đổi từ `x` ra `y`, cho nó học **phần cần chỉnh thêm**:

```
H(x) = x + F(x)        (F(x) = H(x) − x  là "hàm phần dư")
```

**Ví dụ:** thay vì viết lại cả bài văn từ đầu, bạn lấy bản nháp (`x`) rồi chỉ **ghi chú sửa vài chỗ** (`F(x)`). Nếu bản nháp đã tốt, phần sửa xấp xỉ 0 → dễ học. Đường `x` đi thẳng sang đầu ra gọi là **shortcut / identity**.

**ResNet34** gồm: 1 lớp tích chập đầu + **16 khối × 2 lớp** + 1 lớp fully-connected cuối (1 + 32 + 1 = 34 lớp).

### 3.3. Ghép thành U-Net34

U-Net gốc có nửa trái khá đơn giản. Tác giả **thay nửa trái bằng ResNet34** – mạng trích đặc trưng rất mạnh và **đã được huấn luyện sẵn** trên ImageNet.

1. Lấy ResNet34 pretrain, **bỏ phần cuối** dùng để phân loại (từ lớp adaptive pooling trở đi), chỉ giữ phần backbone.
2. **Lưu đầu ra** tại 4 điểm: lớp đầu tiên, khối **thứ 3**, **thứ 8**, **thứ 14** (trên tổng 16 khối). Đây là các điểm sẽ nối tắt sang decoder.
3. Decoder phóng to dần và **concatenate** với 4 đầu ra đã lưu.
4. **Optimizer:** Adam. **Loss:** Binary Cross Entropy with Logits.

> Cài đặt dựa trên thư viện **fast.ai**.

---

## 4. Các chiến lược huấn luyện (phần quan trọng nhất)

Hãy hình dung việc **dạy một học sinh**.

### 4.1. Fine-tuning từ ImageNet – đừng dạy lại từ chữ cái

ResNet34 đã xem hàng triệu ảnh, nên biết nhận ra cạnh, góc, kết cấu, màu, hình khối. Những kỹ năng này dùng lại được cho ảnh da. Chỉ cần dạy thêm "tổn thương da trông thế nào" → **học nhanh hơn, chính xác hơn**, đặc biệt khi dữ liệu ít (chỉ 2.597 ảnh).

### 4.2. Pyramid transfer – học từ dễ đến khó

U-Net là **mạng tích chập hoàn toàn** nên (về lý thuyết) không bị ràng buộc kích thước ảnh cố định, từ đó mượn ý tưởng **image pyramid**:

1. Học trên ảnh **128×128**: nhẹ, nhanh, nắm bức tranh tổng thể (nốt đen ở đâu, to cỡ nào).
2. Lấy chính mô hình đó, học tiếp trên ảnh **256×256**: tinh chỉnh đường biên, chi tiết nhỏ.
3. _(Chỉ đề xuất)_ lên **512×512**. Nhóm không thực hiện vì máy cấu hình thấp, **hết bộ nhớ GPU**.

_Giống học vẽ: phác hình khối trước, tô chi tiết sau._

### 4.3. Learning rate – bước chân khi xuống đồi

Huấn luyện giống như **đi xuống đáy thung lũng trong sương mù** để tìm điểm thấp nhất (loss nhỏ nhất). **Learning rate (LR)** là **độ dài mỗi bước chân**:

- Bước quá ngắn → đi rất lâu mới tới.
- Bước quá dài → nhảy qua đáy, loạng choạng, thậm chí văng ra xa.

**(a) LR finder – tìm độ dài bước tốt nhất** (Smith, 2018; có trong fast.ai)

- Huấn luyện trên một batch, bắt đầu từ LR rất nhỏ và **tăng tuyến tính** sau mỗi iteration, vừa tăng vừa ghi lại loss.
- Nhìn đồ thị (Hình 1a): ban đầu loss giảm; đến khi LR quá lớn thì loss **tăng vọt**.
- Chọn LR ở đoạn loss **đang giảm dốc nhất** (không chọn điểm thấp nhất, vì điểm đó sát mép "vực").

**(b) 1-cycle / STLR – thay đổi bước chân theo thời gian** (Howard & Ruder, 2018)

- Nhìn Hình 1b: LR **tăng vọt rất nhanh** lên đỉnh (~0,15), rồi **giảm dần tuyến tính** về gần 0 trong khoảng 1000 iteration.
- Đoạn LR lớn giúp mạng "khám phá" nhanh, tránh kẹt ở chỗ xấu.
- Đoạn LR giảm dần giúp mạng "đặt chân nhẹ nhàng" vào đáy.
- Kết quả: hội tụ chỉ sau khoảng **30 epoch** (1 epoch = một lượt học hết toàn bộ dữ liệu) – gọi là **super-convergence**.

### 4.4. Freeze và Unfreeze – đừng phá thứ đang tốt

Nửa trái (ResNet) đã thông minh sẵn; nửa phải (Decoder) mới khởi tạo ngẫu nhiên, còn "ngây ngô". Nếu cho cả hai học cùng lúc, sai số lớn của nửa phải có thể **làm hỏng** kiến thức tốt của nửa trái. Quy trình trong bài:

1. **Freeze** nhóm lớp đầu tiên (phần pretrain), chỉ dạy phần mới.
2. Chạy **LR finder** → huấn luyện theo **STLR**.
3. **Unfreeze** mô hình (**vẫn giữ đóng băng các lớp Batch Normalization**).
4. **Lặp lại** LR finder → STLR một lần nữa.

### 4.5. Hàm loss – cách "chấm điểm" mô hình

- Dùng **Binary Cross Entropy (BCE) with Logits**: với **từng pixel**, so sánh dự đoán của AI với đáp án (0 hoặc 1); sai nhiều thì phạt nặng; rồi lấy trung bình.
- **Vì sao không dùng thẳng Jaccard** (thước đo cuối cùng)? Jaccard tính theo ngưỡng cứng nên **không "trơn"**, máy không tính được hướng cải thiện (không khả vi). Có bản "mềm" là **soft Jaccard**, nhưng thử nghiệm sơ bộ của nhóm **không tốt hơn BCE**, nên họ giữ BCE.

### 4.6. Data augmentation – một ảnh thành nhiều ảnh

Chỉ có 2.597 ảnh là ít, nên mỗi lần đưa ảnh vào mạng, họ **biến đổi ngẫu nhiên**:

- Biến đổi dihedral (xoay/lật theo nhóm đối xứng của hình vuông)
- Xoay (tối đa 44°)
- Zoom (tối đa 1,05)
- Lật ảnh
- Thay đổi ánh sáng ngẫu nhiên

Nhờ đó mạng thấy cùng một tổn thương dưới nhiều dạng, **không học vẹt (overfitting)** từng bức ảnh và nhận diện được trong nhiều điều kiện chụp.

**Dữ liệu:** ISIC 2018 Task 1 – **2.597 ảnh train** + **101 ảnh validation** (chính thức), **không dùng dữ liệu ngoài**. Mặt nạ là PNG xám 8-bit: 0 = nền, 255 = tổn thương. Ảnh có đủ lành tính/ác tính (ác tính được đại diện quá mức so với thực tế). Tiền xử lý: resize về 128×128, 256×256, 512×512 và điều chỉnh cân bằng màu.

### 4.7. BestDice và Ensemble

**BestDice – giữ lại "phiên bản giỏi nhất" của mô hình**

Trong khi học, chất lượng mô hình dao động: epoch này tốt, epoch kia tệ. Sau mỗi epoch, mô hình làm "bài kiểm tra" trên tập validation, chấm bằng **Dice**. **Epoch nào điểm cao nhất thì lưu lại** và dùng mô hình ở epoch đó. Vậy **BestDice = mô hình tốt nhất trong quá trình học**.

**Ensemble – hỏi ý kiến 3 người thay vì 1**

Chia dữ liệu train thành **3 fold** (F1, F2, F3), huấn luyện **3 mô hình riêng**, mỗi mô hình lấy bản BestDice:

| Mô hình   | Học trên | Kiểm tra trên |
| --------- | -------- | ------------- |
| Mô hình 1 | F2 + F3  | F1            |
| Mô hình 2 | F1 + F3  | F2            |
| Mô hình 3 | F1 + F2  | F3            |

Vì học trên các tập khác nhau nên **mỗi mô hình mắc lỗi khác nhau**. Khi có ảnh mới, cả 3 cùng dự đoán rồi gộp lại, lỗi lẻ của một mô hình bị hai mô hình kia "kéo lại". Ví dụ bỏ phiếu đa số cho một pixel:

| Pixel | Mô hình 1 | Mô hình 2 | Mô hình 3 | Kết luận             |
| ----- | --------- | --------- | --------- | -------------------- |
| A     | Bệnh      | Bệnh      | Bệnh      | **Bệnh** (chắc chắn) |
| B     | Bệnh      | Bệnh      | Lành      | **Bệnh** (2/3)       |
| C     | Bệnh      | Lành      | Lành      | **Lành** (2/3)       |

> ⚠️ **Lưu ý:** bài báo **không mô tả** cách gộp kết quả 3 mô hình. Hai cách phổ biến là **trung bình xác suất theo pixel** (rồi áp ngưỡng) hoặc **bỏ phiếu đa số**; ví dụ trên chỉ để minh họa ý tưởng, không phải thông tin xác nhận từ bài báo.

---

## 5. Các thước đo đánh giá

Gọi `A` là vùng AI dự đoán, `B` là vùng bác sĩ vẽ (ground truth).

| Thước đo              | Công thức                                                       | Ý nghĩa                                                     |
| --------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| **Jaccard (IoU)**     | `\|A ∩ B\| / \|A ∪ B\|`                                         | Phần chung ÷ phần hợp lại; 100% = khớp hoàn hảo             |
| **Dice**              | `2·\|A ∩ B\| / (\|A\| + \|B\|)`                                 | Tương tự Jaccard nhưng "dễ tính" hơn; dùng để chọn BestDice |
| **Threshold Jaccard** | Jaccard từng ảnh; **< 65% thì tính bằng 0**; rồi lấy trung bình | Thước đo chính thức của ISIC 2018; phạt nặng ảnh làm kém    |

### Ví dụ số

Bác sĩ vẽ vùng bệnh **100 pixel**. AI cũng vẽ **100 pixel**, trong đó **80 pixel trùng**.

- **Jaccard** = 80 / (100 + 100 − 80) = 80 / 120 ≈ **66,7%**
- **Dice** = 2 × 80 / (100 + 100) = **80%**
- **Threshold Jaccard:** 66,7% ≥ 65% → **giữ nguyên 66,7%**.

Nếu AI chỉ trùng **60 pixel**: Jaccard = 60/140 ≈ 43% < 65% → **bị tính là 0**.

→ Mục đích là **buộc mô hình không được bỏ rơi ảnh khó**. Vì vậy Threshold Jaccard luôn thấp hơn Jaccard thường (78,43% so với 85,39% trong bài).

---

## 6. Kết quả – đọc thế nào cho đúng?

| Con số                       | Đo ở đâu                                | Ý nghĩa                                 |
| ---------------------------- | --------------------------------------- | --------------------------------------- |
| **85,39% Jaccard**           | Validation tự chia (Ensemble)           | Mức chồng lấp trung bình rất cao        |
| **78,43% Threshold Jaccard** | Cùng tập trên (ngưỡng 65%)              | Sau khi "phạt" ảnh làm kém              |
| **75,5%**                    | Validation chính thức online (BestDice) | Điểm khách quan hơn vì ban tổ chức chấm |
| **77,5%**                    | Ghi trong abstract                      | Con số tóm tắt của bài                  |
| _76,5%_                      | _Quán quân ISIC 2017 (Jaccard TB)_      | _Tham chiếu, khác tập và khác cách đo_  |

- Về định tính, các phân đoạn tốt nhất **gần như trùng khít** ground truth; sai khác chỉ là đường viền mỏng (Hình 2a).
- Kết luận của nhóm: một mạng **đơn giản, end-to-end** vẫn cho phân đoạn chi tiết nếu huấn luyện tốt.
- ⚠️ Các con số đo trên **các tập khác nhau**, đừng nhầm lẫn. Đặc biệt, so 85,39% với 76,5% của quán quân 2017 **không hoàn toàn công bằng** (khác dữ liệu, khác thước đo).

---

## 7. Những chỗ AI làm sai (và vì sao)

| Trường hợp                        | Vì sao AI sai                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------- |
| **Vết mực bút / viền kính soi**   | Màu đậm, nét rõ nên AI tưởng là tổn thương                                    |
| **Tổn thương quá nhỏ** so với ảnh | Ít pixel nên dễ bị bỏ sót; thu ảnh về 128/256 px càng làm vật nhỏ mờ đi       |
| **Nhãn gốc không chắc chuẩn**     | Ranh giới bác sĩ vẽ đôi khi tùy ý; AI "sai" nhưng có thể do đáp án chưa chuẩn |
| **Ranh giới mờ, vùng đồng màu**   | Dự đoán dễ thiếu hoặc thừa so với nhãn (Hình 2b)                              |

---

## 8. Bảng thuật ngữ

| Thuật ngữ                           | Giải thích ngắn                                                  |
| ----------------------------------- | ---------------------------------------------------------------- |
| **Segmentation**                    | Phân đoạn: gán nhãn cho từng pixel                               |
| **Encoder / Decoder**               | Nhánh thu nhỏ để hiểu ngữ cảnh / nhánh phóng to để vẽ chi tiết   |
| **Skip connection**                 | Đường nối tắt truyền chi tiết từ encoder sang decoder            |
| **Concatenation**                   | Ghép chồng hai khối đặc trưng lại với nhau                       |
| **Residual learning**               | Học phần dư `F(x) = H(x) − x` thay vì học thẳng `H(x)`           |
| **Transfer learning / Fine-tuning** | Dùng mô hình đã học sẵn rồi tinh chỉnh cho bài toán mới          |
| **Image pyramid**                   | Xử lý ảnh ở nhiều độ phân giải, từ thấp đến cao                  |
| **Learning rate (LR)**              | Độ dài "bước chân" khi cập nhật trọng số                         |
| **LR finder**                       | Thử LR tăng dần để chọn LR tốt                                   |
| **1-cycle / STLR**                  | Lịch LR tăng rất nhanh rồi giảm dần đều                          |
| **Super-convergence**               | Hội tụ rất nhanh nhờ LR lớn theo chu kỳ                          |
| **Epoch**                           | Một lượt học hết toàn bộ dữ liệu train                           |
| **Freeze / Unfreeze**               | Khóa / mở khóa trọng số của các lớp                              |
| **Batch Normalization**             | Chuẩn hóa đầu ra từng lớp để huấn luyện ổn định                  |
| **BCE with Logits**                 | Loss nhị phân theo từng pixel, đã gộp sigmoid cho ổn định số học |
| **Soft Jaccard**                    | Biến thể khả vi của Jaccard dùng làm loss                        |
| **Data augmentation**               | Biến đổi ảnh ngẫu nhiên để tăng độ đa dạng dữ liệu               |
| **BestDice**                        | Chọn checkpoint có Dice cao nhất trên tập validation             |
| **Ensemble**                        | Kết hợp dự đoán của nhiều mô hình                                |
| **k-fold cross-validation**         | Chia dữ liệu thành k phần, lần lượt dùng 1 phần để kiểm định     |
| **Overfitting (học vẹt)**           | Mô hình nhớ dữ liệu train mà không tổng quát được                |

---

## 10. Kết luận

- Bài báo dùng **U-Net34** (U-Net + ResNet34 pretrain) cho phân đoạn tổn thương da.
- **Điểm mấu chốt** là **tổ hợp các chiến lược huấn luyện hiện đại**: transfer learning, pyramid transfer, LR finder + 1-cycle (STLR), freeze/unfreeze, augmentation, chọn mô hình theo Dice và ensemble 3-fold → một mạng **tương đối đơn giản** vẫn cho kết quả chi tiết, chính xác.
- **Điểm yếu còn lại:** tổn thương nhỏ, vật thể gây nhiễu (mực bút, viền kính) và nhãn gốc không nhất quán.
- **Hướng mở rộng gợi ý:** thử 512×512, dùng loss gần thước đo hơn (Jaccard/Dice loss), xử lý nhiễu và tổn thương nhỏ, làm ablation study để biết rõ từng kỹ thuật đóng góp bao nhiêu.

---
