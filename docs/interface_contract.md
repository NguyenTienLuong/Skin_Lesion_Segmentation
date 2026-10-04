# Interface Contract: Dữ liệu và Đánh giá (ISIC 2018 Task 1)

- Phiên bản: v0.1 (bản nháp, chờ xác nhận, 04/10/2026)
- Owner: Na
- Phạm vi: kiểm kê cặp ảnh–mask và chốt interface dữ liệu

---

## 1. Nguồn dữ liệu

| Thành phần                                        | Nguồn                                                                                                      | Ghi chú                                                                                                          |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Ảnh Training / Validation / Test và mask Training | Kaggle: `tschandl/isic2018-challenge-task1-data-segmentation` (tải bằng `kagglehub`)                       | code: https://colab.research.google.com/drive/10u1IDBBHMzCvVAfM0_u_bGbIcfHfkBvd?usp=sharing                      |
| Mask Validation và Test (Task 1)                  | challenge.isic-archive.com, tab 2018: https://challenge.isic-archive.com/data/?utm_source=chatgpt.com#2018 | Files: `ISIC2018_Task1_Validation_GroundTruth.zip`, `ISIC2018_Task1_Test_GroundTruth.zip`. Không lấy hàng Task 2 |

Bản Kaggle không có mask cho Validation và Test nên hai file mask phải tải riêng từ ISIC.

## 2. Kiểm kê cặp ảnh–mask

### 2.1 Số lượng

| Tập                    | Ảnh  | Mask | Ghép đủ | Thiếu mask | Thiếu ảnh | ID lặp |
| ---------------------- | ---- | ---- | ------- | ---------- | --------- | ------ |
| Development (Training) | 2594 | 2594 | 2594    | 0          | 0         | 0      |
| Validation chính thức  | 100  | 100  | 100     | 0          | 0         | 0      |
| Test                   | 1000 | 1000 | 1000    | 0          | 0         | 0      |

- Cả 3 tập đều khớp 1-1 giữa ảnh và mask theo ID đã chuẩn hóa
- Mỗi ID có đúng 1 ảnh và 1 mask, không có ảnh mồ côi, mask mồ côi
- Không có ID bị lặp - không có nhiều file cùng map vào một ID
- Overlap theo ID giữa các tập: dev ∩ val = dev ∩ test = val ∩ test = ∅. Không có rò rỉ dữ liệu theo ID giữa development, validation và test.

- Số liệu paper mô tả: 2597 training, 101 validation. Số kiểm đếm thực tế: 2594 và 100.
- Mỗi thư mục Kaggle có thêm 2 file không phải ảnh (txt), bị bỏ qua khi quét. Thư mục `.ipynb_checkpoints` bị loại.

### 2.2 Cách ghép cặp

- Ghép theo **ID**, không ghép theo thứ tự sort.
- Chuẩn hóa ID: lấy `stem` của tên file, loại hậu tố `_segmentation` ở mask. Ví dụ `ISIC_0000000.jpg` ↔ `ISIC_0000000_segmentation.png`.

### 2.3 Kiểm tra kích thước và giá trị mask

| Tập  | Ảnh = mask về kích thước | Giá trị pixel mask                       | Cặp hợp lệ |
| ---- | ------------------------ | ---------------------------------------- | ---------- |
| Dev  | 2594/2594                | `{0, 255}` ở cả 2594 mask                | 2594/2594  |
| Val  | 100/100                  | `{0, 255}`                               | 100/100    |
| Test | 1000/1000                | `{0, 255}` và một số mask chỉ có `{255}` | 1000/1000  |

- 2594/2594 cặp hợp lệ, bảng lỗi theo loại rỗng: không có size*mismatch, mask_not_binary_0_255, missing*\_ hay duplicate\_\_.
- Kích thước ảnh gốc bằng kích thước mask ở cả 2594 cặp.
- Cả 2594 mask chỉ có đúng hai giá trị pixel {0, 255}, khớp mô tả của paper (PNG xám 8-bit, 0 là nền, 255 là tổn thương). Không có mask nào lẫn giá trị trung gian, nên chưa cần xử lý gì thêm.
- Ảnh không đồng nhất về kích thước - có 206 ảnh kích thước ảnh gốc khác nhau

### 2.4 Trùng lặp và overlap

| Hạng mục kiểm tra                | Số lượng vi phạm |
| -------------------------------- | ---------------- |
| **Overlap ID: dev ↔ val**        | 0                |
| **Overlap ID: dev ↔ test**       | 0                |
| **Overlap hash MD5: dev ↔ val**  | 0                |
| **Overlap hash MD5: dev ↔ test** | 0                |
| **Ảnh dev trùng nội dung nhau**  | 0                |

- Overlap hash là kiểm tra xem có ảnh nào ở tập này giống hệt về nội dung với ảnh ở tập kia hay không, dù tên file có khác nhau
- development, validation chính thức và test không giao nhau theo ID, cũng không có file trùng từng byte
- Không có duplicate, overlap

### 2.6 Output

| File                     | Nội dung              | Số dòng | Trạng thái |
| ------------------------ | --------------------- | ------- | ---------- |
| `data/data_manifest.csv` | Development           | 2594    | Đã lưu     |
| `data/val_manifest.csv`  | Validation chính thức | 100     | Đã lưu     |
| `data/test_manifest.csv` | Test                  | 1000    | Đã lưu     |
| `data/data_errors.csv`   | Dòng lỗi của dev      | 0       | Đã lưu     |

Cột: `image_id, image_path, mask_path, image_width, image_height, mask_width, mask_height, mask_values, pair_valid, error_reason`. Đường dẫn tương đối so với thư mục gốc dữ liệu `isic2018/` (`dev_images/`, `dev_masks/`, `val_images/`, `val_masks/`, `test_images/`, `test_masks/`). Sắp xếp theo `image_id`.

## 3. Danh sách việc cần chốt

- Quy tắc dùng từng tập dev, validation, test: có làm giống như bài báo không, chia theo chiến lược như bài báo?
- Interface dữ liệu: Preprocessing và Augmentation như thế nào
- split train: data/tuning_split.csv chưa được tạo cần xem lại vai trò của các tập
- Metric: có dùng các chỉ số như bài báo không? Jaccard, Dice, Threshold Jaccard,...
