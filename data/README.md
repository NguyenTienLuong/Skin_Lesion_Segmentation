# DATA INFORMATION

## 1. Nguồn dữ liệu

Tải từ ISIC, tab năm 2018, hàng 1: https://challenge.isic-archive.com/data/#2018

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

### 2.4 Trùng lặp và overlap

| Hạng mục kiểm tra                | Số lượng vi phạm |
| -------------------------------- | ---------------- |
| **Overlap ID: dev ↔ val**        | 0                |
| **Overlap ID: dev ↔ test**       | 0                |
| **Overlap hash MD5: dev ↔ val**  | 0                |
| **Overlap hash MD5: dev ↔ test** | 0                |
| **Ảnh dev trùng nội dung nhau**  | 0                |

- development, validation chính thức và test không giao nhau theo ID, cũng không có file trùng từng byte
- Không có duplicate, overlap
