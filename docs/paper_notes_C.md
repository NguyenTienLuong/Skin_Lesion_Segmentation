# 1. Tìm hiểu chung

- Phân đoạn tổn thương da (điểm thuộc tổn thương hay thuộc da lành) bằng mạng nơ ron tích chập dựa trên kiến trúc U-Net, dùng dữ liệu trong cuộc thi ISIC Challenge 2018, chỉ số Jaccard đạt ngưỡng 77,7%
- Theo tổ chức Y tế Thế giới, mỗi năm trên toàn cầu có từ 2 đến 3 triệu ca ung thư không phải u hắc tố và 132.000 ca u hắc tố ác tính. Dù chiếm chưa đến 6,5% tổng số ca ung thư da, u hắc tố là loại nguy hiểm nhất, gây khoảng 75% số ca tử vong do ung thư da.
  => Phát hiện sớm là yếu tố then chốt để tăng khả năng sống sót, tuy nhiên quan sát bằng mắt vẫn là kỹ thuật chuẩn đoán phổ biến nhất.
- Soi da là kỹ thuật khám các tổn thương da, nếu được thực hiện đúng cách có thể tăng độ chính xác việc chuẩn đoán từ 60% -> 75-84%.
- Tổ chức International Skin Imaging Collaboration (ISIC) sở hữu một bộ dữ liệu công khai quy mô lớn với hơn 20.000 ảnh soi da, tổ chức này tổ chức cuộc thi đánh giá thường niên về phân tích ảnh soi da từ năm 2016. Cuộc thi gồm 3 nhiệm vụ phân tích tổn thương:
  - Phân đoạn - segmentation
  - Trích xuất đặc trưng soi da
  - Phân loại

# 2. Mô hình U-Net34

- Nền tảng: CNN học hai loại thông tin. Một mạng tích chập dùng các bộ lọc nhỏ trượt trên ảnh để tạo ra các bản đồ đặc trưng (feature map). Càng đi sâu vào mạng, ảnh bị thu nhỏ dần (nhờ pooling hoặc bước nhảy stride) nhưng mỗi điểm nhìn được vùng rộng hơn của ảnh gốc. Vì vậy:

  - Lớp nông: độ phân giải cao, thấy chi tiết mịn (cạnh, kết cấu) nhưng chưa hiểu bức tranh tổng thể.

  - Lớp sâu: độ phân giải thấp, hiểu ngữ cảnh ("đây là một vết sắc tố") nhưng đã mất chi tiết vị trí chính xác.
    => Phân đoạn cần cả hai: biết cái gì, biết chính xác ở đâu => U-Net giải quyết.

- Kết hợp U-Net và ResNet
- U-Net (2015)
  - Là kiến trúc mã hóa-giải mã (encoder-decoder) được thiết kế cho phân đoạn ảnh y sinh, và cũng được dùng cho các bài toán phân đoạn ảnh khác như ảnh vệ tinh.
  - Encoder: thu nhỏ ảnh qua nhiều tầng. Mỗi lần thu nhỏ thì số kênh đặc trưng tăng gấp đôi (ví dụ 64 → 128 → 256 → 512), vì càng sâu càng cần nhiều loại đặc trưng phức tạp để bù cho việc mất kích thước.
  - Decoder: phóng to dần về kích thước ban đầu, nhưng nếu chỉ phóng to thì ảnh sẽ nhòe vì chi tiết đã mất ở encoder.
  - Skip connection (kết nối tắt): Ở mỗi tầng, decoder nối (concatenate) thêm đầu ra của encoder ở tầng cùng độ phân giải. Encoder "trả lại" chi tiết mịn còn decoder mang theo ngữ cảnh, nhờ đó vị trí được xác định chính xác dù mạng khá đơn giản. Hình chữ U của kiến trúc chính là do đường đi xuống rồi đi lên này.
  - Đầu ra là một ảnh có kích thước bằng ảnh đầu vào, một kênh (với bài toán phân đoạn nhị phân).
- ResNet: Người ta từng nghĩ mạng càng sâu càng tốt, nhưng thực tế khi mạng rất sâu thì lỗi trên chính tập huấn luyện lại tăng lên (degradation problem). Nguyên nhân không phải học thuộc, mà là mạng quá khó tối ưu. ResNet giải quyết bằng cách đổi câu hỏi mà mỗi khối phải trả lời:
  - Cách cũ: cho đầu vào x, hãy tự học ra đầu ra mong muốn H(x).
  - Cách ResNet: hãy học phần chênh lệch F(x) = H(x) − x, rồi cộng đầu vào lại: H(x) = F(x) + x.
  - ResNet34 gồm một lớp tích chập ban đầu, 16 khối (mỗi khối 2 lớp) và một lớp kết nối đầy đủ ở cuối.
- U-Net34 dùng ResNet34 đã huấn luyện trước làm nhánh mã hóa của U-Net:
  - Bỏ phần cuối: lớp adaptive pooling và lớp phân loại.
  - Giữ lại phần xương sống của ResNet làm encoder.
  - Bản đồ đặc trưng được lưu ở lớp đầu và ở các khối 3, 8, 14.
  - Trong quá trình tăng kích thước, nối các đầu ra này với đầu ra của các bước up-sampling.
  - Dùng bộ tối ưu Adam, hàm mất mát Binary Cross Entropy with Logits.

# 3. Huấn luyện

- Transfer learning: Encoder được khởi tạo từ ResNet34 đã học trên ImageNet (hơn một triệu ảnh, 1000 loại vật thể). Các lớp nông của nó đã biết nhận cạnh, góc, kết cấu, là những thứ ảnh da cũng có. Với chỉ khoảng 2,6 nghìn ảnh, huấn luyện từ đầu rất dễ bị overfitting, tức là mô hình học thuộc tập huấn luyện nhưng kém trên ảnh mới. Fine-tuning là tiếp tục huấn luyện từ các trọng số có sẵn đó, thay vì bắt đầu từ ngẫu nhiên.
- Pyramid transfer: U-Net là mạng tích chập hoàn toàn, không bị giới hạn bởi độ phân giải của ảnh đầu vào/đầu ra cố định => mạng trước tiên được huấn luyện với dữ liệu độ phân giải thấp để các lớp tích chập học thông tin ngữ cảnh => mạng được tinh chỉnh dần với dữ liệu độ phân giải cao hơn để các lớp tích chập học các chi tiết mịn.
  - Ban đầu huấn luyện mô hình với ảnh 128x128 => huấn luyện cùng mô hình với ảnh 256x256.
    => Nếu cấu hình oke hơn thì huấn luyện 256x256 => 512x512.

# 4. Learning rate schedule

- Quy trình:
  1. Đóng băng nhóm lớp đầu tiên.
  2. Xác định tốc độ học tối ưu: huấn luyện một batch với các tốc độ học khác nhau, bắt đầu từ rất thấp và tăng tuyến tính ở mỗi vòng lặp.
  3. Dùng chính sách tốc độ học tuần hoàn 1 chu kỳ để đạt hội tụ trong 30 epoch, dùng chiến lược Slanted Triangular Learning Rate.
  4. Bỏ đóng băng mô hình, chỉ giữ đóng băng các lớp chuẩn hóa batch, rồi lặp lại bước 2 và 3.

# 5. Chiến lược phân loại

- BestDice: chỉ giữ lại một mô hình có Dice cao nhất trên kiểm định và dùng riêng nó để dự đoán.
- Ensemble: đưa một ảnh mới qua cả 3 mô hình, lấy 3 kết quả dự đoán rồi kết hợp thành một mask cuối cùng.
- Pyramid Transfer: Tập train được chia thành 3 fold, gọi là A, B, C. Bài huấn luyện 3 lần, mỗi lần lấy một fold để kiểm định:

| Mô hình   | Huấn luyện trên | Kiểm định trên |
| :-------- | :-------------- | :------------- |
| Mô hình 1 | B + C           | A              |
| Mô hình 2 | A + C           | B              |
| Mô hình 3 | A + B           | C              |

# 6. Dữ liệu

- Bộ dữ liệu của cuộc thi "ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection":
  - Gồm 2597 ảnh training, 101 ảnh validation.
  - Mỗi mẫu gồm 1 ảnh soi da, một mặt nạ nhị phân khoanh vùng tổn thương.
  - Thu được từ nhiều loại máy soi da khác nhau, nhiều vị trí giải phẫu, nhiều bệnh nhân, nhiều cơ sở khác nhau.
  - Số tổn thương lành tính nhiều hơn ác tính.
  - Ảnh mặt nạ được mã hóa dưới dạng PNG xám 8bit, với 0 là nền, 255 tổn thương.
- Tất cả ảnh được đổi kích thước thành 128x128, 256x256, 512x512, được tiền xử lý để chỉnh cân bằng màu (không rõ phương pháp).
- Tăng cường dữ liệu:
  - Biến đổi dihedral: phép 8 đối xứng của hình vuông (4 góc xoay 90 độ nhân với lật hoặc không lật) => vì ảnh soi da không có hướng trên/dưới cố định.
  - Xoay (tối đa 44 độ): thêm biến thể về vị trí và kích thước của tổn thương.
  - Phóng to (tối đa 1,05): thêm biến thể về vị trí và kích thước của tổn thương.
  - Lật.
  - Thay đổi ánh sáng ngẫu nhiên: bổ sung cho bước cân bằng màu, giúp mô hình bớt phụ thuộc vào điều kiện chiếu sáng của từng thiết bị.
    => Phép biến đổi hình học phải áp dụng giống hệt lên cả ảnh và mask, còn phép đổi ánh sáng chỉ áp lên ảnh.
- Chia dữ liệu:
  - Train chính thức, chia 3 fold. Mỗi mô hình huấn luyện trên 2/3, kiểm định nội bộ trên 1/3 còn lại.
  - Validation chính thức, dùng để chấm điểm trực tuyến (BestDice đạt 75,5%) - ban tổ chức chấm.

# 7. Tiêu chí đánh giá

| Tiêu chí                       | Ý nghĩa                        | Kết quả                                 |
| :----------------------------- | :----------------------------- | :-------------------------------------- |
| Jaccard (IoU) trung bình       | Đo mức chồng khớp của mask     | 85,39%                                  |
| Threshold Jaccard (ngưỡng 65%) | Chỉ số chính thức của cuộc thi | 78,43% nội bộ, 75,5% ban giám khảo chấm |

## a. Chỉ số Jaccard (IoU)

- Với mỗi ảnh, gọi A là tập điểm ảnh mô hình dự đoán là tổn thương và B là tập điểm ảnh nhãn chuẩn cho là tổn thương:J = |A ∩ B| / |A ∪ B|
- Ví dụ: vùng tổn thương thật có 1000 điểm ảnh, mô hình tô 900 điểm ảnh, có 800 điểm trùng => TP = 800, FP = 100, FN = 200 => J = 800/1100 = 0,727

## b. Threshold Jaccard (ngưỡng 65%)

- Quy tắc:
  - Nếu Jaccard < 0,65, điểm của ảnh bằng 0
  - Ngược lại, bằng điểm Jaccard

## c. Chọn mô hình bằng chỉ số Dice

- Công thức:D = 2·|A ∩ B| / (|A| + |B|)
- Ví dụ: vùng tổn thương thật có 1000 điểm ảnh, mô hình tô 900 điểm ảnh, có 800 điểm trùng => TP = 800, FP = 100, FN = 200 => D = 1600/1900 = 0,842

- Chọn mô hình (BestDice) bằng cách giữ mô hình có Dice cao nhất trên tập kiểm định nội bộ (ví dụ: đánh giá bằng tập A, tập B & C được train)

# 8. Kết quả

- Các chiến lược:
  - Chiến lược Ensemble: Jaccard index 85,39%, Jaccard threhold 78,43% (cắt ở 65%).
  - Chiến lược BestDice: đạt điểm do ban giám khảo chấm 75,5% trên tập kiểm định chính thức.
- Các phân đoạn tốt nhất gần như giống hệt nhãn chuẩn.
- Các phân đoạn tệ nhất: thuật toán bị nhầm vì vết bút đánh dấu hoặc tấm kính bác sĩ dùng, tổn thương nhỏ, nghi ngờ nhãn sai.
  => Dùng U-Net để phân đoạn tổn thương da là oke.

# 9. Bảng thuật ngữ:

| Thuật ngữ                 | Nghĩa                                               |
| :------------------------ | :-------------------------------------------------- |
| segmentation              | phân đoạn (gán nhãn từng điểm ảnh)                  |
| encoder / decoder         | nhánh thu nhỏ / nhánh phóng to                      |
| skip connection           | kết nối tắt                                         |
| backbone                  | phần xương sống của mạng                            |
| pretrained                | đã huấn luyện trước                                 |
| fine-tuning               | tinh chỉnh từ trọng số có sẵn                       |
| freeze / unfreeze         | đóng băng / mở băng trọng số                        |
| learning rate             | tốc độ học (độ lớn bước cập nhật)                   |
| epoch / batch / iteration | lượt qua toàn dữ liệu / nhóm ảnh / một lần cập nhật |
| overfitting               | học thuộc tập huấn luyện                            |
| ground truth              | nhãn chuẩn                                          |
| ensemble                  | kết hợp nhiều mô hình                               |

**Author: Na**
