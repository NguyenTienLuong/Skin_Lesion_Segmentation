# HƯỚNG DẪN CHẠY PIPELINE U-NET34: BASELINE → OPTUNA → TOP 5 → 256 → 3-FOLD → TEST

Tài liệu này mô tả **đúng theo các file hiện có trong project** `isic2018_project_from_ipynb`.

> Chạy mọi lệnh từ **thư mục gốc của project**:
>
> ```bash
> cd isic2018_project
> ```

---

## 1. Tổng quan thứ tự chạy

```text
Chuẩn bị dữ liệu
      │
      ▼
Baseline (nếu cần)
      │
      ▼
Optuna HPO @128
      │
      ├── COMPLETE trial
      │       ↓
      │   TopKManager tự cập nhật
      │   current Top 5 @128
      │
      ▼
Dừng HPO khi đủ ngân sách
      │
      ▼
Kiểm tra Top 5 @128
      │
      ▼
Train Top 5 @256
      │
      ▼
confirmation_results.csv
      │
      ▼
Chọn winner @256
      │
      ▼
Khóa winner vào configs/final/unet34.yaml
      │
      ▼
Final 3-fold training
      │
      ▼
3 checkpoint: fold_1, fold_2, fold_3
      │
      ▼
3-fold probability ensemble
      │
      ▼
Held-out test
```

---

# 2. Các file chính dùng để chạy

| Giai đoạn | File chạy |
|---|---|
| Nạp dữ liệu và chia tập lúc chạy | `src/data/loader.py` |
| Baseline | `scripts/baseline/train_unet34.py` |
| Optuna HPO @128 | `scripts/optuna/run_unet34_optuna.py` |
| Export toàn bộ trial | `scripts/optuna/export_trials.py` |
| Train Top 5 @256 | `scripts/optuna/confirm_top5_256.py` |
| Final 3-fold | `scripts/final/train_3fold.py` |
| Ensemble held-out test | `scripts/final/ensemble_test.py` |
| Đánh giá một checkpoint | `scripts/evaluation/evaluation_unet34.py` |

---

# 3. Bước 0 — Cài thư viện

```bash
pip install -r requirements.txt
```

Nên chạy từ root project để các đường dẫn tương đối trong config hoạt động đúng.

---

# 4. Bước 1 — Chuẩn bị dữ liệu

Không cần tạo manifest hoặc các file `data/splits/*.csv`. Loader tự tải/cache
dataset bằng KaggleHub khi `data.data_root: null`, sau đó chia train/validation
hoặc K-fold trực tiếp trong bộ nhớ bằng `seed` trong config.

Nguồn dữ liệu được dùng như sau:

- Train: input và mask từ dataset Kaggle.
- Validation/test chính thức: input từ Kaggle, mask từ `data/` local.
- `data.data_root`: có thể đặt bằng đường dẫn do
  `kagglehub.dataset_download(...)` trả về.
- `data.mask_root`: để `null` nếu mask validation/test nằm trong `data/` của
  project.

`scripts/prepare_data.py` chỉ còn là tiện ích tùy chọn nếu cần xuất CSV để
kiểm toán hoặc lưu lại một split cố định; pipeline training không phụ thuộc
vào các CSV này.

---

# 5. Bước 2 — Chạy baseline

Baseline dùng config:

```text
configs/baseline/unet34.yaml
```

Chạy:

```bash
python scripts/baseline/train_unet34.py \
  --config configs/baseline/unet34.yaml
```

Pipeline baseline:

```text
official training → deterministic 3-fold
  └── mỗi fold:
      train 2 fold (có augmentation)
      LR finder chỉ dùng training loss của train fold
      fold còn lại (không augmentation) chọn Best Dice
      128×128 frozen/unfrozen → 256×256 frozen/unfrozen
official validation (không augmentation)
  ├── đánh giá từng Best-Dice fold
  └── đánh giá ensemble trung bình xác suất 3 fold
```

Artifact chính:

```text
checkpoints/baseline/unet34/
├── fold_1/baseline_best.pt
├── fold_2/baseline_best.pt
└── fold_3/baseline_best.pt

experiments/baseline/unet34/
├── baseline_fold_metrics.csv
└── baseline_validation_summary.json
```

Baseline **không bắt buộc phải chạy trước Optuna về mặt code**, nhưng nên chạy để có mốc so sánh.

---

# 6. Bước 3 — Chạy Optuna @128

File chạy:

```text
scripts/optuna/run_unet34_optuna.py
```

Config:

```text
configs/optuna/unet34.yaml
```

Protocol dữ liệu HPO:

```text
official training
├── train subset: có augmentation, cập nhật trọng số, LR finder dùng training loss
└── validation subset: không augmentation, chỉ prune/chọn Best Dice/xếp trial
```

Official validation và test không tham gia chọn hyperparameter. LR finder không
nhận batch từ validation subset.

## 6.1. Chạy theo time budget

Ví dụ chạy 6 giờ:

```bash
python scripts/optuna/run_unet34_optuna.py \
  --config configs/optuna/unet34.yaml \
  --timeout-seconds 21600
```

## 6.2. Hoặc chạy theo số trial

Ví dụ:

```bash
python scripts/optuna/run_unet34_optuna.py \
  --config configs/optuna/unet34.yaml \
  --n-trials 10
```

> Không nên hiểu `--n-trials 10` là tổng study chỉ có 10 trial. Nếu database đã có trial cũ, lệnh này chạy thêm trial mới.

---

# 7. Nếu hết phiên Colab/Kaggle thì resume Optuna như thế nào?

Chạy lại **chính file này**:

```bash
python scripts/optuna/run_unet34_optuna.py \
  --config configs/optuna/unet34.yaml \
  --timeout-seconds 21600
```

Không dùng file khác để resume.

Config hiện tại dùng:

```text
study.name:
    unet34_isic_hpo_best_dice

study.storage:
    experiments/optuna/unet34/optuna.db
```

`create_or_load_study()` dùng:

```python
load_if_exists=True
```

nên lần sau Optuna đọc lại cùng `optuna.db` và tiếp tục study cũ.

```text
Session 1
   ↓
trial 0, 1, 2, ...
   ↓
optuna.db
   ↓

Session 2
   ↓
load cùng optuna.db
   ↓
trial tiếp theo
```

Trial đang train dở khi runtime chết **không được resume theo epoch**; study tiếp tục bằng trial mới.

---

# 8. Top 5 @128 được lấy ở đâu?

## Không cần chạy một file riêng để "lấy Top 5"

Top 5 được quản lý **tự động trong quá trình Optuna chạy**.

Flow bên trong:

```text
run_unet34_optuna.py
        ↓
UNet34Objective
        ↓
sample_configuration(trial)
        ↓
frozen 30 epoch
        ↓
prune?
   ├── YES → PRUNED
   └── NO
        ↓
unfrozen 30 epoch
        ↓
COMPLETE
        ↓
TopKManager.update(...)
        ↓
cập nhật current Top 5
```

File code quản lý Top 5:

```text
src/hpo/topk_manager.py
```

Sau mỗi trial `COMPLETE`, TopKManager tự:

1. so Dice của trial mới với các candidate đang có;
2. giữ tối đa 5 trial tốt nhất;
3. lưu checkpoint nếu trial nằm trong Top 5;
4. xóa checkpoint bị đẩy khỏi Top 5;
5. cập nhật `top5_manifest_dice.csv`.

Artifact:

```text
experiments/optuna/unet34/top5_manifest_dice.csv
```

và checkpoint:

```text
checkpoints/optuna/unet34/top5_128_dice/
├── trial_XXXX.pt
├── trial_XXXX.pt
├── trial_XXXX.pt
├── trial_XXXX.pt
└── trial_XXXX.pt
```

Do đó sau khi HPO xong, **không cần một bước “extract Top 5” riêng**.

---

# 9. Bước 4 — Export danh sách trial

Bước này **không bắt buộc để train Top 5**, nhưng nên chạy để lưu bảng phục vụ kiểm tra/báo cáo.

```bash
python scripts/optuna/export_trials.py \
  --config configs/optuna/unet34.yaml
```

Output:

```text
experiments/optuna/unet34/trials.csv
```

File này chứa toàn bộ trial trong study.

---

# 10. Kiểm tra trước khi train Top 5 @256

Cần đảm bảo:

```text
experiments/optuna/unet34/top5_manifest_dice.csv
```

có **5 candidate**.

Nếu chưa đủ 5 trial COMPLETE đủ tốt để TopKManager giữ 5 candidate, chạy Optuna tiếp:

```bash
python scripts/optuna/run_unet34_optuna.py \
  --config configs/optuna/unet34.yaml \
  --timeout-seconds 21600
```

Theo protocol của project:

- mục tiêu ít nhất 20 attempted trials;
- Top 5 chỉ lấy từ trial `COMPLETE`;
- `PRUNED` và `FAILED` không nằm trong Top 5.

---

# 11. Bước 5 — Train Top 5 từ 128 lên 256

Khi `top5_manifest_dice.csv` đã có đủ 5 candidate, chạy:

```bash
python scripts/optuna/confirm_top5_256.py \
  --config configs/optuna/unet34.yaml
```

File này đọc:

```text
experiments/optuna/unet34/top5_manifest_dice.csv
```

và với từng candidate:

```text
checkpoint @128
      ↓
load trial.params
      ↓
configuration_from_params(params)
      ↓
khôi phục đúng:
  batch_size
  optimizer
  optimizer params
  loss
  loss params
      ↓
load model weights @128
      ↓
train @256
  frozen 30 epoch
  unfrozen 30 epoch
      ↓
Dice @256
```

Ở bước confirmation @256:

```text
KHÔNG Optuna sample lại
KHÔNG pruning
KHÔNG đổi configuration
```

Mỗi Top-5 candidate giữ đúng configuration mà Optuna đã tìm được ở 128.

---

# 12. Output của Top 5 @256

Checkpoint:

```text
checkpoints/optuna/unet34/confirmation_256_dice/
├── trial_XXXX_256.pt
├── trial_XXXX_256.pt
├── trial_XXXX_256.pt
├── trial_XXXX_256.pt
├── trial_XXXX_256.pt
└── confirmation_results.csv
```

Quan trọng nhất:

```text
checkpoints/optuna/unet34/confirmation_256_dice/confirmation_results.csv
```

File này được sort giảm dần theo:

```text
Dice @256
```

Nên:

```text
dòng đầu tiên = winner
```

Các cột hiện tại:

```text
trial
score
dice
jaccard
params_json
checkpoint
```

Với cấu hình hiện tại, `score` chính là Dice dùng để xếp hạng candidate.

Script cũng in:

```text
winner trial: X
```

---

# 13. Bước 6 — Khóa winner vào final config

## Đây là bước hiện tại CHƯA tự động trong code

Sau:

```bash
python scripts/optuna/confirm_top5_256.py ...
```

script **không tự ghi winner vào**:

```text
configs/final/unet34.yaml
```

Vì vậy cần mở:

```text
checkpoints/optuna/unet34/confirmation_256_dice/confirmation_results.csv
```

lấy `params_json` ở dòng đầu tiên và đưa vào:

```text
configs/final/unet34.yaml
```

---

## 13.1. Ví dụ winner là AdamW + Tversky

Giả sử `params_json`:

```json
{
  "batch_size": 16,
  "optimizer": "AdamW",
  "weight_decay": 0.000031,
  "loss_type": "Tversky",
  "alpha": 0.63
}
```

thì sửa `configs/final/unet34.yaml`:

```yaml
optimizer:
  name: AdamW
  weight_decay: 0.000031

loss:
  name: Tversky
  alpha: 0.63

training:
  sizes: [128, 256]
  batch_size: 16
  frozen_epochs: 30
  unfrozen_epochs: 30
```

---

## 13.2. Nếu winner là SGD

Ví dụ:

```json
{
  "batch_size": 8,
  "optimizer": "SGD",
  "momentum": 0.92,
  "weight_decay": 0.00001,
  "loss_type": "BCE"
}
```

config:

```yaml
optimizer:
  name: SGD
  momentum: 0.92
  weight_decay: 0.00001

loss:
  name: BCE

training:
  batch_size: 8
```

---

## 13.3. Nếu winner là Adam

Optuna protocol hiện tại cố định:

```text
Adam weight_decay = 0
```

nên:

```yaml
optimizer:
  name: Adam
  weight_decay: 0.0
```

---

# 14. Lưu ý rất quan trọng trước final 3-fold

Final 3-fold **không load checkpoint winner @256 để tiếp tục train**.

Nó dùng:

```text
configuration winner
```

đã được khóa trong:

```text
configs/final/unet34.yaml
```

rồi tạo lại model pretrained ImageNet và train độc lập trên từng fold:

```text
Fold 1:
128 frozen + unfrozen
→ 256 frozen + unfrozen

Fold 2:
128 frozen + unfrozen
→ 256 frozen + unfrozen

Fold 3:
128 frozen + unfrozen
→ 256 frozen + unfrozen
```

Đây là đúng mục đích của final cross-validation.

---

# 15. Bước 7 — Train final 3-fold cho optimized configuration

Sau khi đã sửa và khóa:

```text
configs/final/unet34.yaml
```

chạy:

```bash
python scripts/final/train_3fold.py \
  --config configs/final/unet34.yaml \
  --tag optimized
```

Script này tự chạy cả 3 fold.

Không cần gọi:

```text
fold 1
fold 2
fold 3
```

bằng ba lệnh riêng.

Flow:

```text
fold = 0
  train 128
  train 256
  evaluate
  save checkpoint
        ↓
fold = 1
  train 128
  train 256
  evaluate
  save checkpoint
        ↓
fold = 2
  train 128
  train 256
  evaluate
  save checkpoint
```

Output:

```text
checkpoints/final/unet34/
├── fold_1/
│   └── optimized_best.pt
├── fold_2/
│   └── optimized_best.pt
└── fold_3/
    └── optimized_best.pt
```

Metrics:

```text
experiments/final/unet34/
└── optimized_fold_metrics.csv
```

---

# 16. Bước 8 — Baseline 3-fold để so sánh

Nếu báo cáo cần so:

```text
baseline
vs
optimized
```

thì baseline cũng chạy cùng 3-fold protocol, nhưng dùng official validation
để báo cáo từng model và ensemble sau khi các fold đã chọn xong Best Dice.

Chạy:

```bash
python scripts/baseline/train_unet34.py \
  --config configs/baseline/unet34.yaml \
  --tag baseline
```

Output:

```text
checkpoints/baseline/unet34/
├── fold_1/
│   └── baseline_best.pt
├── fold_2/
│   └── baseline_best.pt
└── fold_3/
    └── baseline_best.pt
```

và:

```text
experiments/baseline/unet34/
├── baseline_fold_metrics.csv
└── baseline_validation_summary.json
```

---

# 17. Bước 9 — Ensemble 3 fold trên held-out test

## Optimized

```bash
python scripts/final/ensemble_test.py \
  --config configs/final/unet34.yaml \
  --checkpoint-root checkpoints/final/unet34 \
  --tag optimized
```

Script tìm:

```text
fold_1/optimized_best.pt
fold_2/optimized_best.pt
fold_3/optimized_best.pt
```

sau đó:

```text
model fold 1 ─┐
model fold 2 ─┼─ average probability
model fold 3 ─┘
                ↓
             threshold
                ↓
          held-out metrics
```

---

## Baseline

Nếu muốn ensemble baseline:

```bash
python scripts/final/ensemble_test.py \
  --config configs/baseline/unet34.yaml \
  --checkpoint-root checkpoints/baseline/unet34 \
  --tag baseline
```

---

# 18. Toàn bộ lệnh theo đúng thứ tự

## Trường hợp chạy đầy đủ từ đầu

### 1. Baseline 3-fold + official-validation run

```bash
python scripts/baseline/train_unet34.py \
  --config configs/baseline/unet34.yaml
```

### 2. Optuna session 1

```bash
python scripts/optuna/run_unet34_optuna.py \
  --config configs/optuna/unet34.yaml \
  --timeout-seconds 21600
```

### 3. Optuna session 2, 3, ... nếu cần

Chạy lại đúng lệnh:

```bash
python scripts/optuna/run_unet34_optuna.py \
  --config configs/optuna/unet34.yaml \
  --timeout-seconds 21600
```

### 4. Export trial table

```bash
python scripts/optuna/export_trials.py \
  --config configs/optuna/unet34.yaml
```

### 5. Kiểm tra Top 5

Kiểm tra:

```text
experiments/optuna/unet34/top5_manifest_dice.csv
```

phải có đủ 5 candidate.

### 6. Train Top 5 @256

```bash
python scripts/optuna/confirm_top5_256.py \
  --config configs/optuna/unet34.yaml
```

### 7. Chọn winner

Đọc dòng đầu của:

```text
checkpoints/optuna/unet34/confirmation_256_dice/confirmation_results.csv
```

### 8. Khóa winner

Sửa:

```text
configs/final/unet34.yaml
```

theo `params_json` của winner.

### 9. Final 3-fold optimized

```bash
python scripts/final/train_3fold.py \
  --config configs/final/unet34.yaml \
  --tag optimized
```

### 10. Final 3-fold baseline

```bash
python scripts/baseline/train_unet34.py \
  --config configs/baseline/unet34.yaml \
  --tag baseline
```

### 11. Ensemble optimized test

```bash
python scripts/final/ensemble_test.py \
  --config configs/final/unet34.yaml \
  --checkpoint-root checkpoints/final/unet34 \
  --tag optimized
```

### 12. Ensemble baseline test

```bash
python scripts/final/ensemble_test.py \
  --config configs/baseline/unet34.yaml \
  --checkpoint-root checkpoints/baseline/unet34 \
  --tag baseline
```

---

# 19. File nào gọi `sample_configuration`, file nào dùng `configuration_from_params`?

## HPO @128

```text
scripts/optuna/run_unet34_optuna.py
       ↓
src/hpo/objective.py
       ↓
sample_configuration(trial)
```

Mục đích:

```text
TPE tạo một configuration mới cho trial mới
```

---

## Confirmation Top 5 @256

```text
scripts/optuna/confirm_top5_256.py
       ↓
đọc params_json của trial cũ
       ↓
configuration_from_params(params)
```

Mục đích:

```text
khôi phục đúng configuration đã tìm được
KHÔNG sample lại
```

---

# 20. Các artifact quan trọng cần backup

Trong Optuna:

```text
experiments/optuna/unet34/optuna.db
```

là quan trọng nhất vì chứa lịch sử study.

Ngoài ra cần backup:

```text
experiments/optuna/unet34/top5_manifest_dice.csv

checkpoints/optuna/unet34/top5_128_dice/
```

Sau confirmation:

```text
checkpoints/optuna/unet34/confirmation_256_dice/
```

Sau final:

```text
checkpoints/final/unet34/
experiments/final/unet34/
```

---

# 21. Những file KHÔNG được dùng nhầm

## Không dùng baseline script để chạy Optuna

Sai:

```bash
python scripts/baseline/train_unet34.py
```

khi mục tiêu là HPO.

Đúng:

```bash
python scripts/optuna/run_unet34_optuna.py
```

---

## Không dùng `configuration_from_params()` để tạo trial mới

Trial mới dùng:

```python
sample_configuration(trial)
```

Top-5 confirmation dùng:

```python
configuration_from_params(params)
```

---

## Không chạy held-out test trước khi khóa final config

Thứ tự đúng:

```text
Optuna @128
→ Top 5
→ confirmation @256
→ chọn winner
→ LOCK config
→ final 3-fold
→ ensemble held-out test
```

Không:

```text
Optuna
→ xem test
→ sửa config
→ Optuna tiếp
```

---

# 22. Checklist ngắn trước mỗi phase

## Trước HPO

- [ ] Dataset Kaggle truy cập được qua KaggleHub, Kaggle mount hoặc `data_root`.
- [ ] `seed` và `val_fraction` đã chốt.
- [ ] `configs/optuna/unet34.yaml` đã chốt.
- [ ] `study.name` không đổi giữa các session.
- [ ] `study.storage` trỏ đúng `optuna.db`.
- [ ] Search space không đổi giữa study.

## Trước confirmation @256

- [ ] Đã đạt ngân sách HPO mong muốn.
- [ ] Có ít nhất 5 `COMPLETE` candidate.
- [ ] `top5_manifest_dice.csv` có 5 dòng candidate.
- [ ] 5 checkpoint @128 tồn tại.

## Trước final 3-fold

- [ ] `confirmation_results.csv` đã tạo.
- [ ] Winner được chọn theo Dice @256.
- [ ] `configs/final/unet34.yaml` đã cập nhật đúng winner.
- [ ] Không còn thay đổi optimizer/loss/batch size.
- [ ] `seed` và `n_splits` đã chốt.

## Trước test

- [ ] Có đủ 3 checkpoint fold.
- [ ] Tag checkpoint đúng (`baseline` hoặc `optimized`).
- [ ] Mask test local có đủ và chưa được dùng trong HPO.
- [ ] Configuration đã khóa.

---

# 23. Tóm tắt cực ngắn

```text
1. loader.py tải/cache dữ liệu và chia tập lúc chạy

2. baseline/train_unet34.py            # baseline

3. optuna/run_unet34_optuna.py        # HPO @128
   └── Top 5 được cập nhật TỰ ĐỘNG

4. optuna/export_trials.py             # optional

5. optuna/confirm_top5_256.py          # train 5 candidate @256

6. đọc confirmation_results.csv
   ↓
   sửa configs/final/unet34.yaml       # LOCK winner

7. final/train_3fold.py
   --tag optimized                     # final 3-fold

8. baseline/train_unet34.py
   --tag baseline                      # baseline 3-fold + official validation

9. final/ensemble_test.py
   --tag optimized                     # held-out test optimized

10. final/ensemble_test.py
    --tag baseline                      # held-out test baseline
```

> **Điểm cần nhớ nhất:** `Top 5 @128` không cần một script riêng để lấy. `TopKManager` cập nhật `top5_manifest_dice.csv` và checkpoint Top-5 ngay trong lúc `run_unet34_optuna.py` đang chạy. `confirm_top5_256.py` chỉ việc đọc Top 5 đó và huấn luyện tiếp ở 256×256.
