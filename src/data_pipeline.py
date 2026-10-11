import os, random
import cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from src.augmentation import get_transforms

ROOT = "/content/drive/MyDrive/project/data"   #TODO: FIX PATH
COLS = dict(image="image_path", mask="mask_path", valid="pair_valid")

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_manifest(path):
    """Đọc manifest đã khóa sẵn, chỉ giữ các cặp hợp lệ (pair_valid == TRUE)."""
    df = pd.read_csv(path, sep=None, engine="python")          
    ok = df[COLS["valid"]].astype(str).str.upper().eq("TRUE")
    print(f"{os.path.basename(path)}: giữ {ok.sum()}/{len(df)} cặp hợp lệ")
    return df[ok].reset_index(drop=True)

def read_image(root, rel_path):
    """Đọc ảnh, đổi BGR sang RGB. Trả về uint8 HxWx3."""
    img = cv2.imread(os.path.join(root, rel_path), cv2.IMREAD_COLOR)
    assert img is not None, f"Không đọc được ảnh: {rel_path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_mask(root, rel_path):
    """Đọc mask PNG 0/255, đổi sang 0/1. Trả về uint8 HxW."""
    msk = cv2.imread(os.path.join(root, rel_path), cv2.IMREAD_GRAYSCALE)
    assert msk is not None, f"Không đọc được mask: {rel_path}"
    return (msk > 127).astype(np.uint8)

class ISICSegDataset(Dataset):
    def __init__(self, df, size=256, train=False, root=ROOT):
        self.df, self.root = df.reset_index(drop=True), root
        self.tf = get_transforms(size, train)                  # train=False - không có augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_image(self.root, r[COLS["image"]])
        msk = read_mask(self.root, r[COLS["mask"]])
        out = self.tf(image=img, mask=msk)                    
        return out["image"].float(), out["mask"].unsqueeze(0).float()   # (3,H,W), (1,H,W)

def make_loader(df, size=256, bs=16, workers=2, train=False):
    """train=True: augmentation + shuffle. train=False: chỉ tiền xử lý, giữ thứ tự."""
    return DataLoader(ISICSegDataset(df, size, train), batch_size=bs, shuffle=train,
                      num_workers=workers, pin_memory=True, drop_last=train)

def check_no_leak(train_df, val_df, test_df):
    """Ba tập không được trùng image_id."""
    assert not set(train_df.image_id) & set(val_df.image_id)
    assert not set(train_df.image_id) & set(test_df.image_id)
    assert not set(val_df.image_id)   & set(test_df.image_id)
    print("OK: không rò rỉ giữa train/val/test")

def sanity_check(df, size=128):
    """Kiểm tra nhanh."""
    ds = ISICSegDataset(df.head(8), size, train=True)
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (3, size, size) and y.shape == (1, size, size)
        assert set(y.unique().tolist()) <= {0.0, 1.0}, "mask phải chỉ có 0/1"
    dv = ISICSegDataset(df.head(2), size, train=False)
    assert torch.equal(dv[0][0], dv[0][0]) and torch.equal(dv[0][1], dv[0][1]), "val phải xác định"
    print("OK: shape đúng, mask 0/1, val không ngẫu nhiên")