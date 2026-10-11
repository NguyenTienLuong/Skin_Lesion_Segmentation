"""LOADER TẠM - chỉ để SMOKE TEST pipeline khi chưa có src/data/loader.py của C (W2-C1).
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TempSegDataset(Dataset):
    def __init__(self, df, size, train):
        self.paths = list(zip(df["image_path"], df["mask_path"]))
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img_path, mask_path = self.paths[i]
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.train:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[::-1], mask[::-1]

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return x, y


def make_loader(df, size, train, batch_size, num_workers=0, pin_memory=False, augment_cfg=None):
    return DataLoader(
        TempSegDataset(df, size, train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )