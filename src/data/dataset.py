from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """
    Dataset ảnh-mask cho phân đoạn tổn thương da.
    File được tìm trong image_dir / mask_dir của split, khớp theo TÊN file
    trong manifest.
    """

    REQUIRED_COLUMNS = {
        "image_id",
        "image_path",
        "mask_path",
        "pair_valid",
    }

    def __init__(
        self,
        manifest_path,
        image_dir,
        mask_dir,
        transform=None,
        split="train",
        mask_threshold=127,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.transform = transform
        self.split = split
        self.mask_threshold = mask_threshold
        self.image_dir = Path(image_dir).resolve()
        self.mask_dir = Path(mask_dir).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy manifest: {self.manifest_path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy image_dir: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy mask_dir: {self.mask_dir}")

        df = pd.read_csv(self.manifest_path, dtype={"image_id": str})

        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            raise ValueError(f"Manifest thiếu các cột: {sorted(missing_columns)}")

        if df["image_id"].isna().any():
            raise ValueError(f"[{split}] Manifest có image_id bị thiếu.")
        df["image_id"] = df["image_id"].str.strip()

        if df["image_id"].duplicated().any():
            dup = df.loc[df["image_id"].duplicated(keep=False), "image_id"].tolist()
            raise ValueError(f"image_id bị trùng trong {split}: {dup[:10]}")

        pair_valid = (
            df["pair_valid"].astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes"})
        )
        invalid_count = int((~pair_valid).sum())
        if invalid_count:
            print(f"[{split}] Bỏ qua {invalid_count} cặp có pair_valid không phải True.")
        df = df.loc[pair_valid].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"Không có cặp ảnh-mask hợp lệ trong {self.manifest_path}")
        if df["image_path"].isna().any():
            raise ValueError(f"[{split}] Manifest có image_path bị thiếu.")
        if df["mask_path"].isna().any():
            raise ValueError(f"[{split}] Manifest có mask_path bị thiếu.")

        self.samples = []
        for row in df.itertuples(index=False):
            image_id = str(row.image_id)
            image_path = self._resolve_path(row.image_path, self.image_dir)
            mask_path = self._resolve_path(row.mask_path, self.mask_dir)

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"[{split}] Không tìm thấy ảnh của {image_id}: {image_path}"
                )
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"[{split}] Không tìm thấy mask của {image_id}: {mask_path}"
                )

            self.samples.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

        print(
            f"[{split}] Loaded {len(self.samples)} image-mask pairs "
            f"from {self.manifest_path.name}"
        )

    def _resolve_path(self, path_value, split_dir):
        """Tìm file theo tên trong thư mục của split."""
        path = Path(str(path_value).replace("\\", "/"))
        candidate = (split_dir / path.name).resolve()

        if not candidate.is_file() and path.is_absolute() and path.is_file():
            return path.resolve()

        return candidate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Không đọc được mask: {sample['mask_path']}")

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Kích thước ảnh-mask không khớp: {sample['image_id']} | "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}"
            )

        mask = (mask > self.mask_threshold).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(np.asarray(mask).copy()).long()

        return image, mask, sample["image_id"]