#!/usr/bin/env python
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def scan_pairs(image_dir: Path, mask_dir: Path):
    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Thiếu mask: {mask_path}")
        rows.append({
            "image_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True, type=Path)
    ap.add_argument("--train-masks", required=True, type=Path)
    ap.add_argument("--heldout-images", type=Path)
    ap.add_argument("--heldout-masks", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_manifest = Path("data/manifests/data_manifest.csv")
    out_split = Path("data/splits/tuning_split.csv")
    out_folds = Path("data/splits/folds_3.csv")
    out_test = Path("data/splits/heldout_test.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_split.parent.mkdir(parents=True, exist_ok=True)

    dev = scan_pairs(args.train_images, args.train_masks)
    if dev.empty:
        raise RuntimeError("Không tìm thấy ảnh .jpg trong train-images")
    dev.to_csv(out_manifest, index=False)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(dev))
    n_val = max(1, int(round(len(dev) * args.val_frac)))
    split = dev.copy(); split["split"] = "train"
    split.loc[perm[:n_val], "split"] = "val"
    split.to_csv(out_split, index=False)

    perm = rng.permutation(len(dev))
    fold_ids = np.empty(len(dev), dtype=int)
    for fold, idx in enumerate(np.array_split(perm, 3)):
        fold_ids[idx] = fold
    folds = dev.copy(); folds["fold"] = fold_ids
    folds.to_csv(out_folds, index=False)

    if args.heldout_images and args.heldout_masks:
        test = scan_pairs(args.heldout_images, args.heldout_masks)
        overlap = set(dev.image_id) & set(test.image_id)
        if overlap:
            raise RuntimeError(f"Held-out test trùng {len(overlap)} ID với development data")
        test.to_csv(out_test, index=False)
        print(f"heldout: {len(test)}")

    print(f"development: {len(dev)} | tuning train: {(split['split']=='train').sum()} | val: {(split['split']=='val').sum()}")
    print(f"wrote: {out_manifest}, {out_split}, {out_folds}")


if __name__ == "__main__":
    main()
