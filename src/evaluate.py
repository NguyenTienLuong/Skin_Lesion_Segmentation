import os, json, time, hashlib
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from src.metrics import METRICS

EVAL_SIZE = 256     # lưới chung để so sánh run @128 và @256
THR = 0.5 # ngưỡng nhị phân hóa xác suất

@torch.no_grad()
def predict_masks(model, loader, device="cuda", input_size=256, thr=THR):
    """Loader phải tạo ở size=256 (ảnh và nhãn cùng lưới 256).
    Model chạy ở input_size, logit được phóng về 256x256, rồi sigmoid -> ngưỡng 0.5."""
    model.eval().to(device)
    for x, y in loader:
        x = x.to(device)
        if input_size != EVAL_SIZE:
            x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
        logits = model(x)
        logits = F.interpolate(logits, size=(EVAL_SIZE, EVAL_SIZE), mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logits) > thr).squeeze(1).cpu().numpy()
        gt = (y.squeeze(1) > 0.5).numpy()
        yield pred, gt

def evaluate_per_image(model, loader, image_ids, device="cuda", input_size=256):
    """Trả về DataFrame mỗi dòng một ảnh. loader phải shuffle=False và cùng thứ tự với image_ids."""
    rows, k = [], 0
    for pred_b, gt_b in predict_masks(model, loader, device, input_size):
        for pred, gt in zip(pred_b, gt_b):
            rows.append({"image_id": image_ids[k], **{n: f(pred, gt) for n, f in METRICS.items()}})
            k += 1
    assert k == len(image_ids), "số ảnh đánh giá không khớp manifest"
    return pd.DataFrame(rows)

def summarize(df):
    """Macro mean theo ảnh. NaN bị loại khỏi trung bình và được đếm riêng."""
    out = {"n_images": len(df)}
    for n in METRICS:
        out[n] = float(np.nanmean(df[n]))
        out[f"{n}_n_undefined"] = int(df[n].isna().sum())
    return out

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_run(out_dir, run_name, config, ckpt_path, split_name, per_image_df):
    """Lưu per-image CSV + summary JSON (gồm config, checkpoint, hash).
    """
    os.makedirs(out_dir, exist_ok=True)
    per_image_df.to_csv(f"{out_dir}/{run_name}_{split_name}_per_image.csv", index=False)
    record = {
        "run_name": run_name, "split": split_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,                              # size, lr, epochs, seed, aug, ...
        "eval_protocol": {"eval_grid": EVAL_SIZE, "threshold": THR,
                          "aggregation": "per-image -> macro mean"},
        "checkpoint": {"path": ckpt_path, "sha256": file_sha256(ckpt_path)},
        "summary": summarize(per_image_df),
    }
    with open(f"{out_dir}/{run_name}_{split_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record