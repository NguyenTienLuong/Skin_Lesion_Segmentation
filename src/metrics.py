import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

def counts(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return tp, fp, fn

def jaccard(pred, gt):
    """Raw Jaccard."""
    tp, fp, fn = counts(pred, gt)
    union = tp + fp + fn
    return 1.0 if union == 0 else tp / union

def dice(pred, gt):
    """Dice."""
    tp, fp, fn = counts(pred, gt)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom

def sensitivity(pred, gt):
    """Sensitivity."""
    tp, fp, fn = counts(pred, gt)
    return np.nan if (tp + fn) == 0 else tp / (tp + fn)

def _surface(m):
    return m & ~binary_erosion(m)

def hd95(pred, gt):
    """HD95 (đơn vị pixel, lưới 256x256)."""
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.nan
    sp, sg = _surface(pred), _surface(gt)
    d_p_to_g = distance_transform_edt(~sg)[sp]    # khoảng cách từ biên dự đoán tới biên nhãn
    d_g_to_p = distance_transform_edt(~sp)[sg]    # và ngược lại
    return float(max(np.percentile(d_p_to_g, 95), np.percentile(d_g_to_p, 95)))

METRICS = {"jaccard": jaccard, "dice": dice, "sensitivity": sensitivity, "hd95": hd95}