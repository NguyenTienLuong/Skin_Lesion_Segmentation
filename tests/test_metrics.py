import torch
from src.metrics.segmentation_metrics import per_image_scores, summarize


def test_threshold_jaccard_example_from_notebook():
    target = torch.zeros(2, 1, 8, 8)
    target[0, :, :4] = 1
    target[1, :, :, :4] = 1
    pred = target.clone()
    pred[1] = 0
    pred[1, :, :, :2] = 1

    j, d, s = per_image_scores(pred, target)
    m = summarize(pred, target)
    assert abs(j[0].item() - 1.0) < 1e-6
    assert abs(j[1].item() - 0.5) < 1e-6
    assert abs(m["threshold_jaccard"] - 0.5) < 1e-6
