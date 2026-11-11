"""Dice, IoU, Accuracy, Sensitivity, Specificity computed per image and averaged over images"""

import torch

NAMES = ('dice', 'iou', 'acc', 'sens', 'spec')


def confusion(pred, y, fov):
    """pred, y, fov: bool [B, 1, H, W] -> tp, fp, fn, tn per image, shape [B]"""
    dims = (1, 2, 3)
    tp = (pred & y & fov).sum(dims).float()
    fp = (pred & ~y & fov).sum(dims).float()
    fn = (~pred & y & fov).sum(dims).float()
    tn = (~pred & ~y & fov).sum(dims).float()
    return tp, fp, fn, tn


def per_image(tp, fp, fn, tn):
    return {
        'dice': (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7),
        'iou': (tp + 1e-7) / (tp + fp + fn + 1e-7),
        'acc': (tp + tn) / (tp + fp + fn + tn + 1e-7),
        'sens': (tp + 1e-7) / (tp + fn + 1e-7),
        'spec': (tn + 1e-7) / (tn + fp + 1e-7),
    }


class MetricAccumulator:
    def __init__(self):
        self.values = {k: [] for k in NAMES}

    def update(self, logits, y, fov, threshold=0.5):
        pred = torch.sigmoid(logits) > threshold
        m = per_image(*confusion(pred, y > 0.5, fov > 0.5))
        for k in NAMES:
            self.values[k].append(m[k].cpu())

    def result(self):
        return {k: torch.cat(v).mean().item() for k, v in self.values.items()}
