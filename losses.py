"""Losses on logits. All take (logits, y, w): w masks pixels out of the loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(per_pixel, w):
    return (per_pixel * w).sum() / w.sum().clamp(min=1)


class BCE(nn.Module):
    """Plain cross entropy (binary)"""

    def forward(self, logits, y, w):
        return masked_mean(F.binary_cross_entropy_with_logits(logits, y, reduction='none'), w)


class WeightedBCE(nn.Module):
    """Cross entropy with pos_weight = #neg / #pos, counted on the training set"""

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))

    def forward(self, logits, y, w):
        per_pixel = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=self.pos_weight, reduction='none')
        return masked_mean(per_pixel, w)


class Focal(nn.Module):
    """Focal loss (Lin 2017): (1 - p_t)^gamma * CE, alpha balances the classes"""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y, w):
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        return masked_mean(alpha_t * (1 - p_t) ** self.gamma * ce, w)



def make_loss(name, pos_weight=1.0):
    if name == 'bce':
        return BCE()
    if name == 'wbce':
        return WeightedBCE(pos_weight)
    if name == 'focal':
        return Focal()
        
    raise ValueError(name)
