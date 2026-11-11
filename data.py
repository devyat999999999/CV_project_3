"""PH2 and DRIVE loading, fixed splits, joint augmentation"""
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from glob import glob
from PIL import Image

DATA_ROOT = '/dtu/datasets1/02516'


def _find(root, *patterns):
    for p in patterns:
        hits = sorted(glob(os.path.join(root, p), recursive=True))
        if hits:
            return hits
    return []


def ph2_items(root):
    """[(image, lesion_mask, None)] for every PH2 case

    Layout: PH2_Dataset_images/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp
            PH2_Dataset_images/IMDxxx/IMDxxx_lesion/IMDxxx_lesion.bmp
    """
    images = _find(root, '**/*_Dermoscopic_Image/*.bmp')
    items = []

    for img in images:
        case = os.path.basename(img)[:-4]
        case_dir = os.path.dirname(os.path.dirname(img))
        mask = os.path.join(case_dir, f'{case}_lesion', f'{case}_lesion.bmp')
        if os.path.exists(mask):
            items.append((img, mask, None))

    return items


def drive_items(root):
    """{'training': [...], 'test': [...]} with (image, vessel_mask or None, fov_mask)

    Layout: DRIVE/training/images/21_training.tif
            DRIVE/training/1st_manual/21_manual1.gif   <- vessel mask (the target)
            DRIVE/training/mask/21_training_mask.gif   <- field of view, NOT the target
    """
    out = {}
    for split in ('training', 'test'):
        images = _find(root, f'**/{split}/images/*.tif', f'**/{split}/images/*.png')
        items = []
        for img in images:
            num = os.path.basename(img).split('_')[0]
            base = os.path.dirname(os.path.dirname(img))
            vessel = _find(base, f'1st_manual/{num}_manual1.*')
            fov = _find(base, f'mask/{num}_{split}_mask.*')
            items.append((img, vessel[0] if vessel else None, fov[0] if fov else None))
        out[split] = items

    return out


def make_splits(dataset, root, split_file, seed=0):
    """Fixed train/val/test lists, written once to split_file and reused by every run

    PH2:   200 cases, no official split -> 70 / 15 / 15 %.
    DRIVE: 20 training images have vessel masks. If the 20 test images also
           have masks, they are the test set and training is split 16 / 4.
           Otherwise the 20 labelled images are split 12 / 4 / 4
    """
    if os.path.exists(split_file):
        with open(split_file) as f:
            return json.load(f)
    rng = random.Random(seed)

    if dataset == 'ph2':
        items = ph2_items(os.path.join(root, 'PH2_Dataset_images')
                          if os.path.isdir(os.path.join(root, 'PH2_Dataset_images')) else root)
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = round(0.7 * n), round(0.15 * n)
        splits = {'train': items[:n_tr], 'val': items[n_tr:n_tr + n_va], 'test': items[n_tr + n_va:]}
    
    elif dataset == 'drive':
        d = drive_items(os.path.join(root, 'DRIVE') if os.path.isdir(os.path.join(root, 'DRIVE')) else root)
        labelled = [it for it in d['training'] if it[1]]
        test_labelled = [it for it in d['test'] if it[1]]
        rng.shuffle(labelled)
        if len(test_labelled) == len(d['test']) and test_labelled:
            splits = {'train': labelled[4:], 'val': labelled[:4], 'test': test_labelled}
        else:
            splits = {'train': labelled[8:], 'val': labelled[4:8], 'test': labelled[:4]}

    os.makedirs(os.path.dirname(split_file) or '.', exist_ok=True)

    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=1)
    return splits


def _load(items, size):
    """Decode + resize once. Image: bilinear. Masks: nearest (keeps them binary)"""

    xs, ys, fovs = [], [], []

    for img_p, mask_p, fov_p in items:
        img = Image.open(img_p).convert('RGB').resize((size, size), Image.BILINEAR)
        mask = Image.open(mask_p).convert('L').resize((size, size), Image.NEAREST)
        xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
        ys.append(torch.from_numpy(np.asarray(mask) > 127)[None])
        if fov_p:
            fov = Image.open(fov_p).convert('L').resize((size, size), Image.NEAREST)
            fovs.append(torch.from_numpy(np.asarray(fov) > 127)[None])
        else:
            fovs.append(torch.ones_like(ys[-1]))
    
    return torch.stack(xs), torch.stack(ys), torch.stack(fovs)


def random_resized_crop(x, *maps, scale=(0.5, 1.0)):
    """Same random square crop (50-100% of the area) for image and maps, resized back"""

    H, W = x.shape[-2:]
    s = random.uniform(*scale) ** 0.5
    h, w = max(16, round(H * s)), max(16, round(W * s))
    top, left = random.randint(0, H - h), random.randint(0, W - w)
    
    crop = lambda t: t[..., top:top + h, left:left + w]
    x = F.interpolate(crop(x)[None], size=(H, W), mode='bilinear', align_corners=False)[0]
    maps = [F.interpolate(crop(m)[None], size=(H, W), mode='nearest')[0] for m in maps]
    
    return [x, *maps]


def augment(*tensors):
    """Same random flip / 90-degree rotation for image and all maps of one sample"""

    k = random.randint(0, 3)
    hflip, vflip = random.random() < 0.5, random.random() < 0.5
    out = []
    for t in tensors:
        t = torch.rot90(t, k, dims=(-2, -1))
        if hflip:
            t = t.flip(-1)
        if vflip:
            t = t.flip(-2)
        out.append(t)
    
    return out


class SegDataset(torch.utils.data.Dataset):
    """In-memory dataset with full masks"""

    def __init__(self, items, size, train=False, crop=False):
        self.x, self.y, self.fov = _load(items, size)
        self.train = train
        self.crop = crop

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].float() / 255
        y = self.y[i].float()
        fov = self.fov[i].float()
        if self.train:
            if self.crop:
                x, y, fov = random_resized_crop(x, y, fov)
            x, y, fov = augment(x, y, fov)
            x = (x * (0.9 + 0.2 * random.random())).clamp(0, 1)   # mild brightness jitter
        return x, y, fov
