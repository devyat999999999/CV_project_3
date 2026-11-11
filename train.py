import argparse
import json
import os
import random
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from data import DATA_ROOT, SegDataset, make_splits, resize_check
from losses import make_loss
from metrics import MetricAccumulator
from models import MODELS

DEFAULTS = {
    'ph2': dict(size=256, batch=8, epochs=60),
    'drive': dict(size=512, batch=2, epochs=200),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = MetricAccumulator()
    for x, y, fov in loader:
        acc.update(model(x.to(device)), y.to(device), fov.to(device))
    return acc.result()


@torch.no_grad()
def save_examples(model, ds, device, path, n=6):
    model.eval()
    idx = list(range(min(n, len(ds))))
    x = torch.stack([ds[i][0] for i in idx])
    prob = torch.sigmoid(model(x.to(device))).cpu()
    np.savez_compressed(path, image=(x * 255).byte().numpy(), gt=ds.y[idx].numpy(),
                        prob=(prob * 255).byte().numpy())


def run_name(a):
    name = f'{a.dataset}_{a.model}_{a.loss}'
    if a.padding_mode != 'zeros':
        name += f'_pad{a.padding_mode}'
    if a.crop:
        name += '_crop'
    return name + f'_s{a.seed}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['ph2', 'drive'], required=True)
    p.add_argument('--model', choices=list(MODELS), default='unet')
    p.add_argument('--loss', choices=['bce', 'wbce', 'focal'], default='bce')
    p.add_argument('--padding-mode', choices=['zeros', 'reflect'], default='zeros',
                   help='U-Net conv padding')
    p.add_argument('--crop', action='store_true', help='random resized crop augmentation')
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--size', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()

    for k, v in DEFAULTS[a.dataset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
            
    resize_check(a.size)

    set_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(a.out, run_name(a))
    os.makedirs(out_dir, exist_ok=True)

    # Splits are fixed by a file shared by all runs (split seed is always 0).
    splits = make_splits(a.dataset, a.data_root, os.path.join(a.out, f'splits_{a.dataset}.json'))
    train_ds = SegDataset(splits['train'], a.size, train=True, crop=a.crop)
    train_eval_ds = SegDataset(splits['train'], a.size)
    val_ds = SegDataset(splits['val'], a.size)
    test_ds = SegDataset(splits['test'], a.size)
    print(f'{a.dataset}: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}, size {a.size}')

    fov = train_ds.fov.bool()
    n_pos = (train_ds.y & fov).sum().item()
    pos_weight = (fov.sum().item() - n_pos) / max(n_pos, 1)
    loss_fn = make_loss(a.loss, pos_weight).to(device)
    print(f'loss {a.loss}, positive fraction {n_pos / fov.sum().item():.3f}, pos_weight {pos_weight:.2f}')

    kw = dict(batch_size=a.batch, num_workers=a.workers)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > a.batch, **kw)
    eval_loaders = {s: DataLoader(d, shuffle=False, **kw)
                    for s, d in (('train', train_eval_ds), ('val', val_ds), ('test', test_ds))}

    kwargs = {'padding_mode': a.padding_mode} if a.model == 'unet' else {}
    if a.model != 'unet' and a.padding_mode != 'zeros':
        p.error('--padding-mode is only implemented for the U-Net')
    model = MODELS[a.model](**kwargs).to(device)
    print(f'{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters')
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ckpt = os.path.join(out_dir, 'best.pt')

    history, best = [], -1.0
    for epoch in range(a.epochs):
        model.train()
        tic, total = time.time(), 0.0
        for x, y, fov in train_loader:
            x, y, fov = x.to(device), y.to(device), fov.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y, fov)
            loss.backward()
            opt.step()
            total += loss.item() / len(train_loader)
        sched.step()
        val = evaluate(model, eval_loaders['val'], device)
        history.append({'epoch': epoch + 1, 'loss': total, **{f'val_{k}': v for k, v in val.items()}})
        if val['dice'] > best:
            best = val['dice']
            torch.save(model.state_dict(), ckpt)
        print(f'epoch {epoch + 1:3d}/{a.epochs}  loss {total:.4f}  val dice {val["dice"]:.4f}  '
              f'best {best:.4f}  {time.time() - tic:.1f}s', flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    results = {s: evaluate(model, l, device) for s, l in eval_loaders.items()}
    save_examples(model, test_ds, device, os.path.join(out_dir, 'examples.npz'))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(a), 'results': results, 'history': history}, f, indent=1)
    for s, r in results.items():
        print(s, '  '.join(f'{k} {v:.4f}' for k, v in r.items()))


if __name__ == '__main__':
    main()
