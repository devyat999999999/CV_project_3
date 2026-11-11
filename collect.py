import argparse
import glob
import json
import os
import numpy as np

M = ('dice', 'iou', 'acc', 'sens', 'spec')


def load(out):
    runs = []
    for f in sorted(glob.glob(os.path.join(out, '*', 'results.json'))):
        with open(f) as fh:
            r = json.load(fh)
        if r['args'].get('supervision', 'full') != 'full':
            continue
        r['dir'] = os.path.dirname(f)
        runs.append(r)
    return runs


def is_base(a):
    """Default setup"""
    return a.get('padding_mode', 'zeros') == 'zeros' and not a.get('crop', False)


def variant(a):
    return f"{a.get('padding_mode', 'zeros')} padding" + (' + crop' if a.get('crop') else '')


def row(r, split):
    return ' | '.join(f'{100 * r["results"][split][k]:.1f}' for k in M)


def tables(runs):
    base = [r for r in runs if is_base(r['args'])]
    unet_bce = [r for r in runs if r['args']['model'] == 'unet' and r['args']['loss'] == 'bce']
    hdr = '| ' + ' | '.join(M) + ' |'
    lines = ['## Table 1: EncDec vs U-Net (BCE), metrics in %', '',
             '| dataset | model | split ' + hdr, '|' + '---|' * 8]
    for r in base:
        a = r['args']
        if a['loss'] == 'bce':
            for s in ('train', 'val', 'test'):
                lines.append(f'| {a["dataset"]} | {a["model"]} | {s} | {row(r, s)} |')
    lines += ['', '## Table 2: loss ablation, U-Net, test split', '',
              '| dataset | loss ' + hdr, '|' + '---|' * 7]
    for r in base:
        a = r['args']
        if a['model'] == 'unet':
            lines.append(f'| {a["dataset"]} | {a["loss"]} | {row(r, "test")} |')
    lines += ['', '## Table 3: padding and random crop, U-Net + BCE, test split', '',
              '| dataset | variant ' + hdr, '|' + '---|' * 7]
    for r in sorted(unet_bce, key=lambda r: (r['args']['dataset'], not is_base(r['args']), variant(r['args']))):
        a = r['args']
        lines.append(f'| {a["dataset"]} | {variant(a)} | {row(r, "test")} |')
    return '\n'.join(lines) + '\n'


def plot_examples(runs, dataset, path, n=4):
    """Columns: image, ground truth, EncDec, U-Net (both BCE, default setup)"""

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    pick = {}

    for r in runs:
        a = r['args']
        npz = os.path.join(r['dir'], 'examples.npz')
        if a['dataset'] == dataset and a['loss'] == 'bce' and is_base(a) and os.path.exists(npz):
            pick[a['model']] = np.load(npz)
    pick = {k: pick[k] for k in ('encdec', 'unet') if k in pick}
    
    if not pick:
        return

    first = next(iter(pick.values()))
    n = min(n, len(first['image']))
    cols = 2 + len(pick)
    fig, axes = plt.subplots(n, cols, figsize=(1.2 * cols, 1.2 * n), squeeze=False)
    
    for i in range(n):
        axes[i, 0].imshow(first['image'][i].transpose(1, 2, 0))
        axes[i, 1].imshow(first['gt'][i, 0], cmap='gray')
        for j, ex in enumerate(pick.values()):
            axes[i, 2 + j].imshow(ex['prob'][i, 0] > 127, cmap='gray')
    
    for j, title in enumerate(['image', 'ground truth', *pick]):
        axes[0, j].set_title(title, fontsize=6)
    
    for ax in axes.flat:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results')
    a = p.parse_args()
    runs = load(a.out)
    md = tables(runs)
    
    with open(os.path.join(a.out, 'tables.md'), 'w') as f:
        f.write(md)
    print(md)
    
    for ds in ('ph2', 'drive'):
        plot_examples(runs, ds, os.path.join(a.out, f'examples_{ds}.png'))


if __name__ == '__main__':
    main()
