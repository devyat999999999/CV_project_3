"""Describe PH2 and DRIVE on the HPC (Part 1 task 1). Run this first"""
import argparse
import os
from collections import Counter

import numpy as np
from PIL import Image

from data import DATA_ROOT, drive_items, ph2_items


def tree(root, depth=2):
    for cur, dirs, files in os.walk(root):
        level = cur[len(root):].count(os.sep)
        if level > depth:
            dirs[:] = []
            continue
        print('  ' * level + os.path.basename(cur) + f'/  ({len(files)} files)')


def describe(name, items):
    sizes, fg, values = Counter(), [], set()
    for img, mask, _ in items:
        sizes[Image.open(img).size] += 1
        if mask:
            m = np.asarray(Image.open(mask).convert('L'))
            values.update(np.unique(m).tolist()[:5])
            fg.append((m > 127).mean())
    print(f'\n{name}: {len(items)} images, {sum(1 for i in items if i[1])} with masks')
    print('resolutions (W, H):', dict(sizes))
    if fg:
        print(f'mask values: {sorted(values)[:10]}')
        print(f'foreground fraction: mean {np.mean(fg):.3f}, min {np.min(fg):.3f}, max {np.max(fg):.3f}')


def main():

    p = argparse.ArgumentParser()
    p.add_argument('--root', default=DATA_ROOT)
    a = p.parse_args()
    
    print('Tree of', a.root)
    
    tree(a.root)
    describe('PH2', ph2_items(a.root))
    d = drive_items(a.root)
    
    for split, items in d.items():
        describe(f'DRIVE {split}', items)
        print('  example:', items[0] if items else None)


if __name__ == '__main__':
    main()
