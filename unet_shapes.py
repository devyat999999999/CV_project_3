import argparse
import torch

from models import UNet


def main():
    
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=256)
    a = p.parse_args()
    model = UNet().eval()
    rows = []

    def hook(name):
        return lambda m, i, o: rows.append((name, tuple(o.shape[1:])))

    for i, b in enumerate(model.down):
        b.register_forward_hook(hook(f'enc{i + 1}'))
    
    model.bottleneck.register_forward_hook(hook('bottleneck'))
    
    for i, (u, b) in enumerate(zip(model.upconv, model.up)):
        u.register_forward_hook(hook(f'upconv{i + 1}'))
        b.register_forward_hook(hook(f'dec{i + 1}'))
    
    model.head.register_forward_hook(hook('output'))
    
    with torch.no_grad():
        model(torch.zeros(1, 3, a.size, a.size))
    
    print(f'input (3, {a.size}, {a.size})')
    
    for name, shape in rows:
        print(f'{name:<16} {shape}')


if __name__ == '__main__':
    main()
