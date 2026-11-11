"""Simple encoder-decoder baseline and U-Net. Both output logits [B, 1, H, W]"""
import torch
import torch.nn as nn


def conv_bn(c_in, c_out, padding_mode='zeros'):
    """3x3 conv with padding 1 (size kept). padding_mode: 'zeros' or 'reflect'"""

    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False, padding_mode=padding_mode),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class EncDec(nn.Module):
    """Starter-code encoder-decoder: 4 x (conv, pool), bottleneck, 4 x (upsample, conv).

    No skip connections. Upsampling uses scale_factor=2 instead of fixed sizes,
    so the same network works for 256 (PH2) and 512 (DRIVE) inputs
    """

    def __init__(self, ch=64):
        super().__init__()
        self.enc = nn.ModuleList([conv_bn(3, ch)] + [conv_bn(ch, ch) for _ in range(3)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_bn(ch, ch)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec = nn.ModuleList([conv_bn(ch, ch) for _ in range(3)])
        self.head = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x):
        for conv in self.enc:
            x = self.pool(conv(x))
        x = self.bottleneck(x)
        for conv in self.dec:
            x = conv(self.up(x))

        return self.head(self.up(x))


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out, padding_mode='zeros'):
        super().__init__(conv_bn(c_in, c_out, padding_mode), conv_bn(c_out, c_out, padding_mode))


class UNet(nn.Module):
    """U-Net (Ronneberger 2015) with padded convs, BatchNorm and transposed-conv upsampling.

    Padding keeps H, W unchanged inside each level, so input size must be divisible
    by 2**depth = 16. Channels: base, 2b, 4b, 8b, bottleneck 16b
    """

    def __init__(self, base=32, depth=4, padding_mode='zeros'):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c_prev = 3
        for c in chs[:-1]:
            self.down.append(DoubleConv(c_prev, c, padding_mode))
            c_prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1], padding_mode)
        self.upconv = nn.ModuleList()
        self.up = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.upconv.append(nn.ConvTranspose2d(2 * c, c, 2, stride=2))
            self.up.append(DoubleConv(2 * c, c, padding_mode))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, block, skip in zip(self.upconv, self.up, reversed(skips)):
            x = block(torch.cat([upconv(x), skip], dim=1))
        
        return self.head(x)


MODELS = {'encdec': EncDec, 'unet': UNet}
