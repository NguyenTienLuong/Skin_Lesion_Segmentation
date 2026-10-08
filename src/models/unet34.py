from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, tv_tensors
from torchvision.transforms import v2

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch // 2, 2, stride=2)
        self.x_conv = nn.Conv2d(skip_ch, out_ch //2 , 1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.x_conv(skip)
        if x.shape[-2:] != skip.shape[-2:]: x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        return self.bn(F.relu((torch.cat([x, skip], 1))))

class UNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2,  64ch   (skip "initial layer")
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        # layer1 /4 64ch | layer2 /8 128ch | layer3 /16 256ch | layer4 /32 512ch (bottleneck)
        self.first_layer_group = nn.ModuleList([self.stem, self.layer1, self.layer2])  # "first layer group"
        self.up4, self.up3 = UpBlock(512, 256, 256), UpBlock(256, 128, 256)
        self.up2, self.up1 = UpBlock(256, 64, 256),   UpBlock(256, 64, 256)
        self.up0  = UpBlock(256,3,16)                        # ghép với ảnh đầu vào -> full-res
        self.head = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        s0 = self.stem(x); s1 = self.layer1(self.pool(s0)); s2 = self.layer2(s1); s3 = self.layer3(s2)
        b = self.layer4(s3)
        d = self.up4(b, s3); d = self.up3(d, s2); d = self.up2(d, s1); d = self.up1(d, s0); d = self.up0(d,x)
        return self.head(d)                                    # logits (N,1,H,W)
    
    
BN = nn.BatchNorm2d


def freeze_encoder(model: UNet34) -> None:
    """Mapping notebook: freeze toàn encoder, nhưng BN encoder vẫn cho học ở frozen stage."""
    for module in model.encoder_modules():
        for m in module.modules():
            for p in m.parameters(recurse=False):
                p.requires_grad = isinstance(m, BN)


def unfreeze(model: UNet34, freeze_bn: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = True
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, BN):
                for p in m.parameters():
                    p.requires_grad = False


def set_train_mode(model: nn.Module, bn_frozen: bool) -> None:
    model.train()
    if bn_frozen:
        for m in model.modules():
            if isinstance(m, BN):
                m.eval()
    