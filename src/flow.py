import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbone import LSKBlock

class FineFlow(nn.Module):
    def __init__(self, c2, c3, out_channels=256):
        super().__init__()

        self.p2_proj = nn.Conv2d(c2, out_channels, 1)
        self.p3_proj = nn.Conv2d(c3, out_channels, 1)
        self.d1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, p2, p3):

        p2 = self.p2_proj(p2)
        p3 = F.interpolate(self.p3_proj(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)

        x = p2 + p3

        b1 = self.d1(x)
        b2 = self.d2(x)
        b3 = self.d4(x)

        x = torch.cat([b1, b2, b3], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        return self.act(x)

class ContextFlow(nn.Module):
    def __init__(self, c4, c5, out_channels=256):
        super().__init__()

        self.p4_proj = nn.Conv2d(c4, out_channels, 1)
        self.p5_proj = nn.Conv2d(c5, out_channels, 1)
        self.lsk = LSKBlock(out_channels)

    def forward(self, p4, p5):
        p4 = self.p4_proj(p4)
        p5 = F.interpolate(self.p5_proj(p5), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        x = p4 + p5
        x = self.lsk(x)
        return x