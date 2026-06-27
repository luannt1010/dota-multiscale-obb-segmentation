import torch
import torch.nn as nn

class Upsampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsampler = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.upsampler(x)

class SimpleFPN(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        
        self.up_sampler = Upsampler()
        self.conv_c2 = UpChannel(32, d)
        self.conv_c3 = UpChannel(64, d)
        self.conv_c4 = UpChannel(160, d)
        self.conv_c5 = UpChannel(256, d)

        self.smooth = nn.Conv2d(d, d, kernel_size=3, padding=1, stride=1)
    def forward(self, c2, c3, c4, c5):
        c2 = self.conv_c2(c2)
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        c5 = self.conv_c5(c5)

        p5 = c5
        p4 = self.up_sampler(p5) + c4
        p3 = self.up_sampler(p4) + c3
        p2 = self.up_sampler(p3) + c2

        return self.smooth(p2), self.smooth(p3), self.smooth(p4), self.smooth(p5)

class UpChannel(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, p):
        return self.conv(p)
    
# if __name__ == "__main__":
#     c5 = torch.randn(1, 32, 32, 32)
#     c4 = torch.randn(1, 64, 64, 64)
#     c3 = torch.randn(1, 160, 128, 128)
#     c2 = torch.randn(1, 256, 256, 256)
#     fpn = SimpleFPN(d=512)
#     p2, p3, p4, p5 = fpn(c2, c3, c4, c5)
#     print(p2.shape)
#     print(p3.shape)
#     print(p4.shape)
#     print(p5.shape)
