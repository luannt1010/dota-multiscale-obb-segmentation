import torch
import torch.nn as nn

def norm2d(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LSKBlock(nn.Module):
    """Large Selective Kernel block adapted from the LSKNet paper."""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = attn.mean(dim=1, keepdim=True)
        max_attn = attn.max(dim=1, keepdim=True)[0]
        squeeze = torch.cat([avg_attn, max_attn], dim=1)
        weights = self.conv_squeeze(squeeze).sigmoid()
        selected = attn1 * weights[:, 0:1] + attn2 * weights[:, 1:2]
        selected = self.conv(selected)
        return x * selected

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = norm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = LSKBlock(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = norm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm2d(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x):
        scale1 = self.layer_scale_1.view(1, -1, 1, 1)
        scale2 = self.layer_scale_2.view(1, -1, 1, 1)
        x = x + self.drop_path(scale1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(scale2 * self.mlp(self.norm2(x)))
        return x

class LSKNetBackbone(nn.Module):
    def __init__(self, embed_dims=(32, 64, 160, 256), depths=(3, 3, 5, 2), mlp_ratios=(8, 8, 4, 4), drop_path_rate=0.1):
        super().__init__()
        self.num_stages = len(embed_dims)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        in_chans = 3
        for i, dim in enumerate(embed_dims):
            patch = OverlapPatchEmbed(in_chans, dim, patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2)
            blocks = nn.Sequential(*[
                Block(dim, mlp_ratio=mlp_ratios[i], drop_path=dpr[cur + j])
                for j in range(depths[i])
            ])
            setattr(self, f"patch_embed{i + 1}", patch)
            setattr(self, f"block{i + 1}", blocks)
            in_chans = dim
            cur += depths[i]

    def forward(self, x):
        features = []
        for i in range(self.num_stages):
            x = getattr(self, f"patch_embed{i + 1}")(x)
            x = getattr(self, f"block{i + 1}")(x)
            features.append(x)
        return features