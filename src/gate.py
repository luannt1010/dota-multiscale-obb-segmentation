import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_feat, context_feat):
        B, C, H, W = query_feat.shape
        q = query_feat.flatten(2).transpose(1, 2)
        kv = context_feat.flatten(2).transpose(1, 2)
        out, _ = self.attn(q, kv, kv)
        out = self.norm(out + q)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

class ScaleGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1),
                                  nn.Conv2d(channels, channels // 4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(channels // 4, 2, 1))

    def forward(self, fine, context):

        logits = self.gate(torch.cat([fine, context], dim=1))
        weights = torch.softmax(logits, dim=1)
        fine_w = weights[:, 0:1]
        context_w = weights[:, 1:2]
        fused = (fine_w * fine + context_w * context)

        return fused, weights
