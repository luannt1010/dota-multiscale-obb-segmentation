import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=16):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide dim")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.dim = dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def _partition_windows(self, feature):
        batch_size, channels, height, width = feature.shape
        window_size = self.window_size
        pad_h = (-height) % window_size
        pad_w = (-width) % window_size

        if pad_h or pad_w:
            feature = F.pad(feature, (0, pad_w, 0, pad_h))

        padded_h, padded_w = feature.shape[-2:]
        num_h = padded_h // window_size
        num_w = padded_w // window_size
        windows = feature.reshape(
            batch_size, channels, num_h, window_size, num_w, window_size
        )
        windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = windows.reshape(
            batch_size * num_h * num_w, window_size * window_size, channels
        )

        key_padding_mask = None
        if pad_h or pad_w:
            valid = torch.zeros(
                (padded_h, padded_w), dtype=torch.bool, device=feature.device
            )
            valid[:height, :width] = True
            valid = valid.reshape(num_h, window_size, num_w, window_size)
            valid = valid.permute(0, 2, 1, 3).contiguous()
            valid = valid.reshape(num_h * num_w, window_size * window_size)
            key_padding_mask = (~valid).repeat(batch_size, 1)

        metadata = (batch_size, height, width, padded_h, padded_w)
        return windows, key_padding_mask, metadata

    def _merge_windows(self, windows, metadata):
        batch_size, height, width, padded_h, padded_w = metadata
        window_size = self.window_size
        num_h = padded_h // window_size
        num_w = padded_w // window_size

        feature = windows.reshape(
            batch_size, num_h, num_w, window_size, window_size, self.dim
        )
        feature = feature.permute(0, 5, 1, 3, 2, 4).contiguous()
        feature = feature.reshape(batch_size, self.dim, padded_h, padded_w)
        return feature[:, :, :height, :width]

    def forward(self, query_feat, context_feat):
        if query_feat.ndim != 4 or context_feat.ndim != 4:
            raise ValueError("query_feat and context_feat must have shape [B, C, H, W]")
        if query_feat.shape != context_feat.shape:
            raise ValueError("query_feat and context_feat must have identical shapes")
        if query_feat.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {query_feat.shape[1]}")

        q, key_padding_mask, metadata = self._partition_windows(query_feat)
        kv, _, _ = self._partition_windows(context_feat)
        out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.norm(out + q)
        return self._merge_windows(out, metadata)

class ScaleGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        self.channels = channels
        hidden_channels = max(channels // 4, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1),
                                  nn.Conv2d(channels, hidden_channels, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(hidden_channels, 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, fine, context):
        if fine.ndim != 4 or context.ndim != 4:
            raise ValueError("fine and context must have shape [B, C, H, W]")
        if fine.shape != context.shape:
            raise ValueError("fine and context must have identical shapes")
        if fine.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {fine.shape[1]}")

        logits = self.gate(torch.cat([fine, context], dim=1))
        weights = torch.softmax(logits, dim=1)
        fine_w = weights[:, 0:1]
        context_w = weights[:, 1:2]
        fused = (fine_w * fine + context_w * context)

        return fused, weights
