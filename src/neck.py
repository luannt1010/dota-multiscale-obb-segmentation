import torch.nn as nn
import torch.nn.functional as F
from src import FineFlow, ContextFlow
from src import CrossAttention, ScaleGate


class SDDFBNeck(nn.Module):
    def __init__(self, c2, c3, c4, c5, out_channels=256, num_heads=8):
        super().__init__()

        self.fine_flow = FineFlow(c2, c3, out_channels)
        self.context_flow = ContextFlow(c4, c5, out_channels)
        self.fine_to_context = CrossAttention(out_channels, num_heads)
        self.context_to_fine = CrossAttention(out_channels, num_heads)
        self.scale_gate = ScaleGate(out_channels)

    def forward(self, p2, p3, p4, p5):
        fine_feat = self.fine_flow(p2, p3)
        context_feat = self.context_flow(p4, p5)
        context_feat = F.interpolate(context_feat, size=fine_feat.shape[-2:], mode="bilinear", align_corners=False)
        fine_enhanced = (fine_feat + self.fine_to_context(fine_feat, context_feat))
        context_enhanced = (context_feat + self.context_to_fine(context_feat, fine_feat))
        fused, weights = self.scale_gate(fine_enhanced, context_enhanced)

        return {"fused": fused, "fine": fine_enhanced, "context": context_enhanced, "gate": weights}