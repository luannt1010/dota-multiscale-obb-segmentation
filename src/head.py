import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbone import norm2d


class OBBDetectionHead(nn.Module):
    def __init__(self, in_channels=256, feat_channels=256, num_classes=15, dropout=0.1):
        super().__init__()

        cls_layers = []
        reg_layers = []
        for _ in range(4):
            cls_layers.extend([nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                               norm2d(feat_channels),
                               nn.GELU(),
                               nn.Dropout2d(p=dropout)])
            reg_layers.extend([nn.Conv2d(feat_channels, feat_channels,3, padding=1),
                               norm2d(feat_channels),
                               nn.GELU(),
                               nn.Dropout2d(p=dropout)])
        self.stem = nn.Conv2d(in_channels, feat_channels,1)
        self.cls_branch = nn.Sequential(*cls_layers)
        self.reg_branch = nn.Sequential(*reg_layers)
        self.cls_pred = nn.Conv2d(feat_channels, num_classes,1)
        self.box_pred = nn.Conv2d(feat_channels,4, 1)
        self.angle_pred = nn.Conv2d(feat_channels,1,1)
        self.centerness_pred = nn.Conv2d(feat_channels,1,1)
        nn.init.constant_(self.cls_pred.bias, -4.6)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_branch(x)
        reg_feat = self.reg_branch(x)
        cls_logits = self.cls_pred(cls_feat)
        bbox = F.relu(self.box_pred(reg_feat))
        angle = torch.tanh(self.angle_pred(reg_feat)) * torch.pi
        centerness = torch.sigmoid(self.centerness_pred(reg_feat))

        return {"cls_logits": cls_logits, "bbox": bbox, "angle": angle, "centerness": centerness}

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128, num_classes=15, dropout=0.1):
        super().__init__()

        self.decoder = nn.Sequential(nn.Conv2d(in_channels, hidden_channels,3, padding=1),
                                     norm2d(hidden_channels),
                                     nn.GELU(),
                                     nn.Dropout2d(p=dropout),
                                     nn.Conv2d(hidden_channels, hidden_channels,3, padding=1),
                                     norm2d(hidden_channels),
                                     nn.GELU(),
                                     nn.Dropout2d(p=dropout),
                                     nn.Conv2d(hidden_channels, hidden_channels,3, padding=1),
                                     norm2d(hidden_channels),
                                     nn.GELU(),
                                     nn.Dropout2d(p=dropout))

        self.mask_pred = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, x):
        feat = self.decoder(x)
        masks = self.mask_pred(feat)
        return {"mask_logits": masks}


class DualTaskHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=15, dropout=0.1):
        super().__init__()

        self.obb_head = OBBDetectionHead(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
        self.seg_head = SegmentationHead(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

    def forward(self, neck_out):
        feature = neck_out["fused"] if isinstance(neck_out, dict) else neck_out
        obb_out = self.obb_head(feature)
        seg_out = self.seg_head(feature)
        return {"obb": obb_out, "seg": seg_out}
