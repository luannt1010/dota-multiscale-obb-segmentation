import torch.nn as nn

class SDDFBModel(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, images):
        p2,p3,p4,p5 = self.backbone(images)
        neck_out = self.neck(p2,p3,p4,p5)
        try:
            outputs = self.head(neck_out)
        except TypeError:
            outputs = self.head(neck_out["fused"])

        return outputs
