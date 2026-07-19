import torch.nn as nn

class SDDFBModel(nn.Module):
    def __init__(self, backbone, fpn, neck, head):
        super().__init__()

        self.backbone = backbone
        self.fpn = fpn
        self.neck = neck
        self.head = head

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        neck_out = self.neck(p2, p3, p4, p5)
        outputs = self.head(neck_out)

        return outputs
