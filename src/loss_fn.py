import torch
import torch.nn.functional as F

def heatmap_focal_loss(logits, targets, alpha=2.0, beta=4.0):
    pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    pos = targets.eq(1.0).float()
    neg = targets.lt(1.0).float()
    neg_weights = (1.0 - targets).pow(beta)

    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_l1(pred, target, mask):
    mask = mask.expand_as(pred)
    denom = mask.sum().clamp(min=1.0)
    return F.smooth_l1_loss(pred * mask, target * mask, reduction="sum") / denom


def binary_focal_loss(pred, target, gamma=2.0):
    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    pos = target.eq(1.0).float()
    neg = target.lt(1.0).float()

    pos_loss = -torch.log(pred) * (1.0 - pred).pow(gamma) * pos
    neg_loss = -torch.log(1.0 - pred) * pred.pow(gamma) * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def detection_loss(outputs, targets, cls_w=1, bbox_w=1, angle_w=0.5, centerness_w=0.5):
    obb = outputs["obb"] if "obb" in outputs else outputs
    cls_logits = obb["cls_logits"]
    pred_bbox = obb["bbox"]
    pred_angle = obb["angle"]
    pred_centerness = obb["centerness"]

    heatmap = targets["heatmap"].to(cls_logits.device)
    bbox = targets["bbox"].to(cls_logits.device)
    angle = targets["angle"].to(cls_logits.device)
    centerness = targets["centerness"].to(cls_logits.device)
    mask = targets["mask"].to(cls_logits.device)

    cls_loss = heatmap_focal_loss(cls_logits, heatmap)
    bbox_loss = masked_l1(pred_bbox, bbox, mask)
    target_angle = angle * torch.pi
    angle_diff = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle)) / torch.pi
    angle_loss = masked_l1(angle_diff, torch.zeros_like(angle_diff), mask)
    centerness_loss = binary_focal_loss(pred_centerness, centerness)

    total = cls_w * cls_loss + bbox_w * bbox_loss + angle_w * angle_loss + centerness_w * centerness_loss
    parts = {"cls": cls_loss.detach(), "bbox": bbox_loss.detach(), "angle": angle_loss.detach(), "centerness": centerness_loss.detach()}
    return total, parts


def segmentation_loss(outputs, targets, seg_w=0.25):
    if "seg" not in outputs:
        return torch.zeros((), device=next(iter(outputs.values())).device)
    logits = outputs["seg"]["mask_logits"]
    target = targets["segmentation"].to(logits.device)
    pos = target.eq(1.0).float()
    neg = target.lt(1.0).float()
    pos_count = pos.sum().clamp(min=1.0)
    neg_count = neg.sum().clamp(min=1.0)
    pos_weight = (neg_count / pos_count).clamp(max=20.0)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    bce = (bce * bce_weights).mean()

    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    valid_classes = target.sum(dim=dims) > 0
    dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
    if valid_classes.any():
        dice = dice[valid_classes].mean()
    else:
        dice = dice.mean()

    return seg_w * (0.5 * bce + 0.5 * dice)


