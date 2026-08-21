import torch
import torch.nn.functional as F

def heatmap_focal_loss(logits, targets, alpha=2.0, beta=4.0, ignore_mask=None):
    # Log/sigmoid are numerically fragile in float16 near 0 and 1.
    targets = targets.float()
    pred = logits.float().sigmoid().clamp(1e-4, 1.0 - 1e-4)
    pos = targets.eq(1.0)
    neg = targets.lt(1.0)
    if ignore_mask is not None:
        ignored = ignore_mask.to(device=targets.device, dtype=torch.bool)
        if ignored.ndim == targets.ndim - 1:
            ignored = ignored.unsqueeze(1)
        try:
            ignored = ignored.expand_as(targets)
        except RuntimeError as error:
            raise ValueError("ignore_mask must be broadcastable to logits") from error
        # A true positive remains supervised even if regions overlap.
        neg &= ~ignored
    pos = pos.float()
    neg = neg.float()
    neg_weights = (1.0 - targets).pow(beta)

    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos

def masked_l1(pred, target, mask):
    pred = pred.float()
    target = target.float()
    mask = mask.expand_as(pred)
    denom = mask.sum().clamp(min=1.0)
    return F.smooth_l1_loss(pred * mask, target * mask, reduction="sum") / denom


def binary_focal_loss(pred, target, gamma=2.0, ignore_mask=None):
    target = target.float()
    pred = pred.float().clamp(1e-4, 1.0 - 1e-4)
    pos = target.eq(1.0)
    neg = target.lt(1.0)
    if ignore_mask is not None:
        ignored = ignore_mask.to(device=target.device, dtype=torch.bool)
        if ignored.ndim == target.ndim - 1:
            ignored = ignored.unsqueeze(1)
        try:
            ignored = ignored.expand_as(target)
        except RuntimeError as error:
            raise ValueError("ignore_mask must be broadcastable to prediction") from error
        neg &= ~ignored
    pos = pos.float()
    neg = neg.float()

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
    detection_ignore = targets.get("detection_ignore")
    centerness_ignore = targets.get("centerness_ignore")
    if detection_ignore is not None:
        detection_ignore = detection_ignore.to(cls_logits.device)
    if centerness_ignore is not None:
        centerness_ignore = centerness_ignore.to(cls_logits.device)

    cls_loss = heatmap_focal_loss(
        cls_logits, heatmap, ignore_mask=detection_ignore
    )
    bbox_loss = masked_l1(pred_bbox, bbox, mask)
    target_angle = angle * torch.pi
    # A rectangle is unchanged by a 180-degree rotation. Wrapping 2 * delta
    # makes theta and theta + pi equivalent, which is essential after flips.
    angle_delta = pred_angle.float() - target_angle.float()
    angle_diff = 0.5 * torch.atan2(
        torch.sin(2.0 * angle_delta), torch.cos(2.0 * angle_delta)
    )
    angle_diff = angle_diff / (torch.pi / 2.0)
    angle_loss = masked_l1(angle_diff, torch.zeros_like(angle_diff), mask)
    centerness_loss = binary_focal_loss(
        pred_centerness,
        centerness,
        ignore_mask=centerness_ignore,
    )

    total = cls_w * cls_loss + bbox_w * bbox_loss + angle_w * angle_loss + centerness_w * centerness_loss
    parts = {"cls": cls_loss.detach(), "bbox": bbox_loss.detach(), "angle": angle_loss.detach(), "centerness": centerness_loss.detach()}
    return total, parts


def segmentation_loss(outputs, targets, seg_w=0.25):
    if "seg" not in outputs:
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                return value.new_zeros(())
            if isinstance(value, dict):
                tensor = next(
                    (item for item in value.values() if isinstance(item, torch.Tensor)),
                    None,
                )
                if tensor is not None:
                    return tensor.new_zeros(())
        return torch.zeros(())

    logits = outputs["seg"]["mask_logits"]
    loss_logits = logits.float()
    target = targets["segmentation"].to(logits.device).float()
    ignore = targets.get("segmentation_ignore")
    if ignore is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        ignore = ignore.to(device=logits.device, dtype=torch.bool)
        if ignore.ndim == target.ndim - 1:
            ignore = ignore.unsqueeze(1)
        try:
            valid = ~ignore.expand_as(target)
        except RuntimeError as error:
            raise ValueError(
                "segmentation_ignore must be broadcastable to logits"
            ) from error

    valid_float = valid.float()
    valid_count = valid_float.sum()
    if valid_count == 0:
        return logits.sum() * 0.0

    pos = (target.eq(1.0) & valid).float()
    neg = (target.lt(1.0) & valid).float()
    count_dims = (0, 2, 3)
    pos_count = pos.sum(dim=count_dims, keepdim=True).clamp(min=1.0)
    neg_count = neg.sum(dim=count_dims, keepdim=True).clamp(min=1.0)
    pos_weight = (neg_count / pos_count).clamp(max=20.0)

    bce = F.binary_cross_entropy_with_logits(loss_logits, target, reduction="none")
    bce_weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    bce = (bce * bce_weights * valid_float).sum() / valid_count

    probs = torch.sigmoid(loss_logits) * valid_float
    masked_target = target * valid_float
    dims = (0, 2, 3)
    intersection = (probs * masked_target).sum(dim=dims)
    union = probs.sum(dim=dims) + masked_target.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
    positive_classes = masked_target.sum(dim=dims) > 0
    if positive_classes.any():
        dice = dice[positive_classes].mean()
    else:
        valid_classes = valid_float.sum(dim=dims) > 0
        dice = dice[valid_classes].mean()

    return seg_w * (0.5 * bce + 0.5 * dice)
