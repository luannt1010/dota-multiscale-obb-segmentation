import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image, ImageDraw
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dota_dataset import DotaDataset
from src.loss_fn import detection_loss, segmentation_loss
from src.backbone import LSKNetBackbone
from src.head import DualTaskHead
from src.model import SDDFBModel
from src.neck import SDDFBNeck

DOTA_CLASSES = ["plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", "tennis-court",
                "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter"]
OUTPUT_STRIDE = 4

CLASS_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
    (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GT_YELLOW = (255, 255, 0)

def parse_dota_label(label_path, class_to_id):
    objects = []
    with Path(label_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            class_name = parts[8]
            if class_name not in class_to_id:
                continue
            try:
                difficult = int(float(parts[9]))
            except ValueError:
                difficult = 0
            polygon = np.asarray(coords, dtype=np.float32).reshape(4, 2)
            objects.append({"polygon": polygon, "class_name": class_name,
                            "class_id": class_to_id[class_name], "difficult": difficult})
    return objects


def infer_label_path(image_path):
    image_path = Path(image_path)
    candidates = []
    if image_path.parent.name == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.with_suffix(".txt"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def polygon_to_obb(polygon):
    polygon = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    center = polygon.mean(axis=0)

    edge01 = polygon[1] - polygon[0]
    edge12 = polygon[2] - polygon[1]
    len01 = float(np.linalg.norm(edge01))
    len12 = float(np.linalg.norm(edge12))

    if len01 >= len12:
        width, height = len01, len12
        angle = math.atan2(float(edge01[1]), float(edge01[0]))
    else:
        width, height = len12, len01
        angle = math.atan2(float(edge12[1]), float(edge12[0]))

    return float(center[0]), float(center[1]), max(width, 1.0), max(height, 1.0), normalize_angle(angle)


def normalize_angle(angle):
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def resize_image_and_objects(image, objects, target_size):
    src_w, src_h = image.size
    image = image.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
    sx = target_size / src_w
    sy = target_size / src_h

    scaled = []
    for obj in objects:
        polygon = obj["polygon"].copy()
        polygon[:, 0] *= sx
        polygon[:, 1] *= sy
        scaled.append({**obj, "polygon": polygon})
    return image, scaled


def horizontal_flip(image, objects):
    width = image.size[0]
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped = []
    for obj in objects:
        polygon = obj["polygon"].copy()
        polygon[:, 0] = width - 1 - polygon[:, 0]
        flipped.append({**obj, "polygon": polygon})
    return image, flipped


def vertical_flip(image, objects):
    height = image.size[1]
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped = []
    for obj in objects:
        polygon = obj["polygon"].copy()
        polygon[:, 1] = height - 1 - polygon[:, 1]
        flipped.append({**obj, "polygon": polygon})
    return image, flipped


def rotate90_ccw(image, objects):
    width, _ = image.size
    image = image.transpose(Image.ROTATE_90)
    rotated = []
    for obj in objects:
        polygon = obj["polygon"].copy()
        x = polygon[:, 0].copy()
        y = polygon[:, 1].copy()
        polygon[:, 0] = y
        polygon[:, 1] = width - 1 - x
        rotated.append({**obj, "polygon": polygon})
    return image, rotated


def apply_train_augmentation(image, objects, hflip_prob=0.5,  vflip_prob=0.25, rot90_prob=0.25):
    if random.random() < hflip_prob:
        image, objects = horizontal_flip(image, objects)
    if random.random() < vflip_prob:
        image, objects = vertical_flip(image, objects)
    if random.random() < rot90_prob:
        image, objects = rotate90_ccw(image, objects)
    return image, objects


def draw_gaussian(heatmap, cls_id, cx, cy, radius=2):
    _, out_h, out_w = heatmap.shape
    diameter = 2 * radius + 1
    yy, xx = torch.meshgrid(
        torch.arange(diameter, dtype=torch.float32),
        torch.arange(diameter, dtype=torch.float32),
        indexing="ij",
    )
    center = radius
    sigma = radius / 2 + 1e-6
    gaussian = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))

    left, right = min(cx, radius), min(out_w - cx - 1, radius)
    top, bottom = min(cy, radius), min(out_h - cy - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    patch = heatmap[cls_id, cy - top:cy + bottom + 1, cx - left:cx + right + 1]
    gpatch = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    heatmap[cls_id, cy - top:cy + bottom + 1, cx - left:cx + right + 1] = torch.maximum(patch, gpatch)


def make_segmentation_target(objects, image_size, stride, num_classes):
    out_size = image_size // stride
    masks = torch.zeros(num_classes, out_size, out_size, dtype=torch.float32)
    for obj in objects:
        if obj["difficult"] == 1:
            continue
        cls_id = obj["class_id"]
        polygon = (obj["polygon"] / stride).tolist()
        mask_image = Image.new("L", (out_size, out_size), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.polygon([tuple(point) for point in polygon], outline=1, fill=1)
        mask = torch.from_numpy(np.asarray(mask_image, dtype=np.float32))
        masks[cls_id] = torch.maximum(masks[cls_id], mask)
    return masks


def make_targets(objects, image_size=512, stride=4, num_classes=15, gaussian_radius=2):
    out_h = image_size // stride
    out_w = image_size // stride
    heatmap = torch.zeros(num_classes, out_h, out_w, dtype=torch.float32)
    bbox = torch.zeros(4, out_h, out_w, dtype=torch.float32)
    angle = torch.zeros(1, out_h, out_w, dtype=torch.float32)
    centerness = torch.zeros(1, out_h, out_w, dtype=torch.float32)
    regression = torch.zeros(6, out_h, out_w, dtype=torch.float32)
    mask = torch.zeros(1, out_h, out_w, dtype=torch.float32)
    segmentation = make_segmentation_target(objects, image_size, stride, num_classes)

    for obj in objects:
        if obj["difficult"] == 1:
            continue

        cx, cy, width, height, theta = polygon_to_obb(obj["polygon"])
        gx = int(cx / stride)
        gy = int(cy / stride)
        if gx < 0 or gy < 0 or gx >= out_w or gy >= out_h:
            continue

        cls_id = obj["class_id"]
        draw_gaussian(heatmap, cls_id, gx, gy, radius=gaussian_radius)

        cell_cx = (gx + 0.5) * stride
        cell_cy = (gy + 0.5) * stride
        left = max((cell_cx - (cx - width / 2)) / image_size, 0.0)
        top = max((cell_cy - (cy - height / 2)) / image_size, 0.0)
        right = max(((cx + width / 2) - cell_cx) / image_size, 0.0)
        bottom = max(((cy + height / 2) - cell_cy) / image_size, 0.0)

        bbox[:, gy, gx] = torch.tensor([left, top, right, bottom], dtype=torch.float32)
        angle[:, gy, gx] = theta / math.pi
        centerness[:, gy, gx] = 1.0
        regression[:, gy, gx] = torch.tensor([
            cx / image_size,
            cy / image_size,
            math.log(max(width, 1.0) / image_size),
            math.log(max(height, 1.0) / image_size),
            math.sin(theta),
            math.cos(theta),
        ], dtype=torch.float32)
        mask[:, gy, gx] = 1.0

    return {
        "heatmap": heatmap,
        "bbox": bbox,
        "angle": angle,
        "centerness": centerness,
        "regression": regression,
        "mask": mask,
        "segmentation": segmentation,
    }

def collate_fn(batch):
    images, targets, metas = zip(*batch)
    merged_targets = {key: torch.stack([target[key] for target in targets]) for key in targets[0]}
    return torch.stack(images), merged_targets, list(metas)

def create_dataloaders(train_root="dataset/DOTAv1.0/train", val_root="dataset/DOTAv1.0/val", image_size=512, batch_size=2):
    train_dataset = DotaDataset(train_root, image_size=image_size, stride=OUTPUT_STRIDE, augment=True)
    val_dataset = DotaDataset(val_root, image_size=image_size, stride=OUTPUT_STRIDE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader

def move_targets(targets, device):
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}

def train(model, train_loader, val_loader, optimizer, scheduler, sp, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is training on {device}.")
    best_loss = float("inf")
    best_save_path = os.path.join(sp, "best.pth")
    last_save_path = os.path.join(sp, "last.pth")
    history_save_path = os.path.join(sp, "history.json")
    history = {"train_loss": [], "val_loss": []}
    train_time = 0
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_running_loss = 0
        train_pbar = tqdm(train_loader, desc=f"[Training] Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, targets, _ in train_pbar:
            images = images.to(device, non_blocking=True)
            targets = move_targets(targets, device)

            outputs = model(images)
            det_loss, det_parts = detection_loss(outputs, targets)
            seg_loss = segmentation_loss(outputs, targets)
            loss = det_loss + seg_loss
            train_running_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_epoch_loss = train_running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0
        val_pbar = tqdm(val_loader, desc=f"[Validating] Epoch {epoch+1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for images, targets, _ in val_pbar:
                images = images.to(device, non_blocking=True)
                targets = move_targets(targets, device)

                outputs = model(images)
                det_loss, det_parts = detection_loss(outputs, targets)
                seg_loss = segmentation_loss(outputs, targets)
                loss = det_loss + seg_loss
                val_running_loss += loss.item()
            val_epoch_loss = val_running_loss / len(val_loader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step()

        end = time.time()
        epoch_time = (end - start) / 60
        train_time += epoch_time

        print(f" Epoch {epoch+1}/{num_epochs}: TrainLoss={train_epoch_loss:.4f} ValLoss={val_epoch_loss:.4f} Time={epoch_time:.4f}")


        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "scheduler": scheduler.state_dict()}
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(checkpoint, best_save_path)
            print(f" Best model is saved at {best_save_path} and epoch {epoch+1}.")
        torch.save(checkpoint, last_save_path)

    with open(history_save_path, "w") as f:
        json.dump(history, f)

    print(f"Spent total {train_time:.4f} minutes to train.")
    print("Training completed.")
    print(f"History is saved at {history_save_path}.")
    return history


def build_model(num_classes=15):
    backbone = LSKNetBackbone(embed_dims=(32, 64, 160, 256))
    neck = SDDFBNeck(32, 64, 160, 256, out_channels=256, num_heads=8)
    head = DualTaskHead(in_channels=256, num_classes=num_classes)
    return SDDFBModel(backbone, neck, head)


def obb_to_polygon(cx, cy, width, height, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = width / 2
    dy = height / 2
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


def polygon_mask(polygon, image_size):
    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(point) for point in polygon], outline=1, fill=1)
    return np.asarray(mask, dtype=bool)


def polygon_iou(poly_a, poly_b, image_size):
    mask_a = polygon_mask(poly_a, image_size)
    mask_b = polygon_mask(poly_b, image_size)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def evaluate_detections(detections, gt_objects, image_size, iou_threshold=0.5):
    if not gt_objects:
        for det in detections:
            det["correct"] = False
            det["iou"] = 0.0
        return detections

    matched_gt = set()
    for det in sorted(detections, key=lambda item: item["score"], reverse=True):
        pred_polygon = obb_to_polygon(*det["obb"])
        best_iou = 0.0
        best_index = None
        for index, obj in enumerate(gt_objects):
            if index in matched_gt or obj["class_id"] != det["class_id"] or obj["difficult"] == 1:
                continue
            iou = polygon_iou(pred_polygon, obj["polygon"], image_size)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        det["iou"] = best_iou
        det["correct"] = best_iou >= iou_threshold
        if det["correct"] and best_index is not None:
            matched_gt.add(best_index)
    return detections


def build_gt_class_map(gt_objects, image_size):
    class_map = np.full((image_size, image_size), -1, dtype=np.int32)
    for obj in gt_objects:
        if obj["difficult"] == 1:
            continue
        mask = polygon_mask(obj["polygon"], image_size)
        class_map[mask] = obj["class_id"]
    return class_map


def suppress_duplicate_centers(detections, min_distance=12):
    kept = []
    for det in sorted(detections, key=lambda item: item["score"], reverse=True):
        cx, cy, _, _, _ = det["obb"]
        duplicate = False
        for old in kept:
            if old["class_id"] != det["class_id"]:
                continue
            ox, oy, _, _, _ = old["obb"]
            if math.hypot(cx - ox, cy - oy) < min_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def decode_predictions(outputs, image_size, stride=4, conf_threshold=0.15, topk=50, class_names=DOTA_CLASSES):
    obb = outputs["obb"] if "obb" in outputs else outputs
    cls_scores = obb["cls_logits"].sigmoid()[0]
    centerness = obb["centerness"][0].clamp(0, 1)
    scores = cls_scores * centerness

    pooled = F.max_pool2d(scores.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    scores = scores * (scores == pooled).float()

    bbox = obb["bbox"][0]
    angle = obb["angle"][0, 0]
    flat_scores = scores.flatten()
    k = min(topk * 5, flat_scores.numel())
    values, indices = torch.topk(flat_scores, k)

    detections = []
    _, h, w = scores.shape
    for score_tensor, index_tensor in zip(values.detach().cpu(), indices.detach().cpu()):
        score = float(score_tensor)
        if score < conf_threshold:
            continue

        index = int(index_tensor)
        cls_id = index // (h * w)
        rem = index % (h * w)
        y = rem // w
        x = rem % w

        left, top, right, bottom = bbox[:, y, x].detach().cpu().numpy().tolist()
        theta = float(angle[y, x].detach().cpu())
        cell_cx = (x + 0.5) * stride
        cell_cy = (y + 0.5) * stride
        width = max((left + right) * image_size, 1.0)
        height = max((top + bottom) * image_size, 1.0)
        cx = cell_cx + (right - left) * image_size / 2
        cy = cell_cy + (bottom - top) * image_size / 2

        detections.append({
            "score": score,
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "obb": (cx, cy, width, height, theta),
        })

    detections = suppress_duplicate_centers(detections)
    return detections[:topk]


def draw_obb_image(image, detections, gt_objects=None):
    obb_image = image.copy()
    draw = ImageDraw.Draw(obb_image)
    for obj in gt_objects or []:
        draw.polygon([tuple(point) for point in obj["polygon"]], outline=GT_YELLOW, width=1)

    for det in detections:
        color = GREEN if det.get("correct", False) else RED
        polygon = obb_to_polygon(*det["obb"])
        draw.polygon(polygon, outline=color, width=2)
        label = f"{det['class_name']} {det['score']:.2f}"
        if "iou" in det:
            label = f"{label} IoU={det['iou']:.2f}"
        draw.text(polygon[0], label, fill=color)
    return obb_image


def draw_segment_image(image, outputs, image_size, gt_objects=None, threshold=0.5, alpha=0.45):
    if "seg" not in outputs:
        return image.copy()

    logits = outputs["seg"]["mask_logits"]
    probs = torch.sigmoid(logits)
    probs = F.interpolate(probs, size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    max_probs, class_ids = probs.detach().cpu().max(dim=0)
    valid = max_probs.numpy() > threshold
    class_map = class_ids.numpy()

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    if gt_objects:
        gt_class_map = build_gt_class_map(gt_objects, image_size)
        correct = valid & (gt_class_map == class_map)
        false_positive_or_wrong_class = valid & ~correct
        missed_gt = (~valid) & (gt_class_map >= 0)
        overlay[correct] = (1.0 - alpha) * overlay[correct] + alpha * np.asarray(GREEN, dtype=np.float32)
        overlay[false_positive_or_wrong_class] = (
            (1.0 - alpha) * overlay[false_positive_or_wrong_class] + alpha * np.asarray(RED, dtype=np.float32)
        )
        overlay[missed_gt] = (1.0 - alpha) * overlay[missed_gt] + alpha * np.asarray(RED, dtype=np.float32)
    else:
        for cls_id in np.unique(class_map[valid]):
            mask = valid & (class_map == cls_id)
            color = np.asarray(CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)], dtype=np.float32)
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    segment_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if gt_objects:
        draw = ImageDraw.Draw(segment_image)
        for obj in gt_objects:
            draw.polygon([tuple(point) for point in obj["polygon"]], outline=GT_YELLOW, width=1)
    return segment_image


def inference(model,image_path, label_path=None, image_size=256, conf_threshold=0.15,
              seg_threshold=0.5, obb_iou_threshold=0.5, topk=50, device=None, obb_output_path=None, segment_output_path=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    original_image = Image.open(image_path).convert("RGB")
    image = original_image.resize((image_size, image_size), Image.BILINEAR)
    if label_path is None:
        label_path = infer_label_path(image_path)
    gt_objects = []
    if label_path is not None and Path(label_path).exists():
        raw_objects = parse_dota_label(label_path, CLASS_TO_ID)
        _, gt_objects = resize_image_and_objects(original_image, raw_objects, image_size)

    image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    detections = decode_predictions(outputs, image_size, stride=OUTPUT_STRIDE, conf_threshold=conf_threshold, topk=topk)
    detections = evaluate_detections(detections, gt_objects, image_size, iou_threshold=obb_iou_threshold)
    obb_image = draw_obb_image(image, detections, gt_objects=gt_objects)
    segment_image = draw_segment_image(image, outputs, image_size, gt_objects=gt_objects, threshold=seg_threshold)

    if obb_output_path is not None:
        Path(obb_output_path).parent.mkdir(parents=True, exist_ok=True)
        obb_image.save(obb_output_path)
    if segment_output_path is not None:
        Path(segment_output_path).parent.mkdir(parents=True, exist_ok=True)
        segment_image.save(segment_output_path)

    return {
        "obb_image": obb_image,
        "segment_image": segment_image,
        "detections": detections,
        "groundtruth": gt_objects,
        "label_path": str(label_path) if label_path is not None else None,
    }

def plot_history(history):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = [i+1 for i in range(len(train_loss))]

    idx_min = np.argmin(val_loss)
    min_epoch = epochs[idx_min]
    min_val_loss = val_loss[idx_min]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, color="blue", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_loss, color="red", linewidth=2, label="Val Loss")
    plt.annotate(text=f"Min Val Loss at\n(Epoch: {min_epoch}, Loss: {min_val_loss:.4f})",
                 xy=(min_epoch, min_val_loss), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=10, color="red", xytext=(20, 20))
    plt.title("Training Loss & Validating Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend()

    plt.tight_layout()
    plt.show()


