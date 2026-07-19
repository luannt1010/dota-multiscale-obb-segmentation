"""Evaluation metrics for the DOTA OBB detection and segmentation tasks.

The detection metric uses exact convex-polygon IoU and COCO-style 101-point
interpolation.  It reports AP50 and AP averaged over IoU thresholds 0.50:0.05:0.95.
Ground-truth entries marked by ``ignore``, ``difficult`` or ``iscrowd`` do not
count as positives and detections matched to them do not count as false positives.

Detection inputs are sequences with one dictionary per image::

    prediction = {
        "boxes": Tensor[N, 5],       # (cx, cy, width, height, angle_radians)
        # or "polygons": Tensor[N, 4, 2]
        "scores": Tensor[N],
        "labels": Tensor[N],
    }
    target = {
        "polygons": Tensor[M, 4, 2],
        # or "boxes": Tensor[M, 5]
        "labels": Tensor[M],
        "difficult": Tensor[M],     # optional
    }

The segmentation metric is designed for this project's multi-label masks with
shape ``[B, C, H, W]``. It accumulates a dataset-level confusion matrix and
reports per-class and macro/micro IoU, Dice, precision and recall.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "OBBMeanAveragePrecision",
    "SegmentationMetrics",
    "MultiTaskMetrics",
    "mean_average_precision_obb",
    "segmentation_metrics",
    "obb_to_polygons",
    "polygon_iou",
    "pairwise_polygon_iou",
    "detections_to_metric_input",
    "objects_to_metric_target",
]


def _as_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def obb_to_polygons(boxes: Any) -> np.ndarray:
    """Convert ``(cx, cy, width, height, angle_radians)`` boxes to polygons."""
    boxes_array = _as_numpy(boxes, np.float32).reshape(-1, 5)
    if np.any(boxes_array[:, 2:4] < 0):
        raise ValueError("OBB width and height must be non-negative")
    if len(boxes_array) == 0:
        return np.empty((0, 4, 2), dtype=np.float32)

    cx, cy, width, height, angle = boxes_array.T
    dx = width / 2.0
    dy = height / 2.0
    local_x = np.stack((-dx, dx, dx, -dx), axis=1)
    local_y = np.stack((-dy, -dy, dy, dy), axis=1)
    cos_a = np.cos(angle)[:, None]
    sin_a = np.sin(angle)[:, None]

    x = cx[:, None] + local_x * cos_a - local_y * sin_a
    y = cy[:, None] + local_x * sin_a + local_y * cos_a
    return np.stack((x, y), axis=-1).astype(np.float32, copy=False)


def _convex_polygon(polygon: Any) -> np.ndarray:
    points = _as_numpy(polygon, np.float32).reshape(-1, 2)
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.convexHull(points).reshape(-1, 2).astype(np.float32, copy=False)


def polygon_iou(polygon_a: Any, polygon_b: Any, eps: float = 1e-9) -> float:
    """Return exact IoU of two convex polygons using floating-point geometry."""
    poly_a = _convex_polygon(polygon_a)
    poly_b = _convex_polygon(polygon_b)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0

    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a <= eps or area_b <= eps:
        return 0.0

    intersection, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    intersection = float(np.clip(intersection, 0.0, min(area_a, area_b)))
    union = area_a + area_b - intersection
    if union <= eps:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def pairwise_polygon_iou(polygons_a: Any, polygons_b: Any) -> np.ndarray:
    """Compute the ``[N, M]`` IoU matrix for two polygon collections."""
    polygons_a = _as_numpy(polygons_a, np.float32).reshape(-1, 4, 2)
    polygons_b = _as_numpy(polygons_b, np.float32).reshape(-1, 4, 2)
    ious = np.zeros((len(polygons_a), len(polygons_b)), dtype=np.float32)
    for row, polygon_a in enumerate(polygons_a):
        for col, polygon_b in enumerate(polygons_b):
            ious[row, col] = polygon_iou(polygon_a, polygon_b)
    return ious


def _polygons_from_record(record: Mapping[str, Any]) -> np.ndarray:
    if "polygons" in record:
        return _as_numpy(record["polygons"], np.float32).reshape(-1, 4, 2)
    if "boxes" in record:
        return obb_to_polygons(record["boxes"])
    raise KeyError("Each detection record must contain either 'boxes' or 'polygons'")


def _normalise_prediction(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    polygons = _polygons_from_record(record)
    labels = _as_numpy(record.get("labels", []), np.int64).reshape(-1)
    scores = _as_numpy(record.get("scores", []), np.float64).reshape(-1)
    if not (len(polygons) == len(labels) == len(scores)):
        raise ValueError("Prediction polygons/boxes, labels and scores must have equal lengths")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Prediction scores must be finite")
    return {"polygons": polygons.copy(), "labels": labels.copy(), "scores": scores.copy()}


def _normalise_target(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    polygons = _polygons_from_record(record)
    labels = _as_numpy(record.get("labels", []), np.int64).reshape(-1)
    if len(polygons) != len(labels):
        raise ValueError("Target polygons/boxes and labels must have equal lengths")

    ignored = np.zeros(len(labels), dtype=bool)
    for key in ("ignore", "difficult", "iscrowd"):
        if key in record:
            values = _as_numpy(record[key]).astype(bool, copy=False).reshape(-1)
            if len(values) != len(labels):
                raise ValueError(f"Target '{key}' must have the same length as labels")
            ignored |= values
    return {"polygons": polygons.copy(), "labels": labels.copy(), "ignore": ignored}


def _average_precision_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style AP sampled at 101 recall points after precision enveloping."""
    if len(recall) == 0:
        return 0.0
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    sampled = np.zeros(101, dtype=np.float64)
    for index, recall_level in enumerate(np.linspace(0.0, 1.0, 101)):
        valid = np.flatnonzero(recall >= recall_level)
        if len(valid):
            sampled[index] = precision[valid[0]]
    return float(sampled.mean())


class OBBMeanAveragePrecision:
    """Dataset-level AP for rotated boxes or DOTA quadrilateral polygons.

    Classes without any non-ignored ground truth are excluded from the macro
    mean, matching common COCO evaluation behaviour.
    """

    def __init__(
        self,
        num_classes: int = 15,
        iou_thresholds: Sequence[float] | None = None,
        max_detections: int = 100,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")

        thresholds = (
            np.arange(0.50, 0.96, 0.05, dtype=np.float64)
            if iou_thresholds is None
            else np.asarray(tuple(iou_thresholds), dtype=np.float64)
        )
        if thresholds.ndim != 1 or len(thresholds) == 0:
            raise ValueError("iou_thresholds must be a non-empty one-dimensional sequence")
        if np.any((thresholds <= 0.0) | (thresholds > 1.0)):
            raise ValueError("Every IoU threshold must be in (0, 1]")

        self.num_classes = int(num_classes)
        self.iou_thresholds = thresholds
        self.max_detections = int(max_detections)
        self.class_names = list(class_names) if class_names is not None else None
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        self.reset()

    def reset(self) -> None:
        self._predictions: list[dict[str, np.ndarray]] = []
        self._targets: list[dict[str, np.ndarray]] = []

    def update(
        self,
        predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> None:
        """Add predictions and targets for one image or a batch of images."""
        if isinstance(predictions, Mapping):
            predictions = [predictions]
        if isinstance(targets, Mapping):
            targets = [targets]
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must contain the same number of images")

        for prediction, target in zip(predictions, targets):
            normalised_prediction = _normalise_prediction(prediction)
            normalised_target = _normalise_target(target)
            self._validate_labels(normalised_prediction["labels"], "prediction")
            self._validate_labels(normalised_target["labels"], "target")
            self._predictions.append(normalised_prediction)
            self._targets.append(normalised_target)

    def _validate_labels(self, labels: np.ndarray, source: str) -> None:
        if np.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError(f"{source} labels must be in [0, {self.num_classes - 1}]")

    def _class_ap(self, class_id: int) -> tuple[np.ndarray, int]:
        image_data: list[dict[str, Any]] = []
        entries: list[tuple[float, int, int]] = []
        num_ground_truth = 0

        for image_index, (prediction, target) in enumerate(
            zip(self._predictions, self._targets)
        ):
            pred_indices = np.flatnonzero(prediction["labels"] == class_id)
            pred_indices = pred_indices[
                np.argsort(-prediction["scores"][pred_indices], kind="stable")
            ][: self.max_detections]
            gt_indices = np.flatnonzero(target["labels"] == class_id)

            pred_polygons = prediction["polygons"][pred_indices]
            pred_scores = prediction["scores"][pred_indices]
            gt_polygons = target["polygons"][gt_indices]
            gt_ignore = target["ignore"][gt_indices]
            num_ground_truth += int((~gt_ignore).sum())

            ious = pairwise_polygon_iou(pred_polygons, gt_polygons)
            image_data.append({"ious": ious, "gt_ignore": gt_ignore})
            entries.extend(
                (float(score), image_index, local_index)
                for local_index, score in enumerate(pred_scores)
            )

        if num_ground_truth == 0:
            return np.full(len(self.iou_thresholds), np.nan, dtype=np.float64), 0

        entries.sort(key=lambda item: item[0], reverse=True)
        class_ap = np.zeros(len(self.iou_thresholds), dtype=np.float64)

        for threshold_index, threshold in enumerate(self.iou_thresholds):
            matched = [
                np.zeros(len(data["gt_ignore"]), dtype=bool) for data in image_data
            ]
            true_positives: list[float] = []
            false_positives: list[float] = []

            for _, image_index, prediction_index in entries:
                data = image_data[image_index]
                ious = data["ious"][prediction_index]
                gt_ignore = data["gt_ignore"]
                available = (~gt_ignore) & (~matched[image_index])

                candidate_indices = np.flatnonzero(available & (ious >= threshold))
                if len(candidate_indices):
                    best = candidate_indices[np.argmax(ious[candidate_indices])]
                    matched[image_index][best] = True
                    true_positives.append(1.0)
                    false_positives.append(0.0)
                    continue

                # Difficult/crowd objects can absorb detections without creating FP.
                if np.any(gt_ignore & (ious >= threshold)):
                    continue

                true_positives.append(0.0)
                false_positives.append(1.0)

            tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
            fp = np.cumsum(np.asarray(false_positives, dtype=np.float64))
            recall = tp / max(num_ground_truth, 1)
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            class_ap[threshold_index] = _average_precision_101(recall, precision)

        return class_ap, num_ground_truth

    def _class_key(self, class_id: int) -> str | int:
        return self.class_names[class_id] if self.class_names is not None else class_id

    def compute(self) -> dict[str, Any]:
        """Compute AP from all samples accumulated since the last reset."""
        ap = np.full(
            (len(self.iou_thresholds), self.num_classes), np.nan, dtype=np.float64
        )
        num_gt = np.zeros(self.num_classes, dtype=np.int64)
        for class_id in range(self.num_classes):
            ap[:, class_id], num_gt[class_id] = self._class_ap(class_id)

        valid_classes = num_gt > 0
        map_50_95 = float(np.nanmean(ap[:, valid_classes])) if valid_classes.any() else float("nan")

        index_50 = np.flatnonzero(np.isclose(self.iou_thresholds, 0.50))
        index_75 = np.flatnonzero(np.isclose(self.iou_thresholds, 0.75))
        map_50 = (
            float(np.nanmean(ap[index_50[0], valid_classes]))
            if len(index_50) and valid_classes.any()
            else float("nan")
        )
        map_75 = (
            float(np.nanmean(ap[index_75[0], valid_classes]))
            if len(index_75) and valid_classes.any()
            else float("nan")
        )

        per_class = {
            self._class_key(class_id): (
                float(np.nanmean(ap[:, class_id])) if num_gt[class_id] > 0 else float("nan")
            )
            for class_id in range(self.num_classes)
        }
        per_class_50 = {
            self._class_key(class_id): (
                float(ap[index_50[0], class_id])
                if len(index_50) and num_gt[class_id] > 0
                else float("nan")
            )
            for class_id in range(self.num_classes)
        }

        return {
            "map": map_50_95,
            "map_50": map_50,
            "map_50_95": map_50_95,
            "map_75": map_75,
            "ap_per_class": per_class,
            "ap50_per_class": per_class_50,
            "ap_by_iou": {
                f"{threshold:.2f}": (
                    float(np.nanmean(ap[index, valid_classes]))
                    if valid_classes.any()
                    else float("nan")
                )
                for index, threshold in enumerate(self.iou_thresholds)
            },
            "num_gt_per_class": {
                self._class_key(class_id): int(num_gt[class_id])
                for class_id in range(self.num_classes)
            },
            "iou_thresholds": self.iou_thresholds.tolist(),
            "num_images": len(self._targets),
        }


def mean_average_precision_obb(
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    num_classes: int = 15,
    iou_thresholds: Sequence[float] | None = None,
    max_detections: int = 100,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Functional interface for :class:`OBBMeanAveragePrecision`."""
    metric = OBBMeanAveragePrecision(
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
        max_detections=max_detections,
        class_names=class_names,
    )
    metric.update(predictions, targets)
    return metric.compute()


def detections_to_metric_input(detections: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Convert this project's decoded detection list into the metric input format."""
    boxes = np.asarray([item["obb"] for item in detections], dtype=np.float32).reshape(-1, 5)
    scores = np.asarray([item["score"] for item in detections], dtype=np.float32)
    labels = np.asarray([item["class_id"] for item in detections], dtype=np.int64)
    return {"boxes": boxes, "scores": scores, "labels": labels}


def objects_to_metric_target(objects: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Convert ``meta['objects']`` from :class:`DotaDataset` into metric targets."""
    polygons = np.asarray([item["polygon"] for item in objects], dtype=np.float32).reshape(-1, 4, 2)
    labels = np.asarray([item["class_id"] for item in objects], dtype=np.int64)
    difficult = np.asarray([item.get("difficult", 0) for item in objects], dtype=bool)
    return {"polygons": polygons, "labels": labels, "difficult": difficult}


def _extract_segmentation_prediction(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping) and "seg" in value:
        value = value["seg"]
    if isinstance(value, Mapping):
        for key in ("mask_logits", "masks", "predictions"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _extract_segmentation_target(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        for key in ("segmentation", "masks", "target"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(numerator, torch.nan, dtype=torch.float64)
    valid = denominator > 0
    result[valid] = numerator[valid].double() / denominator[valid].double()
    return result


def _nanmean(values: torch.Tensor) -> float:
    valid = ~torch.isnan(values)
    return float(values[valid].mean()) if valid.any() else float("nan")


class SegmentationMetrics:
    """Streaming metrics for multi-label semantic segmentation masks."""

    def __init__(
        self,
        num_classes: int = 15,
        threshold: float = 0.5,
        from_logits: bool = True,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.from_logits = bool(from_logits)
        self.class_names = list(class_names) if class_names is not None else None
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64)
        self.tn = torch.zeros(self.num_classes, dtype=torch.float64)
        self.num_updates = 0

    def _ensure_bchw(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[0] == self.num_classes:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and self.num_classes == 1:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, C, H, W]")
        if tensor.shape[1] != self.num_classes:
            raise ValueError(f"{name} must contain {self.num_classes} channels")
        return tensor

    @torch.no_grad()
    def update(
        self,
        predictions: Any,
        targets: Any,
        ignore_mask: Any | None = None,
    ) -> None:
        """Accumulate a batch; ``ignore_mask=True`` pixels are excluded."""
        predictions = self._ensure_bchw(
            _extract_segmentation_prediction(predictions), "predictions"
        )
        targets = self._ensure_bchw(_extract_segmentation_target(targets), "targets")

        if predictions.shape[:2] != targets.shape[:2]:
            raise ValueError("predictions and targets must have equal batch and channel dimensions")
        if predictions.shape[-2:] != targets.shape[-2:]:
            predictions = F.interpolate(
                predictions.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
        targets = targets.to(predictions.device)

        predicted = torch.sigmoid(predictions) >= self.threshold if self.from_logits else predictions >= self.threshold
        expected = targets >= 0.5

        valid = torch.ones_like(expected, dtype=torch.bool)
        if ignore_mask is not None:
            ignored = torch.as_tensor(ignore_mask, device=expected.device, dtype=torch.bool)
            if ignored.ndim == 3:
                ignored = ignored.unsqueeze(1)
            try:
                ignored = ignored.expand_as(expected)
            except RuntimeError as error:
                raise ValueError("ignore_mask must be broadcastable to [B, C, H, W]") from error
            valid &= ~ignored

        dimensions = (0, 2, 3)
        self.tp += (predicted & expected & valid).sum(dim=dimensions).cpu().double()
        self.fp += (predicted & ~expected & valid).sum(dim=dimensions).cpu().double()
        self.fn += (~predicted & expected & valid).sum(dim=dimensions).cpu().double()
        self.tn += (~predicted & ~expected & valid).sum(dim=dimensions).cpu().double()
        self.num_updates += 1

    def _class_key(self, class_id: int) -> str | int:
        return self.class_names[class_id] if self.class_names is not None else class_id

    def _per_class_dict(self, values: torch.Tensor) -> dict[str | int, float]:
        return {self._class_key(index): float(values[index]) for index in range(self.num_classes)}

    def compute(self) -> dict[str, Any]:
        union = self.tp + self.fp + self.fn
        predicted_positive = self.tp + self.fp
        actual_positive = self.tp + self.fn
        total = self.tp + self.fp + self.fn + self.tn

        iou = _safe_ratio(self.tp, union)
        dice = _safe_ratio(2.0 * self.tp, 2.0 * self.tp + self.fp + self.fn)
        precision = _safe_ratio(self.tp, predicted_positive)
        recall = _safe_ratio(self.tp, actual_positive)
        accuracy = _safe_ratio(self.tp + self.tn, total)

        total_tp = self.tp.sum()
        total_fp = self.fp.sum()
        total_fn = self.fn.sum()
        total_pixels = total.sum()

        return {
            "mean_iou": _nanmean(iou),
            "miou": _nanmean(iou),
            "mean_dice": _nanmean(dice),
            "micro_iou": float(total_tp / (total_tp + total_fp + total_fn))
            if (total_tp + total_fp + total_fn) > 0
            else float("nan"),
            "micro_dice": float(2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn))
            if (2.0 * total_tp + total_fp + total_fn) > 0
            else float("nan"),
            "mean_precision": _nanmean(precision),
            "mean_recall": _nanmean(recall),
            "pixel_accuracy": float((self.tp.sum() + self.tn.sum()) / total_pixels)
            if total_pixels > 0
            else float("nan"),
            "iou_per_class": self._per_class_dict(iou),
            "dice_per_class": self._per_class_dict(dice),
            "precision_per_class": self._per_class_dict(precision),
            "recall_per_class": self._per_class_dict(recall),
            "accuracy_per_class": self._per_class_dict(accuracy),
            "support_per_class": {
                self._class_key(index): int(actual_positive[index])
                for index in range(self.num_classes)
            },
            "num_updates": self.num_updates,
        }


def segmentation_metrics(
    predictions: Any,
    targets: Any,
    num_classes: int = 15,
    threshold: float = 0.5,
    from_logits: bool = True,
    class_names: Sequence[str] | None = None,
    ignore_mask: Any | None = None,
) -> dict[str, Any]:
    """Functional interface for :class:`SegmentationMetrics`."""
    metric = SegmentationMetrics(
        num_classes=num_classes,
        threshold=threshold,
        from_logits=from_logits,
        class_names=class_names,
    )
    metric.update(predictions, targets, ignore_mask=ignore_mask)
    return metric.compute()


class MultiTaskMetrics:
    """Convenience container for independent OBB and segmentation accumulators."""

    def __init__(
        self,
        num_classes: int = 15,
        class_names: Sequence[str] | None = None,
        iou_thresholds: Sequence[float] | None = None,
        max_detections: int = 100,
        segmentation_threshold: float = 0.5,
    ) -> None:
        self.detection = OBBMeanAveragePrecision(
            num_classes=num_classes,
            iou_thresholds=iou_thresholds,
            max_detections=max_detections,
            class_names=class_names,
        )
        self.segmentation = SegmentationMetrics(
            num_classes=num_classes,
            threshold=segmentation_threshold,
            from_logits=True,
            class_names=class_names,
        )

    def reset(self) -> None:
        self.detection.reset()
        self.segmentation.reset()

    def update_detection(
        self,
        predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> None:
        self.detection.update(predictions, targets)

    def update_segmentation(
        self,
        predictions: Any,
        targets: Any,
        ignore_mask: Any | None = None,
    ) -> None:
        self.segmentation.update(predictions, targets, ignore_mask=ignore_mask)

    def compute(self) -> dict[str, dict[str, Any]]:
        return {
            "detection": self.detection.compute(),
            "segmentation": self.segmentation.compute(),
        }
