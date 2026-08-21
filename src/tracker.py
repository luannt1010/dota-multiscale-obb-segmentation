from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping
import numpy as np
import torch
import torch.nn.functional as F
from .metrics import obb_to_polygons, polygon_iou


_INVALID_COST = 1_000_000.0


def extract_deep_embeddings(
    feature_map: torch.Tensor,
    detections: Iterable[Mapping],
    image_size: int,
    grid_size: tuple[int, int] = (6, 10),
) -> np.ndarray:
    """Pool L2-normalised appearance descriptors from rotated feature ROIs.

    ``feature_map`` is the detector's deep fused feature tensor in ``BCHW``
    format. Sampling each OBB in its own rotated coordinate system preserves
    more appearance information than axis-aligned cropping while requiring no
    extra model or downloaded ReID weights.
    """

    detections = list(detections)
    if feature_map.ndim != 4 or feature_map.shape[0] < 1:
        raise ValueError("feature_map must have BCHW shape with a non-empty batch")
    if image_size <= 1:
        raise ValueError("image_size must be greater than one")
    grid_height, grid_width = grid_size
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("grid_size values must be positive")

    channels = int(feature_map.shape[1])
    if not detections:
        return np.empty((0, channels * 2), dtype=np.float32)

    boxes = torch.as_tensor(
        [detection["obb"] for detection in detections],
        dtype=feature_map.dtype,
        device=feature_map.device,
    ).reshape(-1, 5)
    local_y, local_x = torch.meshgrid(
        torch.linspace(
            -0.5,
            0.5,
            grid_height,
            dtype=feature_map.dtype,
            device=feature_map.device,
        ),
        torch.linspace(
            -0.5,
            0.5,
            grid_width,
            dtype=feature_map.dtype,
            device=feature_map.device,
        ),
        indexing="ij",
    )
    local_x = local_x.unsqueeze(0) * boxes[:, 2, None, None]
    local_y = local_y.unsqueeze(0) * boxes[:, 3, None, None]
    cosine = torch.cos(boxes[:, 4])[:, None, None]
    sine = torch.sin(boxes[:, 4])[:, None, None]
    sample_x = boxes[:, 0, None, None] + local_x * cosine - local_y * sine
    sample_y = boxes[:, 1, None, None] + local_x * sine + local_y * cosine

    scale = 2.0 / float(image_size - 1)
    grid = torch.stack(
        (sample_x * scale - 1.0, sample_y * scale - 1.0),
        dim=-1,
    )
    source = feature_map[:1].expand(len(detections), -1, -1, -1)
    patches = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    average_pool = patches.mean(dim=(-2, -1))
    max_pool = patches.amax(dim=(-2, -1))
    embeddings = F.normalize(
        torch.cat((average_pool, max_pool), dim=1),
        p=2,
        dim=1,
        eps=1e-12,
    )
    return embeddings.float().detach().cpu().numpy()


def _angle_delta(new_angle: float, old_angle: float) -> float:
    """Shortest angular delta for a rectangle, whose orientation repeats at pi."""

    return (new_angle - old_angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def _linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Solve rectangular minimum-cost assignment using the Hungarian method."""

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return []
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost_matrix must contain finite values")

    transposed = rows > columns
    if transposed:
        cost = cost.T
        rows, columns = cost.shape

    # Shortest augmenting-path form of the Hungarian algorithm.  It is O(n^3)
    # and avoids adding SciPy solely for frame-to-frame association.
    row_potential = np.zeros(rows + 1, dtype=np.float64)
    column_potential = np.zeros(columns + 1, dtype=np.float64)
    column_match = np.zeros(columns + 1, dtype=np.int64)
    previous_column = np.zeros(columns + 1, dtype=np.int64)

    for row in range(1, rows + 1):
        column_match[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        column = 0

        while True:
            used[column] = True
            matched_row = column_match[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[matched_row - 1, candidate - 1]
                    - row_potential[matched_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate

            for candidate in range(columns + 1):
                if used[candidate]:
                    row_potential[column_match[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_match[column] == 0:
                break

        while True:
            previous = previous_column[column]
            column_match[column] = column_match[previous]
            column = previous
            if column == 0:
                break

    assignments = [
        (int(column_match[column] - 1), column - 1)
        for column in range(1, columns + 1)
        if column_match[column] != 0
    ]
    if transposed:
        return [(column, row) for row, column in assignments]
    return assignments


@dataclass
class _Track:
    track_id: int
    class_id: int
    class_name: str
    obb: np.ndarray
    score: float
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )
    last_observed_obb: np.ndarray = field(default_factory=lambda: np.zeros(5))
    age: int = 1
    hits: int = 1
    missed: int = 0

    def __post_init__(self):
        self.obb = np.asarray(self.obb, dtype=np.float64).copy()
        self.last_observed_obb = self.obb.copy()

    def predict(self):
        self.obb = self.obb + self.velocity
        self.obb[2:4] = np.maximum(self.obb[2:4], 1.0)
        self.age += 1
        self.missed += 1

    def update(self, detection: Mapping, velocity_momentum: float):
        measurement = np.asarray(detection["obb"], dtype=np.float64).copy()
        measurement[4] = self.last_observed_obb[4] + _angle_delta(
            float(measurement[4]),
            float(self.last_observed_obb[4]),
        )
        frame_gap = max(self.missed, 1)
        measured_velocity = (measurement - self.last_observed_obb) / frame_gap
        measured_velocity[4] = _angle_delta(
            float(measurement[4]),
            float(self.last_observed_obb[4]),
        ) / frame_gap
        self.velocity = (
            velocity_momentum * self.velocity
            + (1.0 - velocity_momentum) * measured_velocity
        )
        self.obb = measurement
        self.last_observed_obb = measurement.copy()
        self.class_name = str(detection["class_name"])
        self.score = float(detection["score"])
        self.hits += 1
        self.missed = 0


class RotatedIoUTracker:
    """Track rotated detections across frames without an appearance model.

    ``requires_embeddings`` lets the inference pipeline avoid collecting deep
    feature maps for this geometry-only algorithm.

    Parameters are expressed in frames/pixels except thresholds.  ``max_missed``
    controls how long an unmatched identity remains available for recovery.
    The returned list contains only detections observed in the current frame;
    predicted-but-missing boxes are retained internally but are not rendered.
    """

    requires_embeddings = False

    def __init__(
        self,
        *,
        high_confidence_threshold: float = 0.30,
        low_confidence_threshold: float = 0.10,
        match_iou_threshold: float = 0.10,
        second_match_iou_threshold: float = 0.05,
        max_center_distance: float = 2.5,
        max_missed: int = 30,
        velocity_momentum: float = 0.65,
    ):
        for name, value in (
            ("high_confidence_threshold", high_confidence_threshold),
            ("low_confidence_threshold", low_confidence_threshold),
            ("match_iou_threshold", match_iou_threshold),
            ("second_match_iou_threshold", second_match_iou_threshold),
            ("velocity_momentum", velocity_momentum),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if low_confidence_threshold > high_confidence_threshold:
            raise ValueError(
                "low_confidence_threshold cannot exceed high_confidence_threshold"
            )
        if max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive")
        if max_missed < 0:
            raise ValueError("max_missed cannot be negative")

        self.high_confidence_threshold = float(high_confidence_threshold)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.match_iou_threshold = float(match_iou_threshold)
        self.second_match_iou_threshold = float(second_match_iou_threshold)
        self.max_center_distance = float(max_center_distance)
        self.max_missed = int(max_missed)
        self.velocity_momentum = float(velocity_momentum)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self):
        """Forget all identities and restart numbering from one."""

        self._tracks.clear()
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def active_tracks(self) -> tuple[dict, ...]:
        """Read-only snapshots useful for diagnostics and tests."""

        return tuple(
            {
                "track_id": track.track_id,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "obb": tuple(float(value) for value in track.obb),
                "score": track.score,
                "age": track.age,
                "hits": track.hits,
                "missed": track.missed,
            }
            for track in self._tracks
        )

    @staticmethod
    def _normalise_detection(detection: Mapping) -> dict:
        required = ("score", "class_id", "class_name", "obb")
        missing = [key for key in required if key not in detection]
        if missing:
            raise KeyError(f"Detection is missing keys: {', '.join(missing)}")

        normalised = dict(detection)
        obb = np.asarray(detection["obb"], dtype=np.float64).reshape(-1)
        if obb.shape != (5,) or not np.all(np.isfinite(obb)):
            raise ValueError("detection['obb'] must contain five finite values")
        if obb[2] <= 0 or obb[3] <= 0:
            raise ValueError("OBB width and height must be positive")
        score = float(detection["score"])
        if not math.isfinite(score):
            raise ValueError("detection score must be finite")

        normalised["obb"] = tuple(float(value) for value in obb)
        normalised["score"] = score
        normalised["class_id"] = int(detection["class_id"])
        normalised["class_name"] = str(detection["class_name"])
        return normalised

    @staticmethod
    def _pair_cost(track: _Track, detection: Mapping) -> tuple[float, float]:
        track_obb = np.asarray(track.obb, dtype=np.float64)
        detection_obb = np.asarray(detection["obb"], dtype=np.float64)
        polygons = obb_to_polygons(np.stack((track_obb, detection_obb)))
        overlap = polygon_iou(polygons[0], polygons[1])

        center_distance = float(
            np.linalg.norm(track_obb[:2] - detection_obb[:2])
        )
        track_scale = math.hypot(float(track_obb[2]), float(track_obb[3]))
        detection_scale = math.hypot(
            float(detection_obb[2]), float(detection_obb[3])
        )
        normalised_distance = center_distance / max(
            0.5 * (track_scale + detection_scale),
            1.0,
        )
        # IoU dominates.  The small distance term breaks ties and permits
        # recovery after a brief miss where predicted boxes no longer overlap.
        return (1.0 - overlap) + 0.20 * normalised_distance, overlap

    def _associate(
        self,
        track_indices: Iterable[int],
        detections: list[dict],
        detection_indices: Iterable[int],
        minimum_iou: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_indices = list(track_indices)
        detection_indices = list(detection_indices)
        if not track_indices or not detection_indices:
            return [], track_indices, detection_indices

        costs = np.full(
            (len(track_indices), len(detection_indices)),
            _INVALID_COST,
            dtype=np.float64,
        )
        valid = np.zeros_like(costs, dtype=bool)
        for row, track_index in enumerate(track_indices):
            track = self._tracks[track_index]
            for column, detection_index in enumerate(detection_indices):
                detection = detections[detection_index]
                if track.class_id != detection["class_id"]:
                    continue
                cost, overlap = self._pair_cost(track, detection)
                distance_term = max(cost - (1.0 - overlap), 0.0) / 0.20
                if (
                    overlap >= minimum_iou
                    or distance_term <= self.max_center_distance
                ):
                    costs[row, column] = cost
                    valid[row, column] = True

        matches = []
        matched_tracks = set()
        matched_detections = set()
        for row, column in _linear_assignment(costs):
            if not valid[row, column]:
                continue
            track_index = track_indices[row]
            detection_index = detection_indices[column]
            matches.append((track_index, detection_index))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        return (
            matches,
            [index for index in track_indices if index not in matched_tracks],
            [
                index
                for index in detection_indices
                if index not in matched_detections
            ],
        )

    def _start_track(self, detection: Mapping) -> _Track:
        track = _Track(
            track_id=self._next_id,
            class_id=int(detection["class_id"]),
            class_name=str(detection["class_name"]),
            obb=np.asarray(detection["obb"], dtype=np.float64),
            score=float(detection["score"]),
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def update(
        self,
        detections: Iterable[Mapping],
        embeddings: np.ndarray | None = None,
    ) -> list[dict]:
        """Associate one frame of detections and attach stable track metadata."""

        _ = embeddings
        normalised = [self._normalise_detection(item) for item in detections]
        for track in self._tracks:
            track.predict()

        eligible = [
            index
            for index, detection in enumerate(normalised)
            if detection["score"] >= self.low_confidence_threshold
        ]
        high = [
            index
            for index in eligible
            if normalised[index]["score"] >= self.high_confidence_threshold
        ]
        high_set = set(high)
        low = [index for index in eligible if index not in high_set]

        matches, unmatched_tracks, unmatched_high = self._associate(
            range(len(self._tracks)),
            normalised,
            high,
            self.match_iou_threshold,
        )
        second_matches, unmatched_tracks, unmatched_low = self._associate(
            unmatched_tracks,
            normalised,
            low,
            self.second_match_iou_threshold,
        )
        matches.extend(second_matches)

        detection_to_track: dict[int, _Track] = {}
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            track.update(
                normalised[detection_index],
                self.velocity_momentum,
            )
            detection_to_track[detection_index] = track

        # Every unmatched detection above the configured low threshold can
        # begin an identity.  Higher-confidence detections are considered first
        # so ID allocation is deterministic when scores tie spatially.
        new_indices = unmatched_high + unmatched_low
        new_indices.sort(
            key=lambda index: (-normalised[index]["score"], index)
        )
        for detection_index in new_indices:
            detection_to_track[detection_index] = self._start_track(
                normalised[detection_index]
            )

        self._tracks = [
            track for track in self._tracks if track.missed <= self.max_missed
        ]

        tracked = []
        for index, detection in enumerate(normalised):
            result = dict(detection)
            track = detection_to_track.get(index)
            if track is not None:
                result["track_id"] = track.track_id
                result["track_age"] = track.age
                result["track_hits"] = track.hits
            tracked.append(result)
        return tracked


__all__ = ["RotatedIoUTracker", "extract_deep_embeddings"]
