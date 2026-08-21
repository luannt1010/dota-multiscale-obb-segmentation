from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping
import numpy as np
from .metrics import obb_to_polygons, polygon_iou
from .tracker import _INVALID_COST, _angle_delta, _linear_assignment


def _normalise_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("appearance embeddings must contain finite values")
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


@dataclass
class _DeepTrack:
    track_id: int
    class_id: int
    class_name: str
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    feature_budget: int
    features: list[np.ndarray] = field(default_factory=list)
    age: int = 1
    hits: int = 1
    missed: int = 0

    @classmethod
    def create(
        cls,
        track_id: int,
        detection: Mapping,
        embedding: np.ndarray,
        feature_budget: int,
    ) -> "_DeepTrack":
        measurement = np.asarray(detection["obb"], dtype=np.float64)
        mean = np.concatenate((measurement, np.zeros(5, dtype=np.float64)))
        scale = max(float(measurement[2]), float(measurement[3]), 2.0)
        standard_deviation = np.asarray(
            (
                0.20 * scale,
                0.20 * scale,
                0.15 * scale,
                0.15 * scale,
                0.15,
                0.50 * scale,
                0.50 * scale,
                0.25 * scale,
                0.25 * scale,
                0.10,
            ),
            dtype=np.float64,
        )
        track = cls(
            track_id=track_id,
            class_id=int(detection["class_id"]),
            class_name=str(detection["class_name"]),
            mean=mean,
            covariance=np.diag(np.square(standard_deviation)),
            score=float(detection["score"]),
            feature_budget=feature_budget,
        )
        track.add_feature(embedding)
        return track

    @property
    def obb(self) -> np.ndarray:
        return self.mean[:5]

    def _motion_matrix(self) -> np.ndarray:
        matrix = np.eye(10, dtype=np.float64)
        matrix[:5, 5:] = np.eye(5, dtype=np.float64)
        return matrix

    def _measurement_noise(self) -> np.ndarray:
        scale = max(float(self.mean[2]), float(self.mean[3]), 2.0)
        standard_deviation = np.asarray(
            (0.08 * scale, 0.08 * scale, 0.10 * scale, 0.10 * scale, 0.08),
            dtype=np.float64,
        )
        return np.diag(np.square(standard_deviation))

    def predict(self):
        motion = self._motion_matrix()
        scale = max(float(self.mean[2]), float(self.mean[3]), 2.0)
        process_deviation = np.asarray(
            (
                0.05 * scale,
                0.05 * scale,
                0.04 * scale,
                0.04 * scale,
                0.025,
                0.02 * scale,
                0.02 * scale,
                0.015 * scale,
                0.015 * scale,
                0.01,
            ),
            dtype=np.float64,
        )
        self.mean = motion @ self.mean
        self.covariance = (
            motion @ self.covariance @ motion.T
            + np.diag(np.square(process_deviation))
        )
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)
        self.age += 1
        self.missed += 1

    def _aligned_measurement(self, obb: Iterable[float]) -> np.ndarray:
        measurement = np.asarray(obb, dtype=np.float64).copy()
        measurement[4] = self.mean[4] + _angle_delta(
            float(measurement[4]),
            float(self.mean[4]),
        )
        return measurement

    def projected_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        measurement_matrix = np.zeros((5, 10), dtype=np.float64)
        measurement_matrix[:, :5] = np.eye(5, dtype=np.float64)
        projected_mean = measurement_matrix @ self.mean
        projected_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + self._measurement_noise()
        )
        return projected_mean, projected_covariance

    def gating_distance(self, obb: Iterable[float]) -> float:
        measurement = self._aligned_measurement(obb)
        projected_mean, projected_covariance = self.projected_distribution()
        innovation = measurement - projected_mean
        try:
            solved = np.linalg.solve(projected_covariance, innovation)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(projected_covariance) @ innovation
        return float(innovation @ solved)

    def update(self, detection: Mapping, embedding: np.ndarray):
        measurement = self._aligned_measurement(detection["obb"])
        measurement_matrix = np.zeros((5, 10), dtype=np.float64)
        measurement_matrix[:, :5] = np.eye(5, dtype=np.float64)
        measurement_noise = self._measurement_noise()
        projected_mean = measurement_matrix @ self.mean
        projected_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_noise
        )
        innovation = measurement - projected_mean
        try:
            kalman_gain = np.linalg.solve(
                projected_covariance,
                measurement_matrix @ self.covariance,
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = (
                self.covariance
                @ measurement_matrix.T
                @ np.linalg.pinv(projected_covariance)
            )

        self.mean = self.mean + kalman_gain @ innovation
        identity = np.eye(10, dtype=np.float64)
        residual_projection = identity - kalman_gain @ measurement_matrix
        # Joseph form keeps covariance symmetric and positive semi-definite.
        self.covariance = (
            residual_projection @ self.covariance @ residual_projection.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)
        self.class_name = str(detection["class_name"])
        self.score = float(detection["score"])
        self.hits += 1
        self.missed = 0
        self.add_feature(embedding)

    def add_feature(self, embedding: np.ndarray):
        self.features.append(_normalise_embedding(embedding))
        if len(self.features) > self.feature_budget:
            del self.features[:-self.feature_budget]

    def appearance_distance(self, embedding: np.ndarray) -> float:
        candidate = _normalise_embedding(embedding)
        if not self.features or not np.any(candidate):
            return 1.0
        gallery = np.stack(self.features)
        similarity = np.clip(gallery @ candidate, -1.0, 1.0)
        return float(1.0 - np.max(similarity))


class DeepSortTracker:
    """Deep appearance + Kalman tracker adapted to rotated detections."""

    requires_embeddings = True

    def __init__(
        self,
        *,
        high_confidence_threshold: float = 0.30,
        low_confidence_threshold: float = 0.10,
        max_age: int = 30,
        n_init: int = 2,
        max_cosine_distance: float = 0.45,
        appearance_weight: float = 0.70,
        minimum_iou: float = 0.02,
        max_center_distance: float = 1.5,
        mahalanobis_threshold: float = 25.0,
        feature_budget: int = 30,
    ):
        for name, value in (
            ("high_confidence_threshold", high_confidence_threshold),
            ("low_confidence_threshold", low_confidence_threshold),
            ("max_cosine_distance", max_cosine_distance),
            ("appearance_weight", appearance_weight),
            ("minimum_iou", minimum_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if low_confidence_threshold > high_confidence_threshold:
            raise ValueError(
                "low_confidence_threshold cannot exceed high_confidence_threshold"
            )
        if max_age < 0:
            raise ValueError("max_age cannot be negative")
        if n_init <= 0:
            raise ValueError("n_init must be positive")
        if max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive")
        if mahalanobis_threshold <= 0:
            raise ValueError("mahalanobis_threshold must be positive")
        if feature_budget <= 0:
            raise ValueError("feature_budget must be positive")

        self.high_confidence_threshold = float(high_confidence_threshold)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.max_cosine_distance = float(max_cosine_distance)
        self.appearance_weight = float(appearance_weight)
        self.minimum_iou = float(minimum_iou)
        self.max_center_distance = float(max_center_distance)
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.feature_budget = int(feature_budget)
        self._tracks: list[_DeepTrack] = []
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def active_tracks(self) -> tuple[dict, ...]:
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
                "confirmed": track.hits >= self.n_init,
            }
            for track in self._tracks
        )

    @staticmethod
    def _normalise_detection(detection: Mapping) -> dict:
        required = ("score", "class_id", "class_name", "obb")
        missing = [key for key in required if key not in detection]
        if missing:
            raise KeyError(f"Detection is missing keys: {', '.join(missing)}")
        result = dict(detection)
        obb = np.asarray(detection["obb"], dtype=np.float64).reshape(-1)
        if obb.shape != (5,) or not np.all(np.isfinite(obb)):
            raise ValueError("detection['obb'] must contain five finite values")
        if obb[2] <= 0 or obb[3] <= 0:
            raise ValueError("OBB width and height must be positive")
        score = float(detection["score"])
        if not math.isfinite(score):
            raise ValueError("detection score must be finite")
        result["obb"] = tuple(float(value) for value in obb)
        result["score"] = score
        result["class_id"] = int(detection["class_id"])
        result["class_name"] = str(detection["class_name"])
        return result

    @staticmethod
    def _geometry(track: _DeepTrack, detection: Mapping) -> tuple[float, float]:
        detection_obb = np.asarray(detection["obb"], dtype=np.float64)
        polygons = obb_to_polygons(np.stack((track.obb, detection_obb)))
        overlap = polygon_iou(polygons[0], polygons[1])
        center_distance = float(
            np.linalg.norm(track.obb[:2] - detection_obb[:2])
        )
        scale = max(
            0.5
            * (
                math.hypot(float(track.obb[2]), float(track.obb[3]))
                + math.hypot(float(detection_obb[2]), float(detection_obb[3]))
            ),
            1.0,
        )
        return overlap, center_distance / scale

    def _associate(
        self,
        track_indices: Iterable[int],
        detections: list[dict],
        embeddings: np.ndarray,
        detection_indices: Iterable[int],
        *,
        relaxed_appearance: bool,
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
                overlap, center_distance = self._geometry(track, detection)
                gating_distance = track.gating_distance(detection["obb"])
                appearance_distance = track.appearance_distance(
                    embeddings[detection_index]
                )
                appearance_ok = (
                    appearance_distance <= self.max_cosine_distance
                    or relaxed_appearance
                    or overlap >= 0.50
                )
                if (
                    gating_distance <= self.mahalanobis_threshold
                    and center_distance <= self.max_center_distance
                    and appearance_ok
                    and (overlap >= self.minimum_iou or center_distance <= 0.75)
                ):
                    appearance_cost = min(
                        appearance_distance / max(self.max_cosine_distance, 1e-6),
                        1.0,
                    )
                    motion_cost = min(
                        gating_distance / self.mahalanobis_threshold,
                        1.0,
                    )
                    geometry_cost = 0.75 * (1.0 - overlap) + 0.25 * motion_cost
                    costs[row, column] = (
                        self.appearance_weight * appearance_cost
                        + (1.0 - self.appearance_weight) * geometry_cost
                    )
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

    def _matching_cascade(
        self,
        track_indices: Iterable[int],
        detections: list[dict],
        embeddings: np.ndarray,
        detection_indices: Iterable[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        remaining_tracks = set(track_indices)
        remaining_detections = list(detection_indices)
        matches = []
        recencies = sorted(
            {self._tracks[index].missed for index in remaining_tracks}
        )
        for recency in recencies:
            if not remaining_detections:
                break
            level_tracks = sorted(
                index
                for index in remaining_tracks
                if self._tracks[index].missed == recency
            )
            level_matches, _, remaining_detections = self._associate(
                level_tracks,
                detections,
                embeddings,
                remaining_detections,
                relaxed_appearance=False,
            )
            matches.extend(level_matches)
            remaining_tracks.difference_update(
                track_index for track_index, _ in level_matches
            )
        return matches, sorted(remaining_tracks), remaining_detections

    def _start_track(
        self,
        detection: Mapping,
        embedding: np.ndarray,
    ) -> _DeepTrack:
        track = _DeepTrack.create(
            self._next_id,
            detection,
            embedding,
            self.feature_budget,
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def update(
        self,
        detections: Iterable[Mapping],
        embeddings: np.ndarray | None = None,
    ) -> list[dict]:
        normalised = [self._normalise_detection(item) for item in detections]
        if embeddings is None:
            raise ValueError("DeepSortTracker requires appearance embeddings")
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(normalised):
            raise ValueError(
                "embeddings must have shape [number_of_detections, dimensions]"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("embeddings must contain finite values")

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

        matches, unmatched_tracks, unmatched_high = self._matching_cascade(
            range(len(self._tracks)),
            normalised,
            embeddings,
            high,
        )
        second_matches, unmatched_tracks, unmatched_low = self._associate(
            unmatched_tracks,
            normalised,
            embeddings,
            low,
            relaxed_appearance=True,
        )
        matches.extend(second_matches)

        detection_to_track: dict[int, _DeepTrack] = {}
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            track.update(normalised[detection_index], embeddings[detection_index])
            detection_to_track[detection_index] = track

        new_indices = unmatched_high + unmatched_low
        new_indices.sort(
            key=lambda index: (-normalised[index]["score"], index)
        )
        for detection_index in new_indices:
            detection_to_track[detection_index] = self._start_track(
                normalised[detection_index],
                embeddings[detection_index],
            )

        self._tracks = [
            track
            for track in self._tracks
            if track.missed <= self.max_age
            and not (track.hits < self.n_init and track.missed > 1)
        ]

        tracked = []
        for index, detection in enumerate(normalised):
            result = dict(detection)
            track = detection_to_track.get(index)
            if track is not None:
                result["obb"] = tuple(float(value) for value in track.obb)
                result["track_id"] = track.track_id
                result["track_age"] = track.age
                result["track_hits"] = track.hits
                result["track_confirmed"] = track.hits >= self.n_init
            tracked.append(result)
        return tracked


__all__ = ["DeepSortTracker"]
