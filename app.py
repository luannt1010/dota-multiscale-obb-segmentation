from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from PyQt6.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src import config
from src.helper_functions import (
    build_model,
    decode_predictions,
    draw_segment_image,
    inference,
    obb_to_polygon,
    resize_image_and_objects,
    round_up_image_size,
)
from src.deepsort import DeepSortTracker
from src.tracker import RotatedIoUTracker, extract_deep_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_FILTER = "Image files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All files (*)"
VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)"
MODEL_FILTER = "PyTorch checkpoints (*.pth *.pt);;All files (*)"
RESULT_MODES = ("Combined", "OBB", "Segmentation")
TRACKER_ALGORITHMS = ("Rotated IoU", "DeepSORT")


class ModelCache:
    """Keep one checkpoint resident for both image and video inference."""

    _lock = threading.RLock()
    _model = None
    _checkpoint_path = None
    _device = None

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: torch.device):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with cls._lock:
            if (
                cls._model is not None
                and cls._checkpoint_path == checkpoint_path
                and cls._device == str(device)
            ):
                return cls._model

            cls._model = None
            cls._checkpoint_path = None
            cls._device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")

            state_dict = checkpoint.get("model", checkpoint)
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint must contain a model state_dict")

            model = build_model()
            model.load_state_dict(state_dict, strict=True)
            model.to(device).eval()
            cls._model = model
            cls._checkpoint_path = checkpoint_path
            cls._device = str(device)
            return model


class ModelLoadWorker(QObject):
    loaded = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, checkpoint_path: Path, device: torch.device):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

    @pyqtSlot()
    def run(self):
        try:
            self.loaded.emit(
                ModelCache.load(self.checkpoint_path, self.device),
                str(self.checkpoint_path),
            )
        except Exception:
            self.error.emit(traceback.format_exc())


def pil_to_qimage(image: Image.Image) -> QImage:
    array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    height, width, _ = array.shape
    result = QImage(
        array.data,
        width,
        height,
        array.strides[0],
        QImage.Format.Format_RGB888,
    )
    return result.copy()


def draw_predictions(image: Image.Image, detections) -> Image.Image:
    """Draw OBB outlines using the same class colors as segmentation."""

    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    image_width, image_height = rendered.size
    for detection in detections:
        class_id = int(detection["class_id"])
        color = config.CLASS_COLORS[class_id % len(config.CLASS_COLORS)]
        points = [
            (float(x), float(y))
            for x, y in obb_to_polygon(*detection["obb"])
        ]
        draw.line(points + [points[0]], fill=color, width=3)

        track_id = detection.get("track_id")
        identity = f" ID:{track_id}" if track_id is not None else ""
        label = (
            f"{detection['class_name']}{identity} "
            f"{detection['score']:.2f}"
        )
        x = min(max(points[0][0], 0.0), max(image_width - 1.0, 0.0))
        y = min(max(points[0][1], 18.0), max(image_height - 1.0, 18.0))
        box = draw.textbbox((x + 3, y - 17), label)
        draw.rectangle(
            (
                max(box[0] - 2, 0),
                max(box[1] - 1, 0),
                min(box[2] + 2, image_width),
                min(box[3] + 1, image_height),
            ),
            fill=(15, 23, 42),
        )
        draw.text((x + 3, y - 17), label, fill=color)
    return rendered


def infer_pil_frame(
    model,
    source_image: Image.Image,
    device: torch.device,
    image_size: int,
    conf_threshold: float,
    seg_threshold: float,
    nms_iou_threshold: float,
    topk: int,
    tracker: RotatedIoUTracker | DeepSortTracker | None = None,
):
    """In-memory form of the existing non-tiled image inference pipeline."""

    image_size = round_up_image_size(image_size)
    model_image, _ = resize_image_and_objects(
        source_image.convert("RGB"),
        [],
        image_size,
    )
    array = (
        np.array(model_image, dtype=np.float32, copy=True)
        .transpose(2, 0, 1)
        / 255.0
    )
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)

    feature_maps = []
    feature_hook = None
    if tracker is not None and tracker.requires_embeddings:
        if not hasattr(model, "neck"):
            raise TypeError(
                "DeepSORT requires a model with a neck feature module"
            )

        def capture_deep_features(_module, _inputs, neck_output):
            fused = (
                neck_output["fused"]
                if isinstance(neck_output, dict)
                else neck_output
            )
            feature_maps.append(fused.detach())

        feature_hook = model.neck.register_forward_hook(capture_deep_features)

    model.eval()
    try:
        with torch.inference_mode():
            outputs = model(tensor)
    finally:
        if feature_hook is not None:
            feature_hook.remove()

    detections = decode_predictions(
        outputs,
        image_size=image_size,
        stride=config.OUTPUT_STRIDE,
        conf_threshold=conf_threshold,
        topk=topk,
        nms_iou_threshold=nms_iou_threshold,
    )
    if tracker is not None:
        embeddings = None
        if tracker.requires_embeddings:
            if len(feature_maps) != 1:
                raise RuntimeError(
                    "Cannot capture the detector feature map for DeepSORT"
                )
            with torch.inference_mode():
                embeddings = extract_deep_embeddings(
                    feature_maps[0],
                    detections,
                    image_size,
                )
        detections = tracker.update(detections, embeddings=embeddings)
    segmentation = draw_segment_image(
        model_image,
        outputs,
        image_size,
        gt_objects=None,
        threshold=seg_threshold,
    ).convert("RGB")
    images = {
        "OBB": draw_predictions(model_image, detections),
        "Segmentation": segmentation,
    }
    images["Combined"] = draw_predictions(segmentation, detections)
    return images, detections


class ImageInferenceWorker(QObject):
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        model,
        input_path: Path,
        output_dir: Path,
        device: torch.device,
        settings: dict,
    ):
        super().__init__()
        self.model = model
        self.input_path = input_path
        self.output_dir = output_dir
        self.device = device
        self.settings = settings

    @pyqtSlot()
    def run(self):
        try:
            self.status.emit(f"Running image inference on {self.device}...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = f"{self.input_path.stem}_inferred_{run_id}"
            missing_label = self.output_dir / f".{stem}.no_label.txt"

            result = inference(
                self.model,
                self.input_path,
                label_path=missing_label,
                image_size=self.settings["image_size"],
                conf_threshold=self.settings["conf_threshold"],
                seg_threshold=self.settings["seg_threshold"],
                nms_iou_threshold=self.settings["nms_iou_threshold"],
                topk=self.settings["topk"],
                device=self.device,
            )
            with Image.open(self.input_path) as loaded:
                source = loaded.convert("RGB")
            model_image, _ = resize_image_and_objects(
                source,
                [],
                self.settings["image_size"],
            )
            detections = result["detections"]
            images = {
                "OBB": draw_predictions(model_image, detections),
                "Segmentation": result["segment_image"].convert("RGB"),
            }
            images["Combined"] = draw_predictions(
                images["Segmentation"],
                detections,
            )

            paths = {}
            for mode, image in images.items():
                path = self.output_dir / f"{stem}_{mode.lower()}.png"
                image.save(path)
                paths[mode] = str(path)

            self.finished.emit(
                {
                    "images": {
                        mode: pil_to_qimage(image)
                        for mode, image in images.items()
                    },
                    "detections": len(detections),
                    "paths": paths,
                }
            )
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class VideoInferenceWorker(QObject):
    frame_ready = pyqtSignal(object, object, int)
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str, bool, int)
    error = pyqtSignal(str)

    def __init__(
        self,
        model,
        input_path: Path,
        output_path: Path,
        device: torch.device,
        settings: dict,
        result_mode: str,
        tracker_algorithm: str = TRACKER_ALGORITHMS[0],
    ):
        super().__init__()
        self.model = model
        self.input_path = input_path
        self.output_path = output_path
        self.device = device
        self.settings = settings
        self.result_mode = result_mode
        self.tracker_algorithm = tracker_algorithm
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    @pyqtSlot()
    def run(self):
        capture = None
        writer = None
        frame_count = 0
        error_details = None
        try:
            capture = cv2.VideoCapture(str(self.input_path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {self.input_path}")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            image_size = self.settings["image_size"]
            confidence = self.settings["conf_threshold"]
            tracker_settings = {
                "high_confidence_threshold": min(
                    1.0,
                    max(0.30, confidence),
                ),
                "low_confidence_threshold": confidence,
            }
            track_lifetime = max(3, int(round(fps)))
            if self.tracker_algorithm == "Rotated IoU":
                tracker = RotatedIoUTracker(
                    **tracker_settings,
                    match_iou_threshold=0.10,
                    second_match_iou_threshold=0.05,
                    max_center_distance=1.5,
                    max_missed=track_lifetime,
                )
            elif self.tracker_algorithm == "DeepSORT":
                tracker = DeepSortTracker(
                    **tracker_settings,
                    max_age=track_lifetime,
                    n_init=2,
                    max_cosine_distance=0.45,
                    appearance_weight=0.70,
                    minimum_iou=0.02,
                    max_center_distance=1.5,
                    mahalanobis_threshold=25.0,
                    feature_budget=30,
                )
            else:
                raise ValueError(
                    f"Unknown tracker algorithm: {self.tracker_algorithm}"
                )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (image_size, image_size),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    "Cannot create output MP4. Check OpenCV codec support."
                )

            self.status.emit(
                f"Running {self.tracker_algorithm} video inference on "
                f"{self.device}; output {image_size}x{image_size}"
            )
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                source = Image.fromarray(rgb)
                results, _ = infer_pil_frame(
                    self.model,
                    source,
                    self.device,
                    **self.settings,
                    tracker=tracker,
                )
                rendered = results[self.result_mode]
                rendered_rgb = np.asarray(rendered, dtype=np.uint8)
                writer.write(
                    cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                )

                frame_count += 1
                self.frame_ready.emit(
                    pil_to_qimage(source),
                    pil_to_qimage(rendered),
                    frame_count,
                )
                self.progress.emit(frame_count, total_frames)

            if frame_count == 0 and not self.stop_event.is_set():
                raise RuntimeError("The video contains no decodable frames")
        except Exception:
            error_details = traceback.format_exc()
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if error_details is not None:
            self.error.emit(error_details)
        else:
            self.finished.emit(
                str(self.output_path),
                self.stop_event.is_set(),
                frame_count,
            )


class MediaView(QFrame):
    def __init__(self, placeholder: str):
        super().__init__()
        self._image = None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(460, 320)
        self.label = QLabel(placeholder)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(
            "background:#111827; color:#94a3b8; border:1px solid #334155;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def set_image(self, image: QImage):
        self._image = image
        self._refresh()

    def set_placeholder(self, text: str):
        self._image = None
        self.label.setPixmap(QPixmap())
        self.label.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._image is None or self._image.isNull():
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setText("")
        self.label.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(
            args.device
            if args.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.image_path = None
        self.video_path = None
        self.image_results = {}
        self.model_thread = None
        self.model_worker = None
        self.image_thread = None
        self.image_worker = None
        self.video_thread = None
        self.video_worker = None

        self.setWindowTitle("DOTA Image and Video Inference")
        self.resize(1480, 900)
        self._build_ui()
        self._apply_style()

        checkpoint = resolve_checkpoint(args.checkpoint)
        if checkpoint is not None:
            QTimer.singleShot(0, lambda: self.load_model(checkpoint))
        else:
            self.model_label.setText("Model: not loaded")
            self.statusBar().showMessage(
                "No checkpoint found. Choose a .pth file."
            )

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        controls = QHBoxLayout()
        self.choose_model_button = QPushButton("Choose model")
        self.image_size_combo = QComboBox()
        image_sizes = ["512", "768", "1024"]
        requested_size = str(self.args.image_size)
        if requested_size not in image_sizes:
            image_sizes.append(requested_size)
        self.image_size_combo.addItems(image_sizes)
        self.image_size_combo.setCurrentText(requested_size)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.001, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(self.args.conf_threshold)
        self.seg_spin = QDoubleSpinBox()
        self.seg_spin.setRange(0.001, 1.0)
        self.seg_spin.setDecimals(3)
        self.seg_spin.setSingleStep(0.05)
        self.seg_spin.setValue(self.args.seg_threshold)
        controls.addWidget(self.choose_model_button)
        controls.addSpacing(18)
        controls.addWidget(QLabel("Input size"))
        controls.addWidget(self.image_size_combo)
        controls.addWidget(QLabel("Confidence"))
        controls.addWidget(self.conf_spin)
        controls.addWidget(QLabel("Seg threshold"))
        controls.addWidget(self.seg_spin)
        controls.addStretch(1)
        root.addLayout(controls)

        self.model_label = QLabel("Model: loading...")
        self.model_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.model_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_image_tab(), "Image")
        self.tabs.addTab(self._build_video_tab(), "Video")
        root.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.choose_model_button.clicked.connect(self.choose_model)
        self.image_upload_button.clicked.connect(self.choose_image)
        self.image_run_button.clicked.connect(self.start_image_inference)
        self.image_result_combo.currentTextChanged.connect(
            self.show_selected_image_result
        )
        self.video_upload_button.clicked.connect(self.choose_video)
        self.video_run_button.clicked.connect(self.start_video_inference)
        self.video_stop_button.clicked.connect(self.stop_video_inference)

    def _build_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.image_upload_button = QPushButton("Upload image")
        self.image_run_button = QPushButton("Run inference")
        self.image_result_combo = QComboBox()
        self.image_result_combo.addItems(RESULT_MODES)
        self.image_run_button.setEnabled(False)
        self.image_result_combo.setEnabled(False)
        row.addWidget(self.image_upload_button)
        row.addWidget(self.image_run_button)
        row.addWidget(QLabel("Display"))
        row.addWidget(self.image_result_combo)
        row.addStretch(1)
        layout.addLayout(row)
        self.image_file_label = QLabel("Image: not selected")
        layout.addWidget(self.image_file_label)

        panels = QHBoxLayout()
        source_group = QGroupBox("Input image")
        source_layout = QVBoxLayout(source_group)
        self.image_source_view = MediaView("Upload an image")
        source_layout.addWidget(self.image_source_view)
        result_group = QGroupBox("Image result")
        result_layout = QVBoxLayout(result_group)
        self.image_result_view = MediaView("The result will appear here")
        result_layout.addWidget(self.image_result_view)
        panels.addWidget(source_group, 1)
        panels.addWidget(result_group, 1)
        layout.addLayout(panels, 1)

        row = QHBoxLayout()
        self.image_progress = QProgressBar()
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        self.image_detection_label = QLabel("Detections: 0")
        row.addWidget(self.image_progress, 1)
        row.addWidget(self.image_detection_label)
        layout.addLayout(row)
        self.image_output_label = QLabel("Output: not created")
        layout.addWidget(self.image_output_label)
        return tab

    def _build_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.video_upload_button = QPushButton("Upload video")
        self.video_run_button = QPushButton("Run inference")
        self.video_stop_button = QPushButton("Stop")
        self.video_result_combo = QComboBox()
        self.video_result_combo.addItems(RESULT_MODES)
        self.video_tracker_combo = QComboBox()
        self.video_tracker_combo.addItems(TRACKER_ALGORITHMS)
        self.video_run_button.setEnabled(False)
        self.video_stop_button.setEnabled(False)
        row.addWidget(self.video_upload_button)
        row.addWidget(self.video_run_button)
        row.addWidget(self.video_stop_button)
        row.addWidget(QLabel("Output mode"))
        row.addWidget(self.video_result_combo)
        row.addWidget(QLabel("Tracker"))
        row.addWidget(self.video_tracker_combo)
        row.addStretch(1)
        layout.addLayout(row)
        self.video_file_label = QLabel("Video: not selected")
        layout.addWidget(self.video_file_label)

        panels = QHBoxLayout()
        source_group = QGroupBox("Source video")
        source_layout = QVBoxLayout(source_group)
        self.video_source_view = MediaView("Upload a video")
        source_layout.addWidget(self.video_source_view)
        result_group = QGroupBox("Video result")
        result_layout = QVBoxLayout(result_group)
        self.video_result_view = MediaView(
            "Processed frames will appear here"
        )
        result_layout.addWidget(self.video_result_view)
        panels.addWidget(source_group, 1)
        panels.addWidget(result_group, 1)
        layout.addLayout(panels, 1)

        row = QHBoxLayout()
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_frame_label = QLabel("Frame 0")
        row.addWidget(self.video_progress, 1)
        row.addWidget(self.video_frame_label)
        layout.addLayout(row)
        self.video_output_label = QLabel("Output: not created")
        layout.addWidget(self.video_output_label)
        return tab

    def _apply_style(self):
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#0f172a; color:#e2e8f0; }
            QTabWidget::pane { border:1px solid #334155; border-radius:7px; }
            QTabBar::tab {
                background:#1e293b; color:#94a3b8; padding:9px 24px;
                border:1px solid #334155;
            }
            QTabBar::tab:selected { background:#2563eb; color:white; }
            QGroupBox {
                border:1px solid #334155; border-radius:8px; margin-top:10px;
                padding-top:10px; font-weight:600;
            }
            QPushButton {
                background:#2563eb; border:none; border-radius:6px;
                padding:8px 14px; font-weight:600;
            }
            QPushButton:hover { background:#3b82f6; }
            QPushButton:disabled { background:#334155; color:#64748b; }
            QComboBox, QDoubleSpinBox {
                background:#1e293b; border:1px solid #475569;
                border-radius:5px; padding:5px;
            }
            QProgressBar {
                border:1px solid #475569; border-radius:5px;
                background:#1e293b; text-align:center;
            }
            QProgressBar::chunk { background:#22c55e; border-radius:4px; }
            """
        )

    def is_busy(self):
        return any(
            thread is not None
            for thread in (
                self.model_thread,
                self.image_thread,
                self.video_thread,
            )
        )

    def settings(self):
        return {
            "image_size": int(self.image_size_combo.currentText()),
            "conf_threshold": self.conf_spin.value(),
            "seg_threshold": self.seg_spin.value(),
            "nms_iou_threshold": self.args.nms_iou_threshold,
            "topk": self.args.topk,
        }

    def output_dir(self):
        path = Path(self.args.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def update_controls(self):
        busy = self.is_busy()
        self.choose_model_button.setEnabled(not busy)
        self.image_upload_button.setEnabled(not busy)
        self.video_upload_button.setEnabled(not busy)
        self.image_size_combo.setEnabled(not busy)
        self.conf_spin.setEnabled(not busy)
        self.seg_spin.setEnabled(not busy)
        self.image_result_combo.setEnabled(
            bool(self.image_results) and not busy
        )
        self.video_result_combo.setEnabled(not busy)
        self.video_tracker_combo.setEnabled(not busy)
        self.image_run_button.setEnabled(
            self.model is not None
            and self.image_path is not None
            and not busy
        )
        self.video_run_button.setEnabled(
            self.model is not None            and self.video_path is not None
            and not busy
        )
        self.video_stop_button.setEnabled(self.video_thread is not None)

    def choose_model(self):
        initial = PROJECT_ROOT / "res"
        if not initial.is_dir():
            initial = PROJECT_ROOT
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose model checkpoint",
            str(initial),
            MODEL_FILTER,
        )
        if path:
            self.load_model(Path(path))

    def load_model(self, checkpoint_path: Path):
        if self.is_busy():
            return
        self.model = None
        self.model_label.setText(f"Model: loading {checkpoint_path.name}...")
        self.statusBar().showMessage(f"Loading model on {self.device}...")
        thread = QThread(self)
        worker = ModelLoadWorker(checkpoint_path, self.device)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._model_loaded)
        worker.error.connect(self._model_error)
        worker.loaded.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._model_thread_finished)
        self.model_thread = thread
        self.model_worker = worker
        self.update_controls()
        thread.start()

    @pyqtSlot(object, str)
    def _model_loaded(self, model, path):
        self.model = model
        self.model_label.setText(f"Model: {path} ({self.device}, cached)")
        self.statusBar().showMessage("Model loaded and cached.", 5000)

    @pyqtSlot(str)
    def _model_error(self, details):
        self.model = None
        self.model_label.setText("Model: load failed")
        QMessageBox.critical(self, "Model load failed", details)

    def _model_thread_finished(self):
        self.model_thread = None
        self.model_worker = None
        self.update_controls()

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input image",
            str(PROJECT_ROOT),
            IMAGE_FILTER,
        )
        if not path:
            return
        image_path = Path(path)
        try:
            with Image.open(image_path) as image:
                preview = image.convert("RGB")
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Image error",
                f"Cannot open image:\n{exc}",
            )
            return

        self.image_path = image_path
        self.image_results = {}
        self.image_source_view.set_image(pil_to_qimage(preview))
        self.image_result_view.set_placeholder("Ready to run inference")
        self.image_file_label.setText(f"Image: {image_path}")
        self.image_output_label.setText("Output: not created")
        self.image_detection_label.setText("Detections: 0")
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        self.update_controls()

    def start_image_inference(self):
        if self.model is None or self.image_path is None or self.is_busy():
            return
        thread = QThread(self)
        worker = ImageInferenceWorker(
            self.model,
            self.image_path,
            self.output_dir(),
            self.device,
            self.settings(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(self.statusBar().showMessage)
        worker.finished.connect(self._image_finished)
        worker.error.connect(self._image_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._image_thread_finished)
        self.image_thread = thread
        self.image_worker = worker
        self.image_progress.setRange(0, 0)
        self.image_result_view.set_placeholder("Running inference...")
        self.update_controls()
        thread.start()

    @pyqtSlot(object)
    def _image_finished(self, payload):
        self.image_results = payload["images"]
        self.image_result_combo.setCurrentText("Combined")
        self.show_selected_image_result("Combined")
        self.image_detection_label.setText(
            f"Detections: {payload['detections']}"
        )
        self.image_output_label.setText(
            f"Output: {payload['paths']['Combined']}"
        )
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(100)

    @pyqtSlot(str)
    def _image_error(self, details):
        self.image_results = {}
        self.image_result_view.set_placeholder("Inference failed")
        self.image_output_label.setText("Output: failed")
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        QMessageBox.critical(self, "Image inference failed", details)

    def _image_thread_finished(self):
        self.image_thread = None
        self.image_worker = None
        self.update_controls()

    def show_selected_image_result(self, name: str):
        image = self.image_results.get(name)
        if image is not None:
            self.image_result_view.set_image(image)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input video",
            str(PROJECT_ROOT),
            VIDEO_FILTER,
        )
        if not path:
            return
        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            QMessageBox.warning(
                self,
                "Video error",
                "Cannot decode this video.",
            )
            return

        self.video_path = video_path
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_source_view.set_image(
            pil_to_qimage(Image.fromarray(rgb))
        )
        self.video_result_view.set_placeholder(
            "Ready to run video inference"
        )
        self.video_file_label.setText(f"Video: {video_path}")
        self.video_output_label.setText("Output: not created")
        self.video_frame_label.setText("Frame 0")
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.update_controls()

    def start_video_inference(self):
        if self.model is None or self.video_path is None or self.is_busy():
            return
        mode = self.video_result_combo.currentText()
        tracker_algorithm = self.video_tracker_combo.currentText()
        tracker_slug = tracker_algorithm.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.output_dir() / (
            f"{self.video_path.stem}_inferred_{timestamp}_"
            f"{tracker_slug}_{mode.lower()}.mp4"
        )
        thread = QThread(self)
        worker = VideoInferenceWorker(
            self.model,
            self.video_path,
            output_path,
            self.device,
            self.settings(),
            mode,
            tracker_algorithm,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frame_ready.connect(self._video_frame_ready)
        worker.progress.connect(self._video_progress)
        worker.status.connect(self.statusBar().showMessage)
        worker.finished.connect(self._video_finished)
        worker.error.connect(self._video_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._video_thread_finished)
        self.video_thread = thread
        self.video_worker = worker
        self.video_progress.setRange(0, 0)
        self.video_result_view.set_placeholder("Processing video...")
        self.video_output_label.setText(f"Output: {output_path}")
        self.update_controls()
        thread.start()

    def stop_video_inference(self):
        if self.video_worker is not None:
            self.video_worker.stop()
            self.video_stop_button.setEnabled(False)
            self.statusBar().showMessage(
                "Stopping after the current frame..."
            )

    @pyqtSlot(object, object, int)
    def _video_frame_ready(self, source, result, frame_index):
        self.video_source_view.set_image(source)
        self.video_result_view.set_image(result)
        self.video_frame_label.setText(f"Frame {frame_index}")

    @pyqtSlot(int, int)
    def _video_progress(self, current, total):
        if total > 0:
            self.video_progress.setRange(0, total)
            self.video_progress.setValue(min(current, total))
        else:
            self.video_progress.setRange(0, 0)

    @pyqtSlot(str, bool, int)
    def _video_finished(self, path, cancelled, frame_count):
        prefix = "Partial output" if cancelled else "Output"
        self.video_output_label.setText(f"{prefix}: {path}")
        state = "stopped" if cancelled else "finished"
        self.statusBar().showMessage(
            f"Video {state}: {frame_count} frames.",
            10000,
        )
        if not cancelled and self.video_progress.maximum() > 0:
            self.video_progress.setValue(self.video_progress.maximum())

    @pyqtSlot(str)
    def _video_error(self, details):
        self.video_result_view.set_placeholder("Video inference failed")
        self.video_output_label.setText("Output: failed")
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        QMessageBox.critical(self, "Video inference failed", details)

    def _video_thread_finished(self):
        self.video_thread = None
        self.video_worker = None
        if self.video_progress.maximum() == 0:
            self.video_progress.setRange(0, 100)
            self.video_progress.setValue(0)
        self.update_controls()

    def closeEvent(self, event: QCloseEvent):
        if self.video_thread is not None:
            if self.video_worker is not None:
                self.video_worker.stop()
            event.ignore()
            self.statusBar().showMessage(
                "Stopping video before closing..."
            )
            QTimer.singleShot(250, self.close)
            return
        if self.is_busy():
            event.ignore()
            self.statusBar().showMessage(
                "Waiting for the current task to finish..."
            )
            QTimer.singleShot(250, self.close)
            return
        event.accept()


def resolve_checkpoint(requested: str | None):
    if requested:
        return Path(requested).expanduser().resolve()
    environment_path = os.environ.get("DOTA_MODEL_PATH")
    if environment_path:
        path = Path(environment_path).expanduser().resolve()
        if path.is_file():
            return path
    candidates = (
        PROJECT_ROOT / "res" / "best.pth",
        PROJECT_ROOT / "res" / "best_map.pth",
        PROJECT_ROOT / "res" / "last.pth",
        PROJECT_ROOT / "runs" / "sddfb" / "best.pth",
        PROJECT_ROOT / "runs" / "sddfb" / "best_map.pth",
        PROJECT_ROOT / "runs" / "sddfb" / "last.pth",
    )
    return next((path for path in candidates if path.is_file()), None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyQt image/video inference app"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--conf-threshold", type=float, default=0.15)
    parser.add_argument("--seg-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--output-dir", default="runs/app")
    args, qt_args = parser.parse_known_args()
    requested_size = args.image_size
    try:
        args.image_size = round_up_image_size(requested_size)
    except ValueError as error:
        parser.error(str(error))
    if args.image_size != requested_size:
        print(
            f"Adjusted --image-size from {requested_size} to "
            f"{args.image_size} (next multiple of 32)."
        )
    if not 0.0 <= args.conf_threshold <= 1.0:
        parser.error("--conf-threshold must be in [0, 1]")
    if not 0.0 <= args.seg_threshold <= 1.0:
        parser.error("--seg-threshold must be in [0, 1]")
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        parser.error("--nms-iou-threshold must be in [0, 1]")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    return args, qt_args


def main():
    args, qt_args = parse_args()
    application = QApplication([sys.argv[0], *qt_args])
    application.setApplicationName("DOTA Image and Video Inference")
    window = MainWindow(args)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
