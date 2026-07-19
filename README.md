# Multi-scale Aerial Object Detection and Segmentation

This project is an end-to-end prototype for **aerial object detection and segmentation** using the **DOTA v1.0** dataset.

The model detects objects in remote sensing images using **Oriented Bounding Boxes (OBB)** and also predicts segmentation masks. It is designed for objects with different scales and rotations, such as planes, ships, vehicles, bridges, and harbors.

## Architecture

You can get this architecture file .PNG at workflow/architecture.png

<image src="workflow/architecture.png" alt="Architecture" width="600"/>

## Project Structure

```text
.
├── src/
│   ├── backbone.py             # LSKNet-style backbone
│   ├── dota_dataset.py         # DOTA parsing and preprocessing
│   ├── fpn_network.py          # Feature pyramid
│   ├── flow.py / gate.py       # Fine/context flows and scale fusion
│   ├── neck.py                 # Dual-flow neck
│   ├── head.py / model.py      # OBB and polygon-mask tasks
│   ├── loss_fn.py              # Multi-task losses
│   ├── metrics.py              # OBB mAP and segmentation metrics
│   └── helper_functions.py     # Targets, training, decoding, inference
├── workflow/architecture.png
├── train.py
├── requirements.txt
└── README.md
```


## Dataset

Prepare the DOTA dataset with the following structure:

```bash
dataset/
└── DOTAv1.0/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

The dataset is not included in this repository because it is too large, available here https://www.kaggle.com/datasets/ntluan3007/dotav1-0/data

## Installation

### Step 1
Clone the repository:

```bash
git clone https://github.com/luannt1010/dota-multiscale-obb-segmentation.git
cd dota-multiscale-obb-segmentation
```

### Step 2
Create and activate a virtual environment:

```bash
python -m venv venv
or
conda create -n yourenv python=3.12
```

For Windows:

```bash
venv\Scripts\activate
or
conda activate yourenv
```

For Linux/macOS:

```bash
source venv/bin/activate
or
conda activate yourenv
```

### Step 3
Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 4
## Training

Run training with default arguments:

```bash
python train.py
```

Or run training with custom arguments:

Source images may have different sizes. The loader resizes each image and its
polygon coordinates to the selected square `--img_size` (default: `1024`).

```bash
python train.py ^
  --train_root dataset/DOTAv1.0/train ^
  --val_root dataset/DOTAv1.0/val ^
  --work_dir runs/sddfb ^
  --epochs 10 ^
  --img_size 1024 ^
  --batch_size 1 ^
  --lr 1e-4 ^
  --weight_decay 1e-4
```

For Linux/macOS, use:

```bash
python train.py \
  --train_root dataset/DOTAv1.0/train \
  --val_root dataset/DOTAv1.0/val \
  --work_dir runs/sddfb \
  --epochs 10 \
  --img_size 1024 \
  --batch_size 1 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

## Training Arguments

| Argument         |                  Default | Description                                  |
| ---------------- | -----------------------: | -------------------------------------------- |
| `--train_root`   | `dataset/DOTAv1.0/train` | Path to the training dataset                 |
| `--val_root`     |   `dataset/DOTAv1.0/val` | Path to the validation dataset               |
| `--work_dir`     |             `runs/sddfb` | Directory for saving outputs and checkpoints |
| `--epochs`       |                     `10` | Number of training epochs                    |
| `--img_size`     |                   `1024` | Resize image and polygons to this square size |
| `--batch_size`   |                      `1` | Training batch size                          |
| `--lr`           |                   `1e-4` | Learning rate                                |
| `--weight_decay` |                   `1e-4` | Weight decay for optimizer                   |
| `--num_workers` |                      `2` | Number of DataLoader workers                 |
| `--seed`        |                     `42` | Random seed for reproducible training        |

## Outputs

During training, the project may generate:

```text
best.pth                 # Best combined mAP50:95 and mIoU
best_map.pth             # Best OBB detection checkpoint
best_seg.pth             # Best polygon-mask checkpoint
last.pth
history.json             # Loss, mAP50, mAP50:95, mIoU and Dice
training_history.png
```

## Evaluation

Validation is computed over the complete validation set on every epoch:

- OBB detection: exact convex-polygon IoU, class-wise rotated NMS, `mAP@50` and COCO-style `mAP@50:95`.
- Segmentation: per-class and mean IoU, Dice, precision, recall and pixel accuracy.
- Objects marked `difficult != 0` are ignored rather than counted as positives or false positives.

The DOTA annotations are oriented polygons, not object silhouette masks. The segmentation task therefore learns filled polygon masks and its scores should not be presented as true silhouette-segmentation quality.
