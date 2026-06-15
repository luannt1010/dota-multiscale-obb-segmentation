# Multi-scale Aerial Object Detection and Segmentation

This project is an end-to-end prototype for **aerial object detection and segmentation** using the **DOTA v1.0** dataset.

The model detects objects in remote sensing images using **Oriented Bounding Boxes (OBB)** and also predicts segmentation masks. It is designed for objects with different scales and rotations, such as planes, ships, vehicles, bridges, and harbors.

## Architecture

You can get this architecture file .PNG at workflow/architecture.png

<image_src="workflow/architecture.png" alt="Architecture" width="200"/>

## Project Structure

```bash
.
├── src/
│   ├── backbone/        # Backbone network
│   ├── datasets/        # DOTA dataset loader and preprocessing
│   ├── head/            # Detection and segmentation heads
│   ├── model_neck/      # Multi-scale feature fusion neck
│   ├── models/          # Full model wrapper
│   ├── utils/           # Losses, target generation, helper functions
│   ├── workflow/        # Workflow figures
│   └── train.py         # Training script
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

```bash
python train.py ^
  --train_root dataset/DOTAv1.0/train ^
  --val_root dataset/DOTAv1.0/val ^
  --work_dir runs/sddfb ^
  --epochs 10 ^
  --img_size 256 ^
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
  --img_size 256 \
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
| `--img_size`     |                    `256` | Input image size                             |
| `--batch_size`   |                      `1` | Training batch size                          |
| `--lr`           |                   `1e-4` | Learning rate                                |
| `--weight_decay` |                   `1e-4` | Weight decay for optimizer                   |

## Outputs

During training, the project may generate:

```bash
best.pth
last.pth
history.json
```


