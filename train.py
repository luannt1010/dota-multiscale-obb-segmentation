import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from src import helper_functions
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default=r"./dataset/DOTAv1.0/train")
    parser.add_argument("--val_root", default=r"./dataset/DOTAv1.0/val")
    parser.add_argument("--work_dir", default="runs/sddfb")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    helper_functions.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = helper_functions.create_dataloaders(
        train_root=args.train_root,
        val_root=args.val_root,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = helper_functions.build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("device:", device)
    print("train samples:", len(train_loader.dataset))
    print("val samples:", len(val_loader.dataset))
    print("work dir:", work_dir)

    num_epochs = args.epochs

    warmup_epochs = 3
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=args.lr * 0.05)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    history = helper_functions.train(model, train_loader, val_loader, optimizer, scheduler, work_dir, num_epochs)
    helper_functions.plot_history(history, work_dir / "training_history.png")

if __name__ == "__main__":
    main()
