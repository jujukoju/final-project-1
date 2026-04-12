"""
ml-backend/train.py — Phase 4.4

Training script for the Siamese fingerprint verification model.

Usage:
    python ml-backend/train.py \
        --data  data/SOCOFing/metadata \
        --root  data/processed_png \
        --epochs 30 \
        --img-size 96 \
        --batch-size 64 \
        --lr 1e-3 \
        --margin 1.0 \
        --distance euclidean \
        --checkpoint-dir ml-backend/checkpoints \
        --log-dir ml-backend/logs

Options:
    --dry-run     Run for 1 epoch on 200 pairs to validate the pipeline.
"""

import sys
import csv
import time
import argparse
import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_backend.models.siamese import SiameseNet
from ml_backend.models.losses import ContrastiveLoss
from ml_backend.pair_generator import PairDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def verification_accuracy(distances: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    """Binary accuracy at a fixed distance threshold."""
    preds = (distances < threshold).long()
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, device, threshold, optimizer=None):
    """Run one training or validation epoch. Returns (avg_loss, avg_acc)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            emb1, emb2, dist = model(img1, img2)
            loss = criterion(dist, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc  += verification_accuracy(dist, labels, threshold)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Siamese fingerprint model")
    parser.add_argument("--data",           default="data/SOCOFing/metadata", help="Directory with metadata.csv")
    parser.add_argument("--root",           default="data/processed_png",     help="Root of processed PNG images")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--img-size",       type=int,   default=96,    help="Square image dimension (e.g. 96 → 96×96)")
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--margin",         type=float, default=1.0,   help="Contrastive loss margin")
    parser.add_argument("--distance",       default="euclidean",       choices=["euclidean", "cosine"])
    parser.add_argument("--threshold",      type=float, default=0.5,   help="Decision threshold for accuracy reporting")
    parser.add_argument("--patience",       type=int,   default=5,     help="Early stopping patience (epochs)")
    parser.add_argument("--embedding-dim",  type=int,   default=128)
    parser.add_argument("--checkpoint-dir", default="ml-backend/checkpoints")
    parser.add_argument("--log-dir",        default="ml-backend/logs")
    parser.add_argument("--dry-run",        action="store_true",       help="1 epoch, 200 pairs — pipeline smoke test")
    args = parser.parse_args()

    # ── Setup directories ────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir);  ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(args.log_dir);         log_dir.mkdir(parents=True, exist_ok=True)
    log_csv  = log_dir / "train_log.csv"

    img_size = (args.img_size, args.img_size)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    pair_kwargs = dict(
        metadata_csv_dir=args.data,
        processed_root=args.root,
        img_size=img_size,
        pairs_per_epoch=200 if args.dry_run else None,
    )
    train_ds = PairDataset(**pair_kwargs, split="train", seed=42)
    val_ds   = PairDataset(**pair_kwargs, split="val",   seed=0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    model     = SiameseNet(embedding_dim=args.embedding_dim, distance=args.distance).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    logger.info("Model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    # ── CSV logger ───────────────────────────────────────────────────────────
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    max_epochs = 1 if args.dry_run else args.epochs

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # Reshuffle pairs each epoch for variety
        if epoch > 1:
            train_ds.reshuffle(seed=42 + epoch)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, args.threshold, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device, args.threshold)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        logger.info(
            "Epoch %3d/%d  |  train loss=%.4f acc=%.3f  |  val loss=%.4f acc=%.3f  |  lr=%.2e  |  %.1fs",
            epoch, max_epochs, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed,
        )

        # CSV log
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                                    f"{val_loss:.6f}", f"{val_acc:.4f}", f"{current_lr:.2e}"])

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = ckpt_dir / "best_siamese.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, best_path)
            logger.info("  ✅ Saved best checkpoint → %s  (val_loss=%.4f)", best_path, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience and not args.dry_run:
                logger.info("Early stopping triggered after %d epochs of no improvement.", epoch)
                break

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Logs: %s", log_csv)
    logger.info("Best checkpoint: %s", ckpt_dir / "best_siamese.pt")


if __name__ == "__main__":
    main()
