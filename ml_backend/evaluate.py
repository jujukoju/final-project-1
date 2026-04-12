"""
ml-backend/evaluate.py — Phase 4.5

Loads a trained Siamese checkpoint and computes:
    - Accuracy at chosen threshold
    - FAR (False Accept Rate) and FRR (False Reject Rate)
    - EER (Equal Error Rate) and the operating threshold at EER
    - ROC curve (saved as PNG)
    - DET curve (saved as PNG)

Usage:
    python ml-backend/evaluate.py \
        --checkpoint ml-backend/checkpoints/best_siamese.pt \
        --data data/SOCOFing/metadata \
        --root data/processed_png
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_backend.models.siamese import SiameseNet
from ml_backend.pair_generator import PairDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn / matplotlib not installed — ROC/DET plots skipped.")


def compute_metrics(distances, labels, thresholds):
    """Return arrays of FAR and FRR over a sweep of thresholds."""
    far_list, frr_list = [], []
    for t in thresholds:
        preds = (distances < t).astype(int)
        # Genuine pairs: label=1 → FRR = genuine rejected / total genuine
        genuine_mask  = labels == 1
        impostor_mask = labels == 0
        frr = 1.0 - preds[genuine_mask].mean()  if genuine_mask.sum() > 0 else 0.0
        far = preds[impostor_mask].mean()        if impostor_mask.sum() > 0 else 0.0
        far_list.append(far)
        frr_list.append(frr)
    return np.array(far_list), np.array(frr_list)


def find_eer(far, frr, thresholds):
    """Find Equal Error Rate (intersection of FAR and FRR curves)."""
    diffs = np.abs(far - frr)
    idx = np.argmin(diffs)
    eer = (far[idx] + frr[idx]) / 2.0
    threshold_at_eer = thresholds[idx]
    return eer, threshold_at_eer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese fingerprint model on test set")
    parser.add_argument("--checkpoint",  default="ml-backend/checkpoints/best_siamese.pt")
    parser.add_argument("--data",        default="data/SOCOFing/metadata")
    parser.add_argument("--root",        default="data/processed_png")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--log-dir",     default="ml-backend/logs")
    parser.add_argument("--n-pairs",     type=int, default=10_000, help="Number of test pairs to evaluate")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_args = ckpt.get("args", {})
    embedding_dim = saved_args.get("embedding_dim", 128)
    distance_type = saved_args.get("distance", "euclidean")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet(embedding_dim=embedding_dim, distance=distance_type)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    img_size_val = saved_args.get("img_size", 96)
    img_size = (img_size_val, img_size_val)
    logger.info("Loaded checkpoint from epoch %d  (val_loss=%.4f)", ckpt.get("epoch", -1), ckpt.get("val_loss", -1))

    # ── Test dataset ─────────────────────────────────────────────────────────
    test_ds = PairDataset(
        metadata_csv_dir=args.data,
        processed_root=args.root,
        split="test",
        img_size=img_size,
        pairs_per_epoch=args.n_pairs,
        seed=99,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Collect distances and labels ─────────────────────────────────────────
    all_distances = []
    all_labels    = []
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            _, _, dist = model(img1, img2)
            all_distances.extend(dist.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    distances = np.array(all_distances)
    labels    = np.array(all_labels)

    # ── Threshold sweep ──────────────────────────────────────────────────────
    thresholds = np.linspace(distances.min(), distances.max(), 500)
    far, frr   = compute_metrics(distances, labels, thresholds)
    eer, eer_threshold = find_eer(far, frr, thresholds)

    # Accuracy at EER threshold
    preds    = (distances < eer_threshold).astype(int)
    accuracy = (preds == labels).mean()

    print("\n" + "─" * 55)
    print("  Evaluation Results — Test Set")
    print("─" * 55)
    print(f"  Pairs evaluated : {len(labels):,}")
    print(f"  EER             : {eer * 100:.2f}%")
    print(f"  Threshold @ EER : {eer_threshold:.4f}")
    print(f"  Accuracy @ EER  : {accuracy * 100:.2f}%")
    far_at_eer = far[np.argmin(np.abs(thresholds - eer_threshold))]
    frr_at_eer = frr[np.argmin(np.abs(thresholds - eer_threshold))]
    print(f"  FAR @ EER       : {far_at_eer * 100:.2f}%")
    print(f"  FRR @ EER       : {frr_at_eer * 100:.2f}%")
    print("─" * 55 + "\n")

    # ── Plots ────────────────────────────────────────────────────────────────
    if HAS_SKLEARN:
        # ROC curve (uses 1-distance as score for genuine)
        fpr_roc, tpr_roc, _ = roc_curve(labels, -distances)
        roc_auc = auc(fpr_roc, tpr_roc)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        axes[0].plot(fpr_roc, tpr_roc, label=f"AUC = {roc_auc:.4f}")
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0].set_xlabel("False Positive Rate (FAR)")
        axes[0].set_ylabel("True Positive Rate (1 - FRR)")
        axes[0].set_title("ROC Curve")
        axes[0].legend()

        # DET (FAR vs FRR)
        axes[1].plot(far * 100, frr * 100, color="darkorange")
        axes[1].scatter([far_at_eer * 100], [frr_at_eer * 100], color="red", zorder=5,
                        label=f"EER = {eer * 100:.2f}%")
        axes[1].set_xlabel("FAR (%)")
        axes[1].set_ylabel("FRR (%)")
        axes[1].set_title("DET Curve")
        axes[1].legend()

        plt.tight_layout()
        out_path = log_dir / "roc_det_curves.png"
        plt.savefig(out_path, dpi=150)
        logger.info("Saved ROC/DET plot → %s", out_path)
        plt.close()

    # Save metrics to text
    metrics_path = log_dir / "eval_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"EER:            {eer * 100:.4f}%\n")
        f.write(f"Threshold_EER:  {eer_threshold:.6f}\n")
        f.write(f"Accuracy_EER:   {accuracy * 100:.4f}%\n")
        f.write(f"FAR_EER:        {far_at_eer * 100:.4f}%\n")
        f.write(f"FRR_EER:        {frr_at_eer * 100:.4f}%\n")
    logger.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    main()
