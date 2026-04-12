import re
import csv
import json
import math
import random
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SOCOFING_RE = re.compile(
    r"^(?P<id>\d+)__(?P<sex>[MF])_(?P<hand>Left|Right)_(?P<finger>\w+)_finger"
    r"(?:_(?P<alteration>CR|Obl|Zcut))?\.BMP$",
    re.IGNORECASE,
)

_DIFFICULTY_MAP = {
    "Real": "real",
    "Altered-Easy": "easy",
    "Altered-Medium": "medium",
    "Altered-Hard": "hard",
}


def parse_socofing_filename(path: Path, difficulty: str) -> Optional[dict]:
    m = _SOCOFING_RE.match(path.name)
    if not m:
        return None
    alteration = m.group("alteration") or "none"
    return {
        "filename": str(path),
        "subject_id": int(m.group("id")),
        "sex": m.group("sex"),
        "hand": m.group("hand"),
        "finger": m.group("finger"),
        "alteration": alteration,
        "difficulty": difficulty,
        "split": "",         
    }


def build_metadata(dataset_root: str | Path) -> list[dict]:
    root = Path(dataset_root)
    records: list[dict] = []
    skipped = 0

    scan_dirs = {
        root / "Real": "real",
        root / "Altered" / "Altered-Easy": "easy",
        root / "Altered" / "Altered-Medium": "medium",
        root / "Altered" / "Altered-Hard": "hard",
    }

    for scan_dir, difficulty in scan_dirs.items():
        if not scan_dir.exists():
            logger.warning("Directory not found, skipping: %s", scan_dir)
            continue
        for p in sorted(scan_dir.rglob("*.BMP")):
            rec = parse_socofing_filename(p, difficulty)
            if rec is None:
                logger.debug("Could not parse filename: %s", p.name)
                skipped += 1
                continue
            records.append(rec)

    logger.info(
        "build_metadata: %d records collected, %d filenames skipped",
        len(records), skipped,
    )
    return records


def split_subjects(
    metadata: list[dict],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    assert abs(train + val + test - 1.0) < 1e-6, "train+val+test must equal 1.0"

    subjects = sorted({r["subject_id"] for r in metadata})
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = math.floor(n * train)
    n_val = math.floor(n * val)

    train_set = set(subjects[:n_train])
    val_set   = set(subjects[n_train : n_train + n_val])
    test_set  = set(subjects[n_train + n_val :])

    for rec in metadata:
        sid = rec["subject_id"]
        if sid in train_set:
            rec["split"] = "train"
        elif sid in val_set:
            rec["split"] = "val"
        else:
            rec["split"] = "test"

    counts = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}
    logger.info(
        "split_subjects: %d subjects → train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
        n,
        counts["train"], 100 * counts["train"] / n,
        counts["val"],   100 * counts["val"]   / n,
        counts["test"],  100 * counts["test"]  / n,
    )
    return metadata


_CSV_FIELDS = [
    "filename", "subject_id", "sex", "hand", "finger",
    "alteration", "difficulty", "split",
]


def save_metadata(records: list[dict], output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "metadata.csv"
    json_path = out / "metadata.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Saved %d records → %s, %s", len(records), csv_path, json_path)
    return csv_path, json_path


def load_metadata(
    metadata_dir: str | Path,
    split: Optional[str] = None,
) -> list[dict]:
    csv_path = Path(metadata_dir) / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in: {metadata_dir}")

    records: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["subject_id"] = int(row["subject_id"])
            if split is None or row["split"] == split:
                records.append(row)

    logger.info(
        "load_metadata: %d records loaded (split=%s)", len(records), split or "all"
    )
    return records


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_and_save_splits(
    dataset_root: str | Path,
    output_dir: Optional[str | Path] = None,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path]:
    root = Path(dataset_root)
    if output_dir is None:
        output_dir = root / "metadata"

    metadata = build_metadata(root)
    metadata = split_subjects(metadata, train=train, val=val, test=test, seed=seed)
    return save_metadata(metadata, output_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate subject-wise 70/15/15 splits for SOCOFing."
    )
    parser.add_argument(
        "--root",
        default="data/SOCOFing",
        help="Path to the SOCOFing dataset root (default: data/SOCOFing)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for metadata files (default: <root>/metadata/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_p, json_p = generate_and_save_splits(
        dataset_root=args.root,
        output_dir=args.out,
        seed=args.seed,
    )

    records = load_metadata(csv_p.parent)
    splits = {"train", "val", "test"}

    bad = [r for r in records if r["split"] not in splits]
    assert not bad, f"{len(bad)} records have an invalid split value!"

    from collections import defaultdict
    subj_splits: dict[int, set] = defaultdict(set)
    for r in records:
        subj_splits[r["subject_id"]].add(r["split"])
    leaking = {sid: ss for sid, ss in subj_splits.items() if len(ss) > 1}
    assert not leaking, f"Subject leakage detected: {leaking}"

    print("\n── Split Summary ─────────────────────────────────────────")
    print(f"  Total images  : {len(records):>7}")
    for sp in ("train", "val", "test"):
        imgs = [r for r in records if r["split"] == sp]
        subs = {r["subject_id"] for r in imgs}
        print(f"  {sp:<5} : {len(imgs):>7} images  |  {len(subs):>4} subjects")
    print(f"\n  No subject leakage detected ✓")
    print(f"  metadata.csv  → {csv_p}")
    print(f"  metadata.json → {json_p}")
