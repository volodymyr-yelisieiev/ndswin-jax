"""Fetch a Hugging Face dataset and optionally export to the repo-standard layout.

Usage examples:
  # For CIFAR-100 classification (saves to data/cifar100/)
  python train/fetch_hf_dataset.py --hf-id cifar100 --outdir data/cifar100

  # For MSD Task01 (saves images/labels per split)
  python train/fetch_hf_dataset.py --hf-id mabdulre9/MSD_Task01_BrainTumour --outdir data/medseg_brain_tumour

Notes:
- This script attempts to detect dataset modality (classification vs segmentation)
  and export into the `DATASET_LAYOUT.md` described layout when possible.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover - optional runtime
    raise RuntimeError("The 'datasets' library is required to fetch HF datasets. Install: pip install datasets") from e


def save_classification(out_dir: Path, split: str, idx: int, image: Any, label: Any) -> None:
    # Save image as npz and a small label file listing (csv or npz)
    split_dir = out_dir / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{idx:05d}.npz"
    np.savez_compressed(images_dir / fname, image=np.array(image))
    np.savez_compressed(labels_dir / fname, label=np.array(label))


def save_segmentation(out_dir: Path, split: str, case_id: str, image: np.ndarray, label: np.ndarray) -> None:
    img_dir = out_dir / split / "images"
    lbl_dir = out_dir / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(img_dir / f"{case_id}.npz", image=image.astype(np.float32))
    np.savez_compressed(lbl_dir / f"{case_id}.npz", label=label.astype(np.int32))


def export_dataset(hf_id: str, outdir: str, limit: int | None = None) -> None:
    out_dir = Path(outdir)
    print(f"Loading dataset {hf_id}...")
    ds = load_dataset(hf_id)

    print("Available splits:", list(ds.keys()))

    for split in ds.keys():
        print(f"Processing split: {split}")
        count = 0
        for ex in ds[split]:
            # Heuristics: classification datasets usually have 'image' (PIL or array) and 'label' (int)
            # Segmentation datasets have 'image' and 'label' where label is an array with spatial dims
            if "label" in ex and ex["label"] is not None and "image" in ex and ex["image"] is not None:
                image = ex["image"]
                label = ex["label"]

                # if label is int -> classification
                if isinstance(label, (int, float)):
                    save_classification(out_dir, split, count, image, label)
                else:
                    # segmentation: try to coerce to arrays and save as image/label npz pair
                    img = np.array(image)
                    lbl = np.array(label)
                    # ensure channel-first for images if 3D/4D
                    if img.ndim == 3:
                        # assume (D,H,W) -> add channel dim
                        img = np.expand_dims(img, 0)
                    elif img.ndim == 4 and img.shape[-1] < 5:
                        img = np.moveaxis(img, -1, 0)
                    # labels expected (D,H,W) or (C,D,H,W) -> convert
                    if lbl.ndim == 4 and lbl.shape[0] == 1:
                        lbl = lbl[0]
                    case_id = ex.get("image_id") or ex.get("id") or f"{split}_{count}"
                    save_segmentation(out_dir, split, str(case_id), img, lbl)

            # Some datasets use 'images' / 'labels' lists
            elif "images" in ex and "labels" in ex and ex["images"] and ex["labels"]:
                # take first or stack
                imgs = ex["images"]
                lbls = ex["labels"]
                img = np.array(imgs[0])
                lbl = np.array(lbls[0])
                case_id = ex.get("id") or f"{split}_{count}"
                save_segmentation(out_dir, split, str(case_id), img, lbl)
            else:
                # Could not recognize example format, skip
                print(f"Warning: skipping example {count} in split {split} (unrecognized format)")
                continue

            count += 1
            if limit is not None and count >= limit:
                break

        print(f"Saved {count} examples under {out_dir / split}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and export HF dataset to repo layout")
    parser.add_argument("--hf-id", type=str, required=True, help="Hugging Face dataset id (e.g., cifar100 or user/Task01_BrainTumour)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to write dataset")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit number of examples per split for testing")

    args = parser.parse_args()
    export_dataset(args.hf_id, args.outdir, limit=args.limit)


if __name__ == "__main__":
    main()
