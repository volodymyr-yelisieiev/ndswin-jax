"""Fetch a Hugging Face dataset and optionally export to the repo-standard layout.

Usage examples:
  # For CIFAR-10 classification (saves to data/cifar10/)
  python train/fetch_hf_dataset.py --hf-id cifar10 --outdir data/cifar10

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


def point_cloud_to_voxel(points: np.ndarray, resolution: int = 32) -> np.ndarray:
    """Convert a point cloud (N, 3) to a binary voxel grid (resolution, resolution, resolution)."""
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    # Normalize to [0, 1]
    points = (points - p_min) / (p_max - p_min + 1e-8)
    # Scale to [0, resolution - 1]
    coords = np.clip(np.floor(points * resolution), 0, resolution - 1).astype(np.int32)
    
    voxel = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return voxel


def save_classification(out_dir: Path, split: str, idx: int, image: Any, label: Any) -> None:
    # Save image as npz in a class subdirectory
    try:
        label_val = int(label)
    except (TypeError, ValueError):
        label_val = hash(str(label)) % 1000  # Fallback
        
    class_dir = out_dir / split / f"class_{label_val:03d}"
    class_dir.mkdir(parents=True, exist_ok=True)

    img_arr = np.array(image)
    if img_arr.ndim == 2 and img_arr.shape[1] == 3:
        # It's a point cloud, voxelize it
        img_arr = point_cloud_to_voxel(img_arr, resolution=32)

    fname = f"{idx:05d}.npz"
    np.savez_compressed(class_dir / fname, image=img_arr)


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

    # First try the normal load (non-streaming) to get splits and iterate easily
    try:
        # Try with authentication token if available to avoid rate limits
        # Removed token=True since we want this to work without auth for public datasets
        ds = load_dataset(hf_id)

        print("Available splits:", list(ds.keys()))

        for split in ds.keys():
            print(f"Processing split: {split}")
            count = 0
            for ex in ds[split]:
                # Heuristics: classification datasets usually have 'image' (PIL or array) and 'label' (int)
                # Segmentation datasets have 'image' and 'label' where label is an array with spatial dims.
                # ModelNet datasets use 'inputs' instead of 'image'.
                image_data = None
                for _k in ("image", "images", "img", "image0", "pixel_values", "inputs"):
                    if _k in ex and ex[_k] is not None:
                        image_data = ex[_k]
                        break
                if "label" in ex and ex["label"] is not None and image_data is not None:
                    image = image_data
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

    except ValueError as e:
        # Some datasets are packaged as WebDataset archives with layouts that the
        # standard loader cannot parse. Fall back to streaming mode which iterates
        # split-by-split and is more tolerant of varied archive layouts.
        print(f"load_dataset failed with: {e}; retrying using streaming fallback...")
        any_saved = False
        for split in ("train", "validation", "val", "test"):
            try:
                print(f"Attempting streaming export for split: {split}")
                it = load_dataset(hf_id, split=split, streaming=True)
                count = 0
                for ex in it:
                    image_data = None
                    for _k in ("image", "images", "img", "image0", "pixel_values", "inputs"):
                        if _k in ex and ex[_k] is not None:
                            image_data = ex[_k]
                            break
                    if "label" in ex and ex["label"] is not None and image_data is not None:
                        image = image_data
                        label = ex["label"]

                        if isinstance(label, (int, float)):
                            save_classification(out_dir, split, count, image, label)
                        else:
                            img = np.array(image)
                            lbl = np.array(label)
                            if img.ndim == 3:
                                img = np.expand_dims(img, 0)
                            elif img.ndim == 4 and img.shape[-1] < 5:
                                img = np.moveaxis(img, -1, 0)
                            if lbl.ndim == 4 and lbl.shape[0] == 1:
                                lbl = lbl[0]
                            case_id = ex.get("image_id") or ex.get("id") or f"{split}_{count}"
                            save_segmentation(out_dir, split, str(case_id), img, lbl)

                    elif "images" in ex and "labels" in ex and ex["images"] and ex["labels"]:
                        img = np.array(ex["images"][0])
                        lbl = np.array(ex["labels"][0])
                        case_id = ex.get("id") or f"{split}_{count}"
                        save_segmentation(out_dir, split, str(case_id), img, lbl)
                    else:
                        print(f"Warning: skipping example {count} in split {split} (unrecognized format)")
                        continue

                    count += 1
                    if limit is not None and count >= limit:
                        break

                if count > 0:
                    print(f"Saved {count} examples under {out_dir / split}")
                    any_saved = True
            except Exception:
                # split not present or streaming failed for this split, continue
                continue

        if not any_saved:
            # Final fallback: attempt to download the dataset tar from HF Hub and extract
            try:
                from huggingface_hub import hf_hub_download

                print("Attempting HF Hub TAR download fallback...")
                tar_path = hf_hub_download(hf_id, filename=hf_id.split('/')[-1] + ".tar", repo_type="dataset")
                print(f"Downloaded archive: {tar_path}; extracting...")
                import tarfile
                import nibabel as nib

                with tarfile.open(tar_path, "r") as tf:
                    members = tf.getmembers()
                    # Extract and convert imagesTr/labelsTr pairs
                    imgs = [m for m in members if "imagesTr" in m.name and m.name.endswith(".nii.gz") and not os.path.basename(m.name).startswith('._')]
                    lbls = [m for m in members if "labelsTr" in m.name and m.name.endswith(".nii.gz") and not os.path.basename(m.name).startswith('._')]

                    def safe_extract(member, target_path):
                        tf.extract(member, path=target_path)

                    import tempfile
                    tmpdir = tempfile.mkdtemp(prefix="hf_extract_")
                    extracted = 0
                    for img_m in imgs:
                        base = os.path.basename(img_m.name)
                        case_id = os.path.splitext(base)[0]
                        try:
                            # Extract to preserve directory structure
                            lbl_name = None
                            for l in lbls:
                                if case_id in l.name and not os.path.basename(l.name).startswith('._'):
                                    lbl_name = l
                                    break
                            if lbl_name is None:
                                # No clean label found for this case
                                continue
                            safe_extract(img_m, tmpdir)
                            safe_extract(lbl_name, tmpdir)

                            img_target = os.path.join(tmpdir, img_m.name)
                            lbl_target = os.path.join(tmpdir, lbl_name.name)

                            # Load NIfTI and save as npz
                            img_arr = nib.load(img_target).get_fdata()
                            lbl_arr = nib.load(lbl_target).get_fdata()
                            # Ensure channel-first for images (C,D,H,W)
                            if img_arr.ndim == 3:
                                img_arr = img_arr.astype(np.float32)[None, ...]
                            save_segmentation(out_dir, "train", case_id, img_arr, lbl_arr)
                            extracted += 1
                        except Exception as e_file:
                            print(f"Skipping {img_m.name} due to error: {e_file}")
                            continue

                    print(f"Extracted and saved {extracted} examples under {out_dir / 'train'}")
            except Exception as e2:
                print(f"Fallback extraction failed: {e2}")
                raise

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and export HF dataset to repo layout")
    parser.add_argument("--hf-id", type=str, required=True, help="Hugging Face dataset id (e.g., cifar10 or user/Task01_BrainTumour)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to write dataset")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit number of examples per split for testing")

    args = parser.parse_args()
    export_dataset(args.hf_id, args.outdir, limit=args.limit)


if __name__ == "__main__":
    main()
