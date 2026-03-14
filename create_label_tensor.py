import numpy as np
import os
import glob
import argparse

# ScanNet-20 indices for the target classes
LABEL_TABLE  = 6
LABEL_WINDOW = 8

CLASS_LABELS_20 = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/obiwan/mirac/sonata/scannet_processed4",
        help="Root directory containing scene* folders.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="window_or_table",
        help="Prefix for output files.",
    )
    args = parser.parse_args()

    scene_paths = sorted(glob.glob(os.path.join(args.data_dir, "scene*")))
    print(f"Found {len(scene_paths)} scenes.")

    scene_names = []
    labels = []

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        seg_path   = os.path.join(scene_path, "segment20.npy")

        if not os.path.exists(seg_path):
            print(f"  [SKIP] {scene_name} — no segment20.npy")
            scene_names.append(scene_name)
            labels.append(0)
            continue

        seg    = np.load(seg_path)
        unique = set(np.unique(seg).tolist())

        has_target = int((LABEL_TABLE in unique) or (LABEL_WINDOW in unique))
        labels.append(has_target)
        scene_names.append(scene_name)

    label_tensor = np.array(labels, dtype=np.int32)

    # Save binary tensor
    npy_path = f"{args.output_prefix}_labels.npy"
    np.save(npy_path, label_tensor)
    print(f"Saved label tensor -> {npy_path}  shape={label_tensor.shape}")

    # Save scene name list (same order)
    txt_path = f"{args.output_prefix}_scenes.txt"
    with open(txt_path, "w") as f:
        for name, lbl in zip(scene_names, label_tensor):
            f.write(f"{name}\t{lbl}\n")
    print(f"Saved scene list   -> {txt_path}")


    n_pos = int(label_tensor.sum())
    n_tot = len(label_tensor)
    print(f"\nSummary:")
    print(f"  Total scenes      : {n_tot}")
    print(f"  Has window/table  : {n_pos}  ({n_pos/n_tot*100:.1f}%)")
    print(f"  Has neither       : {n_tot - n_pos}  ({(n_tot-n_pos)/n_tot*100:.1f}%)")


if __name__ == "__main__":
    main()
