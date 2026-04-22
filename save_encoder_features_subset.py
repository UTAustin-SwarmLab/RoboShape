"""
save_encoder_features_subset.py

For every scene in --data_dir whose coarsest voxel count is in (350, 500):

  Task 1 – Encoder features
      Accumulate all per-scene coarsest-voxel features into one big array
      and save as:  <output_dir>/all_features.npy   shape (N_total, 512)

  Task 2 – Wall label tensor  (voxelwise_pred logic, Solution 2)
      For each coarsest voxel: 1 if predicted label == "wall" (idx 0), else 0
      Saved as:  <output_dir>/wall_labels.npy        shape (N_total,)  int32

  Task 3 – Bedroom scene-type tensor  (create_scene_type_tensor logic)
      For each coarsest voxel: 1 if its scene has scene_type "bedroom", else 0
      Saved as:  <output_dir>/bedroom_labels.npy     shape (N_total,)  int32

All three arrays are aligned row-by-row (same voxel order).
A JSON log of included scenes is written to --json_log.
"""

import numpy as np
import sonata
import torch
import torch.nn as nn
import os
import glob
import json
from tqdm import tqdm

try:
    import flash_attn
except ImportError:
    flash_attn = None

CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)
WALL_IDX = CLASS_LABELS_20.index("wall")   # == 0


# ── Seg head ──────────────────────────────────────────────────────────────────
class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


# ── Data loader ───────────────────────────────────────────────────────────────
def load_custom_data(scene_path):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path not found: {scene_path}")
    point = {}
    point["coord"]  = np.load(os.path.join(scene_path, "coord.npy"))
    point["color"]  = np.load(os.path.join(scene_path, "color.npy"))
    point["normal"] = np.load(os.path.join(scene_path, "normal.npy"))
    seg = os.path.join(scene_path, "segment20.npy")
    if os.path.exists(seg):
        point["segment"] = np.load(seg)
    return point


# ── Per-scene processing ──────────────────────────────────────────────────────
def process_scene(scene_path, model, seg_head, transform, softmax):
    """
    Returns (feat_np, wall_mask, is_bedroom_flag) or None on failure.
      feat_np   : (N_coarsest, 512)  float32  — encoder features
      wall_mask : (N_coarsest,)      int32    — 1 where voxel pred == wall
      is_bedroom: bool               — True if scene_type.txt says bedroom
    """
    # ── Task 3: scene type ────────────────────────────────────────────────────
    type_path = os.path.join(scene_path, "scene_type.txt")
    if os.path.exists(type_path):
        scene_type = open(type_path).read().strip().lower()
        is_bedroom = "bedroom" in scene_type
    else:
        is_bedroom = False

    # ── Load & transform ──────────────────────────────────────────────────────
    try:
        raw = load_custom_data(scene_path)
    except Exception as e:
        print(f"  [SKIP load] {e}")
        return None

    pt = transform(raw)
    for k, v in pt.items():
        if isinstance(v, torch.Tensor):
            pt[k] = v.cuda(non_blocking=True)

    try:
        with torch.inference_mode():
            # ── Encoder forward ───────────────────────────────────────────────
            out = model(pt)

            # ── Task 1: encoder features at bottleneck ────────────────────────
            feat_bottleneck = out.feat.cpu().numpy()   # (N_coarsest, 512)
            N_coarsest = feat_bottleneck.shape[0]

            # Filter by voxel count
            if not (350 < N_coarsest < 500):
                return None

            # ── Unpool to get finest-level features + inverse map ─────────────
            inverses = []
            while "pooling_parent" in out.keys():
                parent  = out.pop("pooling_parent")
                inverse = out.pop("pooling_inverse")
                inverses.append(inverse)
                parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
                out = parent

            feat_finest = out.feat          # (N_finest, C)
            N_final     = feat_finest.shape[0]

            # Compose inverses: finest point → coarsest voxel index
            idx = torch.arange(N_final, device=feat_finest.device)
            for inv in reversed(inverses):
                idx = inv[idx]
            idx_np = idx.cpu().numpy()     # (N_final,)

            # ── Task 2: voxel-level label via Solution 2 ──────────────────────
            probs = softmax(seg_head(feat_finest)).cpu().numpy()   # (N_final, 20)
            voxel_prob_sum = np.zeros((N_coarsest, 20), dtype=np.float32)
            np.add.at(voxel_prob_sum, idx_np, probs)

            voxel_pred = np.argmax(voxel_prob_sum, axis=1)         # (N_coarsest,)
            wall_mask  = (voxel_pred == WALL_IDX).astype(np.int32) # (N_coarsest,)

    except Exception as e:
        print(f"  [SKIP infer] {e}")
        return None

    return feat_bottleneck, wall_mask, is_bedroom


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,
                        default="/home/obiwan/mirac/sonata/scannet_subset1_processed",
                        help="Directory containing scene folders")
    parser.add_argument("--output_dir", type=str,
                        default="/home/obiwan/mirac/sonata/scannet_subset1_tensors",
                        help="Directory to save output tensors")
    parser.add_argument("--json_log",   type=str,
                        default="/home/obiwan/mirac/sonata/scannet_subset1_tensors/scene_log.json",
                        help="Path to JSON log of included scenes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sonata.utils.set_seed(24525867)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading Sonata encoder …")
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        model = sonata.load(
            "sonata", repo_id="facebook/sonata",
            custom_config=dict(enc_patch_size=[1024]*5, enable_flash=False)
        ).cuda()

    print("Loading seg head …")
    ckpt     = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    model.eval(); seg_head.eval()

    transform = sonata.transform.default()
    softmax   = nn.Softmax(dim=-1)

    scene_paths = sorted(glob.glob(os.path.join(args.data_dir, "scene*")))
    print(f"Found {len(scene_paths)} scenes in {args.data_dir}\n")

    # ── Accumulate across scenes ──────────────────────────────────────────────
    all_features  = []   # list of (N_i, 512) arrays
    all_wall      = []   # list of (N_i,) int32 arrays
    all_bedroom   = []   # list of (N_i,) int32 arrays
    scene_log     = []
    global_offset = 0    # running row index into the concatenated arrays

    for scene_path in tqdm(scene_paths, desc="Scenes"):
        scene_name = os.path.basename(scene_path)
        result = process_scene(scene_path, model, seg_head, transform, softmax)
        if result is None:
            continue

        feat_np, wall_mask, is_bedroom = result
        N_i = feat_np.shape[0]

        all_features.append(feat_np)
        all_wall.append(wall_mask)
        all_bedroom.append(np.full(N_i, int(is_bedroom), dtype=np.int32))

        # Read scene_type string for the log
        type_path = os.path.join(scene_path, "scene_type.txt")
        scene_type_str = open(type_path).read().strip() if os.path.exists(type_path) else "Unknown"

        scene_log.append({
            "scene_name":        scene_name,
            "scene_path":        scene_path,
            "scene_type":        scene_type_str,
            "is_bedroom":        is_bedroom,
            "n_coarsest_voxels": N_i,
            "global_start":      global_offset,          # first row index in numpy arrays
            "global_end":        global_offset + N_i,    # exclusive end index
            "wall_voxels":       int(wall_mask.sum()),
            "wall_ratio":        round(float(wall_mask.mean()) * 100, 2),
        })
        global_offset += N_i

    if not all_features:
        print("No scenes passed the filter. Exiting.")
        exit(1)

    # ── Stack & save ──────────────────────────────────────────────────────────
    features_arr = np.concatenate(all_features, axis=0)   # (N_total, 512)
    wall_arr     = np.concatenate(all_wall,     axis=0)   # (N_total,)
    bedroom_arr  = np.concatenate(all_bedroom,  axis=0)   # (N_total,)

    feat_path    = os.path.join(args.output_dir, "all_features.npy")
    wall_path    = os.path.join(args.output_dir, "wall_labels.npy")
    bedroom_path = os.path.join(args.output_dir, "bedroom_labels.npy")

    np.save(feat_path,    features_arr)
    np.save(wall_path,    wall_arr)
    np.save(bedroom_path, bedroom_arr)

    with open(args.json_log, "w") as jf:
        json.dump(scene_log, jf, indent=2)

    N_total = features_arr.shape[0]
    print(f"\n── Saved ──────────────────────────────────────────────────")
    print(f"  all_features.npy   : {features_arr.shape}  float32")
    print(f"  wall_labels.npy    : {wall_arr.shape}  int32  "
          f"({wall_arr.sum():,} wall voxels, {wall_arr.mean()*100:.1f}%)")
    print(f"  bedroom_labels.npy : {bedroom_arr.shape}  int32  "
          f"({bedroom_arr.sum():,} bedroom voxels, {bedroom_arr.mean()*100:.1f}%)")
    print(f"  scene_log.json     : {len(scene_log)} scenes")
    print(f"  Total voxels       : {N_total:,}")
    print("Done!")
