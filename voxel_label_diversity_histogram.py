

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed for batch run
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sonata
import torch
import torch.nn as nn
import os
import collections
import argparse

try:
    import flash_attn
except ImportError:
    flash_attn = None

# ── ScanNet metadata ──────────────────────────────────────────────────────────
CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)


# ── Segmentation head ─────────────────────────────────────────────────────────
class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


# ── Data loader ───────────────────────────────────────────────────────────────
def load_scene(scene_path):
    point = {}
    point["coord"]  = np.load(os.path.join(scene_path, "coord.npy"))
    point["color"]  = np.load(os.path.join(scene_path, "color.npy"))
    point["normal"] = np.load(os.path.join(scene_path, "normal.npy"))
    seg_path = os.path.join(scene_path, "segment20.npy")
    if os.path.exists(seg_path):
        point["segment"] = np.load(seg_path)
    return point


# ── Per-scene inference ───────────────────────────────────────────────────────
def process_scene(scene_path, model, seg_head, transform):
    
    try:
        raw = load_scene(scene_path)
    except Exception as e:
        print(f"  [SKIP] Failed to load: {e}")
        return None

    point_t = transform(raw)
    for key, val in point_t.items():
        if isinstance(val, torch.Tensor):
            point_t[key] = val.cuda(non_blocking=True)

    with torch.inference_mode():
        point_output = model(point_t)

        inverses = []
        while "pooling_parent" in point_output.keys():
            parent  = point_output.pop("pooling_parent")
            inverse = point_output.pop("pooling_inverse")
            inverses.append(inverse)
            parent.feat = torch.cat([parent.feat, point_output.feat[inverse]], dim=-1)
            point_output = parent

        feat     = point_output.feat          # (N_finest, C)
        N_finest = feat.shape[0]

        # Compose inverses: finest point → coarsest voxel
        idx = torch.arange(N_finest, device=feat.device)
        for inv in reversed(inverses):        # finest→coarsest
            idx = inv[idx]

        # Predict labels
        pred   = seg_head(feat).argmax(dim=-1).cpu().numpy()   # (N_finest,)
        idx_np = idx.cpu().numpy()                              # (N_finest,)

    # Count distinct labels per coarsest voxel
    voxel_to_labels = collections.defaultdict(set)
    for fine_i in range(N_finest):
        voxel_to_labels[idx_np[fine_i]].add(pred[fine_i])

    diversity = np.array([len(s) for s in voxel_to_labels.values()], dtype=np.int32)

    # Collect sorted label pairs for voxels with exactly 2 distinct labels
    two_label_pairs = [
        tuple(sorted(s))
        for s in voxel_to_labels.values()
        if len(s) == 2
    ]

    return diversity, two_label_pairs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Voxel label-diversity histogram (all scenes)")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/obiwan/mirac/sonata/scannet_all_processed",
        help="Root directory containing one subdirectory per scene",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/obiwan/mirac/sonata/voxel_label_diversity.png",
        help="Output histogram image path",
    )
    args = parser.parse_args()

    sonata.utils.set_seed(24525867)

    # ── Load model & seg head ─────────────────────────────────────────────────
    print("Loading Sonata encoder …")
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        print("Flash Attention not available — using fallback config.")
        custom_config = dict(enc_patch_size=[1024] * 5, enable_flash=False)
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()

    print("Loading segmentation head …")
    ckpt = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])

    model.eval()
    seg_head.eval()

    transform = sonata.transform.default()

    # ── Gather all scene directories ──────────────────────────────────────────
    scene_dirs = sorted([
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])
    print(f"\nFound {len(scene_dirs)} scenes in {args.data_root}\n")

    # ── Process each scene ────────────────────────────────────────────────────
    all_diversity = []        
    all_two_label_pairs = []  
    n_failed = 0

    for i, scene_path in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_path)
        print(f"[{i+1:4d}/{len(scene_dirs)}] {scene_name} … ", end="", flush=True)

        result = process_scene(scene_path, model, seg_head, transform)
        if result is None:
            n_failed += 1
            print("FAILED")
            continue

        diversity, two_label_pairs = result
        all_diversity.append(diversity)
        all_two_label_pairs.extend(two_label_pairs)
        print(f"{len(diversity):,} coarsest voxels  |  mean diversity {diversity.mean():.2f}")

    print(f"\nDone. {len(all_diversity)} scenes processed, {n_failed} skipped.")

    if not all_diversity:
        print("No scenes were successfully processed. Exiting.")
        return

    # ── Aggregate across all scenes ───────────────────────────────────────────
    diversity_counts = np.concatenate(all_diversity)           # (total_voxels_across_all_scenes,)
    total_voxels     = len(diversity_counts)
    total_scenes     = len(all_diversity)

    max_diversity = int(diversity_counts.max())
    bins         = np.arange(1, max_diversity + 2)
    hist, _      = np.histogram(diversity_counts, bins=bins)
    percentages  = hist / hist.sum() * 100.0
    x_labels     = bins[:-1]

    print("\n── Diversity summary (all scenes) ────────────────────────────────")
    for n_labels, pct in zip(x_labels, percentages):
        print(f"  {n_labels:2d} distinct label(s): {pct:6.2f}%  ({hist[n_labels-1]:,} voxels)")
    print(f"\nTotal coarsest voxels : {total_voxels:,}")
    print(f"Mean diversity        : {diversity_counts.mean():.3f} labels/voxel")
    print(f"Median diversity      : {np.median(diversity_counts):.1f} labels/voxel")
    print(f"Max diversity         : {diversity_counts.max()} labels in one voxel")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(x_labels)))

    bars = ax.bar(
        x_labels, percentages,
        color=palette,
        width=0.65,
        edgecolor="#ffffff22",
        linewidth=0.8,
        zorder=3,
    )

    # Annotate bars
    for bar, pct in zip(bars, percentages):
        if pct > 0.2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{pct:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color="#222222", fontweight="bold",
            )

    # Axes formatting
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xticks(x_labels)
    ax.set_xticklabels([str(n) for n in x_labels], color="#222222", fontsize=11)
    ax.tick_params(axis="y", colors="#222222", labelsize=10)
    ax.grid(axis="y", color="#cccccc", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")

    ax.set_xlabel("Number of Distinct Predicted Labels in Coarsest Voxel",
                  color="#222222", fontsize=12, labelpad=10)
    ax.set_ylabel("% of Coarsest Voxels (all scenes)", color="#222222", fontsize=12, labelpad=10)
    ax.set_title(
        f"Label Diversity per Coarsest Voxel — ScanNet All Processed\n"
        f"({total_scenes:,} scenes  ·  {total_voxels:,} total coarsest voxels)",
        color="#222222", fontsize=13, pad=14,
    )

    stats_text = (
        f"Mean:   {diversity_counts.mean():.2f} labels/voxel\n"
        f"Median: {np.median(diversity_counts):.1f} labels/voxel\n"
        f"Max:    {diversity_counts.max()} labels/voxel"
    )
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10, color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nHistogram saved to: {args.out}")

    # ── Second figure: label-pair frequency TABLE for 2-label voxels ─────────
    if not all_two_label_pairs:
        print("No 2-label voxels found — skipping 2-label pair table.")
        return

    pair_counter = collections.Counter(all_two_label_pairs)
    total_two_label = sum(pair_counter.values())

    print(f"\n── 2-label voxel pair summary ({total_two_label:,} voxels) ──────────────")
    for (a, b), cnt in pair_counter.most_common(20):
        print(f"  {CLASS_LABELS_20[a]:20s} + {CLASS_LABELS_20[b]:20s}: "
              f"{cnt:7,}  ({cnt/total_two_label*100:.2f}%)")

    # Keep only pairs that account for ≥0.1 % of 2-label voxels
    top_pairs = [(pair, cnt) for pair, cnt in pair_counter.most_common()
                 if cnt / total_two_label * 100 >= 0.1]
    if not top_pairs:
        top_pairs = pair_counter.most_common(20)

    # ── Build table data ──────────────────────────────────────────────────────
    col_headers = ["Rank", "Label A", "Label B", "Count", "% of 2-Label Voxels"]
    table_rows = []
    for rank, ((a, b), cnt) in enumerate(top_pairs, start=1):
        pct = cnt / total_two_label * 100.0
        table_rows.append([
            str(rank),
            CLASS_LABELS_20[a],
            CLASS_LABELS_20[b],
            f"{cnt:,}",
            f"{pct:.2f}%",
        ])

    n_rows = len(table_rows)
    row_height = 0.38          # inches per row
    header_height = 0.55       # inches for the header
    fig_height = max(4.0, header_height + n_rows * row_height + 1.6)

    fig2, ax2 = plt.subplots(figsize=(13, fig_height))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    ax2.axis("off")

    ax2.set_title(
        f"Label Pair Combinations in 2-Label Coarsest Voxels — ScanNet All Processed\n"
        f"({total_scenes:,} scenes  ·  {total_two_label:,} two-label voxels  ·  "
        f"{len(pair_counter)} unique pairs)",
        color="#222222", fontsize=13, pad=14, loc="center",
    )

    tbl = ax2.table(
        cellText=table_rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.55)

    # ── Style cells ──────────────────────────────────────────────────────────
    header_bg   = "#e8e8e8"
    header_fg   = "#333333"
    row_bg_even = "white"
    row_bg_odd  = "#f5f5f5"
    cell_fg     = "#222222"
    edge_color  = "#cccccc"

    n_cols = len(col_headers)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.6)
        if row == 0:                          # header row
            cell.set_facecolor(header_bg)
            cell.set_text_props(color=header_fg, fontweight="bold", fontsize=10)
        else:                                 # data rows
            bg = row_bg_even if row % 2 == 0 else row_bg_odd
            cell.set_facecolor(bg)
            cell.set_text_props(color=cell_fg, fontsize=10)
        # Left-align label columns (A and B)
        if col in (1, 2) and row > 0:
            cell._loc = "left"

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out2 = args.out.replace(".png", "_2label_pairs.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"2-label pair table saved to: {out2}")


if __name__ == "__main__":
    main()
