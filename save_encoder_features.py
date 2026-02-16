import numpy as np
import sonata
import torch
import torch.nn as nn
import os
import glob
from tqdm import tqdm

try:
    import flash_attn
except ImportError:
    flash_attn = None

def load_custom_data(scene_path):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path not found: {scene_path}")
        
    point = {}
    point["coord"] = np.load(os.path.join(scene_path, "coord.npy"))
    point["color"] = np.load(os.path.join(scene_path, "color.npy"))
    point["normal"] = np.load(os.path.join(scene_path, "normal.npy"))

    if os.path.exists(os.path.join(scene_path, "segment20.npy")):
        point["segment"] = np.load(os.path.join(scene_path, "segment20.npy"))
    
    return point

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/obiwan/mirac/sonata/sonata/scannet_data_manual/train", help="Directory containing scene folders")
    parser.add_argument("--output_dir", type=str, default="/home/obiwan/mirac/sonata/encoder_features", help="Directory to save output features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    sonata.utils.set_seed(24525867)
    
    # Load Model
    if flash_attn is not None:
        print("Using Flash Attention Config")
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        print("Flash Attention not found, using custom config...")
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()
        
    transform = sonata.transform.default()
    
    # Get List of Scenes
    scene_paths = sorted(glob.glob(os.path.join(args.data_dir, "scene*")))
    print(f"Found {len(scene_paths)} scenes to process.")

    model.eval()
    
    for scene_path in tqdm(scene_paths):
        scene_name = os.path.basename(scene_path)
        output_path = os.path.join(args.output_dir, f"{scene_name}_encoder_features.npy")
        
        if os.path.exists(output_path):
            continue

        try:
            point = load_custom_data(scene_path)
            
            # transform
            point = transform(point)
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)

            with torch.inference_mode():
                # model forward
                point = model(point)
                
                # Save encoder features (bottleneck)
                # point.feat is the feature at the bottleneck
                feat_np = point.feat.cpu().numpy()
                np.save(output_path, feat_np)
                
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            continue
            
    print("Done!")
