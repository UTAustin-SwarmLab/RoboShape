import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def render_3d_isometric(ply_path, output_image):
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found.")
        return

    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    # Subsample for faster rendering and less clutter in matplotlib
    pcd = pcd.voxel_down_sample(voxel_size=0.03)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"Rendering {len(points)} points...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot in 3D
    # Note: Scannet usually has Z as up.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=0.5, alpha=0.8)
    
    # Set view angle (Elevation, Azimuth)
    # elev=30, azim=-60 is standard isometric-ish
    ax.view_init(elev=45, azim=-60)
    
    ax.set_title("Semantic Segmentation Result (3D Isometric View)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Try to set equal aspect ratio manually since maplotlib 3d is tricky with it
    # This is a rough hack for aspect ratio
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(), 
                          points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to {output_image}")

if __name__ == "__main__":
    render_3d_isometric("inference_result.ply", "inference_preview_3d2.png")
