import os
import cv2
import torch
import numpy as np
import open3d as o3d
from glob import glob

# Load MiDaS
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform

# Camera intrinsics from your calibration
K = np.array([[3316.82226, 0.0,        1514.45791],
              [0.0,        3312.61390, 2016.30946],
              [0.0,        0.0,        1.0]])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

# Load and process images
image_dir = "/Users/hariharansureshkumar/3D_Reconstruction_project/data2"
image_paths = sorted(glob(os.path.join(image_dir, "IMG_44*.jpg")))

all_points = []

for path in image_paths:
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        depth = midas(input_tensor).squeeze().cpu().numpy()
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

    # Backproject to 3D
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth.flatten()
    x = (i.flatten() - cx) * z / fx
    y = (j.flatten() - cy) * z / fy
    points = np.vstack((x, y, z)).T
    points = points[(z > 0.1) & (z < 10000)]
    all_points.append(points)

# Merge and visualize
final_pts = np.vstack(all_points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(final_pts)
o3d.io.write_point_cloud("/Users/hariharansureshkumar/3D_Reconstruction_project/output/depth_fused_pointcloud.ply", pcd)
print("Saved point cloud to output/depth_fused_pointcloud.ply")
o3d.visualization.draw_geometries([pcd])
