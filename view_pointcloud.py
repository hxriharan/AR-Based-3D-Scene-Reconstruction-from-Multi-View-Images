"""
Author: Hariharan Sureshkumar
Course: CS 5330 - Pattern Recognition and Computer Vision
Semester: Spring 2025

Purpose: Loads and visualizes .ply files using Open3D.

This file is part of a custom 3D reconstruction pipeline project that
implements feature-based structure-from-motion and sparse point cloud
generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
"""



import open3d as o3d

# Make sure only core Open3D is used (ML module will not be triggered)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Load and visualize the point cloud
pcd = o3d.io.read_point_cloud("/Users/hariharansureshkumar/3D_Reconstruction_project/output/bundle_adjusted_map.ply")
o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud Viewer")


