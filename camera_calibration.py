"""
Author: Hariharan Sureshkumar
Course: CS 5330 - Pattern Recognition and Computer Vision
Semester: Spring 2025

Purpose: Calibrates the camera using checkerboard images and exports intrinsics.

This file is part of a custom 3D reconstruction pipeline project that
implements feature-based structure-from-motion and sparse point cloud
generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
"""


import cv2
import numpy as np
import glob

# Chessboard dimensions
CHECKERBOARD = (9, 6)

# Store object points and image points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('/Users/hariharansureshkumar/3D_Reconstruction_project/calibration_images/*.jpg')  # Update your folder path

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save results
cv_file = cv2.FileStorage("camera_params.yml", cv2.FILE_STORAGE_WRITE)
cv_file.write("K", K)
cv_file.write("distCoeffs", dist)
cv_file.release()

print("Calibration complete.")
print("K:\n", K)
print("Distortion Coefficients:\n", dist)
