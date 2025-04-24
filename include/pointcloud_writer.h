/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: Declares functions for writing triangulated 3D points into .ply files using Open3D.
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#ifndef POINTCLOUD_WRITER_H
#define POINTCLOUD_WRITER_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class PointCloudWriter {
public:
    static bool writePLY(const std::string& filename, const std::vector<cv::Point3f>& points);
};

#endif // POINTCLOUD_WRITER_H
