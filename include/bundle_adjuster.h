/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: Defines the BundleAdjuster class used for Ceres-based global optimization of poses and 3D points.
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#ifndef BUNDLE_ADJUSTER_H
#define BUNDLE_ADJUSTER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Struct used to store observations
struct BAObservation {
    cv::Point2f image_point;
    int camera_index;
    int point_index;
};


class BundleAdjuster {
public:
    BundleAdjuster(const cv::Mat& K);

    void optimize(
        std::vector<cv::Mat>& rotations,
        std::vector<cv::Mat>& translations,
        std::vector<cv::Point3f>& points3D,
        const std::vector<BAObservation>& observations
    );


private:
    cv::Mat K_;
};

#endif // BUNDLE_ADJUSTER_H
