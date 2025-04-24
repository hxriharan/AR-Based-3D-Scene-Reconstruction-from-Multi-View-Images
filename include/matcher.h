/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: Declares functions for FLANN-based feature matching with Lowe’s ratio test.
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#ifndef MATCHER_H
#define MATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureMatcher {
public:
    FeatureMatcher(bool use_flann = true);  // Constructor

    void matchFeatures(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& good_matches
    );

private:
    bool use_flann;
};

#endif  // MATCHER_H
