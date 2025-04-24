/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: [Brief explanation of what the file does.]
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#include "feature_detector.h"

FeatureDetector::FeatureDetector(bool use_sift) {
    if (use_sift) {
        detector = cv::SIFT::create();
    } else {
        detector = cv::ORB::create();
    }
}

void FeatureDetector::detectAndCompute(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors
) {
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}
