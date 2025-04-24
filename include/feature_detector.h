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


#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureDetector {
public:
    FeatureDetector(bool use_sift = true);  // Constructor

    void detectAndCompute(
        const cv::Mat& image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors
    );

private:
    cv::Ptr<cv::Feature2D> detector; // SIFT or ORB detector
};

#endif  // FEATURE_DETECTOR_H
