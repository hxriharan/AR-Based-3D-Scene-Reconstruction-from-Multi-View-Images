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


#include "matcher.h"

FeatureMatcher::FeatureMatcher(bool use_flann_) : use_flann(use_flann_) {}

void FeatureMatcher::matchFeatures(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& good_matches
) {
    std::vector<std::vector<cv::DMatch>> knn_matches;

    if (use_flann) {
        // FLANN requires descriptors to be of type CV_32F
        cv::Mat desc1, desc2;
        descriptors1.convertTo(desc1, CV_32F);
        descriptors2.convertTo(desc2, CV_32F);

        cv::FlannBasedMatcher matcher;
        matcher.knnMatch(desc1, desc2, knn_matches, 2);
    } else {
        cv::BFMatcher matcher(cv::NORM_L2);
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
    }

    // Lowe's ratio test
    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}

