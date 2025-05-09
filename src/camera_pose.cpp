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


#include "camera_pose.h"

CameraPoseEstimator::CameraPoseEstimator(const cv::Mat& K) : cameraMatrix(K.clone()) {}

bool CameraPoseEstimator::estimatePose(
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,
    cv::Mat& R, cv::Mat& t
) {
    // Extract matched 2D points
    std::vector<cv::Point2f> points1, points2;

    for (const auto& match : matches) {
        points1.push_back(kp1[match.queryIdx].pt);
        points2.push_back(kp2[match.trainIdx].pt);
    }

    // Compute Essential matrix
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);

    if (E.empty()) {
        std::cerr << "Failed to compute Essential Matrix." << std::endl;
        return false;
    }

    // Recover pose from Essential matrix
    int inliers = cv::recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

    std::cout << "Recovered pose with " << inliers << " inliers." << std::endl;
    return true;
}
