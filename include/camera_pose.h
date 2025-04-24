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


#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <opencv2/opencv.hpp>
#include <vector>

class CameraPoseEstimator {
public:
    CameraPoseEstimator(const cv::Mat& K);  // Constructor with intrinsic matrix

    bool estimatePose(
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch>& matches,
        cv::Mat& R, cv::Mat& t
    );

private:
    cv::Mat cameraMatrix;
};

#endif // CAMERA_POSE_H
