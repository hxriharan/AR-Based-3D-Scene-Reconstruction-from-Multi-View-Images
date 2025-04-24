/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: Declares triangulation routines using projection matrices and matched keypoints.
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class Triangulator {
public:
    Triangulator(const cv::Mat& K);  // Constructor with intrinsics

    void triangulatePoints(
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& R, const cv::Mat& t,
        std::vector<cv::Point3f>& points3D
    );

private:
    cv::Mat K_;
};

#endif  // TRIANGULATOR_H
