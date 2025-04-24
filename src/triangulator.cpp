/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose:  Implements triangulation logic for converting matched features into 3D points.

 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#include "triangulator.h"

Triangulator::Triangulator(const cv::Mat& K) : K_(K.clone()) {}

void Triangulator::triangulatePoints(
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& R, const cv::Mat& t,
    std::vector<cv::Point3f>& points3D
) {
    // Projection matrix for first camera (reference): [I | 0]
    cv::Mat proj1 = K_ * cv::Mat::eye(3, 4, CV_64F);

    // Projection matrix for second camera: [R | t]
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat proj2 = K_ * Rt;

    // Prepare 2D point correspondences
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    // Triangulate
    cv::Mat points4D;
    cv::triangulatePoints(proj1, proj2, pts1, pts2, points4D);

    // Convert from homogeneous to 3D and filter invalid points
    points3D.clear();
    for (int i = 0; i < points4D.cols; ++i) {
        cv::Mat col = points4D.col(i);
        double w = col.at<double>(3);

        if (std::abs(w) > 1e-5) {
            col /= w;
            double x = col.at<double>(0);
            double y = col.at<double>(1);
            double z = col.at<double>(2);

            // Filter points that are too far, behind camera, or extreme
            if (z > 0 && std::abs(x) < 1000 && std::abs(y) < 1000 && z < 1000) {
                points3D.emplace_back(x, y, z);
            }
        }
    }
}
