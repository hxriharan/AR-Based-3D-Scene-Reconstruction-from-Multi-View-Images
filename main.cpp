/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: Entry point for the 3D reconstruction pipeline; integrates all modules from loading images to outputting 3D results.
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#include <iostream>
#include <opencv2/opencv.hpp>

#include "feature_detector.h"
#include "matcher.h"
#include "camera_pose.h"
#include "triangulator.h"
#include "pointcloud_writer.h"
#include "bundle_adjuster.h"

int main() {
    std::vector<std::string> image_paths = {
        "../data/IMG_4452.jpg", "../data/IMG_4453.jpg", "../data/IMG_4454.jpg",
        "../data/IMG_4455.jpg", "../data/IMG_4456.jpg"
    };

    // Intrinsics from calibration
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        3316.82226, 0.0,        1514.45791,
        0.0,        3312.61390, 2016.30946,
        0.0,        0.0,        1.0);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
        0.359860210, -2.54278576, 0.00220827891, 0.00116188064, 5.64587684);

    std::vector<cv::Mat> images;
    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error loading image: " << path << std::endl;
            return -1;
        }
        cv::Mat undistorted;
        cv::undistort(img, undistorted, K, distCoeffs);
        images.push_back(undistorted);
    }

    FeatureDetector detector(true);  // Use SIFT
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<cv::Mat> all_descriptors;

    for (const auto& img : images) {
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        detector.detectAndCompute(img, kps, desc);
        all_keypoints.push_back(kps);
        all_descriptors.push_back(desc);
    }

    FeatureMatcher matcher(true);  // Use FLANN
    std::vector<std::vector<cv::DMatch>> all_matches;
    for (size_t i = 0; i < images.size() - 1; ++i) {
        std::vector<cv::DMatch> matches;
        matcher.matchFeatures(all_descriptors[i], all_descriptors[i + 1], matches);
        all_matches.push_back(matches);
    }

    CameraPoseEstimator pose_estimator(K);
    Triangulator triangulator(K);
    BundleAdjuster bundle_adjuster(K);

    std::vector<cv::Mat> rotations, translations;
    std::vector<cv::Point3f> structure_points;
    std::vector<BAObservation> observations;
    std::vector<int> camera_indices;
    std::vector<int> point_indices;

    // Initialize first camera at origin
    rotations.push_back(cv::Mat::eye(3, 3, CV_64F));
    translations.push_back(cv::Mat::zeros(3, 1, CV_64F));

    for (size_t i = 0; i < all_matches.size(); ++i) {
        const auto& kp1 = all_keypoints[i];
        const auto& kp2 = all_keypoints[i + 1];
        const auto& matches = all_matches[i];

        cv::Mat R, t;
        bool success = pose_estimator.estimatePose(kp1, kp2, matches, R, t);
        if (!success) {
            std::cerr << "Pose estimation failed between " << i << " and " << i + 1 << std::endl;
            continue;
        }

        t /= cv::norm(t);
        rotations.push_back(R.clone());
        translations.push_back(t.clone());

        std::vector<cv::Point3f> points3D;
        triangulator.triangulatePoints(kp1, kp2, matches, R, t, points3D);

        for (size_t j = 0; j < matches.size(); ++j) {
            const auto& match = matches[j];
            cv::Point2f pt2 = kp2[match.trainIdx].pt;

            observations.emplace_back(BAObservation{ pt2, static_cast<int>(i + 1), static_cast<int>(structure_points.size()) });
            structure_points.push_back(points3D[j]);
            camera_indices.push_back(i + 1);
            point_indices.push_back(structure_points.size() - 1);
        }
    }

    // Run bundle adjustment
    bundle_adjuster.optimize(rotations, translations, structure_points, observations);

    std::string ply_filename = "../output/bundle_adjusted_map.ply";
    PointCloudWriter::writePLY(ply_filename, structure_points);
    std::cout << "Saved final bundle-adjusted map with " << structure_points.size()
              << " points to " << ply_filename << std::endl;

    return 0;
}
