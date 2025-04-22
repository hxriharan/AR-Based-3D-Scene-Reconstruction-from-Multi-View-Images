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

    // Convert from homogeneous to 3D
    points3D.clear();
    for (int i = 0; i < points4D.cols; ++i) {
        cv::Mat col = points4D.col(i);
        col /= col.at<float>(3);  // Normalize by w
        points3D.push_back(cv::Point3f(col.at<float>(0), col.at<float>(1), col.at<float>(2)));
    }
}
