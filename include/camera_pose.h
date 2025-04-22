// camera_pose.h

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
