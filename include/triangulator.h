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
