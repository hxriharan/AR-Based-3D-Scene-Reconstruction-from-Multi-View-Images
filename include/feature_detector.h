// feature_detector.h

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
