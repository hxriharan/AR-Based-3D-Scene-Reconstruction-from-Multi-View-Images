// feature_detector.cpp

#include "feature_detector.h"

FeatureDetector::FeatureDetector(bool use_sift) {
    if (use_sift) {
        detector = cv::SIFT::create();
    } else {
        detector = cv::ORB::create();
    }
}

void FeatureDetector::detectAndCompute(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors
) {
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}
