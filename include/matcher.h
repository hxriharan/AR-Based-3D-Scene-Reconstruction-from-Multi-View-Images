//FLANN and RANSAC parameters

// matcher.h

#ifndef MATCHER_H
#define MATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureMatcher {
public:
    FeatureMatcher(bool use_flann = true);  // Constructor

    void matchFeatures(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& good_matches
    );

private:
    bool use_flann;
};

#endif  // MATCHER_H
