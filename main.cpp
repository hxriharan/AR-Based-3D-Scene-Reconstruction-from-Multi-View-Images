#include <iostream>
#include <opencv2/opencv.hpp>

#include "feature_detector.h"
#include "matcher.h"
#include "camera_pose.h"
#include "triangulator.h"  // Include triangulation module
#include "pointcloud_writer.h"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./reconstruction <image1> <image2>" << std::endl;
        return -1;
    }

    std::string img_path1 = argv[1];
    std::string img_path2 = argv[2];

    cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img_path2, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load input images." << std::endl;
        return -1;
    }

    // Feature detection
    FeatureDetector detector(true);  // true = use SIFT
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    detector.detectAndCompute(img1, kp1, desc1);
    detector.detectAndCompute(img2, kp2, desc2);

    // Feature matching
    FeatureMatcher matcher(true);  // true = use FLANN
    std::vector<cv::DMatch> good_matches;
    matcher.matchFeatures(desc1, desc2, good_matches);

    // Estimate camera pose
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        525.0, 0.0, 319.5,
        0.0, 525.0, 239.5,
        0.0, 0.0, 1.0);  // Intrinsics

    cv::Mat R, t;
    CameraPoseEstimator pose_estimator(K);
    bool success = pose_estimator.estimatePose(kp1, kp2, good_matches, R, t);

    if (success) {
        std::cout << "Estimated camera motion:" << std::endl;
        std::cout << "Rotation matrix R:\n" << R << std::endl;
        std::cout << "Translation vector t:\n" << t << std::endl;

        // Triangulate 3D points
        std::vector<cv::Point3f> points3D;
        Triangulator triangulator(K);
        triangulator.triangulatePoints(kp1, kp2, good_matches, R, t, points3D);

        std::cout << "\nFirst 10 Triangulated 3D Points:" << std::endl;
        for (int i = 0; i < std::min(10, (int)points3D.size()); ++i) {
            std::cout << points3D[i] << std::endl;
        }

        std::string ply_filename = "/Users/hariharansureshkumar/3D_Reconstruction_project/output/points3D.ply";
        PointCloudWriter::writePLY(ply_filename, points3D);
        std::cout << "Saved " << points3D.size() << " 3D points to " << ply_filename << std::endl;

    } else {
        std::cerr << "Pose estimation failed." << std::endl;
    }

    // Visualize matches
    cv::Mat match_img;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, match_img,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Good Matches", match_img);
    cv::waitKey(0);

    return 0;
}
