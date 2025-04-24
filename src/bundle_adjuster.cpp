/*
 * Author: Hariharan Sureshkumar
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 * Semester: Spring 2025
 *
 * Purpose: Implements bundle adjustment using Ceres to refine camera poses and 3D structure.
 *
 * This file is part of a custom 3D reconstruction pipeline project that
 * implements feature-based structure-from-motion and sparse point cloud
 * generation from multi-view RGB images, using OpenCV, Eigen, and Ceres.
 */


#include "bundle_adjuster.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>


struct ReprojectionError {
    ReprojectionError(cv::Point2f observed, cv::Mat K)
        : observed_(observed), fx_(K.at<double>(0, 0)), fy_(K.at<double>(1, 1)),
          cx_(K.at<double>(0, 2)), cy_(K.at<double>(1, 2)) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
        T xp = p[0] / p[2], yp = p[1] / p[2];
        T u = T(fx_) * xp + T(cx_);
        T v = T(fy_) * yp + T(cy_);
        residuals[0] = u - T(observed_.x);
        residuals[1] = v - T(observed_.y);
        return true;
    }

    static ceres::CostFunction* Create(const cv::Point2f& observed, const cv::Mat& K) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(observed, K)));
    }

private:
    const cv::Point2f observed_;
    double fx_, fy_, cx_, cy_;
};

BundleAdjuster::BundleAdjuster(const cv::Mat& K) : K_(K.clone()) {}

void BundleAdjuster::optimize(
    std::vector<cv::Mat>& rotations,
    std::vector<cv::Mat>& translations,
    std::vector<cv::Point3f>& points3D,
    const std::vector<BAObservation>& observations
) {
    ceres::Problem problem;

    std::vector<std::array<double, 6>> camera_params(rotations.size());
    std::vector<std::array<double, 3>> point_params(points3D.size());

    for (int i = 0; i < rotations.size(); ++i) {
        cv::Mat rvec;
        cv::Rodrigues(rotations[i], rvec);
        for (int j = 0; j < 3; ++j) {
            camera_params[i][j] = rvec.at<double>(j);
            camera_params[i][j + 3] = translations[i].at<double>(j);
        }
    }

    for (int i = 0; i < points3D.size(); ++i) {
        point_params[i][0] = points3D[i].x;
        point_params[i][1] = points3D[i].y;
        point_params[i][2] = points3D[i].z;
    }

    for (const auto& obs : observations) {
        ceres::CostFunction* cost_function = ReprojectionError::Create(obs.image_point, K_);
        problem.AddResidualBlock(
            cost_function, nullptr,
            camera_params[obs.camera_index].data(),
            point_params[obs.point_index].data()
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    for (int i = 0; i < rotations.size(); ++i) {
        cv::Mat rvec(3, 1, CV_64F), R;
        for (int j = 0; j < 3; ++j)
            rvec.at<double>(j) = camera_params[i][j];
        cv::Rodrigues(rvec, R);
        rotations[i] = R;
        for (int j = 0; j < 3; ++j)
            translations[i].at<double>(j) = camera_params[i][j + 3];
    }

    for (int i = 0; i < points3D.size(); ++i) {
        points3D[i].x = point_params[i][0];
        points3D[i].y = point_params[i][1];
        points3D[i].z = point_params[i][2];
    }
}
