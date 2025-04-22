#ifndef POINTCLOUD_WRITER_H
#define POINTCLOUD_WRITER_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class PointCloudWriter {
public:
    static bool writePLY(const std::string& filename, const std::vector<cv::Point3f>& points);
};

#endif // POINTCLOUD_WRITER_H
