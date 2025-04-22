#include "pointcloud_writer.h"
#include <fstream>
#include <iostream>

bool PointCloudWriter::writePLY(const std::string& filename, const std::vector<cv::Point3f>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return false;
    }

    // Write PLY header
    file << "ply\n"
         << "format ascii 1.0\n"
         << "element vertex " << points.size() << "\n"
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "end_header\n";

    // Write points
    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z << "\n";
    }

    file.close();
    std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
    return true;
}
