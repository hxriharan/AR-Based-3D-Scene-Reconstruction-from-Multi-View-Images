cmake_minimum_required(VERSION 3.10)
project(AR3DReconstruction)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set OpenCV directory manually (only needed if not found automatically)
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0_1/lib/cmake/opencv4")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Define sources
set(SOURCE_FILES
    main.cpp
    src/feature_detector.cpp
    src/matcher.cpp
    src/camera_pose.cpp
    src/triangulator.cpp
    src/pointcloud_writer.cpp
    src/bundle_adjuster.cpp
)

# Define executable
add_executable(reconstruction ${SOURCE_FILES})

# Link libraries
target_link_libraries(reconstruction
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)
