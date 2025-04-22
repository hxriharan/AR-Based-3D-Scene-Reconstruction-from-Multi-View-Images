cmake_minimum_required(VERSION 3.10)
project(AR3DReconstruction)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 14)

# Set OpenCV directory manually
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0_1/lib/cmake/opencv4")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
add_executable(reconstruction
    main.cpp
    src/feature_detector.cpp
    src/matcher.cpp
    src/camera_pose.cpp
    src/triangulator.cpp
    src/pointcloud_writer.cpp
)

# Link libraries
target_link_libraries(reconstruction ${OpenCV_LIBS})
