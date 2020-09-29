cmake_minimum_required(VERSION 3.10)

# set the project name
project(DepthFromStereo VERSION 0.0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)

# specify OpenCV dependency
find_package( OpenCV REQUIRED )

# add the executable
add_executable(DepthFromStereo stereo_vision_parallel.cpp)

target_link_libraries( DepthFromStereo ${OpenCV_LIBS} )