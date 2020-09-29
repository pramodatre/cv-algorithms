# Build Instructions
If you have CMake setup on your machine, you can build this project using CMake. Other option is to build each C++ file separately -- both options described below.

## Step 1
Clone the `cv-algorithms` repository
```code
git clone https://github.com/pramodatre/cv-algorithms.git
```
## Step 2
Build the project.

### CMake build
Cmake build commands

```shell
cd cv-algorithms/stereo_vision

mkdir build
cd build
cmake ..
make
```

After running the above commands, you will be see a binary `DepthFromStereo` in `build` directory. You can invoke this for test images.

```shell
./DepthFromStereo ../data/left.png ../data/right.png 
```

### Build individual files
If you are using MacOS, I recommend using `pkg-config` for managing complier flags for various dependencies you may have installed using `homebrew`. Assuming that you have installed OpenCV, you can add OpenCV to `pkg-config` paths like this.

```shell
pkg-config --cflags --libs /usr/local/Cellar/opencv/4.1.2/lib/pkgconfig/opencv4.pc
```

Later, you can build C++ files using

```shell
g++ $(pkg-config --cflags --libs opencv4) -std=c++11 stereo_vision_parallel.cpp -o stereo_vision

./stereo_vision
Usage: stereo_vision_executable LEFT_IMG_PATH RIGHT_IMG_PATH
```