#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono> 
#include <string>
#include <thread>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 7
#define SEARCH_BLOCK_SIZE 56

static const int num_threads = 8;
mutex mtx;

/**
 * 
 * Implementing block matching:
 * http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/
 * 
 * First set pkg-config:
 * pkg-config --cflags --libs /usr/local/Cellar/opencv/4.1.2/lib/pkgconfig/opencv4.pc
 * 
 * Compile this code using:
 * g++ $(pkg-config --cflags --libs opencv4) -std=c++11 stereo_vision_parallel.cpp -o stereo_vision
 * Usage: stereo_vision_executable LEFT_IMG_PATH RIGHT_IMG_PATH
 * 
 *   Left image shape: [450 x 375]
 *   Right image shape: [450 x 375]
 *   Initial disparity map: [450 x 375]
 *   Total entries in disparity map: 168750
 *   Using 8 threads for computation...
 *   Execution time: 6 seconds (0.1) mins
 * 
 * Stereo dataset: http://vision.middlebury.edu/stereo/data/scenes2003/
 * 
 * */

void display_image(Mat image)
{
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        exit(1);
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
}

Mat read_image(string path, bool display)
{
    Mat image;
    image = imread(path, IMREAD_GRAYSCALE);
    if(display)
        display_image(image);

    return image;
}

int compare_blocks(const int row, const int col, const int width, const int height, const Mat *left_img, const Mat *right_img)
{
    int sad = 0;
    int min_col = col;
    // compute bounding box for left image with (row, col) as top left point
    // compute bottom right point using (row, col)
    int bottom_row = min(row + BLOCK_SIZE, height - 1); // zero indexed, hence using (height - 1)
    int bottom_col = min(col + BLOCK_SIZE, width - 1);
    // compute bounding box for right image block in which 
    // we will scan and compare left block
    int col_min = max(0, col - SEARCH_BLOCK_SIZE);
    int col_max = min(width, col + SEARCH_BLOCK_SIZE); 
    bool first_block = true;
    int min_sad = 0;
    for (int r_indx = col_min; r_indx < col_max; ++r_indx)
    {
        sad = 0;
        for (int i = row; i < bottom_row; ++i)
        {
            int r_img_col = r_indx;
            for (int j = col; j < bottom_col; ++j)
            {
                Scalar left_pixel = left_img->at<uchar>(i, j);
                // Right image index should be updated using offset
                // since we need to scan both left and right of the
                // block from the left image 
                Scalar right_pixel = right_img->at<uchar>(i, r_img_col);
                sad += abs(left_pixel.val[0] - right_pixel.val[0]);
                ++r_img_col;
            }
        } 

        if(first_block)
        {
            min_sad = sad;
            min_col = r_indx;
            first_block = false;
        }
        else
        {
            if(sad < min_sad)
            {
                min_sad = sad;
                min_col = r_indx;
            }
        }
    }
    //cout << "min sad: " << min_sad << " ";

    return col - min_col;
} 

void compute_disparity(int start_chunk_row, int end_chunk_row, int start_chunk_col, int end_chunk_col, Mat *left_img, Mat *right_img, Mat *disparity_map)
{
    int height = left_img->rows;
    int width = left_img->cols;
    for (int i = start_chunk_row; i < end_chunk_row; ++i)
    {
        for (int j = start_chunk_col; j < end_chunk_col; ++j)
        {
            int disp = compare_blocks(i, j, height, width, left_img, right_img); 
            if(disp < 0)
            {
                mtx.lock();
                disparity_map->at<uchar>(i, j) = 0;
                mtx.unlock();
            }
            else
            {
                mtx.lock();
                disparity_map->at<uchar>(i, j) = disp;
                mtx.unlock();
            }
        } 
    }
}

vector<int> get_chunk_indices(int max, int num_chunks)
{
    vector<int> chunks;
    int step = max / num_chunks;
    for (int i = 0; i < max; i = i + step)
    {
        chunks.push_back(i);
    }
    chunks[chunks.size() - 1] = max - 1;

    return chunks;
}

Mat compute_disparity_map_parallel(Mat *left_img, Mat *right_img)
{
    if (left_img->size() != right_img->size())
    {
        cout << "Image size mismatch!" << endl;
        exit(1);
    }
    int height = left_img->rows;
    int width = left_img->cols;

    Mat disparity_map = Mat::zeros(Size(width, height), CV_8UC1); 
    cout << "Initial disparity map: " << disparity_map.size() << endl;
    cout << "Total entries in disparity map: " << width * height << endl;
    
    auto start = chrono::high_resolution_clock::now(); 

    // chunk the disparity map calculations into smaller 
    // pieces to be computed in parallel
    cout << "Using " << num_threads << " threads for computation..." << endl;
    thread t[num_threads];

    vector<int> height_chunks = get_chunk_indices(height, num_threads);
    //vector<int> width_chunks = get_chunk_indices(width, num_threads); 
    for (int i = 0; i < height_chunks.size() - 1; ++i)
    {

        t[i] = thread(compute_disparity, height_chunks[i], height_chunks[i + 1], 0, 
        width - 1, left_img, right_img, &disparity_map);
    }
    //Join the threads with the main thread
    // for (int i = 0; i < num_threads; ++i) 
    // {
    //     t[i].join();
    // }
    try
    {
        for (auto& th : t) th.join();
    }
    catch(const exception &e)
    {
        cout << "Exception! " << e.what() << endl;
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << "Execution time: " << duration.count() << " seconds" << " (" << duration.count() / 60.0 << ") mins" << endl;

    display_image(disparity_map);
    imwrite("disparity_map.png", disparity_map);

    // Apply color to disparity visualization
    Mat im_color;
    applyColorMap(disparity_map, im_color, COLORMAP_HSV);
    display_image(im_color);
    imwrite("disparity_map_color.png", im_color);

    return disparity_map;
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        cout << "Usage: stereo_vision_executable LEFT_IMG_PATH RIGHT_IMG_PATH" << endl;
        exit(1);
    }
    Mat left_img, right_img;
    // left_img = read_image("data/left.png", false);
    // right_img = read_image("data/right.png", false);
    // left_img = read_image("data/im2.png", false);
    // right_img = read_image("data/im6.png", false);

    left_img = read_image(argv[1], false);
    right_img = read_image(argv[2], false);
    Mat *l, *r;
    l = &left_img;
    r = &right_img;
    cout << "Left image shape: " << left_img.size() << endl;
    cout << "Right image shape: " << right_img.size() << endl;
    //cout << left_img << endl;
    //cout << right_img << endl;
    compute_disparity_map_parallel(l, r);

    return 0;
}