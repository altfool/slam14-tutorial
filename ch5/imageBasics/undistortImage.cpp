//
// Created by xlk on 10/8/21.
//
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

string image_file = "../distorted.png";

int main(int argc, char **argv) {
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0);
    if (image.data == nullptr) {
        cerr << "file " << image_file << " not exist." << endl;
        return 0;
    }
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);

    /*
    // ========= opencv API undistort Func. ==============
    double myCameraMatrix[] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    cv::Mat Camera_Matrix(3, 3, CV_64FC1, myCameraMatrix);
    double distortion_coeff[] = {k1, k2, p1, p2};
    cv::Mat Distortion_Coeff(1, 4, CV_64FC1, distortion_coeff);
    cout << "Camera Matrix: \n" << Camera_Matrix << "\n" << endl;
    cv::undistort(image, image_undistort, Camera_Matrix, Distortion_Coeff);
    */
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            // undistorted plane ==> normal plane [x,y,1] <== distorted plane
            // find matching between undistorted & distorted
            double x = (u - cx) / fx, y = (v - cy) / fy;    // convert to original plane [x,y,1]
            double r = sqrt(x * x + y * y);
            double x_distorted =
                    x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted =
                    y * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p2 * x * y + p1 * (r * r + 2 * y * y);
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    waitKey(0);
    destroyAllWindows();
    return 0;
}