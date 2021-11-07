//
// Created by xlk on 10/9/21.
//
//reference: https://learnopencv.com/camera-calibration-using-opencv/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int CHECKERBOARD[2]{8, 6};

int main(int argc, char **argv) {
    vector<vector<cv::Point3f>> objpoints;
    vector<vector<cv::Point2f>> imgpoints;

    vector<cv::Point3f> objp;
    for (int c = 0; c < CHECKERBOARD[0];c++) {
        for (int r = 0; r < CHECKERBOARD[1]; r++) {
            objp.push_back(cv::Point3f(r, c, 0));   ///???
        }
    }
    vector<cv::String> images;
    string path = "../images/*.png";
    cv::glob(path, images);
    cv::Mat frame, gray;
    vector<cv::Point2f> corner_pts;
    bool success;

    for (int i = 0; i < images.size(); i++) {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
//                                            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts);
        cout << i+1 << "th success: " << success << endl;
        if (success){
            cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001);
            cv::cornerSubPix(gray, corner_pts, cv::Size(11,11), cv::Size(-1,-1), criteria);
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        cv::imshow("image", frame);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();

    cv::Mat cameraMatrix, distCoeffs, R, T;
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
    cout << "cameraMatrix: " << cameraMatrix << endl;
    cout << "distCoeffs: " << distCoeffs << endl;
    cout << "Rotation vector:  " << R << endl;
    cout << "Translation vector: " << T << endl;
    return 0;
}
