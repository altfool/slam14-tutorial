//
// Created by xlk on 10/19/21.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using std::cin, std::cout, std::endl;

int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0, inv_sigma = 1 / w_sigma;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(std::exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }
    int iterations = 100;
    double cost = 0, lastCost = 0;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        cost = 0;
        for (int i = 0; i < N; i++) {
            double xi = x_data[i], yi = y_data[i];
            double error = yi - std::exp(ae * xi * xi + be * xi + ce);
            Vector3d J;
            J[0] = -xi * xi * std::exp(ae * xi * xi + be * xi + ce);
            J[1] = -xi * std::exp(ae * xi * xi + be * xi + ce);
            J[2] = -std::exp(ae * xi * xi + be * xi + ce);
            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * J * error;
            cost += error * error;
        }
        Vector3d dx = H.ldlt().solve(b);
        if (std::isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << " >= last cost: " << lastCost << ", break." << endl;
            break;
        }
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        lastCost = cost;
        cout << "total cost: " << cost << ", \t\t\tupdate: " << dx.transpose() << "\t\t\textimated params: " << ae << ","
             << be << "," << ce << endl;
    }
    std::chrono::steady_clock::time_point t2=std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}


