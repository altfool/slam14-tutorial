//
// Created by xlk on 10/19/21.
//
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <chrono>

using std::cout, std::cin, std::endl;

struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    template<typename T>
    bool operator()(const T *const abc, T *residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0, inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;
    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(std::exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }
    double abc[3] = {ae, be, ce};
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),
                nullptr, abc);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    std::chrono::steady_clock::time_point t1=std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto ex:abc) cout << ex << " ";
    cout << endl;
    return 0;
}




