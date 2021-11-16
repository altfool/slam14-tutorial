//
// Created by xlk on 11/7/21.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "chrono"
#include "sophus/se3.hpp"

using std::cout, std::endl, std::cin;

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R,
                          cv::Mat &t);

void bundleAdjustment(const std::vector<cv::Point3f> &points_3d, const std::vector<cv::Point3f> &points_2d, cv::Mat &R,
                      cv::Mat &t);

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2 " << std::endl;
        return 1;
    }
    // read image
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "totally find " << matches.size() << " pairs" << std::endl;

    // build 3d points
    cv::Mat depth1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    cv::Mat depth2 = cv::imread(argv[4], cv::IMREAD_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts1, pts2;

    for (cv::DMatch m: matches) {
        ushort d1 = depth1.at<ushort>(keypoints_1[m.queryIdx].pt);
        ushort d2 = depth2.at<ushort>(keypoints_2[m.trainIdx].pt);
        if (d1 == 0 || d2 == 0)
            continue;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
//        cout << "p1.y: " << p1.y << "\tqueryIdx.y: " << keypoints_1[m.queryIdx].pt.y << endl;
//        cout << "p2.y: " << p2.y << "\ttrainIdx.y: " << keypoints_2[m.trainIdx].pt.y << endl;
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
//        cout << "dd1 : " << dd1 << endl;
//        cout << "dd2 : " << dd2 << endl;
        pts1.push_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    std::cout << "3d-3d pairs: " << pts1.size() << std::endl;
    cv::Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    std::cout << "ICP via SVD resutls: " << std::endl;
    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << "t = " << t << endl;
    cout << "R_inv = " << endl << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    cout << "\n\ncalling bundle adjustment" << endl;
    bundleAdjustment(pts1, pts2, R, t);
    // verify p1 = R * p2 + t
    for (int i = 0 + 10; i < 5 + 10; i++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "R * p2 + t = " << R * (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << endl;
        cout << endl;
    }
    return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches) {
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    double min_dist = 1e4, max_dist = 0;
    for (auto &m: match) {
        double d = m.distance;
        if (d > max_dist) max_dist = d;
        if (d < min_dist) min_dist = d;
    }
    cout << "max distance: " << max_dist << endl;
    cout << "min distance: " << min_dist << endl;
    for (auto &m: match) {
        if (m.distance <= std::max(2 * min_dist, 30.0))
            matches.push_back(m);
    }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R,
                          cv::Mat &t) {
    cv::Point3f p1, p2;
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = p1 / N;
    p2 = p2 / N;
    // compute distance to center
    std::vector<cv::Point3f> q1(N), q2(N);
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << " W = " << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0)
        R_ = -R_;
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) <<
                                R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2));
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override {}

    virtual bool write(std::ostream &out) const override {}
};

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

    virtual void computeError() override {
        const VertexPose *pose = static_cast<const VertexPose *> (_vertices[0]);
        _error = _measurement - pose->estimate() * _point;
    }

    virtual void linearizeOplus() override {
        VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = pose->estimate();
        Eigen::Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }

    bool read(std::istream &in) {}

    bool write(std::ostream &out) const {}

protected:
    Eigen::Vector3d _point;
};

void
bundleAdjustment(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t) {
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    VertexPose *pose = new VertexPose();
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    //edges
    for (int i = 0; i < pts1.size(); i++) {
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    cout << endl << "after optimization: " << endl;
    cout << "T=\n" << pose->estimate().matrix() << endl;

    // convert to cv::Mat
    Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = pose->estimate().translation();
    R = (cv::Mat_<double>(3, 3) <<
            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}