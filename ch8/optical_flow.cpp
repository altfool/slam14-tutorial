//
// Created by xlk on 11/8/21.
//
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using std::cin, std::cout, std::endl;
std::string file_1 = "./LK1.png";
std::string file_2 = "./LK2.png";

/// optical flow tracker and interface
class OpticalFlowTracker {
public:
    OpticalFlowTracker(const cv::Mat &img1_, const cv::Mat &img2_, const std::vector<cv::KeyPoint> &kp1_,
                       std::vector<cv::KeyPoint> &kp2_, std::vector<bool> &success_, bool inverse_ = true,
                       bool has_initial_ = false) :
            img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
            has_initial(has_initial_) {}

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::KeyPoint> &kp1;
    std::vector<cv::KeyPoint> &kp2;
    std::vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/**
 * single level optical flow
 * @param img1 the first image
 * @param img2 the second image
 * @param kp1 keypoints in img1
 * @param kp2 keypoints in img2
 * @param success true if a keypoint is tracked successfully
 * @param inverse use inverse formulation?
 * @param has_initial_guess true if has initial guess
 */
void OpticalFlowSingleLevel(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1,
                            std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse = false,
                            bool has_initial_guess = false);

void OpticalFlowMultiLevel(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1,
                           std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse = false);

/// bi-linear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > img.cols) x = img.cols - 1;
    if (y > img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {
    cv::Mat img1 = cv::imread(file_1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(file_2, cv::IMREAD_GRAYSCALE);

    // keypoints, using GFTT here
    std::vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);

    // track keypoints with single level LK
    std::vector<cv::KeyPoint> kp2_single;
    std::vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // track keypoints with multi-level LK
    std::vector<cv::KeyPoint> kp2_multi;
    std::vector<bool> success_multi;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);  // multiple level LK
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optical flow by multiple level LK: " << time_used.count() << "seconds. " << endl;

    // use opencv's flow for validation
    std::vector<cv::Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    std::vector<uchar> status;
    std::vector<float> error;
    t1 = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << "seconds. " << endl;


    // plot the tracking results of single-level, multi-level, opencv-api
    cv::Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multiple level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);
    return 0;
}

void OpticalFlowSingleLevel(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1,
                            std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse,
                            bool has_initial_guess) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial_guess);
    cv::parallel_for_(cv::Range(0, kp1.size()),
                      std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {
    // parameters
    int half_path_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if tracking of this keypoint succeed or not

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J; // jacobian
        for (int iter = 0; iter < iterations; iter++) {
            if (!inverse) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                b = Eigen::Vector2d::Zero(); // only reset b
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_path_size; x < half_path_size; x++)
                for (int y = -half_path_size; y < half_path_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);   // Jacobian
                    if (!inverse) {
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kp.pt.x + x + dx + 1, kp.pt.y + y + dy) -
                                       GetPixelValue(img2, kp.pt.x + x + dx - 1, kp.pt.y + y + dy)),
                                0.5 * (GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy + 1) -
                                       GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy - 1))
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps the same for all iterations
                        // NOTE this J doesn't change when dx, dy update
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                       GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                       GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (!inverse || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }
            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);
            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                break;
            }
            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }

        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

void OpticalFlowMultiLevel(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1,
                           std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse) {
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::vector<cv::Mat> pyr1, pyr2;    // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids
    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp: kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = std::chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << "seconds." << endl;

        if(level>0){
            for (auto &kp:kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp:kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }
    for (auto &kp:kp2_pyr)
        kp2.push_back(kp);
}