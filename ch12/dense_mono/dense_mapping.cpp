//
// Created by xlk on 12/28/21.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <boost/timer.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

using namespace std;
using Sophus::SE3d;
using namespace Eigen;
using namespace cv;

/**********************************************
* 本程序演示了单目相机在已知轨迹下的稠密深度估计
* 使用极线搜索 + NCC 匹配的方式，与书本的 12.2 节对应
* 请注意本程序并不完美，你完全可以改进它——我其实在故意暴露一些问题(这是借口)。
***********************************************/

// ----------------------------------
// parameters
const int boarder = 20; // edge width
const int width = 640; // image width
const int height = 480; // image height
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3; // NCC half of window size
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC window area
const double min_cov = 0.1; // convergence test: minimum covariance
const double max_cov = 10; // divergence test: maximum covariance

// -----------------------------------
// important funcs
// read data from REMODE
bool readDatasetFiles(const string &path, vector<string> &color_image_files, vector<SE3d> &poses, cv::Mat &ref_depth);

/**
 * 根据新的图像更新深度估计
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         参考图像到当前图像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2);

/**
 * 极线搜索
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         位姿
 * @param pt_ref        参考图像中点的位置
 * @param depth_mu      深度均值
 * @param depth_cov     深度方差
 * @param pt_curr       当前点
 * @param epipolar_direction  极线方向
 * @return              是否成功
 */
bool epipolarSearch(const Mat &ref, const Mat &curr, const SE3d &T_C_R, const Vector2d &pt_ref, const double &depth_mu,
                    const double &depth_cov, Vector2d &pt_curr, Vector2d &epipolar_direction);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */
bool updateDepthFilter(const Vector2d &pt_ref, const Vector2d &pt_curr, const SE3d &T_C_R,
                       const Vector2d &epipolar_direction, Mat &depth, Mat &depth_cov2);

/**
 * 计算 NCC 评分
 * @param ref       参考图像
 * @param curr      当前图像
 * @param pt_ref    参考点
 * @param pt_curr   当前点
 * @return          NCC评分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 双线性灰度插值
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) + xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[0 + img.step]) + xx * yy * double(d[1 + img.step])) / 255.0;
}

// ----------------------------------
// 一些小工具
// 显示估计的深度图
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素到相机坐标系
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d((px(0, 0) - cx) / fx, (px(1, 0) - cy) / fy, 1);
}

// 相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(p_cam(0, 0) * fx / p_cam(2, 0) + cx, p_cam(1, 0) * fy / p_cam(2, 0) + cy);
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// 显示极线匹配
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// 显示极线
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

// 评测深度估计
void evaluationDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ----------------------------------------------

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: dense_mapping <path_to_test_dataset>" << endl;
        return -1;
    }

    // read data
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (!ret) {
        cout << "reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files. " << endl;

    // first image
    Mat ref = imread(color_image_files[0], 0); // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;    // depth init
    double init_cov2 = 3.0;     // covirance init
    Mat depth(height, width, CV_64F, init_depth); // depth map
    Mat depth_cov2(height, width, CV_64F, init_cov2);

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 坐标转换关系： T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaluationDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }
    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done. " << endl;
    return 0;
}

bool readDatasetFiles(const string &path, vector<string> &color_image_files, vector<SE3d> &poses, Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d: data) fin >> d;
        color_image_files.push_back(path + "/images/" + image);
        poses.push_back(SE3d(Quaterniond(data[6], data[3], data[4], data[5]), Vector3d(data[0], data[1], data[2])));
        if (!fin.good()) break;
    }
    fin.close();

    // load referene depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }
    return true;
}

// update the whole depth map
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = boarder; x < width - boarder; ++x) {
        for (int y = boarder; y < height - boarder; ++y) {
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov)
                continue;   // 深度已收敛或发散
            // 在极线上搜索 (x,y) 的匹配
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(ref, curr, T_C_R, Vector2d(x, y), depth.ptr<double>(y)[x],
                                      sqrt(depth_cov2.ptr<double>(y)[x]), pt_curr, epipolar_direction);
            if (!ret) // match failed
                continue;

            // 取消该注释以显示匹配
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // matching success, update depth map
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    }
}

// 极线搜索
// 方法见书 12.2 12.3 两节
bool epipolarSearch(const Mat &ref, const Mat &curr, const SE3d &T_C_R, const Vector2d &pt_ref, const double &depth_mu,
                    const double &depth_cov, Vector2d &pt_curr, Vector2d &epipolar_direction) {
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;    // reference frame, p vector

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // 按深度均值投影的像素
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); // 按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max)); // 按最大深度投影的像素

    Vector2d epipolar_line = px_max_curr - px_min_curr; // 极线（线段形式）
    epipolar_direction = epipolar_line; // 极线方向
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm(); // 极线线段的半长度
    if (half_length > 100) half_length = 100; // 我们不希望搜索太多东西

    // 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction; // candidate point
        if (!inside(px_curr))
            continue;
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)  // 只相信 NCC 很高的匹配
        return false;
    pt_curr = best_px_curr;
    return true;
}

double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y < ncc_window_size; ++y) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算 Zero mean NCC
    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (int i = 0; i < values_ref.size(); ++i) {
        double n = (values_ref[i] - mean_ref) * (values_ref[i] - mean_curr);
        numerator += n;
        denominator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        denominator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(denominator1 * denominator2 + 1e-10); // 防止分母出现零
}

bool updateDepthFilter(const Vector2d &pt_ref, const Vector2d &pt_curr, const SE3d &T_C_R,
                       const Vector2d &epipolar_direction, Mat &depth, Mat &depth_cov2) {
    // 不知道这段还有没有人看
    // 用三角化计算深度
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;   // ref 侧的结果
    Vector3d xn = t + ans[1] * f2;  // cur 结果
    Vector3d p_esti = (xm + xn) / 2.0;  // P的位置，取两者的平均
    double depth_estimation = p_esti.norm();    // 深度值

    // 计算不确定性（以一个像素为误差）
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 高斯融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2*d_cov2) / (sigma2+d_cov2);
    depth.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))] = sigma_fuse2;

    return true;
}















