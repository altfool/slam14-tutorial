//
// Created by xlk on 9/8/21.
//
#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

int main(int argc, char *argv[]) {
    Matrix3d rotation_matrix = Matrix3d::Identity();
    AngleAxisd rotation_vector (M_PI / 4, Vector3d(0, 0, 1));
    cout.precision(3);
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;
    rotation_matrix = rotation_vector.toRotationMatrix();
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1,0);
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    Isometry3d T = Isometry3d::Identity();
//    T.rotate(rotation_vector);
    T.rotate(rotation_matrix);
    T.pretranslate(Vector3d(1, 3, 4));
    cout << "Transform matrix = \n " << T.matrix() << endl;
    Vector3d v_transformed = T * v;
    cout << "v transformed = " << v_transformed.transpose() << endl;

    Quaterniond q = Quaterniond(rotation_matrix);
    cout << "norm of quaternion from rotation matrix is: " << q.norm() << endl;
    cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;
    q = Quaterniond(rotation_vector);
    cout << "norm of quaternion from rotation vector is: " << q.norm() << endl;
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    cout << "real part w: " << q.w() << endl;
    v_rotated = q * v;
    cout << "(1,0,0) after rotation (by quaternion) = " << v_rotated.transpose() << endl;
    cout << "should be equal to " << (q*Quaterniond(0,1,0,0)*q.inverse()).coeffs().transpose() << endl;
    return 0;
}
