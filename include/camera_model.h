#ifndef __CAMERA_MODEL_H__
#define __CAMERA_MODEL_H__
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
class CameraModel{
private:
    cv::Mat rectified_map1, rectified_map2;
    cv::Mat intrinsic_mat; //3x3
    cv::Mat distortion_mat; //1x5
    Eigen::Matrix<double, 3, 4> projection_mat; // from 3d space to pixel space in calibrated image.
    cv::Size image_size;
public:
    CameraModel();

    CameraModel(double intrinsic_coeffs[9], double distortion_coeffs[5], double projection_coeffs[12], int image_size_[2])
    {
        init(intrinsic_coeffs,distortion_coeffs, projection_coeffs, image_size_);
    }

    void init(double intrinsic_coeffs[9], double distortion_coeffs[5], double projection_coeffs[12], int image_size_[2]){
        intrinsic_mat = cv::Mat(3, 3, CV_64F, intrinsic_coeffs).clone();
        distortion_mat = cv::Mat(1, 5, CV_64F, distortion_coeffs).clone();
        projection_mat = Eigen::Matrix<double, 3, 4>(projection_coeffs);
        image_size = cv::Size(image_size_[0], image_size_[1]);
        initUndistortRectifyMap(intrinsic_mat, distortion_mat, cv::Mat(), intrinsic_mat, image_size, CV_32FC1, rectified_map1, rectified_map2);
    }

    void rectifyImage(const cv::Mat &raw_image, cv::Mat &rectified_image){
        cv::remap(raw_image, rectified_image, rectified_map1, rectified_map2, cv::INTER_LINEAR);
    }
    const Eigen::Matrix<double, 3, 4> & projectionEigenMatrix(){
        return projection_mat;
    }
};


class StereoCameraModel{
public:
    CameraModel left;
    CameraModel right;
    StereoCameraModel(){}
    void initLeft(double intrinsic_coeffs[9], double distortion_coeffs[5], double projection_coeffs[12], int image_size_[2]){
        left.init(intrinsic_coeffs,distortion_coeffs, projection_coeffs, image_size_);
    }
    void initRight(double intrinsic_coeffs[9], double distortion_coeffs[5], double projection_coeffs[12], int image_size_[2]){
        right.init(intrinsic_coeffs,distortion_coeffs, projection_coeffs, image_size_);
    }
};

#endif //__CAMERA_MODEL_H__