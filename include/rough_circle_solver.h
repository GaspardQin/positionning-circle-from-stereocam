#ifndef __ROUGH_CIRCLE_SOLVER_H__
#define __ROUGH_CIRCLE_SOLVER_H__

/*
*   This class implements the algorithm that decribed in the paper
*    "Long Quan. Conic Reconstruction and Correspondence from Two Views",
*   which gives an analytical solution of the circle pose in space. 
*
*   Author: Chuan QIN 
*/
#include "camera_model.h"
#include "circle_model.h"
#include <memory>
#include <vector>
#include <limits>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

#define DEBUG

class RoughCircleSolver{
private:
    #define LEFT_CAMERA 0
    #define RIGHT_CAMERA 1
    std::shared_ptr<StereoCameraModel> stereo_cam_ptr;
public:
    RoughCircleSolver(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):stereo_cam_ptr(stereo_cam_ptr_){
    }

    void getPossibleCircles(const cv::Mat& left_edge, const cv::Mat& right_edge, std::vector<Circle3D>& circles);
    // main function of this class

    void reprojectCircles(cv::Mat& image, const Circle3D& circle, int camera_id, int sample_size, const cv::Scalar& color);
    void reprojectCircles(cv::Mat& image, const std::vector<Circle3D>& circles_vec, int camera_id, int sample_size, const cv::Scalar& color);

private:
    void getPossibleEllipse(const cv::Mat &edge_input, std::vector<Eigen::Matrix3d> &ellipses_vec, std::vector<cv::RotatedRect>* ellipses_box_vec_ptr=nullptr)
; //helper function
    void getCircles(const std::vector<Eigen::Matrix3d>& left_possible_ellipses, const std::vector<Eigen::Matrix3d>& right_possible_ellipses, 
                    std::vector<Circle3D>& circles, const std::vector<cv::RotatedRect>* left_ellipses_box_ptr=nullptr, const std::vector<cv::RotatedRect>* right_ellipses_box_ptr=nullptr,
                    const cv::Mat* left_edge_ptr=nullptr, const cv::Mat* right_edge_ptr=nullptr);//helper function
    typedef std::tuple<int, int, double> CirclePair; // left index, right index, error
    void computeI2I3I4(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B, double& I2, double& I3, double& I4);
    void computeI2I3I4Analytic(const Eigen::Matrix4d &A, const Eigen::Matrix4d &B, double &I_2, double &I_3, double &I_4);

    void computePointsInPlane(const double& u, const double& v, const Eigen::Vector4d& plane, Eigen::Vector4d& point);
    void translateEllipse(const Eigen::Matrix3d& ellipse_quad_form, cv::RotatedRect& ellipse_cv_form);
    void translateEllipse(const cv::RotatedRect& ellipse_cv_form, Eigen::Matrix3d& ellipse_quad_form);

};


#endif //__ROUGH_CIRCLE_SOLVER_H__