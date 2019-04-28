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
#include "base_circle_solver.h"
#include <vector>
#include <limits>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

#define DEBUG

class RoughCircleSolver: public BaseCircleSolver{
private:
    
public:
    RoughCircleSolver(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):BaseCircleSolver(stereo_cam_ptr_){
    }

    void getPossibleCircles(const cv::Mat &left_edge, const cv::Mat &right_edge, std::vector<ConcentricCircles3D> &concentric_circles);
    // main function of this class

    void reprojectCircles(cv::Mat& image, const Circle3D& circle, int camera_id, int sample_size, const cv::Scalar& color);
    void reprojectCircles(cv::Mat& image, const std::vector<Circle3D>& circles_vec, int camera_id, int sample_size, const cv::Scalar& color);
    void reprojectCircles(cv::Mat &image, const ConcentricCircles3D &concentric_circles, int camera_id, int sample_size, const cv::Scalar &color);
    void reprojectCircles(cv::Mat &image, const std::vector<ConcentricCircles3D> &concentric_circles, int camera_id, int sample_size, const cv::Scalar &color);

private:
    void getPossibleEllipse(const cv::Mat &edge_input, std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> &ellipses_vec); //helper function
    bool computeCircle3D(const cv::RotatedRect& left_ellipse_box, const cv::RotatedRect& right_ellipse_box, Circle3D& circle);//helper function
    typedef std::tuple<int, int, double> CirclePair; // left index, right index, error
    void computeI2I3I4(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B, double& I2, double& I3, double& I4);
    void computeI2I3I4Analytic(const Eigen::Matrix4d &A, const Eigen::Matrix4d &B, double &I_2, double &I_3, double &I_4);

    void computePointsInPlane(const double &u, const double &v, const Eigen::Vector4d &plane, Eigen::Vector4d &point, int camera_id);
    
    bool areConcentric(const Circle3D a, const Circle3D b, double center_threshold);
    void getConcentricCircles(const std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> &left_possible_ellipses,
                                   const std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> &right_possible_ellipses,
                                   std::vector<ConcentricCircles3D> &concentric_circles,
                                   const cv::Mat* left_edge_ptr=nullptr,
                                   const cv::Mat* right_edge_ptr=nullptr);


};
template <class T>
bool isIn(T num, T min, T max){
    if(num>=min && num<=max) return true;
    else return false;
}

#endif //__ROUGH_CIRCLE_SOLVER_H__