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
#include <Eigen/Dense>

#define DEBUG

class RoughCircleSolver{
private:
    std::shared_ptr<StereoCameraModel> stereo_cam_ptr;
public:
    RoughCircleSolver(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):stereo_cam_ptr(stereo_cam_ptr_){}

    void getPossibleCircles(const cv::Mat& left_edge, const cv::Mat& right_edge, std::vector<Circle3D>& circles);
    // main function of this class


private:
    void getPossibleEllipse(const cv::Mat &edge, std::vector<Eigen::Matrix3d>& ellipses); //helper function
    void getCircles(const std::vector<Eigen::Matrix3d>& left_possible_ellipses, const std::vector<Eigen::Matrix3d>& right_possible_ellipses, std::vector<Circle3D>& circles);//helper function
};


#endif //__ROUGH_CIRCLE_SOLVER_H__