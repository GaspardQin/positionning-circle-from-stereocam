#ifndef __BASE_CIRCLE_SOLVER_H__
#define __BASE_CIRCLE_SOLVER_H__
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <bitset>
class FittedEllipse{
public:
    int id;
    cv::RotatedRect box;
    double theta_init; // 0 to 2*pi, first theta value of an arc
    double theta_length; // 0 to 2*pi, length of an arc
    std::vector<int> possible_other_parts_id;

    FittedEllipse(int ellipse_id, cv::RotatedRect& ellipse_box)
    :id(ellipse_id), box(ellipse_box)
    {
    }
    void cover(const std::vector<cv::Point>& points){
        double angle = std::atan2(points[0].y-box.center.y, points[0].x-box.center.x) + M_PI; // angle is between 0 to 2*pi
        theta_init = angle;
        theta_length = 0;
        for(int i=1; i<points.size(); i++){
            angle = std::atan2(points[i].y-box.center.y, points[i].x-box.center.x) + M_PI; // angle is between 0 to 2*pi
            double residual = normalizeNegativePiToPositivePi(angle - theta_init);
            if( residual < 0){
                theta_init = normalize(theta_init + residual);
                theta_length = theta_length - residual;
            }    
            else if(residual > theta_length){
                theta_length += residual;
            }
        }
    }
    bool isInArc(const double angle, double threshold=0.0) const{
        if(theta_length >= 2 * M_PI - threshold) return true;
        double residual = normalizeNegativePiToPositivePi(angle - theta_init);
        
        if(residual < threshold) return false;
        if(residual > theta_length - threshold) return false;
        return true;
    }
    bool hasOverlap(const FittedEllipse& another_ellipse) const{
        if(isInArc(another_ellipse.theta_init, M_PI/12.0)) return true;
        if(another_ellipse.isInArc(theta_init, M_PI/12.0)) return true;
        return false;
    }
    static double normalize(double angle){
        // normalize to 0-2pi
        if(angle < 0){
            angle += angle + 2*M_PI;
        }
        else if(angle > 2*M_PI){
            angle -= 2*M_PI;
        }
        return angle;
    }
    static double normalizeNegativePiToPositivePi(double angle){
        // normalize to -pi to pi
        if(angle < -M_PI){
            angle += angle + 2*M_PI;
        }
        else if (angle > M_PI){
            angle -= 2*M_PI;
        }
        return angle;
    }
};

class BaseCircleSolver{
protected:

    #define LEFT_CAMERA 0
    #define RIGHT_CAMERA 1
    
    std::shared_ptr<StereoCameraModel> stereo_cam_ptr;

    void inline getEllipseParams(const cv::RotatedRect& ellipse_cv_form, double& a, double&b ,double &x_c, double &y_c, double& sin_theta, double& cos_theta){
        a = ellipse_cv_form.size.width /2.0;
        b = ellipse_cv_form.size.height /2.0;
        x_c = ellipse_cv_form.center.x;
        y_c = ellipse_cv_form.center.y;
        sin_theta = std::sin(ellipse_cv_form.angle * M_PI / 180.0); // rotatedrect's angle is presented in degree
        cos_theta = std::cos(ellipse_cv_form.angle * M_PI / 180.0);
    }
    double getAlgebraDistance(const cv::RotatedRect& ellipse, double x, double y){
        double a, b, x_c, y_c, sin_theta, cos_theta;
        getEllipseParams(ellipse, a, b, x_c, y_c, sin_theta, cos_theta);
        double equation = -1.0 + std::pow(-(x-x_c)*sin_theta +(y-y_c)*cos_theta, 2)/b/b 
                            + std::pow((x-x_c)*cos_theta + (y-y_c)*sin_theta, 2)/a/a;
        return fabs(equation);
        
    }

    double getAlgebraDistance(const Eigen::Matrix3d & ellipse_quad_form, double x, double y){
        Eigen::Vector3d X;
        X << x,y,1.0;
        return fabs(X.transpose() * ellipse_quad_form * X);
    }
    template <class T>
    double getAlgebraDistance(const T & ellipse, std::vector<cv::Point> points){
        double distance = 0;
        for(int i=0; i<points.size();i++){
            distance += pow(getAlgebraDistance(ellipse, points[i].x, points[i].y),2);
        }
        return distance;
    }
    void translateEllipse(const cv::RotatedRect &ellipse_cv_form, Eigen::Matrix3d &ellipse_quad_form)
    {
        double a, b, x_c, y_c, sin_theta, cos_theta;
        getEllipseParams(ellipse_cv_form, a, b, x_c, y_c, sin_theta, cos_theta);

        double A = (a*a*sin_theta*sin_theta + b*b*cos_theta*cos_theta)/(a*a*b*b);
        double B = (-2*a*a*sin_theta*cos_theta + 2*b*b*sin_theta*cos_theta)/(a*a*b*b);
        double C = (a*a*cos_theta*cos_theta + b*b*sin_theta*sin_theta)/(a*a*b*b);
        double D = (-2*a*a*x_c*sin_theta*sin_theta + 2*a*a*y_c*sin_theta*cos_theta - 2*b*b*x_c*cos_theta*cos_theta - 2*b*b*y_c*sin_theta*cos_theta)/(a*a*b*b);
        double E = (2*a*a*x_c*sin_theta*cos_theta - 2*a*a*y_c*cos_theta*cos_theta - 2*b*b*x_c*sin_theta*cos_theta - 2*b*b*y_c*sin_theta*sin_theta)/(a*a*b*b);
        double F = (-a*a*b*b + a*a*x_c*x_c*sin_theta*sin_theta - 2*a*a*x_c*y_c*sin_theta*cos_theta + a*a*y_c*y_c*cos_theta*cos_theta + b*b*x_c*x_c*cos_theta*cos_theta + 2*b*b*x_c*y_c*sin_theta*cos_theta + b*b*y_c*y_c*sin_theta*sin_theta)/(a*a*b*b);

        ellipse_quad_form << A/F, B / 2.0/F, D / 2.0/F, B / 2.0/F, C/F, E / 2.0/F, D / 2.0/F, E / 2.0/F, 1.0;
        /*
        //verify the translation
        double alpha = 0.0;

        for(; alpha < 2*M_PI; alpha += 0.1){// in rad
            double x = a * cos(alpha) * cos_theta - b * sin(alpha) * sin_theta + x_c;
            double y = a * cos(alpha) * sin_theta + b * sin(alpha) * cos_theta + y_c;
            Eigen::Vector3d X;
            X << x,y,1;
            double result = X.transpose() * ellipse_quad_form * X;
            std::cout<<"verification result: "<<result<<std::endl;
        }
        */
    }



public:
    BaseCircleSolver(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):stereo_cam_ptr(stereo_cam_ptr_){};
};


#endif