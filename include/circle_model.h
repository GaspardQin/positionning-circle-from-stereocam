#ifndef __CIRCLE_MODEL_H__
#define __CIRCLE_MODEL_H__

#include <Eigen/Dense>

class Circle3D{
public:
    Eigen::Vector4d center; //circle's center 
    Eigen::Vector4d plane; //  circle's plane (plane * x = 0)
    double radius; //radius of the circle
    double score; // the quality of the detected circle
    Circle3D(){
        radius = 0;
        score = 0;
    }
    Circle3D(Eigen::Vector4d& circle_center, Eigen::Vector4d& circle_plane, double circle_radius){
        center = circle_center;
        plane = circle_plane;
        radius = circle_radius;
    }
    void getTransformMatrixToOrigin(Eigen::Matrix4d& transform) const{
        /** get transform matrix (4x4) from circle to origin xy plane
         *  X_in_origin_coord = transform * X_in_circles_coord
         *  Attention: the rotation about z-axis is not controlled, thus the transform matrix is not unique.
         *  @ param transform : the 4x4 transform matrix from circle plane to x-o-y plane. 
         *  This function implements a method proposed by https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector 
        */ 
        Eigen::Vector3d normal; 
        normal = plane.head<3>().normalized();
        Eigen::Matrix3d R; 
        double nx_ny_norm = std::sqrt(normal(0) * normal(0) + normal(1) * normal(1));
        R(0,0) = normal(1) / nx_ny_norm;
        R(0,1) = -normal(0) / nx_ny_norm;
        R(0,2) = 0.0;
        R(1,0) = normal(0) * normal(2) / nx_ny_norm;
        R(1,1) = normal(1) * normal(2) / nx_ny_norm;
        R(1,2) = - nx_ny_norm;
        R(2,0) = normal(0);
        R(2,1) = normal(1);
        R(2,2) = normal(2);

        transform.setIdentity();
        transform.block<3,3>(0,0) = R;
        transform.block<3,1>(0,3) = -center.head(3);
    }

};

#endif