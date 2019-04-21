#ifndef __CIRCLE_MODEL_H__
#define __CIRCLE_MODEL_H__


class Circle3D{
public:
    Eigen::Vector4d center; //circle's center 
    Eigen::Vector4d plane; //  circle's plane
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
};

#endif