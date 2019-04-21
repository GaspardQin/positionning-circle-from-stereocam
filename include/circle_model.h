#ifndef __CIRCLE_MODEL_H__
#define __CIRCLE_MODEL_H__


class Circle3D{
public:
    double center[3]; //circle's center 
    double normal[3]; // normal vector of circle's plane
    double radius; //radius of the circle
    double score; // the quality of the detected circle
    Circle3D(){
        for(int i=0; i<3; i++){
            center[i] = 0.0;
            normal[i] = 0.0;
            radius = 0.0;
        }
    }
    Circle3D(double x, double y, double z, double normal_x, double normal_y, double normal_z, double circle_radius){
        center[0] = x;
        center[1] = y;
        center[2] = z;
        normal[0] = normal_x;
        normal[1] = normal_y;
        normal[2] = normal_z;
        radius = circle_radius;
    }
    void setCenter(double x, double y, double z){
        center[0] = x;
        center[1] = y;
        center[2] = z;
    }
    void setNormalVector(double normal_x, double normal_y, double normal_z){
        normal[0] = normal_x;
        normal[1] = normal_y;
        normal[2] = normal_z;
    }
};

#endif