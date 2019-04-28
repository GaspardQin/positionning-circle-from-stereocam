#ifndef __CIRCLE_MODEL_H__
#define __CIRCLE_MODEL_H__

#include <Eigen/Dense>
#include <type_traits>
typedef Eigen::Matrix<double,6,1> EigenVector6d;
typedef Eigen::Matrix<double,7,1> EigenVector7d;
typedef Eigen::Matrix<double,8,1> EigenVector8d;

class CirclePlane{
public:
    Eigen::Vector4d center; //circle's center 
    Eigen::Vector4d plane; //  circle's plane (plane * x = 0)
    CirclePlane(){}
    CirclePlane(Eigen::Vector4d& circle_center, Eigen::Vector4d& circle_plane){
        center = circle_center;
        plane = circle_plane;
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
        transform.block<3,1>(0,3) = center.head(3);
    }
    void setTransform(const Eigen::Matrix4d& transform){
        Eigen::Vector4d normal;
        normal << 0,0,1,1; //z-axis
        plane = transform * normal;
        center = transform.block<4,1>(0,3);
    }
    virtual EigenVector6d getParams() const{
        // params: x, y, z, a, b, c, radius
        EigenVector6d param;
        Eigen::Matrix4d transform;
        getTransformMatrixToOrigin(transform);
        Eigen::Vector3d euler = transform.block<3,3>(0,0).eulerAngles(0,1,2);
        param.head<3>() = transform.block<3,1>(0,3);
        param.segment(3,3) = euler;
        return param;
    }
    virtual void setParams(const EigenVector6d & params){
        Eigen::Matrix4d transform;
        Eigen::Matrix3d rot = Eigen::AngleAxisd(params(3), Eigen::Vector3d::UnitX()).toRotationMatrix()
          * Eigen::AngleAxisd(params(4), Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(params(5), Eigen::Vector3d::UnitZ()).toRotationMatrix();

        transform.block<3,3>(0,0) = rot;
        transform.block<3,1>(0,3) = params.segment(0,3);
        setTransform(transform);
    }
    static int numParams(){
        return 6;
    }
     typedef std::integral_constant<int, 6> num_params;
};


class Circle3D:public CirclePlane{
public:
    double radius; //radius of the circle
    double score; // the quality of the detected circle
    Eigen::Matrix4d transform;
    Circle3D():CirclePlane(){
        radius = 0;
        score = 0;
    }
    Circle3D(Eigen::Vector4d& circle_center, Eigen::Vector4d& circle_plane, double circle_radius):CirclePlane(circle_center, circle_plane){
        radius = circle_radius;
    }
    EigenVector7d getParams(){
        // param: x, y, z, a, b, c, radius
        EigenVector7d param;
        param.head<6>() = ((CirclePlane*)this)->getParams();
        param(6) = radius;
        return param;
    }
    void setParams(const EigenVector7d & param){
        EigenVector6d param_plane = param.head<6>();
        ((CirclePlane*)this)->setParams(param_plane);
        radius = param(6);
    }
    static int numParams(){
        return 7;
    }
     typedef std::integral_constant<int, 7> num_params;

};

class ConcentricCircles3D:public CirclePlane{
public:
    double radius_inner;
    double radius_outer;
    ConcentricCircles3D():CirclePlane(){
        radius_inner = 0.0;
        radius_outer = 0.0;
    }
    ConcentricCircles3D(Eigen::Vector4d& circle_center, Eigen::Vector4d& circle_plane, double circle_radius_inner, double circle_radius_outer)
    :CirclePlane(circle_center, circle_plane)
    {
        radius_inner = circle_radius_inner;
        radius_outer = circle_radius_outer;
    }
    ConcentricCircles3D(Circle3D& a, Circle3D& b){
        center = (a.center + b.center)/2.0;
        plane = (a.plane + b.plane)/2.0;
        radius_inner = a.radius;
        radius_outer = b.radius;
    }
    void splitToCircles(Circle3D circles[2]) const{
        circles[0].center = center;
        circles[0].radius = radius_inner;
        circles[0].plane = plane;

        circles[1].center = center;
        circles[1].radius = radius_outer;
        circles[1].plane = plane;
    }
    EigenVector8d getParams(){
        // params: x, y, z, a, b, c, radius_inner, radius_outer
        EigenVector8d params;
        params.head<6>() = ((CirclePlane*)this)->getParams();
        params(6) = radius_inner;
        params(7) = radius_outer;
        return params;
    }
    void setParams(const EigenVector8d & params){
        EigenVector6d params_plane = params.head<6>();
        ((CirclePlane*)this)->setParams(params_plane);
        radius_inner = params(6);
        radius_outer = params(7);
    }
    static int numParams(){
        return 8;
    }
     typedef std::integral_constant<int, 8> num_params;

};
#endif