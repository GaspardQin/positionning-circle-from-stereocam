#ifndef __PRECISE_CIRCLE_SOLVER_H__
#define __PRECISE_CIRCLE_SOLVER_H__
#include "base_circle_solver.h"
#include "circle_model.h"
#include "camera_model.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

#define INNER_CIRCLE 0
#define OUTER_CIRCLE 1
template <class CircleT>
class PreciseCircleSolverBaseImpl:public BaseCircleSolver{
private:
    cv::Mat edge_left, edge_right;
    CircleT init_circle;
    cv::flann::Index kdtree_left, kdtree_right;
    int n_point;
    
public:
    std::vector<cv::Point2f> non_zeros_coordinates_left_float;   // output, locations of non-zero pixels
    std::vector<cv::Point2f> non_zeros_coordinates_right_float;   // output, locations of non-zero pixels
    Eigen::Matrix3d init_rot;

    PreciseCircleSolverBaseImpl(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):BaseCircleSolver(stereo_cam_ptr_){};
    void translateParam(const double a, const double b, const double c, Eigen::Matrix3d& rot_mat){
        rot_mat = Eigen::AngleAxisd(a, Eigen::Vector3d::UnitX()).toRotationMatrix()
          * Eigen::AngleAxisd(b, Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(c, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    void translateParam(const Eigen::Matrix3d& rot_mat, double& a, double &b, double& c){
        Eigen::Vector3d euler_angle = rot_mat.eulerAngles(0,1,2);
        a = euler_angle(0);
        b = euler_angle(0);
        c = euler_angle(0);
    }
    void init(CircleT& init_circle_input, cv::Mat& edge_left_input, cv::Mat& edge_right_input, int n_point_input)
   {
        edge_left = edge_left_input;
        edge_right = edge_right_input;
        init_circle = init_circle_input; 

        n_point = n_point_input;

        std::vector<cv::Point> non_zeros_coordinates_int_left, non_zeros_coordinates_int_right;
        findNonZero(edge_left, non_zeros_coordinates_int_left);
        
        non_zeros_coordinates_left_float.insert(non_zeros_coordinates_left_float.end(), non_zeros_coordinates_int_left.begin(), non_zeros_coordinates_int_left.end());
        kdtree_left.build(cv::Mat(non_zeros_coordinates_left_float).reshape(1), cv::flann::KDTreeIndexParams(4),cvflann::FLANN_DIST_L2);


        findNonZero(edge_right, non_zeros_coordinates_int_right);
        non_zeros_coordinates_right_float.insert(non_zeros_coordinates_right_float.end(), non_zeros_coordinates_int_right.begin(), non_zeros_coordinates_int_right.end());
        kdtree_right.build(cv::Mat(non_zeros_coordinates_right_float).reshape(1), cv::flann::KDTreeIndexParams(4),cvflann::FLANN_DIST_L2);

    }


    void solve(CircleT& result, double robust_threshold){
        ceres::Problem problem;
        ceres::LossFunction* loss = new ceres::CauchyLoss(robust_threshold);
        const int num_points = n_point;
        double incr_angle = M_PI * 2 / num_points;
        Eigen::VectorXd params = init_circle.getParams();
        translateParam(params[3], params[4], params[5], init_rot);
        params(3) = 0.0;
        params(4) = 0.0;
        params(5) = 0.0;
        double x[8];

        for(int i=0; i<8; i++){
            x[i] = params(i);
        }
        for(int i=0; i<num_points; i++){
            double theta = i * incr_angle;
            ceres::CostFunction* cost_function =
                new ceres::NumericDiffCostFunction<EdgeMinimalDistance, ceres::CENTRAL, 1, 8>(new EdgeMinimalDistance(theta, LEFT_CAMERA, INNER_CIRCLE, this));
            problem.AddResidualBlock(cost_function, loss, x);

            cost_function =
                new ceres::NumericDiffCostFunction<EdgeMinimalDistance, ceres::CENTRAL, 1, 8>(new EdgeMinimalDistance(theta, LEFT_CAMERA, OUTER_CIRCLE, this));
            problem.AddResidualBlock(cost_function, loss, x);

            cost_function =
                new ceres::NumericDiffCostFunction<EdgeMinimalDistance, ceres::CENTRAL, 1, 8>(new EdgeMinimalDistance(theta, RIGHT_CAMERA, INNER_CIRCLE, this));
            problem.AddResidualBlock(cost_function, loss, x);

            cost_function =
                new ceres::NumericDiffCostFunction<EdgeMinimalDistance, ceres::CENTRAL, 1, 8>(new EdgeMinimalDistance(theta, RIGHT_CAMERA, OUTER_CIRCLE, this));
            problem.AddResidualBlock(cost_function, loss, x);
        }



        // Build and solve the problem.
        ceres::Solver::Options options;
        options.max_num_iterations = 500;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //std::cout << summary.FullReport() << "\n";
        std::cout << summary.BriefReport() << "\n";
        
        std::cout << "x  : " << params(0) << " -> " << x[0] << "\n";
        std::cout << "y  : " << params(1) << " -> " << x[1] << "\n";
        std::cout << "z  : " << params(2) << " -> " << x[2] << "\n";
        std::cout << "a  : " << params(3) << " -> " << x[3] << "\n";
        std::cout << "b  : " << params(4) << " -> " << x[4] << "\n";
        std::cout << "c  : " << params(5) << " -> " << x[5] << "\n";
        std::cout << "r_i: " << params(6) << " -> " << x[6] << "\n";
        std::cout << "r_o: " << params(7) << " -> " << x[7] << "\n";

        for(int i=0; i<8; i++){
            params(i) = x[i];
        }
        Eigen::Matrix3d rot_mat;
        translateParam(params(3),params(4),params(5),rot_mat);
        rot_mat = init_rot * rot_mat.eval();
        double a,b,c;
        translateParam(rot_mat, a, b, c);
        params(3) = a;
        params(4) = b;
        params(5) = c;
        result.setParams(params);
    }


    virtual void getReprojectPointLeft(const EigenVector8d& param, const Eigen::Matrix3d& init_rot, const Eigen::Vector4d& point_in_plane,  Eigen::Vector3d& project_point)=0;
    virtual void getReprojectPointRight(const EigenVector8d& param, const Eigen::Matrix3d& init_rot, const Eigen::Vector4d& point_in_plane,  Eigen::Vector3d& project_point)=0;


    struct EdgeMinimalDistance {
        EdgeMinimalDistance(double theta_input, int camera_id_input, int circle_id_input, PreciseCircleSolverBaseImpl* solver_ptr_input)
        :solver_ptr(solver_ptr_input), theta(theta_input), camera_id(camera_id_input), circle_id(circle_id_input){};
        bool operator()(const double* const x, double* residual) const {
            Eigen::Vector3d project_point;
            EigenVector8d param;
            param << x[0], x[1], x[2],x[3],x[4],x[5],x[6],x[7];
            Eigen::Vector4d point_in_plane;
            std::vector<int> indices;
            std::vector<float> dists;
            double radius = (circle_id == INNER_CIRCLE) ? x[6] : x[7];

            point_in_plane << radius * cos(theta), radius * sin(theta), 0.0, 1.0;

            if(camera_id == LEFT_CAMERA){
                solver_ptr->getReprojectPointLeft(param, solver_ptr->init_rot , point_in_plane, project_point);
                std::vector<float> flattern_project_point(2);
                flattern_project_point[0] = project_point(0);
                flattern_project_point[1] = project_point(1);
                solver_ptr->kdtree_left.knnSearch(flattern_project_point, indices, dists, 1, cv::flann::SearchParams());
            }
            else{
                solver_ptr->getReprojectPointRight(param, solver_ptr->init_rot, point_in_plane, project_point);
                std::vector<float> flattern_project_point(2);
                flattern_project_point[0] = project_point(0);
                flattern_project_point[1] = project_point(1);
                solver_ptr->kdtree_right.knnSearch(flattern_project_point, indices, dists, 1, cv::flann::SearchParams());    
            }
            
            residual[0] = std::sqrt(dists[0]);
            //residual[0] = (residual[0] < 5.0)? residual[0]: 5.0;
            if(std::isnan(residual[0])){
                std::cout<<"nan!!!"<<std::endl;
                if(camera_id == LEFT_CAMERA){
                    std::cout<<"x: "<<solver_ptr->non_zeros_coordinates_left_float[indices[0]].x<<", y: "<<solver_ptr->non_zeros_coordinates_right_float[indices[0]].y<<std::endl;
                }
            }
            return true;
        }
    private:
        double theta;
        int camera_id, circle_id;
        PreciseCircleSolverBaseImpl* solver_ptr;
    };

};




class PreciseTwoConcentricCirclesSolver: public PreciseCircleSolverBaseImpl<ConcentricCircles3D>{
private:
    void normalize(Eigen::Vector3d& v){
        v(0) /= v(2);
        v(1) /= v(2);
        v(2) = 0;
    }
public:
    PreciseTwoConcentricCirclesSolver(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):PreciseCircleSolverBaseImpl(stereo_cam_ptr_){};

    virtual void getReprojectPointLeft(const EigenVector8d& param, const Eigen::Matrix3d& init_rot, const Eigen::Vector4d& point_in_plane,  Eigen::Vector3d& project_point){

        Eigen::Matrix4d transform;
        translateParam(param, transform);
        transform.block<3,3>(0,0) = init_rot * transform.block<3,3>(0,0).eval();
        project_point = stereo_cam_ptr->left.projection_mat * transform * point_in_plane;
        normalize(project_point);
    }
    virtual void getReprojectPointRight(const EigenVector8d& param, const Eigen::Matrix3d& init_rot, const Eigen::Vector4d& point_in_plane,  Eigen::Vector3d& project_point){

        Eigen::Matrix4d transform;
        translateParam(param, transform);
        transform.block<3,3>(0,0) = init_rot * transform.block<3,3>(0,0).eval();
        project_point = stereo_cam_ptr->right.projection_mat * transform * point_in_plane;
        normalize(project_point);
    }

    
    void translateParam(const EigenVector8d& param, Eigen::Matrix4d& transform){
        Eigen::Matrix3d rot = Eigen::AngleAxisd(param(3), Eigen::Vector3d::UnitX()).toRotationMatrix()
          * Eigen::AngleAxisd(param(4), Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(param(5), Eigen::Vector3d::UnitZ()).toRotationMatrix();

        transform.block<3,3>(0,0) = rot;
        transform.block<3,1>(0,3) = param.segment(0,3);
    }
};


#endif