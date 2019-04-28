#ifndef __PRECISE_CIRCLE_SOLVER_H__
#define __PRECISE_CIRCLE_SOLVER_H__
#include "base_circle_solver.h"
#include "circle_model.h"
#include "camera_model.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

template <class CircleT>
class PreciseCircleSolverBaseImpl:public BaseCircleSolver{
private:
    cv::Mat edge_left, edge_right;
    CircleT init_circle;
    cv::flann::Index kdtree_left, kdtree_right;
    int n_point;
public:
    PreciseCircleSolverBaseImpl(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr_):BaseCircleSolver(stereo_cam_ptr_){};
    
    void init(CircleT& init_circle_input, cv::Mat& edge_left_input, cv::Mat& edge_right_input, int n_point_input)
   {
        edge_left = edge_left_input;
        edge_right = edge_right_input;
        init_circle = init_circle_input; 
        n_point = n_point_input;


        std::vector<cv::Point> non_zeros_coordinates_int;   // output, locations of non-zero pixels
        findNonZero(edge_left, non_zeros_coordinates_int);
        
        std::vector<cv::Point2f> non_zeros_coordinates_left_float(non_zeros_coordinates_int.begin(), non_zeros_coordinates_int.end());
        kdtree_left.build(cv::Mat(non_zeros_coordinates_left_float).reshape(1), cv::flann::KDTreeIndexParams(4),cvflann::FLANN_DIST_EUCLIDEAN);

        non_zeros_coordinates_int.clear();
        findNonZero(edge_right, non_zeros_coordinates_int);
        std::vector<cv::Point2f> non_zeros_coordinates_right_float(non_zeros_coordinates_int.begin(), non_zeros_coordinates_int.end());
        kdtree_right.build(cv::Mat(non_zeros_coordinates_right_float).reshape(1), cv::flann::KDTreeIndexParams(4),cvflann::FLANN_DIST_EUCLIDEAN);

    }


    void solve(CircleT& result){

        LMFunctor<ConcentricCircles3D::num_params::value> functor(n_point, this);

        Eigen::LevenbergMarquardt<LMFunctor<ConcentricCircles3D::num_params::value>,double> lm(functor);
        Eigen::VectorXd params = init_circle.getParams();
        lm.minimize(params);
        result.setParams(params);
    }


    virtual void getReprojectPoints(const Eigen::VectorXd &param, std::vector<float>& left_reproject_points, 
                                    std::vector<float>& right_reproject_points, const int n_point)=0;
    virtual void getReprojectPoint(const EigenVector8d& param, const Eigen::Vector3d& point_in_plane,  Eigen::Vector3d& left_project_point,  Eigen::Vector3d& right_project_point)=0;
    template<int NX = Eigen::Dynamic,int NY = Eigen::Dynamic>
    struct LMFunctor //used by LM algorithm proposed by Eigen
    {	
        PreciseCircleSolverBaseImpl* solver_ptr;
        typedef double Scalar;
        enum {
            InputsAtCompileTime = NX,
            ValuesAtCompileTime = NY
        };
        typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
        typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
        typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,Eigen::Dynamic> JacobianType;

        std::vector<double> weight;
        LMFunctor(int num_data, PreciseCircleSolverBaseImpl* solver_ptr_input):weight(num_data, 1.0), solver_ptr(solver_ptr_input){
            m = num_data * 2;
            n = NX;
        }
        int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
        {
            /** Compute error
             *  'x' has dimensions n x 1
             * It contains the current estimates for the parameters.
             * 'fvec' has dimensions 1 x 1
             * It will contain the summed error for all data point.
             */
            std::vector<float> reproject_points_left, reproject_points_right;
            std::cout<<"x: "<<x<<std::endl;
            solver_ptr->getReprojectPoints(x, reproject_points_left, reproject_points_right, m/2);
            std::vector<int> indices_left, indices_right;
            std::vector<float> dists, dists_right;
            
            solver_ptr->kdtree_left.knnSearch(reproject_points_left, indices_left, dists, 1, cv::flann::SearchParams());
            solver_ptr->kdtree_right.knnSearch(reproject_points_right, indices_right, dists_right, 1, cv::flann::SearchParams());
            dists.insert(dists.end(), dists_right.begin(), dists_right.end());

            // M-estimator, where the weight function uses "Fair" with c=1.3998
            double error = 0;
            for(int i=0; i<m; i++){              
                error += weight[i] * dists[i] * dists[i]; 
                
            }
            /*
            if(update_weight){
               for(int i=0; i<m; i++){
                    weight[i] = 1/(1+fabs(dists[i])/1.3998);  
                }
            }
            */
            fvec.resize(1,1);
            fvec(0) = error;
            return 0;
        }
        
        double dfPartialNumeric(const Eigen::VectorXd &x, const Eigen::VectorXd& fvec, const Eigen::VectorXd& direction, double epsilon) const{
            std::cout<<"size: "<<x.size()<<" "<<direction.size()<<std::endl;
            Eigen::VectorXd x_1 = x + direction * epsilon; 
            Eigen::VectorXd fvec_1;
            this->operator()(x_1,fvec_1);
            return ((fvec_1- fvec) / epsilon)(0);
        }
        // Compute the jacobian of the errors
        
        int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
        {
            fjac.resize(1, n);
            Eigen::VectorXd direction;
            direction.resize(n,1);
            Eigen::VectorXd fvec;
            this->operator()(x, fvec);
            for(int i=0; i<n; i++){
                direction.setZero();
                direction(i) = 1.0;
                fjac(0,i) = dfPartialNumeric(x, fvec, direction, 0.001);
            }
            return 0;
        }
        

        // Number of data points, i.e. values.
        int m;

        // Returns 'm', the number of values.
        int values() const { return m; }

        // The number of parameters, i.e. inputs.
        int n;

        // Returns 'n', the number of inputs.
        int inputs() const { return n; }

    };




};

struct EdgeMinimalDistance {
    EdgeMinimalDistance(double x, double y, PreciseCircleSolverBaseImpl* solver_ptr_input):solver_ptr(solver_ptr_input){
        point_in_plane<<x, y, 1.0;
    };
    bool operator()(const double* const x, double* residual) const {
        residual[0] = 10.0 - x[0];
        Eigen::Vector3d left_project_point,  Eigen::Vector3d right_project_point;
        EigenVector8d param;
        param << x[0], x[1], x[2],x[3],x[4],x[5],x[6],x[7];
        solver_ptr->getReprojectPoint(param, point_in_plane, left_project_point, right_project_point);
        std::vector<int> indices_left, indices_right;
        std::vector<float> dists_left, dists_right;

        std::vector<float> flattern_left_project_point(2);
        std::vector<float> flattern_right_project_point(2);

        flattern_left_project_point.push_back(left_project_point(0));
        flattern_left_project_point.push_back(left_project_point(1));
        flattern_right_project_point.push_back(right_project_point(0));
        flattern_right_project_point.push_back(right_project_point(0));

        solver_ptr->kdtree_left.knnSearch(flattern_left_project_point, indices_left, dists_left, 1, cv::flann::SearchParams());
        solver_ptr->kdtree_right.knnSearch(flattern_right_project_point, indices_right, dists_right, 1, cv::flann::SearchParams());
        
        residual[0] = dists_left[0] + dists_right[0];
        return true;
    }
private:
    Eigen::Vector3d point_in_plane;
    PreciseCircleSolverBaseImpl* solver_ptr;
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

    virtual void getReprojectPoint(const EigenVector8d& param, const Eigen::Vector3d& point_in_plane,  Eigen::Vector3d& left_project_point,  Eigen::Vector3d& right_project_point){

        Eigen::Matrix4d transform;
        Eigen::Matrix3d rot = Eigen::AngleAxisd(param(3), Eigen::Vector3d::UnitX()).toRotationMatrix()
          * Eigen::AngleAxisd(param(4), Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(param(5), Eigen::Vector3d::UnitZ()).toRotationMatrix();

        transform.block<3,3>(0,0) = rot;
        transform.block<3,1>(0,3) = param.segment(0,3);
        double radius_inner = param(6);
        double radius_outer = param(7);

        left_project_point = stereo_cam_ptr->left.projection_mat * transform * point_in_plane;
        right_project_point = stereo_cam_ptr->right.projection_mat * transform * point_in_plane;
        normalize(left_projection_point);
        normalize(right_projection_point);
    }

    virtual void getReprojectPoints(const Eigen::VectorXd &param, std::vector<float>& left_reproject_points, 
                                    std::vector<float>& right_reproject_points, const int n_point){

        Eigen::Matrix4d transform;
        Eigen::Matrix3d rot = Eigen::AngleAxisd(param(3), Eigen::Vector3d::UnitX()).toRotationMatrix()
          * Eigen::AngleAxisd(param(4), Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(param(5), Eigen::Vector3d::UnitZ()).toRotationMatrix();

        transform.block<3,3>(0,0) = rot;
        transform.block<3,1>(0,3) = param.segment(0,3);
        double radius_inner = param(6);
        double radius_outer = param(7);
        // inner point
        double interval_angle = 2 * M_PI / n_point;
        for(int i=0; i<n_point; i++){
            Eigen::Vector4d point;
            point(0) = cos(i * interval_angle)*radius_inner;
            point(1) = sin(i * interval_angle)*radius_inner;
            point(2) = 0;
            point(3) = 1.0;
            Eigen::Vector3d left_projection_point = stereo_cam_ptr->left.projection_mat * transform * point;
            Eigen::Vector3d right_projection_point = stereo_cam_ptr->right.projection_mat * transform * point;

            normalize(left_projection_point);
            normalize(right_projection_point);
            left_reproject_points.push_back(left_projection_point(0));
            left_reproject_points.push_back(left_projection_point(1));
            right_reproject_points.push_back(right_projection_point(0));
            right_reproject_points.push_back(right_projection_point(1));
        }
        // outer point
        for(int i=0; i<n_point; i++){
            Eigen::Vector4d point;
            point(0) = cos(i * interval_angle)*radius_outer;
            point(1) = sin(i * interval_angle)*radius_outer;
            point(2) = 0;
            point(3) = 1.0;
            Eigen::Vector3d left_projection_point = stereo_cam_ptr->left.projection_mat * transform * point;
            Eigen::Vector3d right_projection_point = stereo_cam_ptr->right.projection_mat * transform * point;

            normalize(left_projection_point);
            normalize(right_projection_point);
            left_reproject_points.push_back(left_projection_point(0));
            left_reproject_points.push_back(left_projection_point(1));
            right_reproject_points.push_back(right_projection_point(0));
            right_reproject_points.push_back(right_projection_point(1));
        }
    }
};


#endif