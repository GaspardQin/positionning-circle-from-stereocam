#include "rough_circle_solver.h"

void RoughCircleSolver::getPossibleEllipse(const cv::Mat &edge, std::vector<Eigen::Matrix3d>& ellipses_vec){
    /**
    * find ellipse from a single image, the ellipses are presented in the pixel coordinate.
    * @param edge an edge image (binary)
    * @param ellipses the possible ellipse in the image, presented in the pixel coordinate. 
    * 
    * The ellipse is presented in the form of quatratic, for example, if C present a ellipse here,
    * thus X^t * C * X =0, where X = [x, y, 1]^t , see https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections 
    */
    #ifdef DEBUG
    cv::Mat show_img(edge.size(), CV_8UC3);
    #endif

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(edge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    const double flattening_treshold = 3.0; // the threshold of flattening of the ellipse
    const double min_area_threshold = 50; // the threshold of minimum area of the ellipse
    for (int i = 0; i < contours.size(); i++)
    {

		int count = contours[i].size();
		if( count < 6 ) continue; //need at least 6 points to fit an ellipse

        RotatedRect box = fitEllipse(contours[i]);
        if( std::max(box.size.width, box.size.height) / std::min(box.size.width, box.size.height) > flattening_treshold)
            continue;
        
        if(box.size.width * box.size.height < min_area_threshold) continue;


        #ifdef DEBUG
        drawContours(show_img, contours, (int)i, Scalar::all(255), 1, 8);
        ellipse(show_img, box, Scalar(0,0,255), 1, CV_AA);
        #endif //DEBUG

        double a = box.size.width;
        double b = box.size.height;
        double x_center = box.center.x;
        double y_center = box.center_y;
        double A = 1/(a*a);
        double B = 0;
        double C = 1/(b*b);
        double D = -2*x_center/(a*a);
        double E = -2*y_center/(b*b);
        double F = x_center * x_center / (a * a) + y_center * y_center /(b*b);
        Eigen::Matrix3d ellipse << A, B/2.0, D/2.0, B/2.0, C, E/2.0, D/2.0, E/2.0, F;
        ellipses_vec.push_back(ellipse);
    }
}

void RoughCircleSolver::computeI2I3I4(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B, double& I2, double& I3, double& I4){
    /** compute I_2, I_3, I_4, which are mentioned in the paper : "Long Quan. Conic Reconstruction and Correspondence from Two Views"
     * to simplify the programming, we choose to calculate I_2, I_3, I_4 numerically
     * by setting lambda = 1, -1, 2
    */
    double y_positive_one = (A + B).determinant();
    double y_negative_one = (A - B).determinant();
    double y_positive_two = (A + B * 2).determinant();
    I_2 = (y_positive_two - y_negative_one)/6.0 - y_positive_one/2.0;
    I_3 = (y_positive_one + y_negative_one)/2.0;
    I_4 = y_positive_one - y_negative_one/3.0 - y_positive_two/6.0;
}
void RoughCircleSolver::computePointsInPlane(const double& u, const double& v, const Eigen::Vector4d& plane, Eigen::Vector4d& point){
    /** compute the point in 3d which is intersected by a line and a plane.
     * the line is passed the focal origin, and projected to the left image plane, with a coordinate (u, v) (in pixel)
     * 
    */ 
    double fx = stereo_cam_ptr->left.projectionEigenMatrix()(0,0);
    double fy = stereo_cam_ptr->left.projectionEigenMatrix()(1,1);

    double m = u/fx; // should be equal to x/z
    double n = v/fy; // should be equal to y/z

    point(2) = - plane(3) / (plane(0)*m + plane(1)*n + plane(2));
    point(1) = point(2) * n;
    point(0) = point(2) * m;
    point(3) = 1;
}

void RoughCircleSolver::getCircles(const std::vector<Eigen::Matrix3d>& left_possible_ellipses, 
                                   const std::vector<Eigen::Matrix3d>& right_possible_ellipses, 
                                   std::vector<Circle3D>& circles){
    /**
     * pairs the ellipses from left/right camera and gives the circles in 3D coordinate.
     * 
    */
    if(left_possible_ellipses.size() == 0 || right_possible_ellipses.size() == 0) return;

    Eigen::Matrix4d A, B; //X^t * A * X =0, same for B

    std::vector<CirclePair> circle_pairs;
    double error_threshold = 5.0;

    Eigen::Matrix<double, 3, 4> left_projection, right_projection;
    left_projection = stereo_cam_ptr->left.projectionEigenMatrix();
    right_projection = stereo_cam_ptr->right.projectionEigenMatrix();
    
    for(int i=0; i<left_possible_ellipses.size(); i++){
        A = left_projection.transpose() * left_possible_ellipses[i] * left_projection;
        double min_error = std::numeric_limits<double>::max();
        int min_index = -1;
        for( int j=0; j<right_possible_ellipses.size(); j++){
            B = right_projection.transpose() * right_possible_ellipses[j] * right_projection;
            double I_2, I3, I4;
            computeI2I3I4(A, B, I_2, I_3, I_4);
            double curr_error = fabs(I_3 * I_3 - 4* I_2 * I_4);
            if(curr_error < min_error){
                min_error = curr_error;
                min_index = j;
            }
        }
        if(min_error < error_threshold){
            circle_pairs.push_back(std::make_tuple(i, min_index, min_error));
        }    
    }

    if(circle_pairs.empty()) return;

    // calculate the circle's pose
    double base_line; // in mm
    base_line = - right_projection(0,3) / right_projection(0,1);
    Eigen::Vector4d right_camera_origin << base_line, 0 ,0, 1;

    for(int i=0; i<circle_pairs.size(); i++){
        double I_2, I3, I4;
        int left_id = std::get<0>(circle_pairs[i]);
        int right_id = std::get<1>(circle_pairs[i]);
        A = left_projection.transpose() * left_possible_ellipses[left_id] * left_projection;
        B = right_projection.transpose() * right_possible_ellipses[right_id] * right_projection;
        computeI2I3I4(A, B, I_2, I_3, I_4);
    
        double lambda = -I_3/(2*I_2);
        Eigen::Matrix4d C = A + lambda * B;
        SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(C);
        if (eigensolver.info() != Success){
            std::cout<<"failed to get the eigen values/vectors"<<std::endl;
            continue;
        };
        Eigen::Vector4d eigen_values = eigensolver.eigenvalues(); //increasing order
        int eigen_non_zero_index[4]; int id_non_zero_index = 0;
    
        for(int eigen_i=0; eigen_i ++; eigen_i < 4){
            if(fabs(eigen_values[eigen_i]) > 0.00001){
                eigen_non_zero_index[id_non_zero_index] = eigen_i;
                id_non_zero_index ++;
            }
        }
        if(id_non_zero_index >=2){
            std::cout<<"has more than 2 non zeros eigen values, seems something wrong..."<<std::endl;
        }
        if(eigen_values[eigen_non_zero_index[0]] * eigen_values[eigen_non_zero_index[1]] >= 0) continue;
        
        // find the plane of the circle
        Eigen::Vector4d p = std::sqrt(-eigen_values[eigen_non_zero_index[0]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[0])
               + std::sqrt(eigen_values[eigen_non_zero_index[1]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[1]);
        if(p.cross(p-right_camera_origin) < 0){
            p = std::sqrt(-eigen_values[eigen_non_zero_index[0]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[0])
               - std::sqrt(eigen_values[eigen_non_zero_index[1]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[1]);
        }
        
        // find the center of the circle
        // Here, we assume the center of the circle is also the center of the ellipse in the image.
        Eigen::Matrix3d ellipse_quadratic = left_possible_ellipses[left_id];
        double center_x = - ellipse_quadratic(0,2) / ellipse_quadratic(0,0);
        double center_y = - ellipse_quadratic(1,2) / ellipse_quadratic(0,0);
        double a = 1.0/sqrt(ellipse_quadratic(0,0));
        double p_x = center_x + a; // point on the ellipse
        double p_y = center_y;

        Eigen::Vector4d center_point_3d, circle_point_3d;
        computePointsInPlane(center_x, center_y, p, center_point_3d);
        computePointsInPlane(p_x, p_y, p, circle_point_3d);
        
        double r = (center_point_3d - circle_point_3d).norm();
        circles.push_back(Circle3D(center_point_3d, p, r);
    }


}