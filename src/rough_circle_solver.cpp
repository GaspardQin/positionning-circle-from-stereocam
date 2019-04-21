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


void RoughCircleSolver::getCircles(const std::vector<Eigen::Matrix3d>& left_possible_ellipses, 
                                   const std::vector<Eigen::Matrix3d>& right_possible_ellipses, 
                                   std::vector<Circle3D>& circles){
    /**
     * pairs the ellipses from left/right camera and gives the circles in 3D coordinate.
     * 
    */
   
}