#include "rough_circle_solver.h"

void RoughCircleSolver::getPossibleEllipse(const cv::Mat &edge_input, std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> &ellipses_vec)
{
/**
    * find ellipse from a single image, the ellipses are presented in the pixel coordinate.
    * @param edge an edge image (binary)
    * @param ellipses the possible ellipse in the image, presented in the pixel coordinate. 
    * 
    * The ellipse is presented in the form of quatratic, for example, if C present a ellipse here,
    * thus X^t * C * X =0, where X = [x, y, 1]^t , see https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections 
    */
#ifdef DEBUG
    cv::Mat show_img;
    cv::cvtColor(edge_input, show_img, CV_GRAY2BGR);
#endif
    cv::Mat edge;
    cv::dilate(edge_input, edge, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 2);

    cv::erode(edge, edge, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    const double flattening_treshold = 2.0; // the threshold of flattening of the ellipse
    const double min_area_threshold = 2000; // the threshold of minimum area of the ellipse


    std::vector<cv::RotatedRect> all_ellipses;
    for (int i = 0; i < contours.size(); i++)
    {

#ifdef DEBUG
        cv::drawContours(show_img, contours, i, cv::Scalar(0, 255, 0), 1);
#endif
        int count = contours[i].size();
        if (count < 6)
            continue; //need at least 6 points to fit an ellipse

        cv::RotatedRect box = fitEllipse(contours[i]);
        if (std::max(box.size.width, box.size.height) / std::min(box.size.width, box.size.height) > flattening_treshold)
            continue;

        if (box.size.width * box.size.height < min_area_threshold)
            continue;

#ifdef DEBUG
        cv::ellipse(show_img, box, cv::Scalar(0, 0, 255), 1, CV_AA);
#endif //DEBUG

        all_ellipses.push_back(box);

    }

    // check whether there are two circles with the same center
    for(int i=0; i<all_ellipses.size(); i++){
        for(int j=i+1; j<all_ellipses.size(); j++){
            // check the centers
            if(fabs(all_ellipses[j].center.x -  all_ellipses[i].center.x) > 5 || fabs(all_ellipses[j].center.y - all_ellipses[j].center.y) > 5) continue;

            // check the flatten ratio
            double flatten_ratio_i = std::max(all_ellipses[i].size.width, all_ellipses[i].size.height) / std::min(all_ellipses[i].size.width, all_ellipses[i].size.height);
            double flatten_ratio_j = std::max(all_ellipses[j].size.width, all_ellipses[j].size.height) / std::min(all_ellipses[j].size.width, all_ellipses[j].size.height);
            if(fabs(flatten_ratio_i - flatten_ratio_j) > 0.2) continue;

            // check the angle (only if flatten ratio is higher than a threshold, otherwise the ellipses are circle, which makes angle with no sense)
            if (std::min(flatten_ratio_i, flatten_ratio_j) > 1.2){
                if(fabs(all_ellipses[i].angle - all_ellipses[j].angle) > 10) continue; //degree
            }

            // chech the radius
            double rad_i = (all_ellipses[i].size.width + all_ellipses[j].size.height)/4;
            double rad_j = (all_ellipses[j].size.width + all_ellipses[j].size.height)/4;
            if(isIn(rad_i/rad_j,  0.6, 0.9)){
                ellipses_vec.push_back(std::make_pair(all_ellipses[i], all_ellipses[j]));

                #ifdef DEBUG
                    cv::ellipse(show_img, all_ellipses[i], cv::Scalar(255, 0, 255), 1, CV_AA);
                    cv::ellipse(show_img, all_ellipses[j], cv::Scalar(255, 0, 0), 1, CV_AA);
                #endif
            }
            else if(isIn(rad_j/rad_i, 1.1, 1.4)){
                ellipses_vec.push_back(std::make_pair(all_ellipses[j], all_ellipses[i]));

                #ifdef DEBUG
                    cv::ellipse(show_img, all_ellipses[j], cv::Scalar(255, 0, 255), 1, CV_AA);
                    cv::ellipse(show_img, all_ellipses[i], cv::Scalar(255, 0, 0), 1, CV_AA);
                #endif
            }


        }
    }
}

void RoughCircleSolver::computeI2I3I4(const Eigen::Matrix4d &A, const Eigen::Matrix4d &B, double &I_2, double &I_3, double &I_4)
{
    /** compute I_2, I_3, I_4, which are mentioned in the paper : "Long Quan. Conic Reconstruction and Correspondence from Two Views"
     * to simplify the programming, we choose to calculate I_2, I_3, I_4 numerically
     * by setting lambda = 1, -1, 2
    */
    double y_positive_one = (A + B).determinant();
    double k = 1.0; //std::exp(std::log(y_positive_one) / 4.0);
    double y_negative_one = ((A - B) / k).determinant();
    double y_positive_two = ((A + B * 2) / k).determinant();
    y_positive_one = ((A + B) / k).determinant();
    I_2 = (y_positive_two - y_negative_one) / 6.0 - y_positive_one / 2.0;
    I_3 = (y_positive_one + y_negative_one) / 2.0;
    I_4 = y_positive_one - y_negative_one / 3.0 - y_positive_two / 6.0;
}

void RoughCircleSolver::computeI2I3I4Analytic(const Eigen::Matrix4d &A, const Eigen::Matrix4d &B, double &I_2, double &I_3, double &I_4)
{
    /** compute I_2, I_3, I_4, which are mentioned in the paper : "Long Quan. Conic Reconstruction and Correspondence from Two Views"
    */
    I_2 = A(0, 0) * B(1, 1) * B(2, 2) * B(3, 3) - A(0, 0) * B(1, 1) * B(2, 3) * B(3, 2) - A(0, 0) * B(1, 2) * B(2, 1) * B(3, 3) 
        + A(0, 0) * B(1, 2) * B(2, 3) * B(3, 1) + A(0, 0) * B(1, 3) * B(2, 1) * B(3, 2) - A(0, 0) * B(1, 3) * B(2, 2) * B(3, 1) 
        - A(0, 1) * B(1, 0) * B(2, 2) * B(3, 3) + A(0, 1) * B(1, 0) * B(2, 3) * B(3, 2) + A(0, 1) * B(1, 2) * B(2, 0) * B(3, 3) 
        - A(0, 1) * B(1, 2) * B(2, 3) * B(3, 0) - A(0, 1) * B(1, 3) * B(2, 0) * B(3, 2) + A(0, 1) * B(1, 3) * B(2, 2) * B(3, 0) 
        + A(0, 2) * B(1, 0) * B(2, 1) * B(3, 3) - A(0, 2) * B(1, 0) * B(2, 3) * B(3, 1) - A(0, 2) * B(1, 1) * B(2, 0) * B(3, 3) 
        + A(0, 2) * B(1, 1) * B(2, 3) * B(3, 0) + A(0, 2) * B(1, 3) * B(2, 0) * B(3, 1) - A(0, 2) * B(1, 3) * B(2, 1) * B(3, 0) 
        - A(0, 3) * B(1, 0) * B(2, 1) * B(3, 2) + A(0, 3) * B(1, 0) * B(2, 2) * B(3, 1) + A(0, 3) * B(1, 1) * B(2, 0) * B(3, 2) 
        - A(0, 3) * B(1, 1) * B(2, 2) * B(3, 0) - A(0, 3) * B(1, 2) * B(2, 0) * B(3, 1) + A(0, 3) * B(1, 2) * B(2, 1) * B(3, 0) 
        - A(1, 0) * B(0, 1) * B(2, 2) * B(3, 3) + A(1, 0) * B(0, 1) * B(2, 3) * B(3, 2) + A(1, 0) * B(0, 2) * B(2, 1) * B(3, 3) 
        - A(1, 0) * B(0, 2) * B(2, 3) * B(3, 1) - A(1, 0) * B(0, 3) * B(2, 1) * B(3, 2) + A(1, 0) * B(0, 3) * B(2, 2) * B(3, 1) 
        + A(1, 1) * B(0, 0) * B(2, 2) * B(3, 3) - A(1, 1) * B(0, 0) * B(2, 3) * B(3, 2) - A(1, 1) * B(0, 2) * B(2, 0) * B(3, 3) 
        + A(1, 1) * B(0, 2) * B(2, 3) * B(3, 0) + A(1, 1) * B(0, 3) * B(2, 0) * B(3, 2) - A(1, 1) * B(0, 3) * B(2, 2) * B(3, 0) 
        - A(1, 2) * B(0, 0) * B(2, 1) * B(3, 3) + A(1, 2) * B(0, 0) * B(2, 3) * B(3, 1) + A(1, 2) * B(0, 1) * B(2, 0) * B(3, 3) 
        - A(1, 2) * B(0, 1) * B(2, 3) * B(3, 0) - A(1, 2) * B(0, 3) * B(2, 0) * B(3, 1) + A(1, 2) * B(0, 3) * B(2, 1) * B(3, 0) 
        + A(1, 3) * B(0, 0) * B(2, 1) * B(3, 2) - A(1, 3) * B(0, 0) * B(2, 2) * B(3, 1) - A(1, 3) * B(0, 1) * B(2, 0) * B(3, 2) 
        + A(1, 3) * B(0, 1) * B(2, 2) * B(3, 0) + A(1, 3) * B(0, 2) * B(2, 0) * B(3, 1) - A(1, 3) * B(0, 2) * B(2, 1) * B(3, 0) 
        + A(2, 0) * B(0, 1) * B(1, 2) * B(3, 3) - A(2, 0) * B(0, 1) * B(1, 3) * B(3, 2) - A(2, 0) * B(0, 2) * B(1, 1) * B(3, 3) 
        + A(2, 0) * B(0, 2) * B(1, 3) * B(3, 1) + A(2, 0) * B(0, 3) * B(1, 1) * B(3, 2) - A(2, 0) * B(0, 3) * B(1, 2) * B(3, 1) 
        - A(2, 1) * B(0, 0) * B(1, 2) * B(3, 3) + A(2, 1) * B(0, 0) * B(1, 3) * B(3, 2) + A(2, 1) * B(0, 2) * B(1, 0) * B(3, 3) 
        - A(2, 1) * B(0, 2) * B(1, 3) * B(3, 0) - A(2, 1) * B(0, 3) * B(1, 0) * B(3, 2) + A(2, 1) * B(0, 3) * B(1, 2) * B(3, 0) 
        + A(2, 2) * B(0, 0) * B(1, 1) * B(3, 3) - A(2, 2) * B(0, 0) * B(1, 3) * B(3, 1) - A(2, 2) * B(0, 1) * B(1, 0) * B(3, 3) 
        + A(2, 2) * B(0, 1) * B(1, 3) * B(3, 0) + A(2, 2) * B(0, 3) * B(1, 0) * B(3, 1) - A(2, 2) * B(0, 3) * B(1, 1) * B(3, 0) 
        - A(2, 3) * B(0, 0) * B(1, 1) * B(3, 2) + A(2, 3) * B(0, 0) * B(1, 2) * B(3, 1) + A(2, 3) * B(0, 1) * B(1, 0) * B(3, 2) 
        - A(2, 3) * B(0, 1) * B(1, 2) * B(3, 0) - A(2, 3) * B(0, 2) * B(1, 0) * B(3, 1) + A(2, 3) * B(0, 2) * B(1, 1) * B(3, 0) 
        - A(3, 0) * B(0, 1) * B(1, 2) * B(2, 3) + A(3, 0) * B(0, 1) * B(1, 3) * B(2, 2) + A(3, 0) * B(0, 2) * B(1, 1) * B(2, 3) 
        - A(3, 0) * B(0, 2) * B(1, 3) * B(2, 1) - A(3, 0) * B(0, 3) * B(1, 1) * B(2, 2) + A(3, 0) * B(0, 3) * B(1, 2) * B(2, 1) 
        + A(3, 1) * B(0, 0) * B(1, 2) * B(2, 3) - A(3, 1) * B(0, 0) * B(1, 3) * B(2, 2) - A(3, 1) * B(0, 2) * B(1, 0) * B(2, 3) 
        + A(3, 1) * B(0, 2) * B(1, 3) * B(2, 0) + A(3, 1) * B(0, 3) * B(1, 0) * B(2, 2) - A(3, 1) * B(0, 3) * B(1, 2) * B(2, 0) 
        - A(3, 2) * B(0, 0) * B(1, 1) * B(2, 3) + A(3, 2) * B(0, 0) * B(1, 3) * B(2, 1) + A(3, 2) * B(0, 1) * B(1, 0) * B(2, 3) 
        - A(3, 2) * B(0, 1) * B(1, 3) * B(2, 0) - A(3, 2) * B(0, 3) * B(1, 0) * B(2, 1) + A(3, 2) * B(0, 3) * B(1, 1) * B(2, 0) 
        + A(3, 3) * B(0, 0) * B(1, 1) * B(2, 2) - A(3, 3) * B(0, 0) * B(1, 2) * B(2, 1) - A(3, 3) * B(0, 1) * B(1, 0) * B(2, 2) 
        + A(3, 3) * B(0, 1) * B(1, 2) * B(2, 0) + A(3, 3) * B(0, 2) * B(1, 0) * B(2, 1) - A(3, 3) * B(0, 2) * B(1, 1) * B(2, 0);

    I_3 = A(0, 0) * A(1, 1) * B(2, 2) * B(3, 3) - A(0, 0) * A(1, 1) * B(2, 3) * B(3, 2) - A(0, 0) * A(1, 2) * B(2, 1) * B(3, 3) 
        + A(0, 0) * A(1, 2) * B(2, 3) * B(3, 1) + A(0, 0) * A(1, 3) * B(2, 1) * B(3, 2) - A(0, 0) * A(1, 3) * B(2, 2) * B(3, 1) 
        - A(0, 0) * A(2, 1) * B(1, 2) * B(3, 3) + A(0, 0) * A(2, 1) * B(1, 3) * B(3, 2) + A(0, 0) * A(2, 2) * B(1, 1) * B(3, 3) 
        - A(0, 0) * A(2, 2) * B(1, 3) * B(3, 1) - A(0, 0) * A(2, 3) * B(1, 1) * B(3, 2) + A(0, 0) * A(2, 3) * B(1, 2) * B(3, 1) 
        + A(0, 0) * A(3, 1) * B(1, 2) * B(2, 3) - A(0, 0) * A(3, 1) * B(1, 3) * B(2, 2) - A(0, 0) * A(3, 2) * B(1, 1) * B(2, 3) 
        + A(0, 0) * A(3, 2) * B(1, 3) * B(2, 1) + A(0, 0) * A(3, 3) * B(1, 1) * B(2, 2) - A(0, 0) * A(3, 3) * B(1, 2) * B(2, 1) 
        - A(0, 1) * A(1, 0) * B(2, 2) * B(3, 3) + A(0, 1) * A(1, 0) * B(2, 3) * B(3, 2) + A(0, 1) * A(1, 2) * B(2, 0) * B(3, 3) 
        - A(0, 1) * A(1, 2) * B(2, 3) * B(3, 0) - A(0, 1) * A(1, 3) * B(2, 0) * B(3, 2) + A(0, 1) * A(1, 3) * B(2, 2) * B(3, 0) 
        + A(0, 1) * A(2, 0) * B(1, 2) * B(3, 3) - A(0, 1) * A(2, 0) * B(1, 3) * B(3, 2) - A(0, 1) * A(2, 2) * B(1, 0) * B(3, 3) 
        + A(0, 1) * A(2, 2) * B(1, 3) * B(3, 0) + A(0, 1) * A(2, 3) * B(1, 0) * B(3, 2) - A(0, 1) * A(2, 3) * B(1, 2) * B(3, 0) 
        - A(0, 1) * A(3, 0) * B(1, 2) * B(2, 3) + A(0, 1) * A(3, 0) * B(1, 3) * B(2, 2) + A(0, 1) * A(3, 2) * B(1, 0) * B(2, 3) 
        - A(0, 1) * A(3, 2) * B(1, 3) * B(2, 0) - A(0, 1) * A(3, 3) * B(1, 0) * B(2, 2) + A(0, 1) * A(3, 3) * B(1, 2) * B(2, 0) 
        + A(0, 2) * A(1, 0) * B(2, 1) * B(3, 3) - A(0, 2) * A(1, 0) * B(2, 3) * B(3, 1) - A(0, 2) * A(1, 1) * B(2, 0) * B(3, 3) 
        + A(0, 2) * A(1, 1) * B(2, 3) * B(3, 0) + A(0, 2) * A(1, 3) * B(2, 0) * B(3, 1) - A(0, 2) * A(1, 3) * B(2, 1) * B(3, 0) 
        - A(0, 2) * A(2, 0) * B(1, 1) * B(3, 3) + A(0, 2) * A(2, 0) * B(1, 3) * B(3, 1) + A(0, 2) * A(2, 1) * B(1, 0) * B(3, 3) 
        - A(0, 2) * A(2, 1) * B(1, 3) * B(3, 0) - A(0, 2) * A(2, 3) * B(1, 0) * B(3, 1) + A(0, 2) * A(2, 3) * B(1, 1) * B(3, 0) 
        + A(0, 2) * A(3, 0) * B(1, 1) * B(2, 3) - A(0, 2) * A(3, 0) * B(1, 3) * B(2, 1) - A(0, 2) * A(3, 1) * B(1, 0) * B(2, 3) 
        + A(0, 2) * A(3, 1) * B(1, 3) * B(2, 0) + A(0, 2) * A(3, 3) * B(1, 0) * B(2, 1) - A(0, 2) * A(3, 3) * B(1, 1) * B(2, 0) 
        - A(0, 3) * A(1, 0) * B(2, 1) * B(3, 2) + A(0, 3) * A(1, 0) * B(2, 2) * B(3, 1) + A(0, 3) * A(1, 1) * B(2, 0) * B(3, 2) 
        - A(0, 3) * A(1, 1) * B(2, 2) * B(3, 0) - A(0, 3) * A(1, 2) * B(2, 0) * B(3, 1) + A(0, 3) * A(1, 2) * B(2, 1) * B(3, 0) 
        + A(0, 3) * A(2, 0) * B(1, 1) * B(3, 2) - A(0, 3) * A(2, 0) * B(1, 2) * B(3, 1) - A(0, 3) * A(2, 1) * B(1, 0) * B(3, 2) 
        + A(0, 3) * A(2, 1) * B(1, 2) * B(3, 0) + A(0, 3) * A(2, 2) * B(1, 0) * B(3, 1) - A(0, 3) * A(2, 2) * B(1, 1) * B(3, 0) 
        - A(0, 3) * A(3, 0) * B(1, 1) * B(2, 2) + A(0, 3) * A(3, 0) * B(1, 2) * B(2, 1) + A(0, 3) * A(3, 1) * B(1, 0) * B(2, 2) 
        - A(0, 3) * A(3, 1) * B(1, 2) * B(2, 0) - A(0, 3) * A(3, 2) * B(1, 0) * B(2, 1) + A(0, 3) * A(3, 2) * B(1, 1) * B(2, 0) 
        + A(1, 0) * A(2, 1) * B(0, 2) * B(3, 3) - A(1, 0) * A(2, 1) * B(0, 3) * B(3, 2) - A(1, 0) * A(2, 2) * B(0, 1) * B(3, 3) 
        + A(1, 0) * A(2, 2) * B(0, 3) * B(3, 1) + A(1, 0) * A(2, 3) * B(0, 1) * B(3, 2) - A(1, 0) * A(2, 3) * B(0, 2) * B(3, 1) 
        - A(1, 0) * A(3, 1) * B(0, 2) * B(2, 3) + A(1, 0) * A(3, 1) * B(0, 3) * B(2, 2) + A(1, 0) * A(3, 2) * B(0, 1) * B(2, 3) 
        - A(1, 0) * A(3, 2) * B(0, 3) * B(2, 1) - A(1, 0) * A(3, 3) * B(0, 1) * B(2, 2) + A(1, 0) * A(3, 3) * B(0, 2) * B(2, 1) 
        - A(1, 1) * A(2, 0) * B(0, 2) * B(3, 3) + A(1, 1) * A(2, 0) * B(0, 3) * B(3, 2) + A(1, 1) * A(2, 2) * B(0, 0) * B(3, 3) 
        - A(1, 1) * A(2, 2) * B(0, 3) * B(3, 0) - A(1, 1) * A(2, 3) * B(0, 0) * B(3, 2) + A(1, 1) * A(2, 3) * B(0, 2) * B(3, 0) 
        + A(1, 1) * A(3, 0) * B(0, 2) * B(2, 3) - A(1, 1) * A(3, 0) * B(0, 3) * B(2, 2) - A(1, 1) * A(3, 2) * B(0, 0) * B(2, 3) 
        + A(1, 1) * A(3, 2) * B(0, 3) * B(2, 0) + A(1, 1) * A(3, 3) * B(0, 0) * B(2, 2) - A(1, 1) * A(3, 3) * B(0, 2) * B(2, 0) 
        + A(1, 2) * A(2, 0) * B(0, 1) * B(3, 3) - A(1, 2) * A(2, 0) * B(0, 3) * B(3, 1) - A(1, 2) * A(2, 1) * B(0, 0) * B(3, 3) 
        + A(1, 2) * A(2, 1) * B(0, 3) * B(3, 0) + A(1, 2) * A(2, 3) * B(0, 0) * B(3, 1) - A(1, 2) * A(2, 3) * B(0, 1) * B(3, 0) 
        - A(1, 2) * A(3, 0) * B(0, 1) * B(2, 3) + A(1, 2) * A(3, 0) * B(0, 3) * B(2, 1) + A(1, 2) * A(3, 1) * B(0, 0) * B(2, 3) 
        - A(1, 2) * A(3, 1) * B(0, 3) * B(2, 0) - A(1, 2) * A(3, 3) * B(0, 0) * B(2, 1) + A(1, 2) * A(3, 3) * B(0, 1) * B(2, 0) 
        - A(1, 3) * A(2, 0) * B(0, 1) * B(3, 2) + A(1, 3) * A(2, 0) * B(0, 2) * B(3, 1) + A(1, 3) * A(2, 1) * B(0, 0) * B(3, 2) 
        - A(1, 3) * A(2, 1) * B(0, 2) * B(3, 0) - A(1, 3) * A(2, 2) * B(0, 0) * B(3, 1) + A(1, 3) * A(2, 2) * B(0, 1) * B(3, 0) 
        + A(1, 3) * A(3, 0) * B(0, 1) * B(2, 2) - A(1, 3) * A(3, 0) * B(0, 2) * B(2, 1) - A(1, 3) * A(3, 1) * B(0, 0) * B(2, 2) 
        + A(1, 3) * A(3, 1) * B(0, 2) * B(2, 0) + A(1, 3) * A(3, 2) * B(0, 0) * B(2, 1) - A(1, 3) * A(3, 2) * B(0, 1) * B(2, 0) 
        + A(2, 0) * A(3, 1) * B(0, 2) * B(1, 3) - A(2, 0) * A(3, 1) * B(0, 3) * B(1, 2) - A(2, 0) * A(3, 2) * B(0, 1) * B(1, 3) 
        + A(2, 0) * A(3, 2) * B(0, 3) * B(1, 1) + A(2, 0) * A(3, 3) * B(0, 1) * B(1, 2) - A(2, 0) * A(3, 3) * B(0, 2) * B(1, 1) 
        - A(2, 1) * A(3, 0) * B(0, 2) * B(1, 3) + A(2, 1) * A(3, 0) * B(0, 3) * B(1, 2) + A(2, 1) * A(3, 2) * B(0, 0) * B(1, 3) 
        - A(2, 1) * A(3, 2) * B(0, 3) * B(1, 0) - A(2, 1) * A(3, 3) * B(0, 0) * B(1, 2) + A(2, 1) * A(3, 3) * B(0, 2) * B(1, 0) 
        + A(2, 2) * A(3, 0) * B(0, 1) * B(1, 3) - A(2, 2) * A(3, 0) * B(0, 3) * B(1, 1) - A(2, 2) * A(3, 1) * B(0, 0) * B(1, 3) 
        + A(2, 2) * A(3, 1) * B(0, 3) * B(1, 0) + A(2, 2) * A(3, 3) * B(0, 0) * B(1, 1) - A(2, 2) * A(3, 3) * B(0, 1) * B(1, 0) 
        - A(2, 3) * A(3, 0) * B(0, 1) * B(1, 2) + A(2, 3) * A(3, 0) * B(0, 2) * B(1, 1) + A(2, 3) * A(3, 1) * B(0, 0) * B(1, 2) 
        - A(2, 3) * A(3, 1) * B(0, 2) * B(1, 0) - A(2, 3) * A(3, 2) * B(0, 0) * B(1, 1) + A(2, 3) * A(3, 2) * B(0, 1) * B(1, 0);


    I_4 = A(0, 0) * A(1, 1) * A(2, 2) * B(3, 3) - A(0, 0) * A(1, 1) * A(2, 3) * B(3, 2) - A(0, 0) * A(1, 1) * A(3, 2) * B(2, 3) 
        + A(0, 0) * A(1, 1) * A(3, 3) * B(2, 2) - A(0, 0) * A(1, 2) * A(2, 1) * B(3, 3) + A(0, 0) * A(1, 2) * A(2, 3) * B(3, 1) 
        + A(0, 0) * A(1, 2) * A(3, 1) * B(2, 3) - A(0, 0) * A(1, 2) * A(3, 3) * B(2, 1) + A(0, 0) * A(1, 3) * A(2, 1) * B(3, 2) 
        - A(0, 0) * A(1, 3) * A(2, 2) * B(3, 1) - A(0, 0) * A(1, 3) * A(3, 1) * B(2, 2) + A(0, 0) * A(1, 3) * A(3, 2) * B(2, 1) 
        + A(0, 0) * A(2, 1) * A(3, 2) * B(1, 3) - A(0, 0) * A(2, 1) * A(3, 3) * B(1, 2) - A(0, 0) * A(2, 2) * A(3, 1) * B(1, 3) 
        + A(0, 0) * A(2, 2) * A(3, 3) * B(1, 1) + A(0, 0) * A(2, 3) * A(3, 1) * B(1, 2) - A(0, 0) * A(2, 3) * A(3, 2) * B(1, 1) 
        - A(0, 1) * A(1, 0) * A(2, 2) * B(3, 3) + A(0, 1) * A(1, 0) * A(2, 3) * B(3, 2) + A(0, 1) * A(1, 0) * A(3, 2) * B(2, 3) 
        - A(0, 1) * A(1, 0) * A(3, 3) * B(2, 2) + A(0, 1) * A(1, 2) * A(2, 0) * B(3, 3) - A(0, 1) * A(1, 2) * A(2, 3) * B(3, 0) 
        - A(0, 1) * A(1, 2) * A(3, 0) * B(2, 3) + A(0, 1) * A(1, 2) * A(3, 3) * B(2, 0) - A(0, 1) * A(1, 3) * A(2, 0) * B(3, 2) 
        + A(0, 1) * A(1, 3) * A(2, 2) * B(3, 0) + A(0, 1) * A(1, 3) * A(3, 0) * B(2, 2) - A(0, 1) * A(1, 3) * A(3, 2) * B(2, 0) 
        - A(0, 1) * A(2, 0) * A(3, 2) * B(1, 3) + A(0, 1) * A(2, 0) * A(3, 3) * B(1, 2) + A(0, 1) * A(2, 2) * A(3, 0) * B(1, 3) 
        - A(0, 1) * A(2, 2) * A(3, 3) * B(1, 0) - A(0, 1) * A(2, 3) * A(3, 0) * B(1, 2) + A(0, 1) * A(2, 3) * A(3, 2) * B(1, 0) 
        + A(0, 2) * A(1, 0) * A(2, 1) * B(3, 3) - A(0, 2) * A(1, 0) * A(2, 3) * B(3, 1) - A(0, 2) * A(1, 0) * A(3, 1) * B(2, 3)
        + A(0, 2) * A(1, 0) * A(3, 3) * B(2, 1) - A(0, 2) * A(1, 1) * A(2, 0) * B(3, 3) + A(0, 2) * A(1, 1) * A(2, 3) * B(3, 0) 
        + A(0, 2) * A(1, 1) * A(3, 0) * B(2, 3) - A(0, 2) * A(1, 1) * A(3, 3) * B(2, 0) + A(0, 2) * A(1, 3) * A(2, 0) * B(3, 1) 
        - A(0, 2) * A(1, 3) * A(2, 1) * B(3, 0) - A(0, 2) * A(1, 3) * A(3, 0) * B(2, 1) + A(0, 2) * A(1, 3) * A(3, 1) * B(2, 0) 
        + A(0, 2) * A(2, 0) * A(3, 1) * B(1, 3) - A(0, 2) * A(2, 0) * A(3, 3) * B(1, 1) - A(0, 2) * A(2, 1) * A(3, 0) * B(1, 3) 
        + A(0, 2) * A(2, 1) * A(3, 3) * B(1, 0) + A(0, 2) * A(2, 3) * A(3, 0) * B(1, 1) - A(0, 2) * A(2, 3) * A(3, 1) * B(1, 0) 
        - A(0, 3) * A(1, 0) * A(2, 1) * B(3, 2) + A(0, 3) * A(1, 0) * A(2, 2) * B(3, 1) + A(0, 3) * A(1, 0) * A(3, 1) * B(2, 2) 
        - A(0, 3) * A(1, 0) * A(3, 2) * B(2, 1) + A(0, 3) * A(1, 1) * A(2, 0) * B(3, 2) - A(0, 3) * A(1, 1) * A(2, 2) * B(3, 0) 
        - A(0, 3) * A(1, 1) * A(3, 0) * B(2, 2) + A(0, 3) * A(1, 1) * A(3, 2) * B(2, 0) - A(0, 3) * A(1, 2) * A(2, 0) * B(3, 1) 
        + A(0, 3) * A(1, 2) * A(2, 1) * B(3, 0) + A(0, 3) * A(1, 2) * A(3, 0) * B(2, 1) - A(0, 3) * A(1, 2) * A(3, 1) * B(2, 0) 
        - A(0, 3) * A(2, 0) * A(3, 1) * B(1, 2) + A(0, 3) * A(2, 0) * A(3, 2) * B(1, 1) + A(0, 3) * A(2, 1) * A(3, 0) * B(1, 2) 
        - A(0, 3) * A(2, 1) * A(3, 2) * B(1, 0) - A(0, 3) * A(2, 2) * A(3, 0) * B(1, 1) + A(0, 3) * A(2, 2) * A(3, 1) * B(1, 0) 
        - A(1, 0) * A(2, 1) * A(3, 2) * B(0, 3) + A(1, 0) * A(2, 1) * A(3, 3) * B(0, 2) + A(1, 0) * A(2, 2) * A(3, 1) * B(0, 3) 
        - A(1, 0) * A(2, 2) * A(3, 3) * B(0, 1) - A(1, 0) * A(2, 3) * A(3, 1) * B(0, 2) + A(1, 0) * A(2, 3) * A(3, 2) * B(0, 1) 
        + A(1, 1) * A(2, 0) * A(3, 2) * B(0, 3) - A(1, 1) * A(2, 0) * A(3, 3) * B(0, 2) - A(1, 1) * A(2, 2) * A(3, 0) * B(0, 3) 
        + A(1, 1) * A(2, 2) * A(3, 3) * B(0, 0) + A(1, 1) * A(2, 3) * A(3, 0) * B(0, 2) - A(1, 1) * A(2, 3) * A(3, 2) * B(0, 0) 
        - A(1, 2) * A(2, 0) * A(3, 1) * B(0, 3) + A(1, 2) * A(2, 0) * A(3, 3) * B(0, 1) + A(1, 2) * A(2, 1) * A(3, 0) * B(0, 3) 
        - A(1, 2) * A(2, 1) * A(3, 3) * B(0, 0) - A(1, 2) * A(2, 3) * A(3, 0) * B(0, 1) + A(1, 2) * A(2, 3) * A(3, 1) * B(0, 0) 
        + A(1, 3) * A(2, 0) * A(3, 1) * B(0, 2) - A(1, 3) * A(2, 0) * A(3, 2) * B(0, 1) - A(1, 3) * A(2, 1) * A(3, 0) * B(0, 2) 
        + A(1, 3) * A(2, 1) * A(3, 2) * B(0, 0) + A(1, 3) * A(2, 2) * A(3, 0) * B(0, 1) - A(1, 3) * A(2, 2) * A(3, 1) * B(0, 0);
}
void RoughCircleSolver::computePointsInPlane(const double &u, const double &v, const Eigen::Vector4d &plane, Eigen::Vector4d &point, int camera_id)
{
    /** compute the point in 3d which is intersected by a line and a plane.
     * the line is passed through the focal origin, and projected to the left image plane, with a coordinate (u, v) (in pixel)
     * 
    */
    Eigen::Matrix<double, 3,4> projection_matrix;
    switch (camera_id)
    {
    case LEFT_CAMERA:
    {
        projection_matrix = stereo_cam_ptr->left.projectionEigenMatrix();
        break;
    }
    case RIGHT_CAMERA:
    {
        projection_matrix = stereo_cam_ptr->right.projectionEigenMatrix();
        break;
    }
    default:
        assert("camera id should be only one of the LEFT_CAMERA or RIGHT_CAMERA");
        break;
    }

    Eigen::Matrix<double, 3,4> A_full; //A_full * [x, y, z, 1]^t = 0
    A_full.row(0) = v * projection_matrix.row(2) - projection_matrix.row(1);
    A_full.row(1) = - u * projection_matrix.row(2) + projection_matrix.row(0);
    A_full.row(2) = plane; 
    std::cout<<"A_full "<<A_full<<std::endl;
    Eigen::Matrix3d A = A_full.block<3,3>(0,0);
    Eigen::Vector3d b = - A_full.col(3);
    
    
    point.head<3>() = A.inverse() * b;
    point(3) = 1;
    
    std::cout << A_full * point <<std::endl;
}
void RoughCircleSolver::translateEllipse(const Eigen::Matrix3d &ellipse_quad_form, cv::RotatedRect &ellipse_cv_form)
{
    double A = ellipse_quad_form(0, 0);
    double D = 2 * ellipse_quad_form(0, 2);
    double C = ellipse_quad_form(1, 1);
    double E = 2 * ellipse_quad_form(1, 2);
    double F = ellipse_quad_form(2, 2);

    ellipse_cv_form.center.x = -D / (2 * A);
    ellipse_cv_form.center.y = -E / (2 * C);
    double k = (F + 1) / (D * D / 4 / A + E * E / 4 / C);
    ellipse_cv_form.size.width = std::sqrt(1 / k / A);
    ellipse_cv_form.size.height = std::sqrt(1 / k / C);
}
void RoughCircleSolver::translateEllipse(const cv::RotatedRect &ellipse_cv_form, Eigen::Matrix3d &ellipse_quad_form)
{
    double a = ellipse_cv_form.size.width /2.0;
    double b = ellipse_cv_form.size.height /2.0;
    double x_c = ellipse_cv_form.center.x;
    double y_c = ellipse_cv_form.center.y;

    double sin_theta = sin(ellipse_cv_form.angle * M_PI / 180.0); // rotatedrect's angle is presented in degree
    double cos_theta = cos(ellipse_cv_form.angle * M_PI / 180.0);
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
    cv::Mat image(stereo_cam_ptr->left.imageSize(), CV_8UC3, cv::Scalar(0,0,0));

    for(; alpha < 2*M_PI; alpha += 0.1){// in rad
        double x = a * cos(alpha) * cos_theta - b * sin(alpha) * sin_theta + x_c;
        double y = a * cos(alpha) * sin_theta + b * sin(alpha) * cos_theta + y_c;

        cv::ellipse(image, ellipse_cv_form, cv::Scalar(0, 0, 255), 1, CV_AA);

        cv::Point point((int)x, (int)y);
        cv::Point point_2((int)y, (int)x);

        cv::circle(image,point,2,cv::Scalar(0,255,255),1);
        cv::circle(image,point_2,2,cv::Scalar(0,255,255),1);

        Eigen::Vector3d X;
        X << x,y,1;
        double result = X.transpose() * ellipse_quad_form * X;
        std::cout<<"verification result: "<<result<<std::endl;
    }

    */
    
}
bool RoughCircleSolver::computeCircle3D(const cv::RotatedRect& left_ellipse_box, const cv::RotatedRect& right_ellipse_box, Circle3D& circle){
    double I_2, I_3, I_4;
    Eigen::Matrix3d left_ellipse_quadratic, right_ellipse_quadratic;
    translateEllipse(left_ellipse_box, left_ellipse_quadratic);
    translateEllipse(right_ellipse_box, right_ellipse_quadratic);

    Eigen::Matrix<double, 3, 4> left_projection_matrix = stereo_cam_ptr->left.projectionEigenMatrix();
    Eigen::Matrix<double, 3, 4> right_projection_matrix = stereo_cam_ptr->right.projectionEigenMatrix();
    std::cout<<"left_projection: "<<left_projection_matrix<<std::endl;
    std::cout<<"right_projection: "<<right_projection_matrix<<std::endl;
    Eigen::Matrix4d A = stereo_cam_ptr->left.projectionEigenMatrix().transpose() * left_ellipse_quadratic * stereo_cam_ptr->left.projectionEigenMatrix();
    Eigen::Matrix4d B = stereo_cam_ptr->right.projectionEigenMatrix().transpose() * right_ellipse_quadratic * stereo_cam_ptr->right.projectionEigenMatrix();
    std::cout<<"A: "<<A<<std::endl;
    std::cout<<"B: "<<B<<std::endl;
    std::cout<<"left ellipse: "<<left_ellipse_quadratic<<std::endl;
    std::cout<<"right ellipse: "<<right_ellipse_quadratic<<std::endl;
    
    // calculate the circle's pose
    Eigen::Matrix4d trans = stereo_cam_ptr->rightCamTransformEigenMatrix();
    Eigen::Vector4d right_camera_origin = stereo_cam_ptr->rightCamTransformEigenMatrix().col(3);


    computeI2I3I4Analytic(A, B, I_2, I_3, I_4);
    std::cout<<"I2: "<<I_2<<" I3: "<<I_3<<" I4: "<<I_4<<std::endl;
    computeI2I3I4(A, B, I_2, I_3, I_4);
    std::cout<<"I2: "<<I_2<<" I3: "<<I_3<<" I4: "<<I_4<<std::endl;

    double lambda = -I_3 / (2 * I_2);
    Eigen::Matrix4d C = A + lambda * B;
    std::cout<<"A: "<<A<<std::endl;
    std::cout<<"B: "<<B<<std::endl;

    std::cout<<"C: "<<C<<std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(C);
    
    if (eigensolver.info() != Eigen::Success)
    {
        std::cout << "failed to get the eigen values/vectors" << std::endl;
        return false;
    };
    Eigen::Vector4d eigen_values = eigensolver.eigenvalues(); //increasing order
    std::cout<<"eigen "<<eigen_values<<std::endl;


    if (eigen_values(0) * eigen_values(3) >= 0)
        return false;
    
    // see if there are two zero eigen values
    double min_non_zero_value = std::min(fabs(eigen_values(0)), fabs(eigen_values(3)));
    if (fabs(eigen_values(1))/min_non_zero_value > 0.1 || fabs(eigen_values(2))/min_non_zero_value > 0.1)
        return false;

    // find the plane of the circle
    Eigen::Vector4d p = std::sqrt(-eigen_values(0)) * eigensolver.eigenvectors().col(0) + std::sqrt(eigen_values(3)) * eigensolver.eigenvectors().col(3);
    if (p.dot(p - right_camera_origin) < 0)
    {
        p = std::sqrt(-eigen_values(0)) * eigensolver.eigenvectors().col(0) - std::sqrt(eigen_values(3)) * eigensolver.eigenvectors().col(3);
    }

    // find the center of the circle
    // Here, we assume the center of the circle is also the center of the ellipse in the image.
    double center_x = left_ellipse_box.center.x;
    double center_y = left_ellipse_box.center.y;

    // x=a*cos(alpha)*cos(theta) - b*sin(alpha)*sin(theta) + x_c
    // y=a*sin(theta)*cos(alpha) + b*sin(alpha)*cos(theta) + y_c
    // where a = left_ellipse_box.size.width / 2.0
    //       b = left_ellipse_box.size.height/ 2.0

    // here, we simply choose a point with alpha=0
    double a = left_ellipse_box.size.width / 2.0;
    double p_x = a*cos(left_ellipse_box.angle) + center_x; // point on the ellipse
    double p_y = a*sin(left_ellipse_box.angle) + center_y;

    Eigen::Vector4d center_point_3d, circle_point_3d;
    computePointsInPlane(center_x, center_y, p, center_point_3d, LEFT_CAMERA);
    computePointsInPlane(p_x, p_y, p, circle_point_3d, LEFT_CAMERA);

    circle.radius = (center_point_3d - circle_point_3d).norm();
    circle.plane = p;
    circle.center = center_point_3d;
    return true;
}

bool RoughCircleSolver::areConcentric(const Circle3D a, const Circle3D b, double center_threshold){
    if((a.center - b.center).norm() < center_threshold) return true;
    else return false;
}
void RoughCircleSolver::getConcentricCircles(const std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> &left_possible_ellipses,
                                   const std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> &right_possible_ellipses,
                                   std::vector<ConcentricCircles3D> &concentric_circles,
                                   const cv::Mat* left_edge_ptr,
                                   const cv::Mat* right_edge_ptr)
{
    /**
     * pairs the ellipses from left/right camera and gives the circles in 3D coordinate.
     * 
    */
    if (left_possible_ellipses.size() == 0 || right_possible_ellipses.size() == 0)
        return;

    Eigen::Matrix4d A, B; //X^t * A * X =0, same for B

    double error_threshold = 0.1;

    Eigen::Matrix<double, 3, 4> left_projection, right_projection;
    left_projection = stereo_cam_ptr->left.projectionEigenMatrix();
    right_projection = stereo_cam_ptr->right.projectionEigenMatrix();

    cv::Mat left_edge, right_edge;
    cv::cvtColor(*left_edge_ptr, left_edge, CV_GRAY2BGR);
    cv::cvtColor(*right_edge_ptr, right_edge, CV_GRAY2BGR);




    for (int i = 0; i < left_possible_ellipses.size(); i++)
    {
        for(int j=0; j<right_possible_ellipses.size(); j++){
            Circle3D circle_inner, circle_outer;
            Eigen::Matrix3d left_ellipse_quadratic, right_ellipse_quadratic;

            cv::ellipse(left_edge, std::get<0>(left_possible_ellipses[i]), cv::Scalar(0, 0, 255), 1, CV_AA);
            cv::ellipse(right_edge, std::get<0>(right_possible_ellipses[j]), cv::Scalar(0, 0, 255), 1, CV_AA);

            computeCircle3D(std::get<0>(left_possible_ellipses[i]), std::get<0>(right_possible_ellipses[j]), circle_inner);

            computeCircle3D(std::get<1>(left_possible_ellipses[i]), std::get<1>(right_possible_ellipses[j]), circle_outer);

            if(areConcentric(circle_inner, circle_outer, 15)){
                concentric_circles.push_back(ConcentricCircles3D(circle_inner, circle_outer));
            }
        }
    }
}

void RoughCircleSolver::getPossibleCircles(const cv::Mat &left_edge, const cv::Mat &right_edge, std::vector<ConcentricCircles3D> &concentric_circles)
{
    /** Main function of this class, get all the possible circles in the image.
     * @ param left_edge : the left edge image
     * @ param right_edge : the right edge image
     * @ param concentric_circles : the concentric circles in 3d space.
    */

    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> left_ellipses_box, right_ellipses_box;
    getPossibleEllipse(left_edge, left_ellipses_box);
    getPossibleEllipse(right_edge, right_ellipses_box);

    getConcentricCircles(left_ellipses_box, right_ellipses_box, concentric_circles,
                &left_edge, &right_edge);
    
}

void RoughCircleSolver::reprojectCircles(cv::Mat &image, const Circle3D &circle, int camera_id, int sample_size, const cv::Scalar &color)
{
    /** reproject a circle in 3D into the image (left/right, indicated by the camera_id)
     *  the circle is discrete by uniformly sampled points, and the 3D points are projected into the image.
     * @ param image : the image needed to be shown, it should be in BGR, attention: this function will change this image.
     * @ param circle : the circle needed to be projected.
     * @ camera_id : it could be LEFT_CAMERA or RIGHT_CAMERA
     * @ sample_size : the number of sampling points.
     * @ color : the color of sampling points which are shown in the image.
    */
    double increment_angle = 2 * M_PI / sample_size;
    Eigen::Matrix4d transform_circle_to_camera;
    Eigen::Vector3d pixel_pos;

    Eigen::Matrix<double, 3, 4> projection_matrix;

    switch (camera_id)
    {
    case LEFT_CAMERA:
    {
        projection_matrix = stereo_cam_ptr->left.projectionEigenMatrix();
        break;
    }
    case RIGHT_CAMERA:
    {
        projection_matrix = stereo_cam_ptr->right.projectionEigenMatrix();
        break;
    }
    default:
        assert("camera id should be only one of the LEFT_CAMERA or RIGHT_CAMERA");
        break;
    }

    circle.getTransformMatrixToOrigin(transform_circle_to_camera);

    for (int i = 0; i < sample_size; i++)
    {
        Eigen::Vector4d point_in_circle_coord; // the point on the circle, which is presented in the circle coordinate. (origin is the circle's center)
        point_in_circle_coord(0) = circle.radius * std::cos(increment_angle * i);
        point_in_circle_coord(1) = circle.radius * std::sin(increment_angle * i);
        point_in_circle_coord(2) = 0.0;
        point_in_circle_coord(3) = 1.0;

        // transform point in circle coordinate into camera's coordinate.
        Eigen::Vector4d point_in_camera_coord = transform_circle_to_camera * point_in_circle_coord;
        pixel_pos = projection_matrix * point_in_camera_coord;

        // normalize pixel pos
        pixel_pos(0) = pixel_pos(0) / pixel_pos(2);
        pixel_pos(1) = pixel_pos(1) / pixel_pos(2);

        // draw points on the image
        cv::Point point((int)pixel_pos(0), (int)pixel_pos(1));    
        cv::circle(image,point,2,color,1);
    }
}

void RoughCircleSolver::reprojectCircles(cv::Mat &image, const std::vector<Circle3D> &circles_vec, int camera_id, int sample_size, const cv::Scalar &color)
{
    for (int i = 0; i < circles_vec.size(); i++)
    {
        reprojectCircles(image, circles_vec[i], camera_id, sample_size, color);
    }
}

void RoughCircleSolver::reprojectCircles(cv::Mat &image, const ConcentricCircles3D &concentric_circles, int camera_id, int sample_size, const cv::Scalar &color){
    Circle3D circles[2];
    concentric_circles.splitToCircles(circles);
    reprojectCircles(image, circles[0], camera_id, sample_size, color);
    reprojectCircles(image, circles[1], camera_id, sample_size, color);
}
void RoughCircleSolver::reprojectCircles(cv::Mat &image, const std::vector<ConcentricCircles3D> &concentric_circles, int camera_id, int sample_size, const cv::Scalar &color)
{
    for (int i = 0; i < concentric_circles.size(); i++)
    {
        reprojectCircles(image, concentric_circles[i], camera_id, sample_size, color);
    }
}
