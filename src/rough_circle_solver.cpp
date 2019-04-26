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


    std::vector<FittedEllipse> all_ellipses;
    for (int i = 0; i < contours.size(); i++)
    {

#ifdef DEBUG
        //cv::drawContours(show_img, contours, i, cv::Scalar(0, 255, 0), 1);
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
        FittedEllipse fitted_ellipse(i, box);
        fitted_ellipse.cover(contours[i]);
        all_ellipses.push_back(fitted_ellipse);
    }

    std::vector<cv::RotatedRect> filtered_ellipses;
    // try to fuse the seperated components of the same ellipse
    std::vector<bool> fused(all_ellipses.size());
    for(int i=0; i<all_ellipses.size(); i++){
        for(int j=i+1; j<all_ellipses.size(); j++){
            if(fused[j]) continue;

            cv::cvtColor(edge_input, show_img, CV_GRAY2BGR);
            cv::drawContours(show_img, contours, all_ellipses[i].id, cv::Scalar(0, 255, 0), 1);
            cv::drawContours(show_img, contours, all_ellipses[j].id, cv::Scalar(0, 255, 0), 1);
            if(all_ellipses[i].hasOverlap(all_ellipses[j])) continue; // can't belong to the same ellipse
            std::vector<cv::Point> fused_contours;
            fused_contours = contours[all_ellipses[i].id];
            fused_contours.insert(fused_contours.end(), contours[all_ellipses[j].id].begin(), contours[all_ellipses[j].id].end());
            cv::RotatedRect fused_box = fitEllipse(fused_contours);
            double error = getAlgebraDistance(fused_box, fused_contours)/(double)(fused_contours.size());

            #ifdef DEBUG

            cv::ellipse(show_img, fused_box, cv::Scalar(0, 255, 255), 1, CV_AA);

            #endif //DEBUG

            if(error < 0.01){
                all_ellipses[i].possible_other_parts_id.push_back(all_ellipses[j].id); 
                fused[j] = true;
            } 
        }    
    }

    for(int i=0; i<all_ellipses.size(); i++){
        cv::ellipse(show_img, all_ellipses[i].box, cv::Scalar(255, 255, 0), 1, CV_AA);

        if(fused[i]) continue;
        if(all_ellipses[i].possible_other_parts_id.size() == 0){
            cv::ellipse(show_img, all_ellipses[i].box, cv::Scalar(255, 0, 0), 1, CV_AA);

            filtered_ellipses.push_back(all_ellipses[i].box);
        }
        else{
            std::vector<cv::Point> fused_contours;
            fused_contours = contours[all_ellipses[i].id];
            for(int other_idx=0; other_idx<all_ellipses[i].possible_other_parts_id.size(); other_idx ++){
                int id = all_ellipses[i].possible_other_parts_id[other_idx]; // contour's index
                fused_contours.insert(fused_contours.end(), contours[id].begin(), contours[id].end());
            } 
            cv::RotatedRect fused_box = fitEllipse(fused_contours);
            cv::ellipse(show_img, fused_box, cv::Scalar(255, 0, 0), 1, CV_AA);

            filtered_ellipses.push_back(fused_box);
        }

    }

    


    // check whether there are two circles with the same center
    for(int i=0; i<filtered_ellipses.size(); i++){
        for(int j=i+1; j<filtered_ellipses.size(); j++){
            // check the centers
            if(fabs(filtered_ellipses[j].center.x -  filtered_ellipses[i].center.x) > 5 || fabs(filtered_ellipses[j].center.y - filtered_ellipses[j].center.y) > 5) continue;

            // check the flatten ratio
            double flatten_ratio_i = std::max(filtered_ellipses[i].size.width, filtered_ellipses[i].size.height) / std::min(filtered_ellipses[i].size.width, filtered_ellipses[i].size.height);
            double flatten_ratio_j = std::max(filtered_ellipses[j].size.width, filtered_ellipses[j].size.height) / std::min(filtered_ellipses[j].size.width, filtered_ellipses[j].size.height);
            if(fabs(flatten_ratio_i - flatten_ratio_j) > 0.2) continue;

            // check the angle (only if flatten ratio is higher than a threshold, otherwise the ellipses are circle, which makes angle with no sense)
            if (std::min(flatten_ratio_i, flatten_ratio_j) > 1.2){
                if(fabs(filtered_ellipses[i].angle - filtered_ellipses[j].angle) > 10) continue; //degree
            }

            // chech the radius
            double rad_i = (filtered_ellipses[i].size.width + filtered_ellipses[j].size.height)/4;
            double rad_j = (filtered_ellipses[j].size.width + filtered_ellipses[j].size.height)/4;
            if(isIn(rad_i/rad_j,  0.5, 0.9)){
                ellipses_vec.push_back(std::make_pair(filtered_ellipses[i], filtered_ellipses[j]));

                #ifdef DEBUG
                    cv::cvtColor(edge_input, show_img, CV_GRAY2BGR);
                    cv::ellipse(show_img, filtered_ellipses[i], cv::Scalar(255, 0, 255), 1, CV_AA);
                    cv::ellipse(show_img, filtered_ellipses[j], cv::Scalar(255, 0, 0), 1, CV_AA);
                #endif
            }
            else if(isIn(rad_j/rad_i, 0.5, 0.9)){
                ellipses_vec.push_back(std::make_pair(filtered_ellipses[j], filtered_ellipses[i]));

                #ifdef DEBUG
                    cv::cvtColor(edge_input, show_img, CV_GRAY2BGR);
                    cv::ellipse(show_img, filtered_ellipses[j], cv::Scalar(255, 0, 255), 1, CV_AA);
                    cv::ellipse(show_img, filtered_ellipses[i], cv::Scalar(255, 0, 0), 1, CV_AA);
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

bool RoughCircleSolver::computeCircle3D(const cv::RotatedRect& left_ellipse_box, const cv::RotatedRect& right_ellipse_box, Circle3D& circle){
    double I_2, I_3, I_4;
    Eigen::Matrix3d left_ellipse_quadratic, right_ellipse_quadratic;
    translateEllipse(left_ellipse_box, left_ellipse_quadratic);
    translateEllipse(right_ellipse_box, right_ellipse_quadratic);

    /*
    if(fabs(left_ellipse_box.angle - 65) < 0.01){
        cv::RotatedRect left_ellipse_box_changed = left_ellipse_box;
        left_ellipse_box_changed.angle = 59.2503;
        translateEllipse(left_ellipse_box_changed, left_ellipse_quadratic);
    }
    */


    Eigen::Matrix<double, 3, 4> left_projection_matrix = stereo_cam_ptr->left.projectionEigenMatrix();
    Eigen::Matrix<double, 3, 4> right_projection_matrix = stereo_cam_ptr->right.projectionEigenMatrix();
    std::cout<<"\n\n\nleft_projection: \n"<<left_projection_matrix<<std::endl;
    std::cout<<"right_projection: \n"<<right_projection_matrix<<std::endl;
    Eigen::Matrix4d A = stereo_cam_ptr->left.projectionEigenMatrix().transpose() * left_ellipse_quadratic * stereo_cam_ptr->left.projectionEigenMatrix();
    Eigen::Matrix4d B = stereo_cam_ptr->right.projectionEigenMatrix().transpose() * right_ellipse_quadratic * stereo_cam_ptr->right.projectionEigenMatrix();
    std::cout<<"left box: center: "<<left_ellipse_box.center.x<<" " <<left_ellipse_box.center.y<<" size: "<<left_ellipse_box.size.width <<" "<<left_ellipse_box.size.height <<" angle: "<<left_ellipse_box.angle<<std::endl;
    std::cout<<"right box: center: "<<right_ellipse_box.center.x<<" " <<right_ellipse_box.center.y<<" size: "<<right_ellipse_box.size.width <<" "<<right_ellipse_box.size.height <<" angle: "<<right_ellipse_box.angle<<std::endl;

    std::cout<<"A: \n"<<A<<std::endl;
    std::cout<<"B: \n"<<B<<std::endl;
    std::cout<<"left ellipse: \n"<<left_ellipse_quadratic<<std::endl;
    std::cout<<"right ellipse: \n"<<right_ellipse_quadratic<<std::endl;
    
    // calculate the circle's pose
    Eigen::Matrix4d trans = stereo_cam_ptr->rightCamTransformEigenMatrix();
    Eigen::Vector4d right_camera_origin = - stereo_cam_ptr->rightCamTransformEigenMatrix().col(3);
    right_camera_origin(3) = -right_camera_origin(3);

    computeI2I3I4Analytic(A, B, I_2, I_3, I_4);
    std::cout<<"I2: "<<I_2<<" I3: "<<I_3<<" I4: "<<I_4<<std::endl;
    computeI2I3I4(A, B, I_2, I_3, I_4);
    std::cout<<"I2: "<<I_2<<" I3: "<<I_3<<" I4: "<<I_4<<std::endl;

    double lambda = -I_3 / (2 * I_2);
    Eigen::Matrix4d C = A + lambda * B;

    std::cout<<"C: \n"<<C<<std::endl;
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
    if (p(3) * p.dot(right_camera_origin) < 0)
    {
        p =  std::sqrt(-eigen_values(0)) * eigensolver.eigenvectors().col(0) - std::sqrt(eigen_values(3)) * eigensolver.eigenvectors().col(3);
    }
    // convention: p(3) > 0 , in our case, it can't be zero
    if(p(3) < 0){
        p = -p;
    }

    std::cout<<"plane :"<<p.transpose()<<std::endl;
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
            cv::ellipse(left_edge, std::get<1>(left_possible_ellipses[i]), cv::Scalar(0, 0, 255), 1, CV_AA);
            cv::ellipse(right_edge, std::get<1>(right_possible_ellipses[j]), cv::Scalar(0, 0, 255), 1, CV_AA);

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
