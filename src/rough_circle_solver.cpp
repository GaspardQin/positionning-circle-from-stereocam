#include "rough_circle_solver.h"

void RoughCircleSolver::getPossibleEllipse(const cv::Mat &edge_input, std::vector<Eigen::Matrix3d> &ellipses_vec)
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

    for (int i = 0; i < contours.size(); i++)
    {

#ifdef DEBUG
        drawContours(show_img, contours, i, cv::Scalar(0, 255, 0), 1);
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
        drawContours(show_img, contours, (int)i, cv::Scalar::all(255), 1, 8);
        ellipse(show_img, box, cv::Scalar(0, 0, 255), 1, CV_AA);
#endif //DEBUG
        Eigen::Matrix3d ellipse;
        translateEllipse(box, ellipse);

        cv::RotatedRect re_translate_box;
        translateEllipse(ellipse, re_translate_box);
        ellipses_vec.push_back(ellipse);
    }
}

void RoughCircleSolver::computeI2I3I4(const Eigen::Matrix4d &A, const Eigen::Matrix4d &B, double &I_2, double &I_3, double &I_4)
{
    /** compute I_2, I_3, I_4, which are mentioned in the paper : "Long Quan. Conic Reconstruction and Correspondence from Two Views"
     * to simplify the programming, we choose to calculate I_2, I_3, I_4 numerically
     * by setting lambda = 1, -1, 2
    */
    double y_positive_one = (A + B).determinant();
    double k = std::exp(std::log(y_positive_one) / 4.0);
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
    I2 = A(0, 0) * B(1, 1) * B(2, 2) * B(3, 3) - A(0, 0) * B(1, 1) * B(2, 3) * B(3, 2) - A(0, 0) * B(1, 2) * B(2, 1) * B(3, 3) 
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

    I3 = A(0, 0) * A(1, 1) * B(2, 2) * B(3, 3) - A(0, 0) * A(1, 1) * B(2, 3) * B(3, 2) - A(0, 0) * A(1, 2) * B(2, 1) * B(3, 3) 
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
void RoughCircleSolver::computePointsInPlane(const double &u, const double &v, const Eigen::Vector4d &plane, Eigen::Vector4d &point)
{
    /** compute the point in 3d which is intersected by a line and a plane.
     * the line is passed the focal origin, and projected to the left image plane, with a coordinate (u, v) (in pixel)
     * 
    */
    double fx = stereo_cam_ptr->left.projectionEigenMatrix()(0, 0);
    double fy = stereo_cam_ptr->left.projectionEigenMatrix()(1, 1);

    double m = u / fx; // should be equal to x/z
    double n = v / fy; // should be equal to y/z

    point(2) = -plane(3) / (plane(0) * m + plane(1) * n + plane(2));
    point(1) = point(2) * n;
    point(0) = point(2) * m;
    point(3) = 1;
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
    double a = ellipse_cv_form.size.width;
    double b = ellipse_cv_form.size.height;
    double x_c = ellipse_cv_form.center.x;
    double y_c = ellipse_cv_form.center.y;

    double sin_theta = sin(ellipse_cv_form.angle);
    double cos_theta = cos(ellipse_cv_form.angle);
    double A = (a*a*sin_theta*sin_theta + b*b*cos_theta*cos_theta)/(a*a*b*b);
    double B = (-2*a*a*sin_theta*cos_theta + 2*b*b*sin_theta*cos_theta)/(a*a*b*b);
    double C = (a*a*cos_theta*cos_theta + b*b*sin_theta*sin_theta)/(a*a*b*b);
    double D = (-2*a*a*x_c*sin_theta*sin_theta + 2*a*a*y_c*sin_theta*cos_theta - 2*b*b*x_c*cos_theta*cos_theta - 2*b*b*y_c*sin_theta*cos_theta)/(a*a*b*b);
    double E = (2*a*a*x_c*sin_theta*cos_theta - 2*a*a*y_c*cos_theta*cos_theta - 2*b*b*x_c*sin_theta*cos_theta - 2*b*b*y_c*sin_theta*sin_theta)/(a*a*b*b);
    double F = (-a*a*b*b + a*a*x_c*x_c*sin_theta*sin_theta - 2*a*a*x_c*y_c*sin_theta*cos_theta + a*a*y_c*y_c*cos_theta*cos_theta + b*b*x_c*x_c*cos_theta*cos_theta + 2*b*b*x_c*y_c*sin_theta*cos_theta + b*b*y_c*y_c*sin_theta*sin_theta)/(a*a*b*b);

    ellipse_quad_form << A, B / 2.0, D / 2.0, B / 2.0, C, E / 2.0, D / 2.0, E / 2.0, F;
}

void RoughCircleSolver::getCircles(const std::vector<Eigen::Matrix3d> &left_possible_ellipses,
                                   const std::vector<Eigen::Matrix3d> &right_possible_ellipses,
                                   std::vector<Circle3D> &circles)
{
    /**
     * pairs the ellipses from left/right camera and gives the circles in 3D coordinate.
     * 
    */
    if (left_possible_ellipses.size() == 0 || right_possible_ellipses.size() == 0)
        return;

    Eigen::Matrix4d A, B; //X^t * A * X =0, same for B

    std::vector<CirclePair> circle_pairs;
    double error_threshold = 0.1;

    Eigen::Matrix<double, 3, 4> left_projection, right_projection;
    left_projection = stereo_cam_ptr->left.projectionEigenMatrix();
    right_projection = stereo_cam_ptr->right.projectionEigenMatrix();

    for (int i = 0; i < left_possible_ellipses.size(); i++)
    {
        A = left_projection.transpose() * left_possible_ellipses[i] * left_projection;
        double min_error = std::numeric_limits<double>::max();
        int min_index = -1;
        for (int j = 0; j < right_possible_ellipses.size(); j++)
        {
            B = right_projection.transpose() * right_possible_ellipses[j] * right_projection;
            double I_2, I_3, I_4;
            computeI2I3I4Analytic(A, B, I_2, I_3, I_4);
            double curr_error = fabs(I_3 * I_3 - 4 * I_2 * I_4);
            if (curr_error < min_error)
            {
                min_error = curr_error;
                min_index = j;
            }
        }
        if (min_error < error_threshold)
        {
            circle_pairs.push_back(std::make_tuple(i, min_index, min_error));
        }
    }

    if (circle_pairs.empty())
        return;

    // calculate the circle's pose
    double base_line; // in mm
    base_line = -right_projection(0, 3) / right_projection(0, 1);
    Eigen::Vector4d right_camera_origin;
    right_camera_origin << base_line, 0, 0, 1;

    for (int i = 0; i < circle_pairs.size(); i++)
    {
        double I_2, I_3, I_4;
        int left_id = std::get<0>(circle_pairs[i]);
        int right_id = std::get<1>(circle_pairs[i]);
        A = left_projection.transpose() * left_possible_ellipses[left_id] * left_projection;
        B = right_projection.transpose() * right_possible_ellipses[right_id] * right_projection;
        computeI2I3I4Analytic(A, B, I_2, I_3, I_4);

        double lambda = -I_3 / (2 * I_2);
        Eigen::Matrix4d C = A + lambda * B;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(C);
        if (eigensolver.info() != Eigen::Success)
        {
            std::cout << "failed to get the eigen values/vectors" << std::endl;
            continue;
        };
        Eigen::Vector4d eigen_values = eigensolver.eigenvalues(); //increasing order
        int eigen_non_zero_index[4];
        int id_non_zero_index = 0;

        for (int eigen_i = 0; eigen_i < 4; eigen_i++)
        {
            if (fabs(eigen_values[eigen_i]) > 0.00001)
            {
                eigen_non_zero_index[id_non_zero_index] = eigen_i;
                id_non_zero_index++;
            }
        }
        if (id_non_zero_index >= 2)
        {
            std::cout << "has more than 2 non zeros eigen values, seems something is wrong..." << std::endl;
        }
        if (eigen_values[eigen_non_zero_index[0]] * eigen_values[eigen_non_zero_index[1]] >= 0)
            continue;

        // find the plane of the circle
        Eigen::Vector4d p = std::sqrt(-eigen_values[eigen_non_zero_index[0]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[0]) + std::sqrt(eigen_values[eigen_non_zero_index[1]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[1]);
        if (p.dot(p - right_camera_origin) < 0)
        {
            p = std::sqrt(-eigen_values[eigen_non_zero_index[0]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[0]) - std::sqrt(eigen_values[eigen_non_zero_index[1]]) * eigensolver.eigenvectors().col(eigen_non_zero_index[1]);
        }

        // find the center of the circle
        // Here, we assume the center of the circle is also the center of the ellipse in the image.
        Eigen::Matrix3d ellipse_quadratic = left_possible_ellipses[left_id];
        double center_x = -ellipse_quadratic(0, 2) / ellipse_quadratic(0, 0);
        double center_y = -ellipse_quadratic(1, 2) / ellipse_quadratic(0, 0);
        double a = 1.0 / sqrt(ellipse_quadratic(0, 0));
        double p_x = center_x + a; // point on the ellipse
        double p_y = center_y;

        Eigen::Vector4d center_point_3d, circle_point_3d;
        computePointsInPlane(center_x, center_y, p, center_point_3d);
        computePointsInPlane(p_x, p_y, p, circle_point_3d);

        double r = (center_point_3d - circle_point_3d).norm();
        circles.push_back(Circle3D(center_point_3d, p, r));
    }
}

void RoughCircleSolver::getPossibleCircles(const cv::Mat &left_edge, const cv::Mat &right_edge, std::vector<Circle3D> &circles)
{
    /** Main function of this class, get all the possible circles in the image.
     * @ param left_edge : the left edge image
     * @ param right_edge : the right edge image
     * @ param circles : the circles contained in the left/right image.
    */
    std::vector<Eigen::Matrix3d> left_ellipses_vec, right_ellipses_vec;
    getPossibleEllipse(left_edge, left_ellipses_vec);
    getPossibleEllipse(right_edge, right_ellipses_vec);

    getCircles(left_ellipses_vec, right_ellipses_vec, circles);
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
        projection_matrix = stereo_cam_ptr->left.projectionEigenMatrix();
        break;
    }
    default:
        assert("camera id should be only one of the LEFT_CAMERA or RIGHT_CAMERA");
        break;
    }

    circle.getTransformMatrixToOrigin(transform_circle_to_camera);

    cv::Vec3b pixel_color(color(0), color(1), color(2));
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
        image.at<cv::Vec3b>((int)pixel_pos(0), (int)pixel_pos(1)) = pixel_color;
    }
}

void RoughCircleSolver::reprojectCircles(cv::Mat &image, const std::vector<Circle3D> &circles_vec, int camera_id, int sample_size, const cv::Scalar &color)
{
    for (int i = 0; i < circles_vec.size(); i++)
    {
        reprojectCircles(image, circles_vec[i], camera_id, sample_size, color);
    }
}
