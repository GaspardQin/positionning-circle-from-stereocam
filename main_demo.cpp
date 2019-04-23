#include <iostream>
#include "rough_circle_solver.h"
#include "track_bar.h"

#include <string>
#include <cassert>

void setCamera(std::shared_ptr<StereoCameraModel>& stereo_cam_ptr){
    
	int image_size[2] = {2208, 1242};
	//left camera
	double left_intrinsic[9] = { 1404.6805178647157, 0, 1093.7708076703914, 0, 1404.9562302837180, 650.19805634924830, 0, 0, 1 };
	double left_distortion[5] = { -0.15884502830994698, -0.017019860223729141, -0.00050845821974763127, -0.00049227193638006373, 0.042091851482026002 };
    double left_projection[12] = {1404.680517864716, 0, 1093.770807670391, 0, 0, 1404.956230283718, 650.1980563492483, 0, 0, 0, 1, 0};

	//right camera
	double right_intrinsic[9] = { 1395.0427725909017, 0, 1065.3453232201614, 0, 1393.9191585585920, 681.76116308075120, 0, 0, 1 };
	double right_distortion[5] = { -0.15815821923496060, -0.023252410765977595, -0.00088122276582216434, -0.00048619722968107478, 0.047581138754438562 };
    double right_projection[12] = {1376.915308200754, 2.002081978942329, 1088.671309797108, -169123.8601871156, -14.5707534390388, 1393.226512711618, 683.0201179347162, -157.3182194049588, -0.01682694049012899, -0.001009946069590211, 0.9998579069461211, -1.584014773188137};

    stereo_cam_ptr->initLeft(left_intrinsic, left_distortion, left_projection, image_size);
    stereo_cam_ptr->initRight(right_intrinsic, right_distortion, right_projection, image_size);
}
 
int main(int argc, char** argv) {
    // read raw image
    if(argc <= 1){
        assert("please input the id of images");
    }
    std::string path = "images/";
    std::string left_raw_image_name = "raw_left";
    left_raw_image_name += std::string(argv[1]) + ".png";
    std::string right_raw_image_name = "raw_right";
    right_raw_image_name += std::string(argv[1]) + ".png";

    cv::Mat left_raw_image = cv::imread(path+left_raw_image_name);
    if(left_raw_image.empty()){
        std::cout << "cannot load image: "<<path <<left_raw_image_name;
        abort();
    }
    cv::Mat right_raw_image = cv::imread(path+right_raw_image_name);
    if(right_raw_image.empty()){
        std::cout << "cannot load image: "<<path <<right_raw_image_name;
        abort();
    }

    cv::Mat left_rectified_image, right_rectified_image;
    std::shared_ptr<StereoCameraModel> stereo_cam_ptr = std::make_shared<StereoCameraModel>();
    setCamera(stereo_cam_ptr);

    stereo_cam_ptr->left.rectifyImage(left_raw_image, left_rectified_image);
    stereo_cam_ptr->right.rectifyImage(right_raw_image, right_rectified_image);

    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_rectified_image, left_gray, CV_BGR2GRAY);
    cv::cvtColor(right_rectified_image, right_gray, CV_BGR2GRAY);

    cv::Mat left_edge, right_edge;
    CannyTrackBar canny_bar_left(left_gray, "canny_bar_of_left_image", 500, 1500, 3000, 5000);
    CannyTrackBar canny_bar_right(right_gray, "canny_bar_of_right_image", 500, 1500, 3000, 5000);
    left_edge = canny_bar_left.edge;
    right_edge = canny_bar_right.edge;

    RoughCircleSolver solver(stereo_cam_ptr);
    std::vector<Circle3D> circles;
    solver.getPossibleCircles(left_edge, right_edge, circles);

    cv::Mat left_show_image = left_rectified_image.clone();
    cv::Mat right_show_image = right_rectified_image.clone();

    solver.reprojectCircles(left_show_image, circles, LEFT_CAMERA, 50, cv::Scalar(255,0,0));
    solver.reprojectCircles(right_show_image, circles, RIGHT_CAMERA, 50, cv::Scalar(255,0,0));
    cv::imshow("left_reproject", left_show_image);
    cv::imshow("right_reproject", right_show_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
