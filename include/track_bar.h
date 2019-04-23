#ifndef __TRACK_BAR_H__
#define __TRACK_BAR_H__

#include <opencv2/opencv.hpp>

class TrackBar{
public:
    int value;
    TrackBar(){};
    void setTrackBar(std::string window_name, std::string bar_name, int default_value, int max_value, void (&f_input)(int, void*), void* data_ptr){
        value = default_value;
        cv::createTrackbar(bar_name, window_name, &value, max_value, 
                        f_input, data_ptr);
    }
};

class CannyTrackBar{
public:
    TrackBar thresh1_bar, thresh2_bar;
    cv::Mat gray;
    std::string window_name;
    cv::Mat edge;

    CannyTrackBar(cv::Mat& gray_input, std::string window_name_input, int default_thresh1, int default_thresh2, int max_thresh1, int max_thresh2)
    {
        gray = gray_input.clone();
        window_name = window_name_input;
        cv::namedWindow(window_name, 0);
        thresh1_bar.setTrackBar(window_name_input,"tresh1", default_thresh1, max_thresh1, onTrackBar, reinterpret_cast<void*>(this));
        thresh2_bar.setTrackBar(window_name_input,"tresh2", default_thresh2, max_thresh2, onTrackBar, reinterpret_cast<void*>(this));
        applyCanny();
        cv::waitKey(0);        
        cv::destroyWindow(window_name);

    }
    void applyCanny(){
        cv::Canny(gray, edge, thresh1_bar.value, thresh2_bar.value, 5);
        cv::imshow(window_name, edge);

    }
    void trackbarCallback(int v){
        applyCanny();
    }
    static void onTrackBar(int v, void* ptr){
        reinterpret_cast<CannyTrackBar*>(ptr)->trackbarCallback(v);
    }
};


#endif