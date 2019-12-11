//
// Created by cmf on 2019/12/5.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int main(int args, char **argv){
    cv::VideoCapture capture("/home/cmf/xfl.mp4");

    while(true){
        cv::Mat frame;
        capture >> frame;

        cv::imshow("show", frame);
        cv::waitKey(1);
    }
}