//
// Created by cmf on 2019/12/5.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "classify/interface.h"


int main(int args, char **argv){
    cv::VideoCapture capture("/home/cmf/my_video.avi");


    while(true){
        cv::Mat frame;
        capture >> frame;

        ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR,
                                              frame.cols, frame.rows);

        int index = classify(in);

        printf("index is %d\r\n", index);

        cv::imshow("show", frame);
        cv::waitKey(1);
    }
}