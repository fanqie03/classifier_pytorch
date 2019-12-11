//
// Created by cmf on 2019/12/5.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "classify/interface.h"


int main(int args, char **argv){


//    cv::Mat frame = cv::imread("/home/cmf/datasets/spoof/NUAADetectedface/Detectedface/ImposterFace/0001/0001_00_00_01_0.jpg");

//    cv::Mat frame = cv::imread("/home/cmf/datasets/spoof/NUAADetectedface/Detectedface/ClientFace/0001/0001_00_00_01_0.jpg");

    cv::Mat frame = cv::imread("/home/cmf/ir-avator.jpg");

    printf("frame width %d, frame height %d\r\n", frame.cols, frame.rows);

//    frame = frame(cv::Range(0, frame.rows - 1),
//            cv::Range(0, frame.cols - 1));

    ncnn::Mat in = ncnn::Mat::from_pixels(frame.data,
            ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);

    int index = classify(in);

    printf("index is %d\r\n", index);

    cv::imshow("show", frame);
    cv::waitKey();

}