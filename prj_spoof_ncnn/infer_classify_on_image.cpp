//
// Created by cmf on 2019/12/5.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"


int main(int args, char **argv){
//    cv::VideoCapture capture("/home/cmf/my_video.avi");


//    cv::Mat frame = cv::imread("/home/cmf/datasets/spoof/NUAADetectedface/Detectedface/ImposterFace/0001/0001_00_00_01_0.jpg");
    cv::Mat frame = cv::imread("/home/cmf/ir-avator.jpg");

    printf("frame width %d, frame height %d\r\n", frame.cols, frame.rows);

    ncnn::Net net;
    net.load_param("classify.param");
    net.load_model("classify.bin");

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR2RGB,
            frame.cols, frame.rows, 224, 224);

    float mean[3] = {0.0f, 0.0f, 0.0f};
    float norm[3] = {1/255.f, 1/255.f, 1/255.f};

    in.substract_mean_normalize(mean, norm);


    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);
    printf("channel: %d, width: %d, height: %d \r\n", out.c, out.w, out.h);

    printf("out %f, %f \r\n", out[0], out[1]);

    cv::imshow("show", frame);
    cv::waitKey();
}