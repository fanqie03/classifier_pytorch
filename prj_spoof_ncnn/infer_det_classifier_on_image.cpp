//
// Created by cmf on 2019/12/5.
//

#include <iostream>
//#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mat.h"
#include "net.h"
#include <string>

#include "arcface/interface.h"
#include "classify/interface.h"

#define FACE_DETECT_SIZEH   448
using namespace std;
Arcface arc;


cv::Mat ncnn2cv(ncnn::Mat in, bool show= false){
    cv::Mat out(in.h, in.w, CV_8UC3);
    in.to_pixels(out.data, ncnn::Mat::PIXEL_BGR);
    if (show){
        cv::imshow("ncnn2cv", out);
        cv::waitKey(0);
    }
    return out;
}


int argmax(vector<float> feature){

    // find max index
    int index = 0;
    for(int i = 1; i < feature.size(); i++){
        if(feature[i] > feature[index]){
            index = i;
        }
    }

    return index;

}

vector<float> mat2vector(ncnn::Mat mat, int size=2){
    vector<float> feature;
    for(int i = 0; i < size; i++){
        feature.push_back(mat[i]);
    }
    return feature;
}

int main(int args, char **argv) {

//    cv::Mat frame = cv::imread("/home/cmf/datasets/spoof/NUAADetectedface/Detectedface/ImposterFace/0001/0001_00_00_01_0.jpg");

    cv::Mat frame = cv::imread("/home/cmf/datasets/spoof/NUAADetectedface/Detectedface/ClientFace/0001/0001_00_00_01_0.jpg");

    int src_width = frame.cols;
    int src_height = frame.rows;
    printf("%d, %d\n", src_width, src_height);

    // det

    int sizew = FACE_DETECT_SIZEH;
    int sizeh = FACE_DETECT_SIZEH * src_height / src_width;


    cv::Mat resize_frame;
    cv::resize(frame, resize_frame, cv::Size(sizew, sizeh));

    cv::Mat resize_frame2(resize_frame);

//      检测人脸
    ncnn::Mat in = ncnn::Mat::from_pixels(resize_frame2.data, ncnn::Mat::PIXEL_BGR, sizew, sizeh);
//        in.substract_mean_normalize(mean_vals,norm_vals);
    vector<FaceInfo> faceinfo = face_detect(in);
//        printf("%d", faceinfo.size());
    for (int i = 0; i < faceinfo.size(); i++) {
        FaceInfo face = faceinfo[i];

        cv::Mat crop_img(resize_frame, cv::Range(face.y[0], face.y[1]),
                cv::Range(face.x[0], face.x[1]));

        printf("%d\r\n", resize_frame.isContinuous());

        cv::imshow("crop", crop_img);

        cv::imwrite("test.jpg", crop_img);

        crop_img=cv::imread("test.jpg");

        cv::rectangle(resize_frame,
                cv::Point(face.x[0], face.y[0]),
                cv::Point(face.x[1], face.y[1]),
                cv::Scalar(0, 255, 0));


        ncnn::Mat inn = ncnn::Mat::from_pixels(crop_img.data,
                ncnn::Mat::PIXEL_BGR, crop_img.cols, crop_img.rows);
//        ncnn::Mat inn = ncnn::Mat::from_pixels(resize_frame.data,
//               ncnn::Mat::PIXEL_BGR, resize_frame.cols, resize_frame.rows);

        int index = classify(inn);

        printf("max index is %d \r\n", index);

        cv::putText(resize_frame, to_string(index), cv::Point2d(20, 20),
                1, 1., cv::Scalar(0, 255, 0));

    }

//        show
    cv::imshow("read video", resize_frame);
    cv::waitKey();

}