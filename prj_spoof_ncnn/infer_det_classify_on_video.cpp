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
//        feature[i] = mat[i];
        feature.push_back(mat[i]);
    }
    return feature;
}

int main(int args, char **argv) {
    // 确定capture
//    cv::VideoCapture capture;
//    if(args >= 2){
//        string video_path = argv[1];
//        capture = cv::VideoCapture(video_path);
//    }else{
//        capture = cv::VideoCapture(0);
//    }
    cv::VideoCapture capture("/home/cmf/my_video.avi");


    int src_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int src_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    printf("%d, %d\n", src_width, src_height);

    // det

    int sizew = FACE_DETECT_SIZEH;
    int sizeh = FACE_DETECT_SIZEH * src_height / src_width;

    // classify
    ncnn::Net net;
    net.load_param("spoof-Mb_Tiny_RFB-_Mb_Tiny_RFB.param");
    net.load_model("spoof-Mb_Tiny_RFB-_Mb_Tiny_RFB.bin");


    while (true) {
        cv::Mat frame;
        capture >> frame;
        // 检查
        if (frame.empty()) {
            printf("播放完成\r\n");
            break;
        }
        cv::Mat resize_frame;
        cv::resize(frame, resize_frame, cv::Size(sizew, sizeh));

//      检测人脸
        ncnn::Mat in = ncnn::Mat::from_pixels(resize_frame.data, ncnn::Mat::PIXEL_BGR, sizew, sizeh);
//        in.substract_mean_normalize(mean_vals,norm_vals);
        vector<FaceInfo> faceinfo = face_detect(in);
//        printf("%d", faceinfo.size());
        for (int i = 0; i < faceinfo.size(); i++) {
            FaceInfo face = faceinfo[i];
//            cout << face.score << endl;
            cv::rectangle(resize_frame, cv::Point(face.x[0], face.y[0]), cv::Point(face.x[1], face.y[1]),
                          cv::Scalar(0, 255, 0));

            cv::Mat crop_img = resize_frame(cv::Range(face.y[0], face.y[1]), cv::Range(face.x[0], face.x[1]));

            cv::imshow("crop", crop_img);

            ncnn::Mat in = ncnn::Mat::from_pixels_resize(crop_img.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                         crop_img.cols, crop_img.rows, 224, 224);

            float mean[3] = {0.0f, 0.0f, 0.0f};
            float std[3] = {1/255.f, 1/255.f, 1/255.f};

            in.substract_mean_normalize(mean, std);


            ncnn::Extractor ex = net.create_extractor();
            ex.set_light_mode(true);
            ex.set_num_threads(4);

            ex.input("input", in);

            ncnn::Mat out;
            ex.extract("output", out);
            printf("channel: %d, width: %d, height: %d \r\n", out.c, out.w, out.h);

            vector<float> feature = mat2vector(out);
            int index = argmax(feature);

            printf("out %f, %f, max index is %d \r\n", out[0], out[1], index);

//            string text = "";
//            fprintf(, to_string(index))
            cv::putText(resize_frame, to_string(index), cv::Point2d(20, 20),
                    1, 1., cv::Scalar(0, 255, 0));

        }

//        show
        cv::imshow("read video", resize_frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}