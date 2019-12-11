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

float distance(int p1x, int p1y, int p2x, int p2y) {
    int dx = p1x-p2x;
    int dy = p1y - p2y;
    return sqrt(dx * dx + dy * dy);
}

double getPos(FaceInfo faceInfo){
    int *p = faceInfo.landmark;

    int le = p[0], re = p[2], n = p[4], lm=p[6], rm=p[8];
    double range =abs(le-n)*1.0/abs(re-le);

    float chinDirection = abs(1.0 * (p[1] - p[3]) / (le - re + 1e-8));
    float noseDistance = abs(n-le) * 1.0 / abs(n-re + 1e-8);
//    float noseDistance = distance(p[4], p[5], p[0], p[1]) / distance(p[4], p[5], p[2], p[3]);
//    return range;
    return noseDistance;


    if(le > n and re < n and n < lm and n > rm) return 1; //扭头
    if (noseDistance > 1.5 || noseDistance < 0.5) return 2; //侧脸
    if (chinDirection > 0.3) return 3;//歪头



//    int ac = abs(p[4] - p[0]);
//    int bc = abs(p[5] - p[1]);
//    return bc*1.0/ac * 180 / 3.1415926;


    return 0;
}

cv::Mat ncnn2cv(ncnn::Mat in, bool show= false){
    cv::Mat out(in.h, in.w, CV_8UC3);
    in.to_pixels(out.data, ncnn::Mat::PIXEL_BGR);
    if (show){
        cv::imshow("ncnn2cv", out);
        cv::waitKey(0);
    }
    return out;
}

vector<float> detect_single_img(cv::Mat img) {
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    vector<FaceInfo> faceinfo = face_detect(in);
//    vector<float> feature = face_exactfeature(in, faceinfo[0]);
    vector<float> feature;
    int st, et, cnt;
    double costtime;
    st = clock();
    ncnn::Mat det = preprocess(in, faceinfo[0]);

//    et = clock();
//    costtime = et - st;
////    LOGD("face_exactfeature preprocess cost %fs\n", costtime / CLOCKS_PER_SEC);
//    st = clock();
    feature = arc.getFeature(det);
//    et = clock();
//    costtime = et - st;

    ncnn2cv(det, false);
    return feature;

}


int main(int args, char **argv) {
    string img_path = argv[1];

    cv::Mat img = cv::imread(img_path);
    printf("%d %d \r\n", img.cols, img.rows);

    vector<float> my_feature = detect_single_img(img);
    printf("%d\r\n", my_feature.size());
    cv::VideoCapture capture(0);
    int src_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int src_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    printf("%d, %d\n", src_width, src_height);

    int sizew = FACE_DETECT_SIZEH;
    int sizeh = FACE_DETECT_SIZEH * src_height / src_width;
    while (true) {
        cv::Mat frame;
        capture >> frame;
//        检查
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
            vector<float> feature = face_exactfeature(in, face);
//            printf("%d\r\n", feature.size());
//            cout << feature[0] << endl;
            float sim = face_calcSimilar(feature, my_feature);
//            printf("sim is %f\r\n", sim);
            double pos = getPos(face);
            cv::putText(resize_frame, to_string(sim), cv::Point(face.x[0], face.y[0]), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(0, 255, 0),1, 8,false);

            int *landmark = face.landmark;
            for(int i=0; i < 10; i+=2){
                cv::circle(resize_frame, cv::Point(landmark[i], landmark[i+1]), 1 , cv::Scalar(0, 255, 0));
            }


//            cv::putText(resize_frame, to_string(sim), cv::Point(face.x[0], face.y[0]), cv::FONT_HERSHEY_SIMPLEX, 1,
//                        cv::Scalar(0, 255, 0));
        }

//        show
        cv::imshow("read video", resize_frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}