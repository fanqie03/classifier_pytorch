//
// Created by cmf on 2019/12/5.
//

#include "interface.h"
#include "net.h"

Classifier g_classifier;

int classify(const ncnn::Mat& img){

    int st, et, cnt;
    double costtime;
    st = clock();
    printf("classify start\r\n");

    int ret = g_classifier.classify(img);

    printf("ret is %d\r\n", ret);
    et = clock();
    costtime = et - st;
    printf("classify cost %f\r\n", costtime/CLOCKS_PER_SEC);

//    vector<float> ret;
    return ret;
}