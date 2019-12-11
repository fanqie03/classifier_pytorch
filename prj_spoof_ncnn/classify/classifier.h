//
// Created by cmf on 2019/12/5.
//

#ifndef CLASSIFIER_CLASSIFIER_H
#define CLASSIFIER_CLASSIFIER_H

#include "net.h"
#include <vector>
#include <string>

using namespace std;


class Classifier {
public:
    Classifier();
    ~Classifier();

    /**
     *
     * @return network output of feature vector
     */
    vector<float> getFeature(const ncnn::Mat& img);
    /**
     *
     * @return index
     */
    int classify(const ncnn::Mat& img);
private:
    ncnn::Net net;

    const int feature_dim = 2;

    int argmax(vector<float> feature);

    float mean[3] = {0.0f, 0.0f, 0.0f};
    float norm[3] = {1/255.f, 1/255.f, 1/255.f};

    ncnn::Mat resize(ncnn::Mat src, int w, int h);

    ncnn::Mat bgr2rgb(ncnn::Mat src);

    ncnn::Mat rgb2bgr(ncnn::Mat src);

    void print(vector<float> feature);


};


#endif //CLASSIFIER_CLASSIFIER_H
