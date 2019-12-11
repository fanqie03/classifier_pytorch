//
// Created by cmf on 2019/12/5.
//

#include "classifier.h"
#include "classify.id.h"
#include "classify.h"

Classifier::Classifier() {
    this->net.load_param(classify_param_bin);
    this->net.load_model(classify_bin);
}

Classifier::~Classifier() {
    this->net.clear();
}

/**
 *
 * @param img need bgr format, didn't normalize img
 * @return
 */
vector<float> Classifier::getFeature(const ncnn::Mat& img) {

    vector<float> feature;

    ncnn::Mat in = img;

    in = resize(in, 224, 224);
    in = bgr2rgb(in);


    in.substract_mean_normalize(this->mean, this->norm);

    ncnn::Extractor ex = this->net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(1);
    ex.input(classify_param_id::BLOB_input, in);

    ncnn::Mat out;
    ex.extract(classify_param_id::BLOB_output, out);

    feature.resize(this->feature_dim);
    for(int i = 0; i < this->feature_dim; i++)
        feature[i] = out[i];
    print(feature);


    return feature;
}

int Classifier::classify(const ncnn::Mat& img) {
    vector<float> feature = this->getFeature(img);
    return this->argmax(feature);
}

int Classifier::argmax(vector<float> feature){

    // find max index
    int index = 0;
    for(int i = 1; i < feature.size(); i++){
        if(feature[i] > feature[index]){
            index = i;
        }
    }

    return index;

}

ncnn::Mat Classifier::resize(ncnn::Mat src, int w, int h)
{
    int src_w = src.w;
    int src_h = src.h;
    unsigned char* u_src = new unsigned char[src_w * src_h * 3];
    src.to_pixels(u_src, ncnn::Mat::PIXEL_RGB);
    unsigned char* u_dst = new unsigned char[w * h * 3];
    ncnn::resize_bilinear_c3(u_src, src_w, src_h, u_dst, w, h);
    ncnn::Mat dst = ncnn::Mat::from_pixels(u_dst, ncnn::Mat::PIXEL_RGB, w, h);
    delete[] u_src;
    delete[] u_dst;
    return dst;
}

ncnn::Mat Classifier::bgr2rgb(ncnn::Mat src)
{
    int src_w = src.w;
    int src_h = src.h;
    unsigned char* u_rgb = new unsigned char[src_w * src_h * 3];
    src.to_pixels(u_rgb, ncnn::Mat::PIXEL_BGR2RGB);
    ncnn::Mat dst = ncnn::Mat::from_pixels(u_rgb, ncnn::Mat::PIXEL_RGB, src_w, src_h);
    delete[] u_rgb;
    return dst;
}

ncnn::Mat Classifier::rgb2bgr(ncnn::Mat src)
{
    return bgr2rgb(src);
}

void Classifier::print(vector<float> feature){
    string a = "";
    for(int i = 0; i < feature.size(); i++){
//        printf("%f, ", feature[i]);
        a += to_string(feature[i]) + ", ";
    }
//    printf("\r\n");
    a += "\r\n";
    printf(a.c_str());
}
