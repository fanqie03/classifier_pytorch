//
// Created by cmf on 2019/10/31.
//

#include <iostream>
#include <stdio.h>

#include <dirent.h>
#include <sys/io.h>
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
    int dx = p1x - p2x;
    int dy = p1y - p2y;
    return sqrt(dx * dx + dy * dy);
}

double getPos(FaceInfo faceInfo) {
    int *p = faceInfo.landmark;

    int le = p[0], re = p[2], n = p[4], lm = p[6], rm = p[8];
    double range = abs(le - n) * 1.0 / abs(re - le);

    float chinDirection = abs(1.0 * (p[1] - p[3]) / (le - re + 1e-8));
    float noseDistance = abs(n - le) * 1.0 / abs(n - re + 1e-8);
    return noseDistance;
}

cv::Mat ncnn2cv(ncnn::Mat in, int show = false) {
    cv::Mat out(in.h, in.w, CV_8UC3);
    in.to_pixels(out.data, ncnn::Mat::PIXEL_BGR);
    if (show) {
        cv::imshow("ncnn2cv", out);
        cv::waitKey(0);
    }
    return out;
}

void print_vector(vector<float> vector) {
    for (int i = 0; i < vector.size(); i++) {
        printf("%f ", vector[i]);
    }
    printf("\r\n");
}

vector<float> detect_single_img(cv::Mat img, int show_img=false) {
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    vector<FaceInfo> faceinfo = face_detect(in);
    vector<float> feature;
    int st, et, cnt;
    double costtime;
    st = clock();
//    feature = face_exactfeature(in, faceinfo[0]);
    ncnn::Mat det = preprocess(in, faceinfo[0]);
    feature = arc.getFeature(det);
    print_vector(feature);
    ncnn2cv(det, show_img);
    return feature;
}

vector<vector<float>> detect_imgs(vector<cv::Mat> imgs, int show_img = false) {
    vector<vector<float>> embeddings;
    vector<float> embedding;
    int i, size = imgs.size();
    for (i = 0; i < size; i++) {
        embedding = detect_single_img(imgs[i], show_img);
        embeddings.push_back(embedding);
    }
    return embeddings;
}

vector<string> list_file(string root, bool absolute_path = true) {

//    std::string inPath = "E:\\image\\image\\*.jpg";//遍历文件夹下的所有.jpg文件
    vector<string> v;
    // 目录信息结构体，用于保存正在被读取的目录的有关信息
    DIR *dirptr = NULL;
    // dirent结构体不仅指向目录，还指向目录中的具体文件
    struct dirent *entry;
// 使用opendir打开一个目录，该函数返回指向DIR结构体的指针
    if ((dirptr = opendir(root.c_str())) == NULL) {
        cout << "Open dir error!" << endl;
    } else {
        // 使用readdir读取目录中的所有文件
        // 默认情况下，包括了'.'和'..'目录文件
        while (entry = readdir(dirptr)) {
            // 判断是否为普通类型的文件
            // 这里用于区别于其他类型（如目录类型、管道类型等）的文件
            if (DT_REG == entry->d_type) {
                // 打印文件名
                string file_path;

                file_path = absolute_path ? root + "/" + entry->d_name : entry->d_name;

                cout << "DIR '" << root << "' include FILE: " << entry->d_name << " " << file_path << endl;

                v.push_back(file_path);
            }
        }
        // 关闭目录
        closedir(dirptr);
    }
    return v;
}

vector<cv::Mat> read_imgs(vector<string> imgs_path) {
    vector<cv::Mat> r;
    cv::Mat m;
    for (int i = 0; i < imgs_path.size(); i++) {
        m = cv::imread(imgs_path[i]);
        r.push_back(m);
    }
    return r;
}


vector<float> compare_othres(vector<float> embedding, vector<vector<float>> embeddings) {
    vector<float> dist;
    int i = 0, size = embeddings.size();
    for (i = 0; i < size; i++) {
        float d = face_calcSimilar(embedding, embeddings[i]);
        dist.push_back(d);
    }
    return dist;
}

vector<vector<float>> compare_each(vector<vector<float>> embeddings) {
    vector<vector<float>> dist_array;
    int i = 0, size = embeddings.size();
    for (i = 0; i < size; i++) {
        dist_array.push_back(compare_othres(embeddings[i], embeddings));
    }
    return dist_array;
}

int main(int argc, char **argv) {
    string img_path = argv[1];
    int show_img = atoi(argv[2]);
    vector<string> imgs_path = list_file(img_path);
    vector<string> imgs_name = list_file(img_path, false);
    vector<cv::Mat> imgs = read_imgs(imgs_path);
    vector<vector<float>> embeddings = detect_imgs(imgs, show_img);
    printf("embeddings size is %d\r\n", embeddings.size());
    vector<vector<float>> dist_array = compare_each(embeddings);

    // 打印结果

    printf("%-8s ", "name");

    for (int i = 0; i < imgs_name.size(); i++) {
        printf("%-8s ", imgs_name[i].c_str());
    }

    printf("\r\n");

    for (int i = 0; i < dist_array.size(); i++) {
        printf("%-8s ", imgs_name[i].c_str());
        for (int j = 0; j < dist_array.size(); j++) {
            printf("%f ", dist_array[i][j]);
        }
        printf("\r\n");
    }

    return 0;
}