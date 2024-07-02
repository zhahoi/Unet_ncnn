#ifndef _METER_SEG_H_
#define _METER_SEG_H_

#include "net.h"
#include "mat.h"
#include "layer.h"
#include <opencv2/opencv.hpp>

#define UNET_TARGET_SIZE 416  // target image size after resize, might use 416 for small model

class MeterSegmentation {

public:
    MeterSegmentation(const char* param, const char* bin);
    ~MeterSegmentation();
    bool run(const cv::Mat& img, ncnn::Mat& res);
    cv::Mat Process(const cv::Mat& input_image);
    float ResizeImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color);
    void Softmax(ncnn::Mat& res);
    // 可视化
    cv::Mat Visualizer(ncnn::Mat res);

private:
    ncnn::Net meterSeg;

    const float mean[3] = { 0., 0., 0. };
    const float std[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
};

#endif
