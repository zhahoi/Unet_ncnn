#include "meter_seg.h"
#include <stdio.h>
#include <iostream>

#define seg_param "C:/CPlusPlus/MeterSeg/weights/unet_416.param"
#define seg_bin "C:/CPlusPlus/MeterSeg/weights/unet_416.bin"

MeterSegmentation* meterSeg = new MeterSegmentation(seg_param, seg_bin);

void processSegmentation(const cv::Mat& input_image)
{
    if (input_image.empty()) {
        printf("cv::imread read image failed\n");
        return;
    }

    cv::Mat seg_result = meterSeg->Process(input_image);

    cv::imshow("seg_result", seg_result);
    cv::waitKey(0);
}

void runSegmentation(const std::string& file_path) {
   
    cv::Mat input_image = cv::imread(file_path);
    if (input_image.empty()) {
        std::cerr << "Failed to load image: " << file_path << std::endl;
        exit;
    }
    processSegmentation(input_image);

    printf("segmentation end\n");
}

int main(int argc, char** argv)
{
    std::string file_path = "C:/CPlusPlus/MeterSeg/images/seg_result_40.jpg";
    runSegmentation(file_path);

    printf("meter end\n");
    return 0;
}