#include "meter_seg.h"
#include "gpu.h"
#include <mutex>
#include <iostream>


MeterSegmentation::MeterSegmentation(const char* param, const char* bin)
{
    meterSeg.opt.use_vulkan_compute = false;
    meterSeg.opt.use_bf16_storage = false;
    meterSeg.load_param(param);
    meterSeg.load_model(bin);
}


bool MeterSegmentation::run(const cv::Mat& img, ncnn::Mat& res)
{
    if (img.empty())
        return false;

    // 预处理
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, UNET_TARGET_SIZE, UNET_TARGET_SIZE);
    input.substract_mean_normalize(mean, std);

    ncnn::Extractor ex = meterSeg.create_extractor();
    ex.input("images", input);
    ex.extract("output", res);

    // Softmax
    Softmax(res);

    return true;
}

MeterSegmentation::~MeterSegmentation()
{
    meterSeg.clear();
}

float MeterSegmentation::ResizeImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color = cv::Scalar(128, 128, 128)) {
    cv::Size shape = image.size();
    float scale = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);

    int new_width = static_cast<int>(shape.width * scale);
    int new_height = static_cast<int>(shape.height * scale);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);

    out_image = cv::Mat(new_shape, image.type(), color);
    resized_image.copyTo(out_image(cv::Rect((new_shape.width - new_width) / 2, (new_shape.height - new_height) / 2, new_width, new_height)));

    return scale;
}


void MeterSegmentation::Softmax(ncnn::Mat& res)
{
    for (int i = 0; i < res.h; i++)
    {
        for (int j = 0; j < res.w; j++)
        {
            float max = -FLT_MAX;
            for (int q = 0; q < res.c; q++)
            {
                max = std::max(max, res.channel(q).row(i)[j]);
            }

            float sum = 0.0f;
            for (int q = 0; q < res.c; q++)
            {
                res.channel(q).row(i)[j] = exp(res.channel(q).row(i)[j] - max);
                sum += res.channel(q).row(i)[j];
            }

            for (int q = 0; q < res.c; q++)
            {
                res.channel(q).row(i)[j] /= sum;
            }
        }
    }
}

inline void random_color(std::vector<std::vector<int>>& color_list, int num)
{
    for (int i = 0; i < num; i++)
    {
        int B = rand() % 255;
        int G = rand() % 255;
        int R = rand() % 255;
        color_list.push_back({ B, G, R });
    }
}

cv::Mat MeterSegmentation::Visualizer(ncnn::Mat res)
{
    std::vector<std::vector<int>> color_list;
    random_color(color_list, res.c);
    cv::Mat seg_result(res.h, res.w, CV_8UC3);

    for (int i = 0; i < res.h; i++)
    {
        for (int j = 0; j < res.w; j++)
        {
            float max = res.channel(0)[res.w * i + j];
            int index = 0;
            for (int k = 0; k < res.c; k++)
            {
                if (res.channel(k)[res.w * i + j] > max)
                {
                    index = k;
                    max = res.channel(k)[res.w * i + j];
                }
            }
            seg_result.at<cv::Vec3b>(i, j)[0] = color_list[index][0];  // B
            seg_result.at<cv::Vec3b>(i, j)[1] = color_list[index][1];  // G
            seg_result.at<cv::Vec3b>(i, j)[2] = color_list[index][2];  // R
        }
    }

    return seg_result;
}

cv::Mat MeterSegmentation::Process(const cv::Mat& input_image)
{

    std::cout << "current image shape: " << input_image.rows << ", " << input_image.cols << std::endl;

#ifdef VISUALIZE
    cv::imshow("input_image: ", input_image);
    cv::waitKey(0);
#endif

    // 将图像大小调整为目标大小
    cv::Mat resize_image;
    float scale = ResizeImage(input_image, resize_image, cv::Size(UNET_TARGET_SIZE, UNET_TARGET_SIZE), cv::Scalar(128, 128, 128));
    std::cout << "current scale: " << scale << std::endl;

#ifdef VISUALIZE
    cv::imshow("resize_image", resize_image);
    cv::waitKey(0);
#endif

    // 运行分割模型
    ncnn::Mat res;
    run(resize_image, res);

#ifdef VISUALIZE
    std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;
#endif

    /*
    // 代码后处理(针对仪表检测任务)
    cv::Mat mask = cv::Mat(UNET_TARGET_SIZE, UNET_TARGET_SIZE, CV_8UC1, cv::Scalar(0));

    const float* class0mask = res.channel(0);  // background
    const float* class1mask = res.channel(1);  // pointer
    const float* class2mask = res.channel(2);  // scale

    // 遍历每个像素，确定其类别
    for (int i = 0; i < UNET_TARGET_SIZE; i++) {
        for (int j = 0; j < UNET_TARGET_SIZE; j++) {
            int num = i * UNET_TARGET_SIZE + j;
            if ((class1mask[num] > class2mask[num]) && (class1mask[num] > class0mask[num])) {
                mask.at<uchar>(i, j) = 1;
            }
            else if ((class2mask[num] > class1mask[num]) && (class2mask[num] > class0mask[num])) {
                mask.at<uchar>(i, j) = 2;
            }
        }
    }
    */

    cv::Mat seg_result = Visualizer(res);

    res.release();

    return seg_result;
}