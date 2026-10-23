#pragma once
#include <opencv2/core.hpp>

void gaussian_blur_cpu(const cv::Mat& input, cv::Mat& output);
void sobel_cpu(const cv::Mat& input, cv::Mat& output);
