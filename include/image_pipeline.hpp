#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

class GpuPipeline;

struct ImageJob {
  std::string input_path;
  std::string stem;
  cv::Mat input;
  cv::Mat gaussian_cpu;
  cv::Mat sobel_cpu;
  cv::Mat gaussian_gpu;
  cv::Mat sobel_gpu;
  double preprocess_ms = 0.0;
  double cpu_ms = 0.0;
  double gpu_setup_ms = 0.0;
  double gpu_h2d_ms = 0.0;
  double gpu_kernels_ms = 0.0;
  double gpu_d2h_ms = 0.0;
  double gpu_ms = 0.0;
  double max_gaussian_diff = 0.0;
  double max_sobel_diff = 0.0;
};

bool load_and_preprocess(ImageJob& job, std::string& error);
void run_cpu_filters(ImageJob& job);
void run_gpu_filters(ImageJob& job, GpuPipeline& pipeline);
void aggregate_and_save(ImageJob& job);
void save_cpu_only(ImageJob& job);
void save_gpu_only(ImageJob& job);
double max_abs_difference(const cv::Mat& a, const cv::Mat& b);
