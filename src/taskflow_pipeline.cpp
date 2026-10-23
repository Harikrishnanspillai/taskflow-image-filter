#include "image_pipeline.hpp"
#include "cpu_filters.hpp"
#include "gpu_filters.cuh"

#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;

double max_abs_difference(const cv::Mat& a, const cv::Mat& b) {
  if (a.size() != b.size() || a.type() != b.type()) return -1.0;
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  double max_value = 0.0;
  cv::minMaxLoc(diff, nullptr, &max_value);
  return max_value;
}

void run_cpu_filters(ImageJob& job) {
  const auto start = Clock::now();
  gaussian_blur_cpu(job.input, job.gaussian_cpu);
  sobel_cpu(job.gaussian_cpu, job.sobel_cpu);
  job.cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void run_gpu_filters(ImageJob& job, GpuPipeline& pipeline) {
  const GpuTimings t = pipeline.run(job.input, job.gaussian_gpu, job.sobel_gpu);
  job.gpu_setup_ms = t.setup_ms;
  job.gpu_h2d_ms = t.h2d_ms;
  job.gpu_kernels_ms = t.kernels_ms;
  job.gpu_d2h_ms = t.d2h_ms;
  job.gpu_ms = t.total_ms;
}

void aggregate_and_save(ImageJob& job) {
  std::filesystem::create_directories("results");
  job.max_gaussian_diff = max_abs_difference(job.gaussian_cpu, job.gaussian_gpu);
  job.max_sobel_diff = max_abs_difference(job.sobel_cpu, job.sobel_gpu);

  const std::string prefix = "results/" + job.stem;
  if (!cv::imwrite(prefix + "_gaussian_cpu.png", job.gaussian_cpu) ||
      !cv::imwrite(prefix + "_sobel_cpu.png", job.sobel_cpu) ||
      !cv::imwrite(prefix + "_gaussian_gpu.png", job.gaussian_gpu) ||
      !cv::imwrite(prefix + "_sobel_gpu.png", job.sobel_gpu)) {
    throw std::runtime_error("Failed to save output images for: " + job.input_path);
  }

  std::cout << "\n" << job.input_path << " (" << job.input.cols << "x"
            << job.input.rows << ")\n"
            << "  Preprocessing: " << job.preprocess_ms << " ms\n"
            << "  CPU filters:   " << job.cpu_ms << " ms\n"
            << "  GPU setup:     " << job.gpu_setup_ms << " ms\n"
            << "  GPU H2D:       " << job.gpu_h2d_ms << " ms\n"
            << "  GPU kernels:   " << job.gpu_kernels_ms << " ms (CUDA events)\n"
            << "  GPU D2H:       " << job.gpu_d2h_ms << " ms\n"
            << "  GPU run total: " << job.gpu_ms << " ms (excludes one-time setup)\n"
            << "  Gaussian max pixel difference: " << job.max_gaussian_diff << '\n'
            << "  Sobel max pixel difference:    " << job.max_sobel_diff << '\n';
  if (job.max_gaussian_diff == 0 && job.max_sobel_diff == 0)
    std::cout << "  Validation: PASS (exact pixel match)\n";
  else
    std::cout << "  Validation: CHECK output differences\n";
}


void save_cpu_only(ImageJob& job) {
  std::filesystem::create_directories("results");
  const std::string prefix = "results/" + job.stem;
  if (!cv::imwrite(prefix + "_gaussian_cpu.png", job.gaussian_cpu) ||
      !cv::imwrite(prefix + "_sobel_cpu.png", job.sobel_cpu)) {
    throw std::runtime_error("Failed to save CPU-only output for: " + job.input_path);
  }
}


void save_gpu_only(ImageJob& job) {
  std::filesystem::create_directories("results");
  const std::string prefix = "results/" + job.stem;
  if (!cv::imwrite(prefix + "_gaussian_gpu.png", job.gaussian_gpu) ||
      !cv::imwrite(prefix + "_sobel_gpu.png", job.sobel_gpu)) {
    throw std::runtime_error("Failed to save GPU output for: " + job.input_path);
  }
}
