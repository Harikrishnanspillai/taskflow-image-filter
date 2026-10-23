#include "benchmark.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>

namespace {
std::string manifest_for(const std::vector<std::shared_ptr<ImageJob>>& jobs) {
  std::string manifest;
  for (const auto& job : jobs) {
    manifest += job->input_path;
    manifest += '\n';
  }
  return manifest;
}

void print_throughput(size_t count, double ms) {
  if (ms > 0.0) {
    std::cout << "Batch throughput: "
              << (1000.0 * static_cast<double>(count) / ms)
              << " images/s\n";
  }
}
}

void print_benchmark_summary(
    const std::vector<std::shared_ptr<ImageJob>>& jobs,
    double end_to_end_ms) {
  double preprocess_sum = 0.0, cpu_sum = 0.0, setup_sum = 0.0;
  double h2d_sum = 0.0, kernels_sum = 0.0, d2h_sum = 0.0, gpu_run_sum = 0.0;
  size_t gaussian_mismatches = 0, sobel_mismatches = 0;

  for (const auto& ptr : jobs) {
    const auto& job = *ptr;
    preprocess_sum += job.preprocess_ms;
    cpu_sum += job.cpu_ms;
    setup_sum += job.gpu_setup_ms;
    h2d_sum += job.gpu_h2d_ms;
    kernels_sum += job.gpu_kernels_ms;
    d2h_sum += job.gpu_d2h_ms;
    gpu_run_sum += job.gpu_ms;
    if (job.max_gaussian_diff != 0.0) ++gaussian_mismatches;
    if (job.max_sobel_diff != 0.0) ++sobel_mismatches;
  }

  std::cout << "\n=== CPU/GPU correctness validation summary ===\n"
            << "Images processed: " << jobs.size() << '\n'
            << "Sum of preprocessing task times: " << preprocess_sum << " ms\n"
            << "Sum of CPU reference filter times: " << cpu_sum << " ms\n"
            << "GPU setup/allocation: " << setup_sum << " ms\n"
            << "GPU H2D transfer sum: " << h2d_sum << " ms\n"
            << "GPU kernel graph sum (CUDA events): " << kernels_sum << " ms\n"
            << "GPU D2H transfer sum: " << d2h_sum << " ms\n"
            << "GPU per-image run wall-time sum: " << gpu_run_sum << " ms\n"
            << "Validation-run wall time (NOT used for speedup): " << end_to_end_ms << " ms\n"
            << "Gaussian outputs with nonzero max difference: " << gaussian_mismatches << '\n'
            << "Sobel outputs with nonzero max difference: " << sobel_mismatches << '\n';
}

void print_cpu_only_benchmark_summary(
    const std::vector<std::shared_ptr<ImageJob>>& jobs,
    double end_to_end_ms) {
  double preprocess_sum = 0.0, cpu_sum = 0.0;
  for (const auto& ptr : jobs) {
    preprocess_sum += ptr->preprocess_ms;
    cpu_sum += ptr->cpu_ms;
  }

  std::cout << "\n=== CPU-only benchmark summary ===\n"
            << "Images processed: " << jobs.size() << '\n'
            << "Sum of preprocessing task times: " << preprocess_sum << " ms\n"
            << "Sum of CPU filter task times: " << cpu_sum << " ms\n"
            << "CPU-only end-to-end wall time: " << end_to_end_ms << " ms\n";
  print_throughput(jobs.size(), end_to_end_ms);
}

void print_gpu_only_benchmark_summary(
    const std::vector<std::shared_ptr<ImageJob>>& jobs,
    double end_to_end_ms) {
  double preprocess_sum = 0.0, setup_sum = 0.0, h2d_sum = 0.0;
  double kernels_sum = 0.0, d2h_sum = 0.0, gpu_run_sum = 0.0;
  for (const auto& ptr : jobs) {
    const auto& job = *ptr;
    preprocess_sum += job.preprocess_ms;
    setup_sum += job.gpu_setup_ms;
    h2d_sum += job.gpu_h2d_ms;
    kernels_sum += job.gpu_kernels_ms;
    d2h_sum += job.gpu_d2h_ms;
    gpu_run_sum += job.gpu_ms;
  }

  std::cout << "\n=== GPU-only filtering pipeline benchmark ===\n"
            << "Images processed: " << jobs.size() << '\n'
            << "Sum of preprocessing task times: " << preprocess_sum << " ms\n"
            << "GPU setup/allocation: " << setup_sum << " ms\n"
            << "GPU H2D transfer sum: " << h2d_sum << " ms\n"
            << "GPU kernel graph sum (CUDA events): " << kernels_sum << " ms\n"
            << "GPU D2H transfer sum: " << d2h_sum << " ms\n"
            << "GPU per-image run wall-time sum: " << gpu_run_sum << " ms\n"
            << "GPU-only end-to-end wall time: " << end_to_end_ms << " ms\n";
  print_throughput(jobs.size(), end_to_end_ms);
}


void record_and_print_speedup(double cpu_ms, double gpu_ms) {
  if (cpu_ms <= 0.0 || gpu_ms <= 0.0) {
    std::cerr << "Invalid benchmark times; CSV not updated.\n";
    return;
  }

  std::filesystem::create_directories("results");

  const std::string csv_file = "results/benchmark.csv";
  const bool needs_header =
      !std::filesystem::exists(csv_file) ||
      std::filesystem::file_size(csv_file) == 0;

  int run_number = 1;

  if (!needs_header) {
    std::ifstream existing_csv(csv_file);
    std::string line;

    while (std::getline(existing_csv, line)) {
      if (!line.empty() && line.rfind("label,", 0) != 0) {
        ++run_number;
      }
    }
  }

  std::ofstream csv(csv_file, std::ios::app);

  if (!csv) {
    std::cerr << "Could not open " << csv_file << '\n';
    return;
  }

  if (needs_header) {
    csv << "label,cpu_ms,gpu_ms,end_to_end_ms\n";
  }

  csv << "Run " << run_number << ','
      << std::fixed << std::setprecision(3)
      << cpu_ms << ','
      << gpu_ms << ','
      << gpu_ms << '\n';

  std::cout << "\n=== CPU vs GPU comparison ===\n"
            << "CPU-only wall time: " << cpu_ms << " ms\n"
            << "GPU filtering phase wall time: " << gpu_ms << " ms\n"
            << "Speedup: " << cpu_ms / gpu_ms << "x\n"
            << "Saved comparison as Run " << run_number
            << " in " << csv_file << '\n';
}
