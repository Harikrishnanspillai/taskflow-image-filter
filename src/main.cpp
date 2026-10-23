#include "image_pipeline.hpp"
#include "gpu_filters.cuh"
#include "benchmark.hpp"

#include <taskflow/taskflow.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  bool validate = false;
  std::vector<std::string> image_paths;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--validate") {
      validate = true;
    } else {
      image_paths.push_back(arg);
    }
  }

  if (image_paths.empty()) {
    std::cerr
      << "Usage:\n"
      << "  " << argv[0] << " <images...>           # CPU + GPU benchmark\n"
      << "  " << argv[0] << " --validate <images...> # Correctness check\n";
    return 1;
  }

  tf::Executor executor;

  // Create independent jobs for the CPU and GPU measurements.
  std::vector<std::shared_ptr<ImageJob>> cpu_jobs;
  cpu_jobs.reserve(image_paths.size());

  for (const auto& path : image_paths) {
    auto job = std::make_shared<ImageJob>();
    job->input_path = path;
    cpu_jobs.push_back(std::move(job));
  }

  // Phase 1: complete CPU pipeline.
  tf::Taskflow cpu_flow;

  for (const auto& job : cpu_jobs) {
    auto preprocess = cpu_flow.emplace([job] {
      const auto start = Clock::now();
      std::string error;

      if (!load_and_preprocess(*job, error)) {
        throw std::runtime_error(error);
      }

      job->preprocess_ms =
        std::chrono::duration<double, std::milli>(
          Clock::now() - start).count();
    }).name("CPU preprocessing");

    auto cpu = cpu_flow.emplace([job] {
      run_cpu_filters(*job);
    }).name("CPU Gaussian and Sobel");

    auto save = cpu_flow.emplace([job] {
      save_cpu_only(*job);
    }).name("Save CPU outputs");

    preprocess.precede(cpu);
    cpu.precede(save);
  }

  double cpu_wall_ms = 0.0;

  try {
    const auto start = Clock::now();
    executor.run(cpu_flow).get();

    cpu_wall_ms =
      std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
  } catch (const std::exception& e) {
    std::cerr << "CPU pipeline failed: " << e.what() << '\n';
    return 2;
  }

  print_cpu_only_benchmark_summary(cpu_jobs, cpu_wall_ms);

  // Correctness mode: compare GPU output with the CPU reference.
  if (validate) {
    auto gpu_pipeline = std::make_unique<GpuPipeline>();
    tf::Taskflow validation_flow;
    tf::Task previous_gpu;
    bool has_previous_gpu = false;

    for (const auto& job : cpu_jobs) {
      auto gpu = validation_flow.emplace([job, &gpu_pipeline] {
        run_gpu_filters(*job, *gpu_pipeline);
      }).name("GPU reference execution");

      auto compare = validation_flow.emplace([job] {
        aggregate_and_save(*job);
      }).name("Compare CPU/GPU and save outputs");

      gpu.precede(compare);

      if (has_previous_gpu) {
        previous_gpu.precede(gpu);
      }

      previous_gpu = gpu;
      has_previous_gpu = true;
    }

    try {
      executor.run(validation_flow).get();
    } catch (const std::exception& e) {
      std::cerr << "Validation pipeline failed: " << e.what() << '\n';
      return 2;
    }

    std::cout << "\nValidation complete. "
                 "No benchmark comparison was recorded.\n";
    return 0;
  }

  // Phase 2: GPU pipeline with independent jobs.
  std::vector<std::shared_ptr<ImageJob>> gpu_jobs;
  gpu_jobs.reserve(image_paths.size());

  for (const auto& path : image_paths) {
    auto job = std::make_shared<ImageJob>();
    job->input_path = path;
    gpu_jobs.push_back(std::move(job));
  }

  double gpu_wall_ms = 0.0;

  try {
    // Construct the pipeline before starting the timed execution.
    auto gpu_pipeline = std::make_unique<GpuPipeline>();
    tf::Taskflow gpu_flow;
    tf::Task previous_gpu;
    bool has_previous_gpu = false;

    for (const auto& job : gpu_jobs) {
      auto preprocess = gpu_flow.emplace([job] {
        const auto start = Clock::now();
        std::string error;

        if (!load_and_preprocess(*job, error)) {
          throw std::runtime_error(error);
        }

        job->preprocess_ms =
          std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
      }).name("GPU pipeline CPU preprocessing");

      auto gpu = gpu_flow.emplace([job, &gpu_pipeline] {
        run_gpu_filters(*job, *gpu_pipeline);
      }).name("GPU Gaussian and Sobel");

      auto save = gpu_flow.emplace([job] {
        save_gpu_only(*job);
      }).name("Save GPU outputs");

      preprocess.precede(gpu);
      gpu.precede(save);

      // One reusable GPU buffer set: serialize GPU processing.
      if (has_previous_gpu) {
        previous_gpu.precede(gpu);
      }

      previous_gpu = gpu;
      has_previous_gpu = true;
    }
    const auto gpu_start = Clock::now();
    
    executor.run(gpu_flow).get();

    gpu_wall_ms =
      std::chrono::duration<double, std::milli>(
        Clock::now() - gpu_start).count();

  } catch (const std::exception& e) {
    std::cerr << "GPU pipeline failed: " << e.what() << '\n';
    return 2;
  }

  print_gpu_only_benchmark_summary(gpu_jobs, gpu_wall_ms);

  // Exactly one CSV row per paired CPU/GPU benchmark invocation.
  record_and_print_speedup(cpu_wall_ms, gpu_wall_ms);

  return 0;
}
