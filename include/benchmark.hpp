#pragma once
#include <memory>
#include <vector>
#include "image_pipeline.hpp"

void print_benchmark_summary(
    const std::vector<std::shared_ptr<ImageJob>>& jobs,
    double end_to_end_ms);
void print_cpu_only_benchmark_summary(
    const std::vector<std::shared_ptr<ImageJob>>& jobs,
    double end_to_end_ms);
void print_gpu_only_benchmark_summary(
    const std::vector<std::shared_ptr<ImageJob>>& jobs,
    double end_to_end_ms);
void record_and_print_speedup(
    double cpu_ms,
    double gpu_ms);
