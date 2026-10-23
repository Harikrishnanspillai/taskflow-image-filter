#pragma once

#include <opencv2/core.hpp>
#include <memory>

struct GpuTimings {
  double setup_ms = 0.0;       // Allocation + graph creation; normally paid once per image size.
  double h2d_ms = 0.0;         // Host-to-device copy wall time.
  double kernels_ms = 0.0;     // CUDA-event time for the graph containing Gaussian + Sobel.
  double d2h_ms = 0.0;         // Device-to-host copy wall time.
  double total_ms = 0.0;       // Per-image run time, excluding one-time setup.
};

// Reuses device buffers, CUDA graph executable, stream, and events for images
// of the same dimensions. Calls to run() on one instance must be serialized.
class GpuPipeline {
public:
  GpuPipeline();
  ~GpuPipeline();
  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;
  GpuPipeline(GpuPipeline&&) = delete;
  GpuPipeline& operator=(GpuPipeline&&) = delete;

  GpuTimings run(const cv::Mat& input, cv::Mat& gaussian, cv::Mat& sobel);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
