#include "gpu_filters.cuh"

#include <taskflow/cuda/cudaflow.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

__device__ int clampd(int v, int limit) {
  return max(0, min(v, limit - 1));
}

__global__ void gaussian_kernel(const unsigned char* input,
                                unsigned char* output, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int w[3] = {1, 2, 1};
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    const int yy = clampd(y + ky, height);
    for (int kx = -1; kx <= 1; ++kx) {
      const int xx = clampd(x + kx, width);
      sum += w[ky + 1] * w[kx + 1] * input[yy * width + xx];
    }
  }
  output[y * width + x] = static_cast<unsigned char>((sum + 8) / 16);
}

__global__ void sobel_kernel(const unsigned char* input,
                             unsigned char* output, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int sx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  const int sy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
  int gx = 0, gy = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    const int yy = clampd(y + ky, height);
    for (int kx = -1; kx <= 1; ++kx) {
      const int xx = clampd(x + kx, width);
      const int p = input[yy * width + xx];
      gx += sx[ky + 1][kx + 1] * p;
      gy += sy[ky + 1][kx + 1] * p;
    }
  }
  const int mag = min(255, abs(gx) + abs(gy));
  output[y * width + x] = static_cast<unsigned char>(mag);
}
} // namespace

struct GpuPipeline::Impl {
  int width = 0;
  int height = 0;
  size_t bytes = 0;
  unsigned char* d_input = nullptr;
  unsigned char* d_gaussian = nullptr;
  unsigned char* d_sobel = nullptr;

  // Members persist across images. The graph captures the stable device pointers.
  std::unique_ptr<tf::cudaGraph> graph;
  std::unique_ptr<tf::cudaGraphExec> executable;
  tf::cudaStream stream;
  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_stop = nullptr;

  Impl() {
    cuda_check(cudaEventCreate(&event_start), "cudaEventCreate start");
    try {
      cuda_check(cudaEventCreate(&event_stop), "cudaEventCreate stop");
    } catch (...) {
      cudaEventDestroy(event_start);
      event_start = nullptr;
      throw;
    }
  }

  ~Impl() {
    stream.synchronize();
    executable.reset();
    graph.reset();
    if (d_sobel) cudaFree(d_sobel);
    if (d_gaussian) cudaFree(d_gaussian);
    if (d_input) cudaFree(d_input);
    if (event_stop) cudaEventDestroy(event_stop);
    if (event_start) cudaEventDestroy(event_start);
  }

  void ensure_size(int new_width, int new_height) {
    if (new_width == width && new_height == height && executable) return;

    stream.synchronize();
    // The executable/graph must be destroyed before freeing the pointers they capture.
    executable.reset();
    graph.reset();
    if (d_sobel) { cuda_check(cudaFree(d_sobel), "free old Sobel buffer"); d_sobel = nullptr; }
    if (d_gaussian) { cuda_check(cudaFree(d_gaussian), "free old Gaussian buffer"); d_gaussian = nullptr; }
    if (d_input) { cuda_check(cudaFree(d_input), "free old input buffer"); d_input = nullptr; }

    width = new_width;
    height = new_height;
    bytes = static_cast<size_t>(width) * static_cast<size_t>(height);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes), "cudaMalloc input");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gaussian), bytes), "cudaMalloc Gaussian");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sobel), bytes), "cudaMalloc Sobel");

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    graph = std::make_unique<tf::cudaGraph>();
    auto blur_task = graph->kernel(grid, block, 0, gaussian_kernel,
                                   d_input, d_gaussian, width, height);
    auto sobel_task = graph->kernel(grid, block, 0, sobel_kernel,
                                    d_gaussian, d_sobel, width, height);
    blur_task.precede(sobel_task);
    executable = std::make_unique<tf::cudaGraphExec>(*graph);
  }
};

GpuPipeline::GpuPipeline() : impl_(std::make_unique<Impl>()) {}
GpuPipeline::~GpuPipeline() = default;

GpuTimings GpuPipeline::run(const cv::Mat& input, cv::Mat& gaussian, cv::Mat& sobel) {
  CV_Assert(input.type() == CV_8UC1 && input.isContinuous());
  CV_Assert(input.cols > 0 && input.rows > 0);

  GpuTimings timings;
  const bool needs_setup = !impl_->executable ||
                           impl_->width != input.cols || impl_->height != input.rows;
  auto setup_start = Clock::now();
  impl_->ensure_size(input.cols, input.rows);
  if (needs_setup) timings.setup_ms = elapsed_ms(setup_start, Clock::now());

  // Per-image run timing deliberately starts after one-time size-dependent setup.
  const auto total_start = Clock::now();
  auto stage_start = Clock::now();
  cuda_check(cudaMemcpy(impl_->d_input, input.data, impl_->bytes,
                        cudaMemcpyHostToDevice), "copy input to device");
  timings.h2d_ms = elapsed_ms(stage_start, Clock::now());

  // Events are recorded on the SAME Taskflow-managed CUDA stream as the graph.
  // The graph contains only the Gaussian and Sobel kernels, so this measures
  // device execution of those kernels without host-side launch/setup time.
  impl_->stream.record(impl_->event_start);
  impl_->stream.run(*impl_->executable);
  impl_->stream.record(impl_->event_stop);
  impl_->stream.synchronize();
  float kernel_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&kernel_ms, impl_->event_start, impl_->event_stop),
             "cudaEventElapsedTime");
  timings.kernels_ms = static_cast<double>(kernel_ms);

  gaussian.create(input.size(), CV_8UC1);
  sobel.create(input.size(), CV_8UC1);
  stage_start = Clock::now();
  cuda_check(cudaMemcpy(gaussian.data, impl_->d_gaussian, impl_->bytes,
                        cudaMemcpyDeviceToHost), "copy Gaussian to host");
  cuda_check(cudaMemcpy(sobel.data, impl_->d_sobel, impl_->bytes,
                        cudaMemcpyDeviceToHost), "copy Sobel to host");
  timings.d2h_ms = elapsed_ms(stage_start, Clock::now());
  timings.total_ms = elapsed_ms(total_start, Clock::now());
  return timings;
}
