#include "cpu_filters.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
inline int clamp_index(int v, int limit) {
  return std::max(0, std::min(v, limit - 1));
}
}

void gaussian_blur_cpu(const cv::Mat& input, cv::Mat& output) {
  CV_Assert(input.type() == CV_8UC1);
  output.create(input.size(), CV_8UC1);
  const int weights[3] = {1, 2, 1};
  for (int y = 0; y < input.rows; ++y) {
    auto* out = output.ptr<unsigned char>(y);
    for (int x = 0; x < input.cols; ++x) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        const auto* row = input.ptr<unsigned char>(clamp_index(y + ky, input.rows));
        for (int kx = -1; kx <= 1; ++kx) {
          sum += weights[ky + 1] * weights[kx + 1] *
                 row[clamp_index(x + kx, input.cols)];
        }
      }
      out[x] = static_cast<unsigned char>((sum + 8) / 16);
    }
  }
}

void sobel_cpu(const cv::Mat& input, cv::Mat& output) {
  CV_Assert(input.type() == CV_8UC1);
  output.create(input.size(), CV_8UC1);
  const int sx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  const int sy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
  for (int y = 0; y < input.rows; ++y) {
    auto* out = output.ptr<unsigned char>(y);
    for (int x = 0; x < input.cols; ++x) {
      int gx = 0, gy = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        const auto* row = input.ptr<unsigned char>(clamp_index(y + ky, input.rows));
        for (int kx = -1; kx <= 1; ++kx) {
          const int p = row[clamp_index(x + kx, input.cols)];
          gx += sx[ky + 1][kx + 1] * p;
          gy += sy[ky + 1][kx + 1] * p;
        }
      }
      out[x] = static_cast<unsigned char>(std::min(255, std::abs(gx) + std::abs(gy)));
    }
  }
}
