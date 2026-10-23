#include "image_pipeline.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

bool load_and_preprocess(ImageJob& job, std::string& error) {
  cv::Mat loaded = cv::imread(job.input_path, cv::IMREAD_GRAYSCALE);
  if (loaded.empty()) {
    error = "Could not decode image: " + job.input_path;
    return false;
  }
  // Normalize to 8-bit grayscale. imread(IMREAD_GRAYSCALE) already yields CV_8UC1.
  if (loaded.type() != CV_8UC1) {
    loaded.convertTo(job.input, CV_8UC1);
  } else {
    job.input = loaded;
  }
  job.stem = std::filesystem::path(job.input_path).stem().string();
  return true;
}
