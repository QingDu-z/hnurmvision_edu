// 从仿真器抓 N 帧，跑 lesson03_detection，把结果画出来存成 jpg。
// 用法：./build/detect_preview configs/simulation.yaml [帧数]
#include "io_context.hpp"
#include "lessons.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const int want = argc > 2 ? std::stoi(argv[2]) : 5;

  training::io::IoContext devices{config};
  devices.open();

  int saved = 0;
  for (int i = 0; i < 6000 && saved < want; ++i) {
    const auto input = devices.read();
    if (!input) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    const auto armors = training::lesson03_detection(input->image);
    cv::Mat vis = input->image.clone();
    for (const auto& a : armors) {
      std::vector<cv::Point> pts;
      pts.reserve(a.corners.size());
      for (const auto& c : a.corners) pts.emplace_back(cvRound(c.x), cvRound(c.y));
      cv::polylines(vis, pts, true, cv::Scalar(0, 255, 0), 2);
      const std::string tag = std::string(a.large_armor ? "L" : "S") + " " +
                              cv::format("%.2f", a.confidence);
      cv::putText(vis, tag, pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    const std::string name = "detect_" + std::to_string(saved) + ".jpg";
    cv::imwrite(name, vis);
    std::cout << "armors=" << armors.size() << " -> " << name << "\n";
    ++saved;
  }

  devices.close();
  return saved > 0 ? 0 : 1;
}
