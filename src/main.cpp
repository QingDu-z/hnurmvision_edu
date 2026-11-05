#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <csignal>
#include <opencv2/imgproc.hpp>
namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
}

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  while (!interrupted) {
    const auto input = devices.read();
    training::Command command{};
    std::vector<training::Armor> armors;
    if (input) {
      armors = training::lesson03_detection(input->image);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
    }
    if (interrupted) break;
    devices.write(command);

    if (input && devices.config.display) {
            cv::Mat vis = input->image.clone();
      for (const auto& a : armors) {
        std::vector<cv::Point> pts;
        for (const auto& c : a.corners) pts.emplace_back(cvRound(c.x), cvRound(c.y));
        cv::polylines(vis, pts, true, cv::Scalar(0, 255, 0), 2);
        cv::putText(vis, a.large_armor ? "L" : "S", pts[0],
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
      }
      cv::imshow("RoboMaster training", vis);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }
  }

  devices.close();
}
