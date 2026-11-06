#pragma once
#include "config.hpp"
#include "types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>

#include <cstdint>
#include <memory>
#include <optional>

namespace training::io {
struct IoContext {
  Config config;

  // ==== 第二讲 P3 新增：ROS2 通信对象 ====
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr recv_sub_;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr send_pub_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;

  // 全局 context 是不是本对象 init 的：谁 init，谁 shutdown
  bool owns_context_ = false;

  // ==== 第二讲 P4 新增：最新输入缓存 ====
  cv::Mat image_;                       // 最近一帧（CV_8UC3，BGR，独立内存）
  training::TimePoint image_stamp_{};   // 该帧的采集时刻（steady_clock）
  bool has_image_ = false;              // 有一帧还没交给算法

  struct Feedback {
    double yaw = 0;           // 偏航度/z轴
    double pitch = 0;         // 俯仰度/y轴
    double roll = 0;          // 横滚度/x轴
    double bullet_speed = 0;  // m/s
    std::uint8_t work_mode = 0; //unsigned int 8bit （0～255）
    training::TimePoint stamp{};   // 最近一次收到反馈的时刻
    bool valid = false;   //有效
  };
  Feedback feedback_;

  std::uint64_t loop_count_ = 0;   // read() 被调用的次数
  std::uint64_t frame_count_ = 0;  // 交给算法的帧数
  training::TimePoint last_report_{};   //上次打印统计报告的时刻
  bool warned_encoding_ = false;

  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();

  // 回调：由 executor 在 read() 里调度，和 read() 同一个线程
  void on_image(sensor_msgs::msg::Image::SharedPtr msg);
  void on_feedback(hnurm_interfaces::msg::VisionRecvData::SharedPtr msg);
};
}  // namespace training::io