#include "io_context.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp/contexts/default_context.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace training::io {
namespace {
//---- constexpr 常量表达式，让计算发生在编译期 ---------//
constexpr double kDeg2Rad = kPi / 180.0;   
constexpr double kRad2Deg = 180.0 / kPi;
// 与仿真器 config.toml 的 vehicle.gimbal_pitch_limit 一致（±45°）：
// 超出机械限位的指令一律降级为"停止"，不能送到真实电机上（课件 3.4）。
constexpr double kPitchLimitRad = 0.7853981633974483;   //45度
// 消息里的角度是"度"，这里统一换成本工程使用的弧度制。
// 旋转顺序：R = Rz(yaw) * Ry(pitch) * Rx(roll)
Eigen::Quaterniond euler_deg_to_quaternion(double yaw_deg, double pitch_deg,
                                           double roll_deg) {
  const Eigen::AngleAxisd yaw(yaw_deg * kDeg2Rad, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd pitch(pitch_deg * kDeg2Rad, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd roll(roll_deg * kDeg2Rad, Eigen::Vector3d::UnitX());
  return Eigen::Quaterniond(yaw * pitch * roll).normalized();
}
}  // namespace

void IoContext::open() {
  // rclcpp::init/shutdown 操作的是进程级全局 context，不是某个节点。
  // 只有在"全局 context 还不可用"时才由 IO 层负责 init，并记住这件事。
  if (!rclcpp::contexts::get_global_default_context()->is_valid()) {
    rclcpp::init(0, nullptr);
    owns_context_ = true;
  }

  node_ = std::make_shared<rclcpp::Node>("training_io");

  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::SharedPtr msg) {
        on_image(std::move(msg));
      });

  recv_sub_ =
      node_->create_subscription<hnurm_interfaces::msg::VisionRecvData>(
          "/vision_receive_data", rclcpp::SensorDataQoS(),
          [this](hnurm_interfaces::msg::VisionRecvData::SharedPtr msg) {
            on_feedback(std::move(msg));
          });

  send_pub_ = node_->create_publisher<hnurm_interfaces::msg::VisionSendData>(
      "/vision_send_data", rclcpp::QoS(10));

  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);

  last_report_ = Clock::now();

  RCLCPP_INFO(node_->get_logger(),
              "IO 已启动：等待 /image_raw 与 /vision_receive_data");
}

void IoContext::on_image(sensor_msgs::msg::Image::SharedPtr msg) {
  const std::string& encoding = msg->encoding;
  cv::Mat frame;
  if (encoding == "bgr8") {
    // 必须 clone：msg 释放后 data 指针就悬空了（课件 3.3）
    frame = cv::Mat(static_cast<int>(msg->height), static_cast<int>(msg->width),
                    CV_8UC3, msg->data.data(),
                    static_cast<std::size_t>(msg->step))
                .clone();
  } else if (encoding == "rgb8") {
    cv::Mat rgb(static_cast<int>(msg->height), static_cast<int>(msg->width),
                CV_8UC3, msg->data.data(),
                static_cast<std::size_t>(msg->step));
    cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);  // cvtColor 自己分配内存
  } else {
    if (!warned_encoding_) {
      RCLCPP_WARN(node_->get_logger(), "不支持的图像编码：%s", encoding.c_str());
      warned_encoding_ = true;
    }
    return;
  }

  // 时间戳：把 ROS 时间换成本工程的 steady_clock。
  // 采集时刻 = 收到消息的时刻 - (ROS 当前时刻 - 采集时刻)
  const auto age = node_->now() - rclcpp::Time(msg->header.stamp);
  const auto age_ns = std::chrono::nanoseconds(
      std::max<std::int64_t>(0, age.nanoseconds()));

  image_ = std::move(frame);
  image_stamp_ = Clock::now() - age_ns;
  has_image_ = true;
}

void IoContext::on_feedback(
    hnurm_interfaces::msg::VisionRecvData::SharedPtr msg) {
  feedback_.yaw = msg->yaw;              // 度
  feedback_.pitch = msg->pitch;          // 度
  feedback_.roll = msg->roll;            // 度
  feedback_.bullet_speed = static_cast<double>(msg->bullet_speed.data);  // m/s
  feedback_.work_mode = msg->work_mode.data;
  feedback_.valid = true;
}

std::optional<SensorData> IoContext::read() {
  // 1) 让回调有机会执行。带超时，保证 Ctrl+C 随时能中断（不能 spin() 死等）
  executor_->spin_once(std::chrono::milliseconds(2));
  ++loop_count_;

  const auto now = Clock::now();

  // 2) 观测：每秒打印一次姿态与弹速（课件 6.3）
  if (now - last_report_ >= std::chrono::seconds(1)) {
    last_report_ = now;
    if (feedback_.valid) {
      std::cout << "yaw=" << feedback_.yaw << " deg pitch=" << feedback_.pitch
                << " deg bullet_speed=" << feedback_.bullet_speed << " m/s\n";
    } else {
      std::cout << "等待仿真器反馈...\n";
    }
  }

  // 3) 观测：每 100 轮打印一次 IO 状态（课件 6.3）
  if (loop_count_ % 100 == 0) {
    std::cout << "[IO] loop=" << loop_count_ << " delivered=" << frame_count_
              << " image=" << (has_image_ ? "yes" : "no")
              << " feedback=" << (feedback_.valid ? "yes" : "no") << "\n";
  }

  // 4) 图像和反馈都到齐才交付，否则交出"没有"
  if (!has_image_ || !feedback_.valid) {
    return std::nullopt;
  }

  SensorData data;
  data.image = std::move(image_);  // 所有权转移，不额外拷贝；image_ 变为空
  has_image_ = false;              // 同一帧只交一次
// 时间对齐：timestamp 由 header.stamp 换算到 steady_clock（见 docs/io-notes.md §2）。

  data.timestamp = image_stamp_;
  data.imu_q = euler_deg_to_quaternion(feedback_.yaw, feedback_.pitch,
                                       feedback_.roll);
  data.bullet_speed = feedback_.bullet_speed;

  ++frame_count_;
  return data;
}

void IoContext::write(const Command& command) { 
  // ---- 1) 安全门控：任一项不满足就发停止（课件 3.4 / 5.4）----
  const bool angles_finite =
      std::isfinite(command.yaw) && std::isfinite(command.pitch);
  const bool within_limit = std::abs(command.pitch) <= kPitchLimitRad;  //abs绝对值
  const bool have_feedback = feedback_.valid;
  const bool auto_aim_mode =
      have_feedback &&
      feedback_.work_mode == hnurm_interfaces::msg::WorkMode::AUTO_AIM;
  const bool allow_control =
      command.control && angles_finite && within_limit && have_feedback && auto_aim_mode;

  // ---- 2) 三态映射：0 不控制 / 1 控制不发射 / 2 控制并发射 ----
  std::uint8_t state = hnurm_interfaces::msg::TargetState::LOST_TARGET;
  if (allow_control) {
    state = (command.shoot && config.allow_fire)
                ? hnurm_interfaces::msg::TargetState::FIRE
                : hnurm_interfaces::msg::TargetState::CONVERGING;
  }

  // ---- 3) 填消息 ----
  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = node_->now();  // ROS 时间，不是 steady_clock
  msg.header.frame_id = "gimbal_link";
  msg.target_state.data = state;
  msg.target_type.data = hnurm_interfaces::msg::TargetType::NONE;

  if (allow_control) {
    // Command 是绝对角、弧度；消息要的是度和协议方向
    msg.yaw = static_cast<float>(command.yaw * kRad2Deg);
    msg.pitch = static_cast<float>(command.pitch * kRad2Deg);
    // 按文档 4.2：target_distance = -1 表示"输入数据非法，云台不再接受控制"，
    // 所以控制时给 0，"停止"由 target_state = 0 表达（实现见 ros2/plugin.rs:405）。
    msg.target_distance = 0.0F;
  } else {
    msg.yaw = 0.0F;
    msg.pitch = 0.0F;
    msg.target_distance = 0.0F;  // 停止
  }

  send_pub_->publish(msg);
} 

void IoContext::close() {
  // 1) 先发一帧停止：退出后不允许再控制或发射（课件 5.5 / 5.6）
  if (node_ && send_pub_ && rclcpp::ok()) {
    write(Command{});  // control=false → target_state=0，yaw/pitch 归零
    // 给 DDS 一点时间把这一帧真正送出去，避免刚 publish 就 shutdown 被丢掉
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

  // 2) 结束回调处理：节点不再被 executor 调度，后面不会再有回调执行
  if (executor_ && node_) {
    executor_->remove_node(node_);
  }

  // 3) 释放通信对象：订阅/发布器析构后通信自动取消（课件 4.3）
  send_pub_.reset();
  recv_sub_.reset();
  image_sub_.reset();
  node_.reset();
  executor_.reset();

  // 4) 只关自己 init 的那份全局 context；调用方的 ROS2 不归我们管。
  if (owns_context_) {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    owns_context_ = false;
  }

  std::cout << "[IO] 已关闭：停止指令已发出，通信资源已释放\n";
}
}  // namespace training::io