#include "lessons.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace training {
namespace {

// ---- 参数：按仿真画面标定（1440x1080，3 m 处小装甲板约 58x24 px）----
constexpr int kBinThreshold = 150;        // LED 灰度阈值：仿真器把灯条提亮到 ~195
constexpr int kBarMinHeight = 8;          // 灯条最短（px），太短的当噪声
constexpr double kBarMinAspect = 1.6;     // 灯条 高/宽 下限
constexpr double kBarMaxAspect = 12.0;    // 上限：太细的是反光条纹
constexpr double kPairMaxHeightDiff = 0.35;  // 两灯条高度相对差
constexpr double kPairMaxCenterDiff = 0.55;  // 两灯条中心 y 相对差
constexpr double kPairMinGapRatio = 0.8;     // 间距 / 灯条高度 的下限
constexpr double kPairMaxGapRatio = 5.0;     // 上限
constexpr double kLargeArmorAspect = 3.2;  // 板宽/板高 超过此值判大装甲板
// 板高阈值：像素单位，按相机原始画面标定（1080p，fy=1303.7）
//   h_px = fy * 0.055 / Z  →  3 m ≈ 24 px，5 m ≈ 14 px，0.5 m ≈ 143 px
// 若上游改变分辨率（resize），这两个值必须重标定；若只是 ROI 裁剪，
// px/m 不变、阈值仍然有效。本函数接口拿不到相机内参，无法自动换算。
constexpr double kPlateMinHeightPx = 14.0;   // ≈5 m 处小装甲板
constexpr double kPlateMaxHeightPx = 216.0;  // ≈0.4 m 处，防近距离误杀
constexpr double kBarMaxTiltDeg = 20.0;    // 灯条长轴允许偏离竖直的最大角度

// 标定基准：像素阈值只在"相机原始画面"这个分辨率下有物理意义。
// 上游做 ROI 裁剪时 px/m 不变（阈值仍有效，警告可忽略）；做 resize 或换相机时阈值失效。
constexpr int kCalibCols = 1440;
constexpr int kCalibRows = 1080;

// ---- 板面验证参数（3 m 静态靶 53 帧标定；近距离/远距离需复测）----
constexpr double kFaceThrRatio     = 0.30;   // 数字阈值 = 该比例 × 灯条均值
constexpr double kFaceThrAbs       = 25.0;   // 阈值绝对下限，压住传感器噪声
constexpr double kFaceRatioLo      = 0.03;   // 数字亮像素占比下限
constexpr double kFaceRatioHi      = 0.70;   // 占比上限，防"整块亮面"
constexpr double kFaceMaxMeanRatio = 0.65;   // 中间均值不得超过灯条均值的该比例
constexpr double kFaceMinBarMean   = 1.0;    // 灯条过暗直接判无效
struct Bar {
  cv::Rect rect;
  cv::Point2f center;
  double aspect = 0;
  double tilt=0;   //长轴相对竖直方向的偏差，单位：度
};

// 步骤 1-3：亮度阈值 + 形态学 + 灯条筛选
std::vector<Bar> find_bars(const cv::Mat& gray) {
  cv::Mat bin;
  cv::threshold(gray, bin, kBinThreshold, 255, cv::THRESH_BINARY);

  // 竖向闭运算：把被反光切断的灯条连起来，同时不把横向干扰连成片
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
  cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<Bar> bars;
  bars.reserve(contours.size());
  for (const auto& c : contours) {
    const cv::Rect r = cv::boundingRect(c);
    if (r.height < kBarMinHeight || r.width <= 0) continue;
    const double aspect = static_cast<double>(r.height) / r.width;
    if (aspect < kBarMinAspect || aspect > kBarMaxAspect) continue;

    // 长轴方向：竖灯条 ~90°，场地上的横/斜亮线会被挡掉
    const cv::RotatedRect rr = cv::minAreaRect(c);
    double long_axis = (rr.size.width >= rr.size.height) ? rr.angle : rr.angle + 90.0;
    if (long_axis >= 180.0) long_axis -= 180.0;
    const double d = std::abs(long_axis - 90.0);
    if (std::min(d, 180.0 - d) > kBarMaxTiltDeg) continue;

        // tilt 存"相对竖直的偏差"，单位度；long_axis 本身是 [0,180) 的绝对角
    bars.push_back({r, {r.x + r.width * 0.5F, r.y + r.height * 0.5F}, aspect,
                    long_axis - 90.0});
  }
  return bars;
}

// 步骤 4：两两配对成装甲板，返回四角点（左上、右上、右下、左下）
bool try_pair(const Bar& a, const Bar& b, cv::Point2f corners[4], bool& is_large) {
  const Bar& left = a.center.x <= b.center.x ? a : b;
  const Bar& right = a.center.x <= b.center.x ? b : a;

  const double h = std::max(left.rect.height, right.rect.height);
  const double hdiff = std::abs(left.rect.height - right.rect.height) / static_cast<double>(h);
  if (hdiff > kPairMaxHeightDiff) return false;

  const double cdiff = std::abs(left.center.y - right.center.y) / h;
  if (cdiff > kPairMaxCenterDiff) return false;

  const double gap = right.rect.x - (left.rect.x + left.rect.width);
  const double g = gap / h;
  if (g < kPairMinGapRatio || g > kPairMaxGapRatio) return false;

  const float top = std::min(left.rect.y, right.rect.y);
  const float bottom = std::max(left.rect.y + left.rect.height,
                                right.rect.y + right.rect.height);
  const float x0 = static_cast<float>(left.rect.x);
  const float x1 = static_cast<float>(right.rect.x + right.rect.width);

  corners[0] = {x0, top};      // 左上
  corners[1] = {x1, top};      // 右上
  corners[2] = {x1, bottom};   // 右下
  corners[3] = {x0, bottom};   // 左下

  const double plate_w = x1 - x0;
  const double plate_h = bottom - top;
  if (plate_h < kPlateMinHeightPx || plate_h > kPlateMaxHeightPx) return false;
  is_large = plate_h > 0 && (plate_w / plate_h) > kLargeArmorAspect;
  return true;
}

// 板面验证：两灯条之间必须存在"数字"亮斑（装甲板印刷的数字），
// 否则说明这两根灯条来自相邻的两块板（中间是车体，纯黑）。
// 阈值来自实测：误配帧中间均值 3.2 / 最大 12；灯条均值约 145。
bool plate_face_ok(const cv::Mat& gray, const Bar& L, const Bar& R) {
  const Bar& a = L.center.x <= R.center.x ? L : R;
  const Bar& b = L.center.x <= R.center.x ? R : L;

  const double tilt_deg = std::max(std::abs(a.tilt), std::abs(b.tilt));   // 度
  const double half_w = 0.5 * (b.rect.x - (a.rect.x + a.rect.width));
  const int inset_y = static_cast<int>(std::tan(tilt_deg * kPi / 180.0) * half_w) + 2;

  int x0 = std::max(a.rect.x + a.rect.width + 2, 0);
  int x1 = std::min(b.rect.x - 2, gray.cols);
  int y0 = std::max(std::max(a.rect.y, b.rect.y) + inset_y, 0);
  int y1 = std::min(std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height) - inset_y,
                    gray.rows);
  if (x1 - x0 < 6 || y1 - y0 < 4) return false;

  const cv::Rect inner(x0, y0, x1 - x0, y1 - y0);
  const cv::Mat roi = gray(inner);

  const double bar_mean =
      0.5 * (cv::mean(gray(a.rect))[0] + cv::mean(gray(b.rect))[0]);
  if (bar_mean < kFaceMinBarMean) return false;

  const double thr = std::max(kFaceThrRatio * bar_mean, kFaceThrAbs);
  const double bright_ratio = cv::countNonZero(roi > thr) /
                              (static_cast<double>(inner.width) * inner.height);

  if (cv::mean(roi)[0] > kFaceMaxMeanRatio * bar_mean) return false;
  return bright_ratio > kFaceRatioLo && bright_ratio < kFaceRatioHi;
}
// 可选：把中间结果存成图，方便调参（父目录必须已存在）
void debug_dump(const cv::Mat& image, const std::vector<Bar>& bars,
                const std::vector<Armor>& armors) {
  const char* dir = std::getenv("DETECT_DEBUG_DIR");
  if (dir == nullptr) return;
  static int seq = 0;
  static int counter = 0;
  constexpr int kEvery = 30;    // 每 30 帧存一次
  constexpr int kMax   = 40;    // 最多 40 组
  if (seq >= kMax) return;
  if ((counter++ % kEvery) != 0) return;
  const std::string prefix = std::string(dir) + "/det_" + std::to_string(seq++);

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::Mat bin;
  cv::threshold(gray, bin, kBinThreshold, 255, cv::THRESH_BINARY);
  cv::imwrite(prefix + "_bin.png", bin);

  cv::Mat vis = image.clone();
  for (const auto& b : bars) {
    cv::rectangle(vis, b.rect, cv::Scalar(0, 255, 255), 1);
  }
  for (const auto& a : armors) {
    std::vector<cv::Point> pts;
    pts.reserve(a.corners.size());
    for (const auto& c : a.corners) {
      pts.emplace_back(cvRound(c.x), cvRound(c.y));  // polylines 只接受整点
    }
    cv::polylines(vis, pts, true, cv::Scalar(0, 255, 0), 2);
  }
  cv::imwrite(prefix + "_vis.jpg", vis);
}

}  // namespace

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  std::vector<Armor> armors;
  if (image.empty() || image.type() != CV_8UC3) return armors;

  // 只检查一次，避免每帧刷屏：分辨率变了要让人看见，而不是静默降级。
  static bool resolution_checked = false;
  if (!resolution_checked) {
    resolution_checked = true;
    if (image.cols != kCalibCols || image.rows != kCalibRows) {
      std::cerr << "[lesson03] 警告：输入 " << image.cols << "x" << image.rows
                << " 与标定基准 " << kCalibCols << "x" << kCalibRows
                << " 不一致；若是 resize/换相机，kPlateMinHeightPx / "
                   "kPlateMaxHeightPx / kBarMinHeight 需重标定（ROI 裁剪可忽略本警告）\n";
    }
  }

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  const std::vector<Bar> bars = find_bars(gray);

  // 贪心配对：按灯条从左到右，尽量一一配对，避免同一根被重复使用
  std::vector<bool> used(bars.size(), false);
  for (std::size_t i = 0; i < bars.size(); ++i) {
    if (used[i]) continue;
    for (std::size_t j = i + 1; j < bars.size(); ++j) {
      if (used[j]) continue;
      cv::Point2f corners[4];
      bool is_large = false;
      if (!try_pair(bars[i], bars[j], corners, is_large)) continue;
      if (!plate_face_ok(gray, bars[i], bars[j])) continue;
      Armor armor;
      armor.corners = {corners[0], corners[1], corners[2], corners[3]};
      armor.large_armor = is_large;
      armor.class_id = -1;    // 编号交给 P4.5 的分类器，先按课件允许填 -1
      armor.confidence = 1.0; // 传统管线的"置信度"暂时恒为 1
      armors.push_back(armor);

      used[i] = used[j] = true;
      break;
    }
  }

  debug_dump(image, bars, armors);
  return armors;
}

}  // namespace training