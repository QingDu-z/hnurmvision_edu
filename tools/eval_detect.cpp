// 用仿真器导出的 GT 评估 lesson03_detection 的检出率与角点误差。
//
// 数据来源：仿真器里按住数字键 1（DatasetPlugin，最多 4 Hz），会在
//   simulation/simulator/dataset/{images,label}/ 下生成 frame_XXXXXX.jpg/.txt
// 标签格式（10 列，角点已按宽高归一化）：
//   color typ label x0 y0 x1 y1 x2 y2 x3 y3
//
// 用法：
//   ./build/eval_detect simulation/simulator/dataset            # 默认 IoU>=0.5
//   ./build/eval_detect simulation/simulator/dataset --iou 0.6 --verbose
#include "lessons.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct GtArmor {
  std::array<cv::Point2f, 4> corners{};  // 像素坐标
  int color = -1, type = -1, label = -1;
};

std::vector<GtArmor> load_gt(const std::string& path) {
  std::vector<GtArmor> out;
  std::ifstream in(path);
  if (!in) return out;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    GtArmor g;
    float x, y;
    if (!(ss >> g.color >> g.type >> g.label)) continue;
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
      if (!(ss >> x >> y)) { ok = false; break; }
      g.corners[i] = {x, y};   // 归一化，稍后乘宽高
    }
    if (ok) out.push_back(g);
  }
  return out;
}

void order_quad(std::array<cv::Point2f, 4>& p) {
  // 先按 y 分上下两排，再各自按 x 分左右，得到 TL, TR, BR, BL
  std::sort(p.begin(), p.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    return a.y == b.y ? a.x < b.x : a.y < b.y;
  });
  if (p[0].x > p[1].x) std::swap(p[0], p[1]);   // 上排：TL, TR
  if (p[2].x > p[3].x) std::swap(p[2], p[3]);   // 下排：BL, BR
  const std::array<cv::Point2f, 4> q{p[0], p[1], p[3], p[2]};
  p = q;
}

double iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float w = std::max(0.f, x2 - x1), h = std::max(0.f, y2 - y1);
  const float inter = w * h;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0 ? inter / uni : 0.0;
}

cv::Rect2f bbox(const std::array<cv::Point2f, 4>& p) {
  // 用浮点包围盒：cv::boundingRect 传浮点会返回整数 Rect（ceil(max)-floor(min)），
  // 宽度比真实值多 0~1 px，会让 29.x px 的板被算进 ">=30 px" 口径。
  float x0 = p[0].x, x1 = p[0].x, y0 = p[0].y, y1 = p[0].y;
  for (int i = 1; i < 4; ++i) {
    x0 = std::min(x0, p[i].x);
    x1 = std::max(x1, p[i].x);
    y0 = std::min(y0, p[i].y);
    y1 = std::max(y1, p[i].y);
  }
  return cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: eval_detect <dataset_dir> [--iou 0.5] [--verbose]\n";
    return 2;
  }
  const std::string dir = argv[1];
  double iou_thresh = 0.5;
  double min_w = 0.0;          // 只统计宽度 >= min_w 像素的 GT（默认全部）
  bool verbose = false;  (void)verbose;
  std::string dump_dir;        // 非空则把 GT(蓝) 与检出(绿) 画到图里
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iou" && i + 1 < argc) iou_thresh = std::stod(argv[++i]);
    else if (a == "--min-w" && i + 1 < argc) min_w = std::stod(argv[++i]);
    else if (a == "--verbose") verbose = true;
    else if (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
  }

  struct Stats {
    int gt = 0, det = 0, matched = 0;
    double err = 0.0;
    int err_n = 0;
  };
  // 两个口径一起统计：0 = 全量 GT，1 = 只看宽度 >= min_w 的 GT
  Stats stats[2];
  int frames = 0, empty_frames = 0, fp_frames = 0;

  for (int idx = 0;; ++idx) {
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%06d", idx + 1);
    cv::Mat img = cv::imread(dir + "/images/" + name + ".jpg");
    if (img.empty()) break;   // 序号不连续就停

    ++frames;
    const auto gts_all = load_gt(dir + "/label/" + name + ".txt");
    const auto dets = training::lesson03_detection(img);

    for (int pass = 0; pass < 2; ++pass) {
      const double thr = pass == 0 ? 0.0 : (min_w > 0.0 ? min_w : 30.0);
      std::vector<GtArmor> gts;
      for (const auto& g : gts_all) {
        if (thr > 0.0) {
          auto c = g.corners;
          for (auto& pt : c) { pt.x *= img.cols; pt.y *= img.rows; }
          if (bbox(c).width < thr) continue;
        }
        gts.push_back(g);
      }
      if (pass == 0 && gts.empty()) ++empty_frames;

      std::vector<bool> used(gts.size(), false);
      for (const auto& d : dets) {
        int best = -1;
        double best_iou = 0.0;
        for (std::size_t g = 0; g < gts.size(); ++g) {
          if (used[g]) continue;
          auto gc = gts[g].corners;
          for (auto& c : gc) { c.x *= img.cols; c.y *= img.rows; }
          const double i = iou(bbox(d.corners), bbox(gc));
          if (i > best_iou) { best_iou = i; best = static_cast<int>(g); }
        }
        if (best >= 0 && best_iou >= iou_thresh) {
          used[best] = true;
          ++stats[pass].matched;
          auto gc = gts[best].corners;
          for (auto& c : gc) { c.x *= img.cols; c.y *= img.rows; }
          auto dc = d.corners;
          order_quad(gc);
          order_quad(dc);
          for (int k = 0; k < 4; ++k) {
            stats[pass].err += cv::norm(dc[k] - gc[k]);
            ++stats[pass].err_n;
          }
        }
      }
      stats[pass].gt += static_cast<int>(gts.size());
      stats[pass].det += static_cast<int>(dets.size());
      if (pass == 0 && gts.empty() && !dets.empty()) ++fp_frames;
    }
  }

  if (frames == 0) {
    std::cerr << "没找到帧。先在仿真器里按住数字键 1 采集，再指向 dataset 目录。\n";
    return 1;
  }

  auto report = [&](const char* tag, const Stats& st) {
    const double recall = st.gt > 0 ? 100.0 * st.matched / st.gt : 0.0;
    const double precision = st.det > 0 ? 100.0 * st.matched / st.det : 0.0;
    const double err = st.err_n > 0 ? st.err / st.err_n : 0.0;
    std::printf("%-12s GT %4d / 检出 %4d / 匹配 %4d | recall %5.1f%% | precision %5.1f%% | 角点 %.2f px\n",
                tag, st.gt, st.det, st.matched, recall, precision, err);
  };
  std::printf("\n帧数 %d（无目标帧 %d，误检帧 %d）\n", frames, empty_frames, fp_frames);
  report("全量 GT", stats[0]);
  char tag[64];
  std::snprintf(tag, sizeof(tag), ">=\u0025.0fpx GT", min_w > 0 ? min_w : 30.0);
  report(tag, stats[1]);
  std::printf("门槛：>=30 px 那一行即“3 m 正面目标”口径，验收看这一行。\n");
  return 0;
}
