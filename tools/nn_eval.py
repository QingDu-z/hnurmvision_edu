#!/usr/bin/env python3
"""用与传统 eval_detect 相同的口径评估 NN 预测（IoU>=0.5、浮点包围盒、>=30px）。

用法：
    python3 tools/nn_eval.py --gt simulation/simulator/dataset_B_rot \
        --pred runs/pose/nn_B/labels --min-w 30
"""
import argparse
import pathlib


def read_gt(path, w_img, h_img):
    boxes = []
    for line in pathlib.Path(path).read_text().strip().split("\n"):
        v = line.split()
        if len(v) < 11:
            continue
        xs = [float(v[3 + i * 2]) * w_img for i in range(4)]
        ys = [float(v[4 + i * 2]) * h_img for i in range(4)]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def read_pred(path, w_img, h_img):
    boxes = []
    if not pathlib.Path(path).exists():
        return boxes
    for line in pathlib.Path(path).read_text().strip().split("\n"):
        v = line.split()
        if len(v) < 5:
            continue
        cx, cy, w, h = (float(x) for x in v[1:5])
        boxes.append(((cx - w / 2) * w_img, (cy - h / 2) * h_img,
                      (cx + w / 2) * w_img, (cy + h / 2) * h_img))
    return boxes


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = w * h
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uni if uni > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True, help="YOLO 输出的 labels 目录")
    ap.add_argument("--min-w", type=float, default=30.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--w", type=int, default=1440)
    ap.add_argument("--h", type=int, default=1080)
    a = ap.parse_args()

    gt_dir = pathlib.Path(a.gt) / "label"
    pred_dir = pathlib.Path(a.pred)
    gt30 = det = matched = 0
    for f in sorted(gt_dir.glob("*.txt")):
        gts = [b for b in read_gt(f, a.w, a.h) if b[2] - b[0] >= a.min_w]
        preds = read_pred(pred_dir / f.name, a.w, a.h)
        gt30 += len(gts)
        det += len(preds)
        used = [False] * len(gts)
        for p in preds:
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                v = iou(p, g)
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= a.iou:
                used[bi] = True
                matched += 1
    recall = 100.0 * matched / gt30 if gt30 else 0.0
    precision = 100.0 * matched / det if det else 0.0
    print(f"GT(>= %g px) %d | 检出 %d | 匹配 %d | recall %5.1f%% | precision %5.1f%%"
          % (a.min_w, gt30, det, matched, recall, precision))


if __name__ == "__main__":
    main()
