#!/usr/bin/env python3
"""融合原型：传统检测 + NN 类别（按位置匹配）。

前置：
  1) eval_detect <dataset> --save-json trad.txt    导出传统检出
  2) yolo pose predict ... save_txt=True           导出 NN 标签（13 列）

用法：
  python3 tools/fuse_proto.py --gt <dataset目录> --trad trad.txt --nn runs/pose/xxx/labels
"""
import argparse
import pathlib

W, H = 1440, 1080
NAMES = ["B1","B2","B3","B4","B5","BO","BS","R1","R2","R3","R4","R5","RO","RS"]


def load_trad(path):
    d = {}
    for line in pathlib.Path(path).read_text().strip().split("\n"):
        v = line.split()
        if len(v) < 9:
            continue
        xs = [float(v[1 + 2 * i]) for i in range(4)]
        ys = [float(v[2 + 2 * i]) for i in range(4)]
        d.setdefault(v[0], []).append((min(xs), min(ys), max(xs), max(ys)))
    return d


def load_gt(path):
    out = []
    for line in pathlib.Path(path).read_text().strip().split("\n"):
        v = line.split()
        if len(v) < 11:
            continue
        xs = [float(v[3 + 2 * i]) * W for i in range(4)]
        ys = [float(v[4 + 2 * i]) * H for i in range(4)]
        # 仿真器 label 为 1 基编号：cls = (label-1) + 7*color
        out.append(((min(xs), min(ys), max(xs), max(ys)), (int(v[2]) - 1) + 7 * int(v[0])))
    return out


def load_nn(path):
    out = []
    if not pathlib.Path(path).exists():
        return out
    for line in pathlib.Path(path).read_text().strip().split("\n"):
        v = line.split()
        if len(v) < 13:
            continue
        kp = v[5:13]
        xs = [float(kp[2 * i]) * W for i in range(4)]
        ys = [float(kp[2 * i + 1]) * H for i in range(4)]
        out.append(((min(xs), min(ys), max(xs), max(ys)), int(float(v[0]))))
    return out


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = w * h
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uni if uni > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="仿真 dataset 目录（含 label/）")
    ap.add_argument("--trad", required=True, help="eval_detect --save-json 的输出")
    ap.add_argument("--nn", required=True, help="YOLO 的 labels 目录")
    ap.add_argument("--match-iou", type=float, default=0.3, help="传统↔NN 匹配阈值")
    ap.add_argument("--gt-iou", type=float, default=0.5, help="与 GT 判定的阈值")
    a = ap.parse_args()

    base = pathlib.Path(a.gt)
    trad = load_trad(a.trad)
    n_trad = n_asg = n_ok = n_nn = n_only = n_only_true = 0
    for f in sorted((base / "label").glob("*.txt")):
        gb = load_gt(f)
        tb = trad.get(f.stem, [])
        nb = load_nn(pathlib.Path(a.nn) / f.name)
        n_trad += len(tb)
        n_nn += len(nb)
        matched = set()
        for t in tb:
            best, bi = 0.0, -1
            for k, (gg, _) in enumerate(gb):
                v = iou(t, gg)
                if v > best:
                    best, bi = v, k
            gt_cls = gb[bi][1] if bi >= 0 and best >= a.gt_iou else None
            bb, bj = 0.0, -1
            for j, (nnb, _) in enumerate(nb):
                v = iou(t, nnb)
                if v > bb:
                    bb, bj = v, j
            if bj >= 0 and bb >= a.match_iou:
                matched.add(bj)
                n_asg += 1
                if gt_cls is not None and nb[bj][1] == gt_cls:
                    n_ok += 1
        for j, (nnb, _) in enumerate(nb):
            if j in matched:
                continue
            n_only += 1
            if max([iou(nnb, gg) for gg, _ in gb], default=0.0) >= a.gt_iou:
                n_only_true += 1

    cov = 100.0 * n_asg / n_trad if n_trad else 0.0
    acc = 100.0 * n_ok / n_asg if n_asg else 0.0
    print(f"传统检出 {n_trad} | 拿到类别 {n_asg}（覆盖 {cov:.1f}%）| 类别正确 {n_ok}（准确 {acc:.1f}%）")
    print(f"NN 检出 {n_nn} | 与传统文化匹配 | NN 独有 {n_only} 个（其中命中 GT {n_only_true} 个 → 可补检）")
    print(f"融合后检出 = {n_trad + n_only_true}（传统 {n_trad} + 补检 {n_only_true}）")


if __name__ == "__main__":
    main()
