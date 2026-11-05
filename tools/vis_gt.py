#!/usr/bin/env python3
"""P0：数据集核对与 GT 角点可视化。

用法：
    python3 tools/vis_gt.py                       # 默认校验 train，抽 20 张，存到 /tmp/gt_vis
    python3 tools/vis_gt.py --split val --n 30
    python3 tools/vis_gt.py --root ~/hnurmvision_edu/dataset_norm   # 也适用于 P3.5 规范化后的数据
    python3 tools/vis_gt.py --order-check         # 只做统计，不画图

输出：
    <out>/000_<stem>.jpg       每张图上画出四角多边形 + 类别号 + 角点序号
    <out>/montage.jpg          前 4 张拼成 2x2，方便一眼看
    终端打印数量、标签格式、角点绕序统计
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# classes 0..13 = B1 B2 B3 B4 B5 BO BS R1 R2 R3 R4 R5 RO RS
CLASS_NAMES = ["B1", "B2", "B3", "B4", "B5", "BO", "BS",
               "R1", "R2", "R3", "R4", "R5", "RO", "RS"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / "hnurmvision_edu/dataset"),
                   help="数据集根目录（下面有 train/val）")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--n", type=int, default=20, help="抽样张数")
    p.add_argument("--out", default="/tmp/gt_vis", help="输出目录")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order-check", action="store_true", help="只统计不画图")
    return p.parse_args()


def load_labels(txt: Path):
    """返回 [(cls, 4x2 归一化角点), ...]；格式非法返回 None。"""
    rows = []
    for line in txt.read_text().strip().split("\n"):
        if not line.strip():
            continue
        v = line.split()
        if len(v) != 13:          # class + cx cy w h + 4*(x,y)
            return None
        cls = int(v[0])
        pts = np.array([[float(v[5 + i * 2]), float(v[6 + i * 2])] for i in range(4)])
        rows.append((cls, pts))
    return rows


def order_pattern(pts):
    """把四角按相对质心的位置标成 TL/TR/BL/BR，得到绕序字符串。"""
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    return ",".join(("T" if y < cy else "B") + ("L" if x < cx else "R") for x, y in pts)


def draw(img, rows, path: Path):
    h, w = img.shape[:2]
    for cls, pts in rows:
        px = np.array([[int(x * w), int(y * h)] for x, y in pts], np.int32)
        cv2.polylines(img, [px], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(px):
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        cv2.putText(img, name, (int(px[:, 0].min()), int(px[:, 1].min()) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(path), img)


def main():
    a = parse_args()
    root = Path(a.root).expanduser()
    img_dir, lbl_dir = root / a.split / "images", root / a.split / "labels"

    if not img_dir.is_dir():
        raise SystemExit(f"找不到图片目录：{img_dir}（先按 P0 的解压步骤放好数据）")

    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    lbls = sorted(lbl_dir.glob("*.txt"))
    print(f"[核对] {a.split}: 图片 {len(imgs)} 张，标签 {len(lbls)} 个")
    if len(imgs) != len(lbls):
        print("  ⚠ 图片与标签数量不一致，先查缺哪个")

    random.seed(a.seed)
    sample = random.sample(imgs, min(a.n, len(imgs)))
    stats = {"empty": 0, "bad": 0, "instances": 0}
    pattern, cls_cnt, per_img = Counter(), Counter(), Counter()

    out = Path(a.out)
    if not a.order_check:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        tiles = []

    for i, p in enumerate(sample):
        txt = lbl_dir / f"{p.stem}.txt"
        if not txt.exists():
            stats["bad"] += 1
            continue
        rows = load_labels(txt)
        if rows is None:
            stats["bad"] += 1
            continue
        if not rows:
            stats["empty"] += 1
        stats["instances"] += len(rows)
        per_img[len(rows)] += 1
        for cls, pts in rows:
            cls_cnt[cls] += 1
            pattern[order_pattern(pts)] += 1

        if not a.order_check:
            img = cv2.imread(str(p))
            if img is None:
                stats["bad"] += 1
                continue
            draw(img, rows, out / f"{i:03d}_{p.stem}.jpg")
            if len(tiles) < 4:
                tiles.append(cv2.resize(img, (640, 384)))

    print(f"[核对] 抽样 {len(sample)} 张：非法标签 {stats['bad']}，空标签 {stats['empty']}，实例 {stats['instances']}")
    print(f"[核对] 每图实例数：{dict(sorted(per_img.items()))}")
    print("[核对] 类别出现次数：" + ", ".join(
        f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else c}:{n}" for c, n in cls_cnt.most_common()))
    print("[核对] 角点绕序前 4 种：")
    for k, n in pattern.most_common(4):
        print(f"    {k}: {n}")

    if not a.order_check and len(tiles) == 4:
        montage = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(str(out / "montage.jpg"), montage)
        print(f"[输出] {out}/  （逐张图 + montage.jpg）")


if __name__ == "__main__":
    main()
