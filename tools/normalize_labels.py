#!/usr/bin/env python3
"""P3.5：把装甲板标注的角点顺序规范化成 TL,TR,BR,BL（图像坐标）。

背景：数据集标签是"板体顺序"（多为 左上,左下,右下,右上），而 lesson03 的
Armor.corners 要求 左上,右上,右下,左下。训练前重排，模型直接学目标顺序。

标签格式（13 列，与 dataset.zip 一致）：
    class cx cy w h  x0 y0 x1 y1 x2 y2 x3 y3      # 后 8 个数是 4 个归一化角点

用法：
    python3 tools/normalize_labels.py                     # 默认 ~/hnurmvision_edu/dataset → dataset_norm
    python3 tools/normalize_labels.py --src A --dst B --link    # 图片用软链，省 600 MB
    python3 tools/normalize_labels.py --dry-run                 # 只统计，不写文件
"""
import argparse
import pathlib
import shutil
import sys
from collections import Counter

import numpy as np


def parse_args():
    home = pathlib.Path.home() / "hnurmvision_edu"
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(home / "dataset"), help="原数据集（含 train/ val/）")
    p.add_argument("--dst", default=str(home / "dataset_norm"), help="输出目录")
    p.add_argument("--link", action="store_true", help="图片用软链接而不是复制")
    p.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    p.add_argument("--sample", type=int, default=0, help="只处理前 N 个标签（调试用）")
    return p.parse_args()


def order_rank(pts: np.ndarray) -> np.ndarray:
    """按 y 分上下排、各自按 x 分左右；同高时退化为极角排序。"""
    idx = np.argsort(pts[:, 1])
    top, bot = pts[idx[:2]], pts[idx[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    if abs(top[0][1] - top[1][1]) < 0.004:          # 上排两点几乎同高（窄板/正对镜头）
        c = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        pts = pts[order]
        top, bot = pts[:2], pts[2:]
        top = top[np.argsort(top[:, 0])]
        bot = bot[np.argsort(bot[:, 0])]
    cand = np.array([top[0], top[1], bot[1], bot[0]])   # TL, TR, BR, BL（按排）
    if is_canonical(cand):
        return cand
    # 兜底：极对角排序，两种绕向里挑满足 TL,TR,BR,BL 的那个（处理横滚/倒装板）
    c = pts.mean(axis=0)
    cyc = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    k = int(np.argmin(cyc[:, 0] + cyc[:, 1]))
    cyc = np.roll(cyc, -k, axis=0)
    for candidate in (cyc, cyc[[0, 3, 2, 1]]):
        if is_canonical(candidate):
            return candidate
    return cand


def is_canonical(p: np.ndarray) -> bool:
    """四个不等式同时成立才算 TL,TR,BR,BL。"""
    return (p[1][0] > p[0][0] and p[2][1] > p[1][1]
            and p[3][0] < p[2][0] and p[0][1] < p[3][1])


def pattern(p: np.ndarray) -> str:
    """相对质心的粗标签，仅用于前后对比（窄板会有退化组合，不代表错误）。"""
    cx, cy = p[:, 0].mean(), p[:, 1].mean()
    return ",".join(("T" if y < cy else "B") + ("L" if x < cx else "R") for x, y in p)


def main():
    a = parse_args()
    src, dst = pathlib.Path(a.src).expanduser(), pathlib.Path(a.dst).expanduser()
    if not (src / "train" / "labels").is_dir():
        sys.exit(f"找不到 {src}/train/labels，先确认 --src")

    before, after = Counter(), Counter()
    bad_before = bad_after = changed = total_inst = 0
    files = 0

    for split in ("train", "val"):
        ldir, idir = src / split / "labels", src / split / "images"
        if not ldir.is_dir():
            continue
        lines_out_dir = dst / split / "labels"
        img_out_dir = dst / split / "images"
        if not a.dry_run:
            lines_out_dir.mkdir(parents=True, exist_ok=True)
            img_out_dir.mkdir(parents=True, exist_ok=True)

        txts = sorted(ldir.glob("*.txt"))
        if a.sample:
            txts = txts[:a.sample]

        for txt in txts:
            rows, out = [], []
            for line in txt.read_text().strip().split("\n"):
                v = line.split()
                if len(v) != 13:
                    continue
                pts = np.array([[float(v[5 + i * 2]), float(v[6 + i * 2])] for i in range(4)])
                before[pattern(pts)] += 1
                if not is_canonical(pts):
                    bad_before += 1
                new = order_rank(pts)
                after[pattern(new)] += 1
                if not is_canonical(new):
                    bad_after += 1
                if not np.allclose(pts, new):
                    changed += 1
                total_inst += 1
                rows.append(new)
                out.append(" ".join(v[:5]) + " " + " ".join(f"{x:.6f}" for p in new for x in p))
            if a.dry_run:
                continue

            (lines_out_dir / txt.name).write_text("\n".join(out) + ("\n" if out else ""))
            img_in = idir / f"{txt.stem}.jpg"
            if img_in.exists():
                target = img_out_dir / img_in.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                if a.link:
                    target.symlink_to(img_in.resolve())
                else:
                    shutil.copy(img_in, target)
            files += 1

    print(f"处理文件 {files} 个，实例 {total_inst} 个")
    print(f"非规范实例：前 {bad_before} → 后 {bad_after}（重排了 {changed} 个）")
    print("前 3 种绕序：")
    for k, n in before.most_common(3):
        print(f"  {k}: {n}")
    print("后 3 种绕序：")
    for k, n in after.most_common(3):
        print(f"  {k}: {n}")
    if not a.dry_run:
        print(f"输出：{dst}")


if __name__ == "__main__":
    main()
