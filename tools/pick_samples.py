#!/usr/bin/env python3
"""从验证集里挑代表性样本，供 predict 生成可视化。

挑选规则（各取一张，去重）：
  1. 最大目标   —— 近距离、大装甲板
  2. 最小目标   —— 远距离、小装甲板（检验小目标性能）
  3. 目标最多   —— 一图多目标
  4. 含小样本类 —— BO / RS（训练实例最少的类）
  5. 中等目标   —— 常规距离，作为"普通情况"

用法：
    python3 tools/pick_samples.py                        # 只打印，不复制
    python3 tools/pick_samples.py --copy-to /tmp/picks   # 复制选中的图到目录，便于 predict
"""
import argparse
import pathlib
import shutil

import numpy as np

NAMES = ["B1","B2","B3","B4","B5","BO","BS","R1","R2","R3","R4","R5","RO","RS"]
RARE = {5, 13}   # BO, RS


def parse_args():
    home = pathlib.Path.home() / "hnurmvision_edu"
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(home / "dataset_norm" / "val"))
    p.add_argument("--copy-to", default="", help="复制选中的图片到这个目录")
    return p.parse_args()


def main():
    a = parse_args()
    root = pathlib.Path(a.data).expanduser()
    imgs, lbls = root / "images", root / "labels"
    if not lbls.is_dir():
        raise SystemExit(f"找不到 {lbls}")

    rows = []
    for txt in sorted(lbls.glob("*.txt")):
        img = next((imgs / (txt.stem + ext) for ext in (".jpg", ".png")), None)
        if img is None or not img.exists():
            continue
        boxes = []
        for line in txt.read_text().split("\n"):
            v = line.split()
            if len(v) != 13:
                continue
            pts = np.array([[float(v[5 + i * 2]), float(v[6 + i * 2])] for i in range(4)])
            boxes.append((float(pts[:, 0].max() - pts[:, 0].min()),
                          float(pts[:, 1].max() - pts[:, 1].min()), int(v[0])))
        if boxes:
            rows.append((txt.stem, img, boxes))

    picks, used = [], set()

    def take(tag, pred):
        for stem, img, boxes in pred(rows):
            if stem in used:
                continue
            used.add(stem)
            picks.append((tag, stem, img, boxes))
            return
    take("最大目标（近距离）", lambda rs: sorted(rs, key=lambda r: -max(b[0] for b in r[2])))
    take("最小目标（远距离，>=8px）",
         lambda rs: sorted([r for r in rs if 0.008 < max(b[0] for b in r[2]) < 0.06],
                           key=lambda r: max(b[0] for b in r[2])))
    take("多目标", lambda rs: sorted(rs, key=lambda r: -len(r[2])))
    take("小样本类 BO/RS",
         lambda rs: [r for r in rs if any(b[2] in RARE and b[0] > 0.02 for b in r[2])])
    take("中等目标（常规）", lambda rs: sorted(rs, key=lambda r: abs(max(b[0] for b in r[2]) - 0.08)))

    for tag, stem, img, boxes in picks:
        w = max(b[0] for b in boxes) * 1440
        cls = ",".join(NAMES[b[2]] for b in boxes)
        print(f"{tag:22s} {img.name:16s} 目标 {len(boxes)} 个（{cls}），最大板宽 ~{w:.0f}px")

    if a.copy_to:
        dst = pathlib.Path(a.copy_to).expanduser()
        dst.mkdir(parents=True, exist_ok=True)
        for _, _, img, _ in picks:
            shutil.copy(img, dst / img.name)
        print(f"\n已复制 {len(picks)} 张到 {dst}")
        print("接着跑：yolo pose predict model=... source=%s imgsz=640 conf=0.25 save=True name=picks" % dst)


if __name__ == "__main__":
    main()
