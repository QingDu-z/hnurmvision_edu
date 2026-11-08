#!/usr/bin/env python3
"""仿真器 10 列 GT → YOLO-Pose 13 列（图像顺序 TL,TR,BR,BL）。

标签格式：color typ label x1 y1 x2 y2 x3 y3 x4 y4（归一化，板体顺序 TL,BL,BR,TR）
输出：cls cx cy w h kx1 ky1 ... kx4 ky4，cls = (label-1) + 7*color（label 为 1 基）
"""
import argparse
import pathlib
import shutil

NAMES = ["B1", "B2", "B3", "B4", "B5", "BO", "BS",
         "R1", "R2", "R3", "R4", "R5", "RO", "RS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="仿真器导出的 dataset 目录")
    ap.add_argument("--dst", required=True, help="输出目录（会被清空重建）")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()

    n_img = n_box = 0
    for lbl in sorted((src / "label").glob("*.txt")):
        img = None
        for ext in (".jpg", ".png"):
            cand = src / "images" / (lbl.stem + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        lines = []
        for line in lbl.read_text().strip().split("\n"):
            v = line.split()
            if len(v) < 11:
                continue
            color, label = int(v[0]), int(v[2])
            pts = [(float(v[3 + i * 2]), float(v[4 + i * 2])) for i in range(4)]
            tl, bl, br, tr = pts                    # 板体顺序
            ordered = [tl, tr, br, bl]              # → 图像顺序
            xs = [p[0] for p in ordered]
            ys = [p[1] for p in ordered]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            # 仿真器的 label 是 1 基编号（1..5），类索引 = label-1 + 7*color
            cls = (label - 1) + 7 * color
            if not 0 <= cls <= 13:
                continue
            kp = " ".join(f"{x:.6f} {y:.6f}" for x, y in ordered)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp}")
            n_box += 1
        if lines:
            (dst / "labels" / (lbl.stem + ".txt")).write_text("\n".join(lines) + "\n")
            shutil.copy(img, dst / "images" / img.name)
            n_img += 1

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve()}\ntrain: images\nval: images\n"
        "kpt_shape: [4, 2]\nflip_idx: [1, 0, 3, 2]\n"
        f"names: {NAMES}\n")
    print(f"转换完成：{n_img} 张图，{n_box} 个目标 → {dst}")


if __name__ == "__main__":
    main()
