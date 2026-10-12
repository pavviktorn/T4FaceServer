#!/usr/bin/env python3
import argparse
import glob
import ntpath
import os
import time

import cv2


def get_frame(video_path, index):
    try:
        head, fname = ntpath.split(video_path)
        pp = fname.split(".")
        path = os.path.join(head, pp[0])
        path = path + f"/{index}.jpg"
        if os.path.exists(path):
            return path
        file_name_without_ext, _ = os.path.splitext(fname)
        path = os.path.join(head, file_name_without_ext)
        path = path + f"/{index}.jpg"
        if os.path.exists(path):
            return path
        return ""
    except Exception:
        return ""


def parse_info_to_path_and_label(info, img_root):
    elem = info.split(",")
    if len(elem) < 4:
        return "", None

    try:
        class_label = int(elem[0])
    except ValueError:
        return "", None

    video_or_frame = elem[1]
    org_path = elem[3]

    if video_or_frame == "f":
        path = (img_root + org_path).replace("\\", "/")
        head, fname = ntpath.split(path)
        pp = fname.split(".")
        path = head + f"/{pp[0]}.jpg"
        if not os.path.exists(path):
            file_name_without_ext, _ = os.path.splitext(fname)
            path = head + f"/{file_name_without_ext}.jpg"
            if not os.path.exists(path):
                path = ""
    else:
        pp = org_path.split(":")
        if len(pp) < 2:
            return "", None
        path = (img_root + pp[0]).replace("\\", "/")
        try:
            frame = int(pp[1])
        except ValueError:
            return "", None
        path = get_frame(path, frame)

    return path, class_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export original left images from batch ids using split info files."
    )
    parser.add_argument(
        "--save-root",
        default="/datasets/newout/vqa_info_2+13+4+3_fmt",
        help="Root folder containing batches/, splited_info/, and left_images/.",
    )
    parser.add_argument(
        "--img-root",
        default="/datasets/newout",
        help="Image root used to resolve org_path from split info lines.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images in left_images.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of info lines to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    batches_dir = os.path.join(args.save_root, "batches")
    splited_info_dir = os.path.join(args.save_root, "splited_info")
    left_images_dir = os.path.join(args.save_root, "left_images")
    os.makedirs(left_images_dir, exist_ok=True)

    batch_files = sorted(glob.glob(os.path.join(batches_dir, "*.jsonl")))
    print(f"[INFO] batches_dir={batches_dir}")
    print(f"[INFO] splited_info_dir={splited_info_dir}")
    print(f"[INFO] left_images_dir={left_images_dir}")
    print(f"[INFO] found {len(batch_files)} batch files")

    stats = {
        "batches_seen": 0,
        "lines_seen": 0,
        "saved": 0,
        "skipped_exists": 0,
        "missing_split": 0,
        "empty_split": 0,
        "missing_path": 0,
        "read_failed": 0,
        "write_failed": 0,
    }
    t0 = time.time()

    stop = False
    for batch_file in batch_files:
        batch_id = os.path.splitext(os.path.basename(batch_file))[0]
        stats["batches_seen"] += 1

        split_path = os.path.join(splited_info_dir, f"{batch_id}.txt")
        if not os.path.exists(split_path):
            stats["missing_split"] += 1
            print(f"[WARN] missing split file: {split_path}")
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            info_lines = [line.strip() for line in f if line.strip()]

        if not info_lines:
            stats["empty_split"] += 1
            print(f"[WARN] empty split file: {split_path}")
            continue

        multi = len(info_lines) > 1
        for idx, info in enumerate(info_lines):
            if args.max is not None and stats["lines_seen"] >= args.max:
                stop = True
                break

            stats["lines_seen"] += 1
            img_path, class_label = parse_info_to_path_and_label(info, args.img_root)
            if img_path == "" or not os.path.exists(img_path):
                stats["missing_path"] += 1
                print(f"[WARN] missing image for {batch_id} line {idx}: {img_path}")
                continue

            if class_label is None:
                stats["missing_path"] += 1
                print(f"[WARN] missing class label for {batch_id} line {idx}")
                continue

            out_name = (
                f"{class_label}_{batch_id}.jpg"
                if not multi
                else f"{class_label}_{batch_id}_{idx}.jpg"
            )
            out_path = os.path.join(left_images_dir, out_name)

            if os.path.exists(out_path) and not args.overwrite:
                stats["skipped_exists"] += 1
                continue

            image = cv2.imread(img_path)
            if image is None:
                stats["read_failed"] += 1
                print(f"[WARN] cv2.imread failed: {img_path}")
                continue

            ok = cv2.imwrite(out_path, image)
            if not ok:
                stats["write_failed"] += 1
                print(f"[WARN] cv2.imwrite failed: {out_path}")
            else:
                stats["saved"] += 1

            if stats["lines_seen"] % 1000 == 0:
                dt = time.time() - t0
                print(
                    f"[INFO] progress lines={stats['lines_seen']} "
                    f"saved={stats['saved']} skipped={stats['skipped_exists']} "
                    f"missing_path={stats['missing_path']} elapsed={dt:.1f}s"
                )

        if stop:
            break

    dt = time.time() - t0
    print("[INFO] done")
    print(
        f"[INFO] batches_seen={stats['batches_seen']} "
        f"lines_seen={stats['lines_seen']} saved={stats['saved']} "
        f"skipped_exists={stats['skipped_exists']} "
        f"missing_split={stats['missing_split']} empty_split={stats['empty_split']} "
        f"missing_path={stats['missing_path']} read_failed={stats['read_failed']} "
        f"write_failed={stats['write_failed']} elapsed={dt:.1f}s"
    )


if __name__ == "__main__":
    main()
