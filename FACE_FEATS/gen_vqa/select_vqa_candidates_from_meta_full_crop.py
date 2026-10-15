import base64
import cv2
import os
import time
import json
import ntpath
import numpy as np
import random


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return sorted(file_paths)


def get_group(info):
    elem = info.split(',')
    class_label = int(elem[0])
    video_or_frame = elem[1]
    quality = elem[2]
    org_path = elem[3]
    src_path = elem[4]

    if video_or_frame == 'f':
        head, fname = ntpath.split(org_path)
        group_str = head
    else:
        pp = org_path.split(':')
        group_str = pp[0].replace('\\', '/')

    return group_str, class_label


def main():
    in_root = '/datasets/newout/meta'
    out_root = '/datasets/newout'
    # in_root = r'D:\zLiveness_data\out\temp'
    # out_root = 'D:/zLiveness_data/out'

    full_file_paths = get_filepaths(in_root)

    sel_percent_df = 0.05
    sel_percent_pad = 0.05
    sel_percent_real = 0.3

    n_df = 0
    n_pad = 0
    n_real = 0

    f_a = open(os.path.join(out_root, 'vqa_candidates.txt'), 'w')

    for src_path in full_file_paths:
        head, fname = ntpath.split(src_path)
        if os.path.isdir(src_path) is False and fname.endswith('.txt'):
            if "new_test_20250821" in src_path:
                continue
            src_path = src_path.replace("\\", "/")
            print(src_path)
            f = open(src_path, 'r+')
            meta_info = f.readlines()
            f.close()

            mi_size = len(meta_info)

            prev_group_str, label = get_group(meta_info[0])
            cur_group = []
            for idx in range(mi_size):
                info = meta_info[idx]
                cur_group_str, label = get_group(info)
                if cur_group_str != prev_group_str:
                    cg_size = len(cur_group)
                    if label == 0: #real
                        sel_num = max(int(cg_size * sel_percent_real), min(cg_size, 10))
                        n_real += sel_num
                    elif label == 1: #pad
                        sel_num = max(int(cg_size * sel_percent_pad), min(cg_size, 3))
                        n_pad += sel_num
                    elif label == 2: #df
                        sel_num = max(int(cg_size * sel_percent_df), min(cg_size, 2))
                        n_df += sel_num

                    selected = random.sample(cur_group, k=sel_num)
                    for line in selected:
                        f_a.write(line)
                    cur_group = []

                cur_group.append(info)
                prev_group_str = cur_group_str

    f_a.close()

    print(f"sel real: {n_real}")
    print(f"sel pad: {n_pad}")
    print(f"sel deepfake: {n_df}")

    return



if __name__ == '__main__':
    main()


