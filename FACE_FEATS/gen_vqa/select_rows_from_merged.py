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

    return class_label


def main():
    info_path = '/datasets/newout/vqa_info_full_fmt/1_baseinfo/merged.txt'

    f = open(info_path, 'r+')
    info_meta = f.readlines()
    f.close()

    df_arr = []
    pad_arr = []
    real_arr = []
    makeup_arr = []
    for info in info_meta:
        label = get_group(info)
        if label == 0:
            real_arr.append(info)
        elif label == 1:
            pad_arr.append(info)
        elif label == 2:
            df_arr.append(info)
        elif label == 3:
            makeup_arr.append(info)
        else:
            continue

    sel_all = []
    rest_all = []

    random.shuffle(df_arr)
    selected = df_arr[:350000]
    rest = df_arr[350000:]
    sel_all.extend(selected)
    rest_all.extend(rest)

    random.shuffle(pad_arr)
    selected = pad_arr[:350000]
    rest = pad_arr[350000:]
    sel_all.extend(selected)
    rest_all.extend(rest)

    random.shuffle(real_arr)
    selected = real_arr[:350000]
    rest = real_arr[350000:]
    sel_all.extend(selected)
    rest_all.extend(rest)

    sel_all.extend(makeup_arr)

    f = open(info_path, 'w')
    f.writelines(rest_all)
    f.close()

    f_a = open('/datasets/newout/vqa_info_full_fmt/selected_1M.txt', 'w')
    f_a.writelines(sel_all)
    f_a.close()

    return



if __name__ == '__main__':
    main()


