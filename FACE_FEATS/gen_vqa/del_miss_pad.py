import json
import random
import time
import os
from utils import *
import shutil
import ntpath
import re
from conversation import conv_templates
from constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)


# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
img_root = '/datasets/newout'
save_root = '/datasets/newout/vqa_info/miss&makeup'

# Directory for request and result files
results_dir = os.path.join(save_root, "results")
splited_info_dir = os.path.join(save_root, "splited_info")

def parse_info(info):
    elem = info.split(',')
    class_label = int(elem[0])
    video_or_frame = elem[1]
    quality = elem[2]
    org_path = elem[3]
    source_path = elem[4]
    if source_path != ':':
        source_path = (img_root + source_path).replace('\\', '/')
    else:
        source_path = ''

    bbox = np.zeros((4,)).astype(int)
    landmark = np.zeros((106, 2)).astype(int)

    if video_or_frame == 'f':
        path = (img_root + org_path).replace('\\', '/')
        frame = 0
        head, fname = ntpath.split(path)
        pp = fname.split('.')
        path = head + f'/{pp[0]}.jpg'
        if not os.path.exists(path):
            file_name_without_ext, ext = os.path.splitext(fname)
            path = head + f'/{file_name_without_ext}.jpg'
            if not os.path.exists(path):
                path = ""
        # if source_path != '':
        #     head, fname = ntpath.split(source_path)
        #     pp = fname.split('.')
        #     source_path = head + f'/{pp[0]}.jpg'
        #     if not os.path.exists(source_path):
        #         file_name_without_ext, ext = os.path.splitext(fname)
        #         source_path = head + f'/{file_name_without_ext}.jpg'
        #         if not os.path.exists(source_path):
        #             source_path = ""
    else: # "v"
        pp = org_path.split(':')
        path = (img_root + pp[0]).replace('\\', '/')
        frame = int(pp[1])
        path = get_frame(path, frame)
        # if source_path != '':
        #     source_path = get_frame(source_path, frame)
        # else:
        #     source_path = ""

    be = elem[5].split(':')
    bbox[0], bbox[1], bbox[2], bbox[3] = int(be[0]), int(be[1]), int(be[2]), int(be[3]),
    le = elem[6].split(':')
    for i in range(landmark.shape[0]):
        xy = le[i].split('_')
        landmark[i][0], landmark[i][1] = xy[0], xy[1]

    return path, {'labels': class_label, 'vf': video_or_frame, 'frame': frame, 'landmark': landmark, 'bbox': bbox,
                        'source_path': source_path,
                        'quality': quality}


def parse_infoline(split_path):
    f = open(split_path, 'r+')
    lines = f.readlines()
    f.close()
    path, info = parse_info(lines[0])
    path = path.replace(img_root, "")
    path = path[1:]

    return path, info


split_files = get_filepaths(splited_info_dir)

n_total = 0
n_pad = 0
n_makeup = 0

data = []

for split_path in split_files:
    head, fname = ntpath.split(split_path)
    pp = fname.split('.')
    id = pp[0]
    res_path = os.path.join(results_dir, f"{id}.jsonl")
    if os.path.exists(res_path):
        img_fn, info = parse_infoline(split_path)
        if info['labels'] == 2 and '/pad/' in img_fn:
            n_pad += 1
            os.remove(res_path)
            if os.path.exists(split_path):
                os.remove(split_path)
        elif 'Makeup' in img_fn:
            n_makeup += 1

        n_total += 1
        if n_total % 1000 == 0:
            print(n_total)

print(f"n_total={n_total}")
print(f"n_pad={n_pad}")
print(f"n_makeup={n_makeup}")
