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
save_root = '/datasets/newout/vqa_info_2+13+4+3_fmt'

# Directory for request and result files
results_dir = os.path.join(save_root, "results")
splited_info_dir = os.path.join(save_root, "splited_info")

prompt_list = read_txt_file("prompts.txt")

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


def check_format(contents, info):
    global n_fe_1, n_fe_2, n_fe_3, n_fe_4, n_fe_5
    sorted_response_json, formatted_response = decode_response(contents)
    if len(sorted_response_json) != 5:
        print(f"format error 1: {res_path}")
        n_fe_1 += 1
        return False, None, None
    if info['labels'] != 0 and sorted_response_json['Analysis result'].lower() == 'real':
        print(f"format error 2: {res_path}")
        n_fe_2 += 1
        return False, None, None
    if info['labels'] == 0 and sorted_response_json['Analysis result'].lower() == 'fake':
        print(f"format error 3: {res_path}")
        n_fe_3 += 1
        return False, None, None
    if info['labels'] == 0 and not "none" in sorted_response_json['Forgery type'].lower():
        print(f"format error 4: {res_path}")
        n_fe_4 += 1
        return False, None, None
    if info['labels'] == 1 and not "pad" in sorted_response_json['Forgery type'].lower():
        if sorted_response_json['Analysis result'].lower() == 'fake':
            sorted_response_json['Forgery type'] = "PAD attacks"
            formatted_response = "\n".join(f"{key}: {value}" for key, value in sorted_response_json.items())
        else:
            print(f"format error 5: {res_path}")
            n_fe_5 += 1
            return False, None, None
    if float(sorted_response_json['Probability']) < 0.7:
        print(f"format error 5: {res_path}")
        n_fe_5 += 1
        return False, None, None
    return True, info['labels'], formatted_response


def extract_fields(input_string):
    pattern = r'(\w+): ([^\n]+)'
    matches = re.findall(pattern, input_string)
    result = {key: value.strip() for key, value in matches}
    return result


def quality_label(q: float) -> str:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0.0 and 1.0")

    if q < 0.2:
        return "Poor"
    elif q < 0.4:
        return "Low"
    elif q < 0.6:
        return "Fair"
    elif q < 0.87:
        return "Good"
    else:
        return "Excellent"
    

def quality_value(label: str) -> float:
    label = label.lower()

    mapping = {
        "poor": 0.1,       # midpoint of [0.0, 0.2)
        "low": 0.3,        # midpoint of [0.2, 0.4)
        "fair": 0.5,       # midpoint of [0.4, 0.6)
        "good": 0.7,     # midpoint of [0.6, 0.87)
        "excellent": 0.9 # midpoint of [0.87, 1.0]
    }

    if label not in mapping:
        raise ValueError(f"Unknown label: {label}")

    return mapping[label]
    

def has_none(value):
    if value is None:
        return True
    parts = str(value).lower().replace(" ", "").split("/")
    return any(p == "none" or p == "no" for p in parts)


def decode_response(resp_json):
    """
    Get the formatted answer from MLLM's response
    """
    response_json = {k.lower(): v for k, v in resp_json.items()}
    # if 'short_description' in response_json:
    #     response_json['Image description'] = response_json.pop('short_description')
    if 'long_forgery_reasoning' in response_json:
        response_json['Forgery reasoning'] = response_json.pop('long_forgery_reasoning')
    if 'result' in response_json:
        response_json['Analysis result'] = response_json.pop('result')
    if 'score' in response_json:
        response_json['Probability'] = response_json.pop('score')
    if 'forgery_type' in response_json:
        response_json['Forgery type'] = response_json.pop('forgery_type')
    if 'hair_color' in response_json:
        response_json['hair'] = response_json.pop('hair_color')
    if 'face_decorations' in response_json:
        response_json['decorations'] = response_json.pop('face_decorations')

    key_order = ['age', 'gender', 'expression', 'pose', 'ethnicity', 'skin',
                 'hair', 'hat', 'decorations', 'occlusions',
                 'focus', 'resolution', 'quality']
    # key_order = ['decorations', 'occlusions', 'focus', 'resolution', 'quality']
    # response_json['quality'] = quality_label(float(response_json['quality']))
    # temp_json = {key: response_json[key] for key in key_order if key in response_json}
    temp_json = {
        key: response_json[key]
        for key in key_order
        if key in response_json and not has_none(response_json[key])
    }
    description = ", ".join(f"{key}-{value}" for key, value in temp_json.items())
    response_json['Image description'] = description
    # response_json['Forgery reasoning'] = response_json['forgery_hint']
    
    key_order = ['Image description']
    # key_order = ['Image description', 'Forgery reasoning', 'Analysis result', 'Probability', 'Forgery type']
    sorted_response_json = {key: response_json[key] for key in key_order if key in response_json}
    formatted_response = "\n".join(f"{key}: {value}" for key, value in sorted_response_json.items())

    return sorted_response_json, formatted_response


def get_llava_prompt(qs, conv_mode):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def make_dataset_json(item_id, img_fn, answer_result):
    qs = random.choice(prompt_list)
    prompt = get_llava_prompt(qs, "v1")
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    entry = {
        "id": item_id,
        "image": img_fn,
        "conversations": [
            {
                "from": "human",
                # "value": prompt
                "value": qs
            },
            {
                "from": "gpt",
                "value": answer_result
            }
        ]
    }

    return entry


result_files = get_filepaths(results_dir)

n_total = 0
n_real = 0
n_pad = 0
n_deepfake = 0
n_makeup = 0
n_format_error = 0

n_fe_0 = 0
n_fe_1 = 0
n_fe_2 = 0
n_fe_3 = 0
n_fe_4 = 0
n_fe_5 = 0
n_badreal = 0
n_blur = 0

data = []

for res_path in result_files:
    head, fname = ntpath.split(res_path)
    pp = fname.split('.')
    id = pp[0]

    #eval dataset
    # if int(id) < 127655:
    #     continue

    split_path = os.path.join(splited_info_dir, f"{id}.txt")
    if os.path.exists(split_path):
        img_fn, info = parse_infoline(split_path)

        if "sel_for_mids/fake/deepfake/DFGC-2022/DetectionDataset" in img_fn \
                or ("/fake" in img_fn and "/real" in img_fn) \
                or "PAD/dataset/mywebcam/real" in img_fn:
            n_fe_0 += 1
            # print(f"bad label:{img_fn}")
            continue # bad label

        f = open(res_path, 'r+')
        contents = f.readlines()
        f.close()

        if "{" in contents[0]:
            contents[0] = contents[0][contents[0].find("{"): contents[0].rfind("}") + 1]
        response = contents[0].replace("#", "\n")
        try:
            obj = json.loads(response)
        except Exception as e:
            print(f"Error: {e}, {response}")
            print(f"format error 1: {img_fn}")
            n_fe_1 += 1
            continue

        sorted_response_json, formatted_response = decode_response(obj)

        n_total += 1

        entry = make_dataset_json(id, img_fn, formatted_response)
        data.append(entry)

        if n_total % 1000 == 0:
            print(n_total)

        # if n_total > 100:
        #     break

print(f"n_total={n_total}")

print(f"\n\nn_format_error_1={n_fe_1}")

# Save as JSON
with open(os.path.join(save_root, "eFFAA_feats.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

selected = random.sample(data, int(len(data)*0.02))
with open(os.path.join(save_root, "eFFAA_feats_eval.json"), "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)
