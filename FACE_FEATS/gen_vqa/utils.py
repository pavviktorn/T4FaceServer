import base64
from threading import Thread
import cv2
import os
import time
import json
import ntpath
import numpy as np
import random
from insightface.app import FaceAnalysis


app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))


def get_face(frame):
    if len(frame.shape) < 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    new_height = 640
    new_width = 640
    exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    exp_scale = 1.0
    if frame.shape[0] >= frame.shape[1] and 480 < frame.shape[0]:
        half_h = 640
        half_w = int(frame.shape[1] * half_h / frame.shape[0])
        exp_scale = half_h / frame.shape[0]
        half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
        x_pos = int(new_height / 2 - half_img.shape[0] / 2)
        y_pos = int(new_width / 2 - half_img.shape[1] / 2)
        exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img
        faces = app.get(exp_img)
        if len(faces) == 0:
            half_h = 320
            half_w = int(frame.shape[1] * half_h / frame.shape[0])
            exp_scale = half_h / frame.shape[0]
            half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
            x_pos = int(new_height / 2 - half_img.shape[0] / 2)
            y_pos = int(new_width / 2 - half_img.shape[1] / 2)
            exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img
            faces = app.get(exp_img)
            if len(faces) == 0:
                return [], []

    elif frame.shape[0] <= frame.shape[1] and 480 < frame.shape[1]:

        half_w = 640
        half_h = int(frame.shape[0] * half_w / frame.shape[1])
        exp_scale = half_w / frame.shape[1]
        half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
        x_pos = int(new_height / 2 - half_img.shape[0] / 2)
        y_pos = int(new_width / 2 - half_img.shape[1] / 2)
        exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img

        faces = app.get(exp_img)
        if len(faces) == 0:
            half_w = 320
            half_h = int(frame.shape[0] * half_w / frame.shape[1])
            exp_scale = half_w / frame.shape[1]
            half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
            x_pos = int(new_height / 2 - half_img.shape[0] / 2)
            y_pos = int(new_width / 2 - half_img.shape[1] / 2)
            exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img
            faces = app.get(exp_img)
            if len(faces) == 0:
                return [], []

    else:
        exp_scale = 1.0
        x_pos = int(new_height / 2 - frame.shape[0] / 2)
        y_pos = int(new_width / 2 - frame.shape[1] / 2)
        exp_img[x_pos: x_pos + frame.shape[0], y_pos: y_pos + frame.shape[1], :] = frame
        faces = app.get(exp_img)
        if len(faces) == 0:
            return [], []

    # get biggest face
    max_idx = 0
    max_area = 0
    for face_idx in range(0, len(faces)):
        face = faces[face_idx]
        area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        if area > max_area:
            max_idx = face_idx
            max_area = area

    face = faces[max_idx]

    bbox = face.bbox
    bbox[0] = int((bbox[0] - y_pos) / exp_scale)
    bbox[1] = int((bbox[1] - x_pos) / exp_scale)
    bbox[2] = int((bbox[2] - y_pos) / exp_scale)
    bbox[3] = int((bbox[3] - x_pos) / exp_scale)

    kps = face.kps
    for i in range(5):
        kps[i][0] = int((kps[i][0] - y_pos) / exp_scale)
        kps[i][1] = int((kps[i][1] - x_pos) / exp_scale)

    return bbox, kps


def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    # scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        # right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        # right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w - 1:
        # left_top_x -= right_bottom_x-src_w+1
        right_bottom_x = src_w - 1

    if right_bottom_y > src_h - 1:
        # left_top_y -= right_bottom_y-src_h+1
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def get_cropped(org_img, face_bbox, scale, out_w=256, out_h=256, crop=True):

    if not crop:
        dst_img = cv2.resize(org_img, (out_w, out_h))
    else:
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, face_bbox, scale)

        img = org_img[left_top_y: right_bottom_y+1, left_top_x: right_bottom_x+1]
        dst_img = cv2.resize(img, (out_w, out_h))

    return dst_img


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return sorted(file_paths)


def read_txt_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_base64(img):
    # Convert the image to a binary buffer
    _, img_buffer = cv2.imencode('.jpg', img)

    # Convert the buffer to bytes
    img_bytes = img_buffer.tobytes()

    # Encode the bytes to base64
    base64_str = base64.b64encode(img_bytes).decode("utf-8")

    return base64_str


def resize_image_keep_aspect_opencv(image, target_width=240, target_height=320):
    # image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    # Calculate aspect ratio for width and height scaling
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    # Use the smaller ratio to keep aspect ratio in bounds
    scale_ratio = min(width_ratio, height_ratio)
    # Calculate new dimensions
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    # Resize the image with interpolation
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Create a new image with the target dimensions and white background
    output_image = 255 * np.ones(shape=[target_height, target_width, 3], dtype=np.uint8)
    # Calculate top-left corner to paste resized image centered
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    # Paste resized image onto output image
    output_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return output_image


def get_forgery_specific_prompt(img_path):
    if "/real/" in img_path:
        return "The image is a real face. Please analyze the reason and answer the question."
    elif "/pad/" in img_path:
        return "The image is a fake face whose forgery type is PAD attacks. Please analyze the reason and answer the question."
    elif "/deepfake/" in img_path:
        return "The image is a fake face whose forgery type is identity exchange. Please analyze the reason and answer the question."


def get_forgery_specific_prompt1(img_path, source_path, class_label):
    if source_path == "":
        if class_label == 0:
            return ("The image is a real face. It can assist in your analysis, but you cannot mention it in your answer. "
                    "Please analyze the reason and answer the question.")
        elif class_label == 1:
            return ("The image is a fake face whose forgery type is PAD attacks. It can assist in your analysis, but you cannot mention it in your answer. "
                    "Please analyze the reason and answer the question.")
        elif class_label == 2:
            return ("The image is a fake face whose forgery type is identity exchange. It can assist in your analysis, but you cannot mention it in your answer. "
                    "Please analyze the reason and answer the question.")
        else:
            print("class label error")
            exit(0)
    else:
        if class_label == 0:
            return "The image is a real face. It can assist in your analysis, but you cannot mention it in your answer. " \
                   "Please analyze the reason and answer the question. If you can't get a clear answer, please tell me simply that you can't get a result."
        elif class_label == 1:
            return "The image is a fake face whose forgery type is PAD attacks. It can assist in your analysis, but you cannot mention it in your answer. " \
                   "Please analyze the reason and answer the question. If you can't get a clear answer, please tell me simply that you can't get a result."
        elif class_label == 2:
            return "The first image is a fake face whose forgery type is identity exchange. " \
                   "It can assist in your analysis, but you cannot mention it in your answer. " \
                   "Please analyze the reason and answer the question. If you can't get a clear answer, please tell me simply that you can't get a result.\n" \
                   "And the second image shows the original, unaltered version before manipulation. It can assist in your analysis, but you cannot mention it in your answer."
        else:
            print("class label error")
            exit(0)


def get_forgery_specific_prompt2(img_path, source_path, class_label):
    if class_label == 0:
        return ("The image is a real face. It can assist in your analysis, but you cannot mention it in your answer. "
                "Please analyze the reason and answer the question.")
    elif class_label == 1:
        return ("The image is a fake face whose forgery type is PAD attacks. "
                "It can assist in your analysis, but you cannot mention it in your answer. "
                "Please analyze the reason and answer the question.")
    elif class_label == 2:
        return ("The image is a fake face whose forgery type is identity exchange. "
                "It can assist in your analysis, but you cannot mention it in your answer. "
                "Please analyze the reason and answer the question.")
    elif class_label == 3:
        return ("The image is a fake face whose forgery type is makeup manipulation. "
                "It can assist in your analysis, but you cannot mention it in your answer. "
                "Please analyze the reason and answer the question.")
    else:
        print("class label error")
        exit(0)


def get_frame(video_path, index):
    try:
        head, fname = ntpath.split(video_path)
        pp = fname.split('.')
        path = os.path.join(head, pp[0])
        path = path + f'/{index}.jpg'
        if os.path.exists(path):
            return path
        else:
            file_name_without_ext, ext = os.path.splitext(fname)
            path = os.path.join(head, file_name_without_ext)
            path = path + f'/{index}.jpg'
            if os.path.exists(path):
                return path

            return ''
    except:
        print(f'get_frame except : {video_path} {index}')
        return None, ''

    return ''

