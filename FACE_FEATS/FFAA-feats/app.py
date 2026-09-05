import ntpath
import random
import string
import json
from datetime import datetime
from io import BytesIO
from typing import List

import cv2
import base64
import os
from PIL import Image, UnidentifiedImageError
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import torch
# torch.set_flush_denormal(True)
from models import *
from transformers import AutoTokenizer, CLIPProcessor

from flask import (Flask, render_template, request, jsonify, g)
from werkzeug.utils import secure_filename

# from insightface.app import FaceAnalysis
from yolo11_cls_onnx import Yolo11ClsONNX
from flask_cors import CORS
from mids.selector import make_decision_batch

# # Initialize globally (do this once at startup)
# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU, 0 for first GPU

det_model_path = "./rot_det_model/yolo11n-rotation2/weights/best.onnx"
det_model = Yolo11ClsONNX(
    onnx_path=det_model_path,
    imgsz=224,
    class_names=["0", "180", "270", "90"],
)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = APP_ROOT + "/images"
DEFAULT_FEATURES_PROMPT = "The image is a human face image. Please analyze facial features."
IMAGE_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
}

# model path
llava_groups = {
    'mistral': "checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_4",
}

g_crop = 0
device_id = 0

# load mllm
mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["mistral"], device_id)


app = Flask(__name__)
CORS(app) # Enable CORS for all routes and origins
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH_MB', '64')) * 1024 * 1024


# force browser to hold no cache. Otherwise old result might return.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def homepage():
    # return render_template('home.html')
    return render_template('home_two.html')


def getRotatedAngle(img):

    rot_angle, conf = det_model.predict(img)  # angle:(int 0,90,180,270) conf:(float)

    return rot_angle


def upload_request_images_face(content, subdir1, subdir2):
    """ Upload request image to folder """
    file_name = secure_filename(content.filename)

    file_name_without_ext, ext = os.path.splitext(file_name)
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext

    subdir = get_request_image_subdir()

    # subdir = os.path.join(subdir, subdir2)
    # if not os.path.exists(subdir):
    #     os.makedirs(subdir)

    file_path = os.path.join(subdir, fname)
    content.save(file_path)

    return file_path


def get_request_image_subdir():
    ipaddr = request.headers.get("X-Forwarded-For", request.remote_addr)
    if ipaddr:
        ipaddr = ipaddr.split(",")[0].strip()
    if not ipaddr:
        ipaddr = "unknown"

    subdir = os.path.join('./images', ipaddr)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def get_image_suffix_from_bytes(image_bytes):
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Failed to decode image") from exc

    return IMAGE_FORMAT_TO_EXT.get(image_format, ".png")


def save_request_image_bytes(image_bytes, suffix):
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix
    file_path = os.path.join(get_request_image_subdir(), fname)
    with open(file_path, "wb") as file_obj:
        file_obj.write(image_bytes)
    return file_path


def get_random_string():
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(8))
    return result_str


def get_features_prompt():
    prompt_list = read_txt_file("playground/prompts.txt")
    if len(prompt_list) == 0:
        return DEFAULT_FEATURES_PROMPT
    return prompt_list[0]


def build_face_features_string_response(answer):
    output = get_jsonfmt(answer)
    json_output = json.dumps(output, indent=2)
    print(json_output)
    return {
        'success': True,
        'face_features': json_output,
    }


def build_face_features_object_response(answer):
    output = get_jsonfmt(answer)
    json_output = json.dumps(output, indent=2)
    print(json_output)
    return {
        'success': True,
        'face_features': output,
    }


def inference(args):
    # args
    image_path = args.image_path
    crop = args.crop

    start_t = time.time()

    print(f'USER: {args.prompt}\n')
    device = torch.device(f'cuda:{device_id}')

    def run_mistral(image, answer_holder):
        with torch.no_grad():
            answers = get_llava_answer(mistral_model, mistral_tokenizer, mistral_image_processor,
                                       image, args.prompt, args.temperature, args.top_p, args.num_beams,
                                       args.max_new_tokens, args.generate_num['mistral'], 'v1')
            answer_holder.extend(answers)

    image = load_image(image_path)
    if crop == 1:
        image = crop_face(image)
        if image is None:
            print('No face detected')
            return
    else:
        image.save(TMP_IMG_PATH)

    threads = []

    mistral_answer_holder = []

    mistral_thread = threading.Thread(target=run_mistral, args=(image, mistral_answer_holder))
    threads.append(mistral_thread)
    mistral_thread.start()

    # wait all threads complete
    for thread in threads:
        thread.join()

    answers = mistral_answer_holder

    end_t = time.time()
    print(f"total time : {end_t - start_t}s")

    return answers[0]


def get_face_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    angle = getRotatedAngle(img)
    if angle > 0:
        res = {'error': 'The image appears to be rotated. Please try again with a straightened image.'}
        print(f"error: The image appears to be rotated. Please try again with a straightened image.")
        return res

    # select the image and write your prompt here
    prompt = get_features_prompt()

    crop = g_crop
    visualize = 0

    genarate_num_dict = {
        'mistral': 1,
    }

    args = type('Args', (), {
        "device": 0,
        "image_path": image_path,
        "prompt": prompt,
        "crop": crop,
        "conv_mode": None,
        "llava_groups": llava_groups,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "generate_num": genarate_num_dict,
        "max_new_tokens": 512
    })()

    answer = inference(args)

    # print(answer)

    res = {'success': True, 'face_features': answer}

    head, fname = ntpath.split(image_path)
    file_name_without_ext, ext = os.path.splitext(fname)
    text_path = os.path.join(head, file_name_without_ext+'.txt')
    with open(text_path, "w") as f:
        f.write(answer)

    return res


@app.route('/face_features', methods=['POST'])
def receive_face():
    file_list = []
    # print(request.form)
    # if ('user_id' not in request.form):
    #     return jsonify({'error': 'no user id.'})
    # user_id = request.form['user_id']
    user_id = get_random_string()

    if ('face' not in request.files):
        return jsonify({'error': 'no face image file.'})
    face_image = request.files['face']

    if face_image.filename == '':
        return jsonify({'error': 'no face image file.'})

    print(face_image.filename)
    img_path = upload_request_images_face(face_image, user_id, 'features')

    res = get_face_features(img_path)

    if 'error' in res:
        return jsonify(res)

    answer = res['face_features']
    return jsonify(build_face_features_string_response(answer))


@app.route('/face_features_base64', methods=['POST'])
def receive_face_base64():
    data = request.get_json(silent=True)

    if not data or 'image_base64' not in data:
        return jsonify({'error': 'image_base64 is required'}), 400

    image_base64 = data['image_base64']

    # Remove data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        return jsonify({'error': 'Invalid base64 image'}), 400

    try:
        suffix = get_image_suffix_from_bytes(image_bytes)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    img_path = save_request_image_bytes(image_bytes, suffix)

    # Reuse existing logic
    res = get_face_features(img_path)

    if 'error' in res:
        return jsonify(res)

    answer = res['face_features']
    return jsonify(build_face_features_object_response(answer))


def get_ssl_context():
    cert_path = os.environ.get("SSL_CERT_PATH", "certs/cert.pem")
    key_path = os.environ.get("SSL_KEY_PATH", "certs/key.pem")

    if (
        os.path.isfile(cert_path)
        and os.path.isfile(key_path)
        and os.access(cert_path, os.R_OK)
        and os.access(key_path, os.R_OK)
    ):
        return (cert_path, key_path)

    print(f"SSL cert/key not readable ({cert_path}, {key_path}); using ad-hoc self-signed cert.")
    return "adhoc"


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    # app.run(host='127.0.0.1', port=5000)
    app.run(host='0.0.0.0', port=3000, ssl_context=get_ssl_context())
