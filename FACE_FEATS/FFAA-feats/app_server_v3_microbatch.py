import ntpath
import random
import string
import json
import time
import threading
import queue
from datetime import datetime
from io import BytesIO
from typing import List

import cv2
import numpy as np
import base64
import os
from PIL import Image, UnidentifiedImageError
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.set_flush_denormal(True)
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
FEATURES_PROMPT = None
SAVE_RESULTS_TO_DISK = os.environ.get("SAVE_RESULTS_TO_DISK", "0") == "1"
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

# MLLM is loaded lazily to avoid long Gunicorn worker boot / apparent freezes.
mistral_model = None
mistral_image_processor = None
mistral_tokenizer = None
MODEL_LOAD_ERROR = None
MODEL_LOADED_AT = None
MODEL_LOCK = threading.Lock()

MAX_CONCURRENT_GPU_REQUESTS = int(os.environ.get("MAX_CONCURRENT_GPU_REQUESTS", "1"))
GPU_ACQUIRE_TIMEOUT_SECONDS = float(os.environ.get("GPU_ACQUIRE_TIMEOUT_SECONDS", "30"))
GPU_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_GPU_REQUESTS)

# Micro-batching settings. Keep TRUE_BATCH disabled by default because some
# custom get_llava_answer implementations only accept one PIL image. Even with
# TRUE_BATCH disabled, this worker gives controlled backpressure and prevents
# multiple Flask threads from fighting over CUDA. Enable TRUE_BATCH only after
# verifying your models.get_llava_answer supports a list of images.
MICRO_BATCH_ENABLED = os.environ.get("MICRO_BATCH_ENABLED", "1") == "1"
MICRO_BATCH_MAX_SIZE = int(os.environ.get("MICRO_BATCH_MAX_SIZE", "4"))
MICRO_BATCH_MAX_WAIT_MS = float(os.environ.get("MICRO_BATCH_MAX_WAIT_MS", "40"))
MICRO_BATCH_QUEUE_MAX_SIZE = int(os.environ.get("MICRO_BATCH_QUEUE_MAX_SIZE", "64"))
MICRO_BATCH_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("MICRO_BATCH_REQUEST_TIMEOUT_SECONDS", "180"))
MICRO_BATCH_TRUE_BATCH = os.environ.get("MICRO_BATCH_TRUE_BATCH", "0") == "1"
MICRO_BATCH_FALLBACK_TO_SINGLE = os.environ.get("MICRO_BATCH_FALLBACK_TO_SINGLE", "1") == "1"

INFERENCE_QUEUE = queue.Queue(maxsize=MICRO_BATCH_QUEUE_MAX_SIZE)
BATCHER_STARTED = False
BATCHER_LOCK = threading.Lock()


app = Flask(__name__)
CORS(app) # Enable CORS for all routes and origins
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH_MB', '64')) * 1024 * 1024


@app.before_request
def mark_request_start():
    g.request_start_time = time.time()


# force browser to hold no cache. Otherwise old result might return.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    if hasattr(g, 'request_start_time'):
        response.headers['X-Response-Time-Seconds'] = f"{time.time() - g.request_start_time:.4f}"
    return response


@app.route('/')
def homepage():
    # return render_template('home.html')
    return render_template('home_two.html')


def is_model_loaded():
    return mistral_model is not None and mistral_image_processor is not None and mistral_tokenizer is not None


def ensure_model_loaded():
    """Load the large LLaVA/Mistral model once, on demand. Thread-safe."""
    global mistral_model, mistral_image_processor, mistral_tokenizer
    global MODEL_LOAD_ERROR, MODEL_LOADED_AT

    if is_model_loaded():
        return True

    with MODEL_LOCK:
        if is_model_loaded():
            return True

        start_t = time.time()
        MODEL_LOAD_ERROR = None
        print(f"[startup] Loading LLaVA model from {llava_groups['mistral']} on cuda:{device_id}...", flush=True)
        try:
            mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(
                llava_groups["mistral"],
                device_id,
            )
            MODEL_LOADED_AT = datetime.utcnow().isoformat() + "Z"
            print(f"[startup] Model loaded in {time.time() - start_t:.2f}s", flush=True)
            return True
        except Exception as exc:
            MODEL_LOAD_ERROR = str(exc)
            print(f"[startup] Model load failed: {MODEL_LOAD_ERROR}", flush=True)
            return False


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'success': True,
        'status': 'ok',
        'model_loaded': is_model_loaded(),
        'micro_batch_enabled': MICRO_BATCH_ENABLED,
        'micro_batch_max_size': MICRO_BATCH_MAX_SIZE,
        'micro_batch_max_wait_ms': MICRO_BATCH_MAX_WAIT_MS,
        'micro_batch_queue_size': INFERENCE_QUEUE.qsize(),
        'micro_batch_true_batch': MICRO_BATCH_TRUE_BATCH,
    })


@app.route('/ready', methods=['GET'])
def ready():
    status = 200 if is_model_loaded() else 503
    return jsonify({
        'success': is_model_loaded(),
        'model_loaded': is_model_loaded(),
        'model_loaded_at': MODEL_LOADED_AT,
        'model_load_error': MODEL_LOAD_ERROR,
    }), status


@app.route('/load_model', methods=['POST'])
def load_model_route():
    ok = ensure_model_loaded()
    status = 200 if ok else 500
    return jsonify({
        'success': ok,
        'model_loaded': is_model_loaded(),
        'model_loaded_at': MODEL_LOADED_AT,
        'model_load_error': MODEL_LOAD_ERROR,
    }), status


@app.route('/batch_status', methods=['GET'])
def batch_status():
    return jsonify({
        'success': True,
        'micro_batch_enabled': MICRO_BATCH_ENABLED,
        'micro_batch_max_size': MICRO_BATCH_MAX_SIZE,
        'micro_batch_max_wait_ms': MICRO_BATCH_MAX_WAIT_MS,
        'micro_batch_queue_max_size': MICRO_BATCH_QUEUE_MAX_SIZE,
        'micro_batch_queue_size': INFERENCE_QUEUE.qsize(),
        'micro_batch_request_timeout_seconds': MICRO_BATCH_REQUEST_TIMEOUT_SECONDS,
        'micro_batch_true_batch': MICRO_BATCH_TRUE_BATCH,
        'batcher_started': BATCHER_STARTED,
    })


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
    """Return cached prompt text. Avoids reading prompts.txt on every request."""
    global FEATURES_PROMPT
    if FEATURES_PROMPT is None:
        prompt_list = read_txt_file("playground/prompts.txt")
        FEATURES_PROMPT = prompt_list[0] if prompt_list else DEFAULT_FEATURES_PROMPT
    return FEATURES_PROMPT


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


class InferenceTask:
    def __init__(self, args, image):
        self.args = args
        self.image = image
        self.event = threading.Event()
        self.answer = None
        self.error = None
        self.enqueued_at = time.time()


def start_batcher_once():
    """Start the background micro-batch worker once per Gunicorn worker."""
    global BATCHER_STARTED
    if not MICRO_BATCH_ENABLED:
        return
    if BATCHER_STARTED:
        return
    with BATCHER_LOCK:
        if BATCHER_STARTED:
            return
        worker = threading.Thread(target=micro_batch_worker, name="micro_batch_worker", daemon=True)
        worker.start()
        BATCHER_STARTED = True
        print(
            f"[batcher] started max_size={MICRO_BATCH_MAX_SIZE} "
            f"max_wait_ms={MICRO_BATCH_MAX_WAIT_MS} queue_max={MICRO_BATCH_QUEUE_MAX_SIZE} "
            f"true_batch={MICRO_BATCH_TRUE_BATCH}",
            flush=True,
        )


def enqueue_for_micro_batch(args, image):
    start_batcher_once()
    task = InferenceTask(args, image)
    try:
        INFERENCE_QUEUE.put(task, timeout=GPU_ACQUIRE_TIMEOUT_SECONDS)
    except queue.Full as exc:
        raise TimeoutError("Inference queue is full. Try again shortly.") from exc

    finished = task.event.wait(timeout=MICRO_BATCH_REQUEST_TIMEOUT_SECONDS)
    if not finished:
        raise TimeoutError("Inference timed out while waiting for the micro-batch worker.")
    if task.error is not None:
        raise task.error
    return task.answer


def micro_batch_worker():
    while True:
        first = INFERENCE_QUEUE.get()
        tasks = [first]
        deadline = time.time() + (MICRO_BATCH_MAX_WAIT_MS / 1000.0)

        while len(tasks) < MICRO_BATCH_MAX_SIZE:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                tasks.append(INFERENCE_QUEUE.get(timeout=remaining))
            except queue.Empty:
                break

        process_inference_batch(tasks)
        for _ in tasks:
            INFERENCE_QUEUE.task_done()


def process_inference_batch(tasks):
    if not tasks:
        return

    batch_start = time.time()
    acquired = GPU_SEMAPHORE.acquire(timeout=GPU_ACQUIRE_TIMEOUT_SECONDS)
    if not acquired:
        err = TimeoutError("GPU is busy. Try again shortly.")
        for task in tasks:
            task.error = err
            task.event.set()
        return

    try:
        if not ensure_model_loaded():
            raise RuntimeError(f"Model is not loaded: {MODEL_LOAD_ERROR}")

        if MICRO_BATCH_TRUE_BATCH and len(tasks) > 1:
            try:
                run_true_batch(tasks)
            except Exception as exc:
                if not MICRO_BATCH_FALLBACK_TO_SINGLE:
                    raise
                print(f"[batcher] true batch failed, falling back to single inference: {exc}", flush=True)
                run_sequential_batch(tasks)
        else:
            run_sequential_batch(tasks)

    except Exception as exc:
        for task in tasks:
            task.error = exc
            task.event.set()
    finally:
        GPU_SEMAPHORE.release()
        print(
            f"[batcher] processed batch_size={len(tasks)} "
            f"latency={time.time() - batch_start:.3f}s queue_size={INFERENCE_QUEUE.qsize()}",
            flush=True,
        )


def run_true_batch(tasks):
    """Experimental path: requires get_llava_answer to support list[PIL.Image]."""
    args = tasks[0].args
    images = [task.image for task in tasks]
    with torch.inference_mode():
        answers = get_llava_answer(
            mistral_model,
            mistral_tokenizer,
            mistral_image_processor,
            images,
            args.prompt,
            args.temperature,
            args.top_p,
            args.num_beams,
            args.max_new_tokens,
            args.generate_num['mistral'],
            'v1',
        )

    if not isinstance(answers, (list, tuple)) or len(answers) != len(tasks):
        raise RuntimeError(
            f"True batch returned {type(answers).__name__} with length "
            f"{len(answers) if hasattr(answers, '__len__') else 'unknown'} for {len(tasks)} tasks"
        )

    for task, answer in zip(tasks, answers):
        task.answer = answer
        task.event.set()


def run_sequential_batch(tasks):
    """Safe path: one CUDA owner processes queued requests one by one."""
    for task in tasks:
        args = task.args
        try:
            with torch.inference_mode():
                answers = get_llava_answer(
                    mistral_model,
                    mistral_tokenizer,
                    mistral_image_processor,
                    task.image,
                    args.prompt,
                    args.temperature,
                    args.top_p,
                    args.num_beams,
                    args.max_new_tokens,
                    args.generate_num['mistral'],
                    'v1',
                )
            task.answer = answers[0] if answers else None
        except Exception as exc:
            task.error = exc
        finally:
            task.event.set()


def inference(args, image=None):
    """Run inference through the micro-batch queue.

    image may be passed directly as a PIL image to avoid saving/reloading from disk.
    """
    start_t = time.time()
    print(f'USER: {args.prompt}\n')

    if image is None:
        image = load_image(args.image_path)

    if args.crop == 1:
        image = crop_face(image)
        if image is None:
            print('No face detected')
            return None
    elif args.image_path:
        # Avoid this path in production unless SAVE_RESULTS_TO_DISK is required.
        image.save(TMP_IMG_PATH)

    if MICRO_BATCH_ENABLED:
        answer = enqueue_for_micro_batch(args, image)
    else:
        if not ensure_model_loaded():
            raise RuntimeError(f"Model is not loaded: {MODEL_LOAD_ERROR}")
        acquired = GPU_SEMAPHORE.acquire(timeout=GPU_ACQUIRE_TIMEOUT_SECONDS)
        if not acquired:
            raise TimeoutError("GPU is busy. Try again shortly.")
        try:
            with torch.inference_mode():
                answers = get_llava_answer(
                    mistral_model,
                    mistral_tokenizer,
                    mistral_image_processor,
                    image,
                    args.prompt,
                    args.temperature,
                    args.top_p,
                    args.num_beams,
                    args.max_new_tokens,
                    args.generate_num['mistral'],
                    'v1',
                )
            answer = answers[0] if answers else None
        finally:
            GPU_SEMAPHORE.release()

    print(f"total time : {time.time() - start_t}s")
    return answer

def build_inference_args(image_path=None):
    genarate_num_dict = {
        'mistral': 1,
    }

    return type('Args', (), {
        "device": 0,
        "image_path": image_path,
        "prompt": get_features_prompt(),
        "crop": g_crop,
        "conv_mode": None,
        "llava_groups": llava_groups,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "generate_num": genarate_num_dict,
        "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "512")),
    })()


def decode_image_bytes(image_bytes):
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        pil_image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Failed to decode image") from exc

    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return pil_image, cv_image


def get_face_features_from_images(pil_image, cv_image, image_path=None):
    angle = getRotatedAngle(cv_image)
    if angle > 0:
        res = {'error': 'The image appears to be rotated. Please try again with a straightened image.'}
        print("error: The image appears to be rotated. Please try again with a straightened image.")
        return res

    args = build_inference_args(image_path=image_path)
    try:
        answer = inference(args, image=pil_image)
    except TimeoutError as exc:
        return {'error': str(exc), 'status_code': 503}
    except RuntimeError as exc:
        return {'error': str(exc), 'status_code': 503}

    if answer is None:
        return {'error': 'No face detected'}

    res = {'success': True, 'face_features': answer}

    if SAVE_RESULTS_TO_DISK and image_path:
        head, fname = ntpath.split(image_path)
        file_name_without_ext, ext = os.path.splitext(fname)
        text_path = os.path.join(head, file_name_without_ext + '.txt')
        with open(text_path, "w") as f:
            f.write(answer)

    return res


def get_face_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {'error': 'Failed to decode image'}

    pil_image = load_image(image_path)
    return get_face_features_from_images(pil_image, img, image_path=image_path)


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

    image_bytes = face_image.read()
    try:
        pil_image, cv_image = decode_image_bytes(image_bytes)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    img_path = None
    if SAVE_RESULTS_TO_DISK:
        suffix = os.path.splitext(secure_filename(face_image.filename))[1] or get_image_suffix_from_bytes(image_bytes)
        img_path = save_request_image_bytes(image_bytes, suffix)

    # Run in memory to avoid request-time disk I/O.
    res = get_face_features_from_images(pil_image, cv_image, image_path=img_path)

    if 'error' in res:
        return jsonify(res), res.get('status_code', 200)

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
        pil_image, cv_image = decode_image_bytes(image_bytes)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    img_path = None
    if SAVE_RESULTS_TO_DISK:
        suffix = get_image_suffix_from_bytes(image_bytes)
        img_path = save_request_image_bytes(image_bytes, suffix)

    # Run in memory to avoid request-time disk I/O.
    res = get_face_features_from_images(pil_image, cv_image, image_path=img_path)

    if 'error' in res:
        return jsonify(res), res.get('status_code', 200)

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


if os.environ.get('PRELOAD_MODEL', '0') == '1':
    ensure_model_loaded()


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    # app.run(host='127.0.0.1', port=5000)
    app.run(host='0.0.0.0', port=3001, ssl_context=get_ssl_context())
