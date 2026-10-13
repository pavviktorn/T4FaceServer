import json
import time

import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils import *
import shutil
from multiprocessing import Pool, cpu_count


load_dotenv()
client = OpenAI()

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
# img_root = '/datasets/newout'
# save_root = '/datasets/newout/vqa_info/test'
# info_path = '/datasets/newout/new_mytest.txt'
img_root = '/datasets/newout'
# save_root = '/datasets/newout/vqa_info_full_fmt_1'
# info_path = '/datasets/newout/vqa_info_full_fmt_1/temp.txt'
save_root = '/datasets/newout/vqa_info_2+13+4+3_fmt'
info_path = '/datasets/newout/vqa_info_2+13+4+3_fmt/3D_ATTACKS.txt'

f = open(info_path, 'r+')
info_meta = f.readlines()
f.close()
# random.shuffle(info_meta)

bias_start_idx=1303887#1293165#1282441#1269962#1257483

TOTAL_REQUESTS = len(info_meta)     # total images
CHUNK_SIZE = 1                   # requests per batch file (tune for <100MB)
MAX_ACTIVE_JOBS = 1#50                # how many batch jobs to keep running in parallel
CHECK_INTERVAL = 2                 # seconds between polling
CROP = False
crop_scale = 1.4
# Directory for request and result files
batches_dir = os.path.join(save_root, "batches")
os.makedirs(batches_dir, exist_ok=True)
results_dir = os.path.join(save_root, "results")
os.makedirs(results_dir, exist_ok=True)
completed_dir = os.path.join(save_root, "completed")
os.makedirs(completed_dir, exist_ok=True)
splited_info_dir = os.path.join(save_root, "splited_info")
os.makedirs(splited_info_dir, exist_ok=True)

prompt_list = read_txt_file("prompts.txt")
system_prompt = """You are an AI master specialized in face forgery analysis. You’ll be provided with a human face image.
You need to think step by step to determine whether the face is real or fake.
Below are the four types of face forgery:
1. Entire face synthesis: Generate an entire non-existent face through some technologies.
2. Identity exchange: Use existing videos or images to replace the source character's face with the target character's face, thereby changing the source character's identity.
3. Facial attribute manipulation: This involves altering facial expressions or features without changing the identity, such as skin color, hair color, gender, age, decorations, etc.
4. PAD attacks: This involves PAD attack types such as printed photo (matte and glossy), phone screen, tablet screen, high-resolution monitor, replay video, 3D mask, or cutout with eye/mouth holes, etc.
5. Makeup manipulation: This means manipulating the face with heavy makeup, such as cosmetic, impersonation, obfuscation, or concealment, rather than normal makeup.

Below are the format and requirements of answers:

Full_Description: Describe this image in terms of gender, age, skin color, expression, face orientation, lighting, resolution, quality, face decoration, etc. Answer in one paragraph of 30 words.

Short_description: Describe this image in terms of gender, age, skin color, expression, face orientation, lighting, resolution, quality, face decoration, etc. Answer in exactly 10 words.

Age: 3~9 | 10~19 | 20~29 | 30~39 | 40~49 | 50~59 | 60~69 | 70+

Gender: male | female

Expression: neutral | happiness | surprise | sadness | fright | disgust | contempt | anger

Pose: frontal | profile | slightly profile | slightly tilted | tilted | upward | slightly upward | downward | slightly  downward

Ethnicity: White | Black | Indian | East Asian | Southeast Asian | Middle Eastern | Latino

Skin: smoothness | wrinkles | scars | freckles | moles | tattoos | normal

Hair_color: Black | Dark brown | Light brown | Blonde | Red/Auburn | Gray/White | Dyed colors | Multi-color/highlights | Bald

Hat: yes | no

Face_decorations: output decorations and locations like the following format: makeup/eyes
makeup | jewelry | paint | sticker | tattoo | cultural | medical | digital | other | None
eyes | ears | lips | cheeks | nose | forehead | None

Occlusions: That means whether there are occlusions such as masks, sunglasses, hat, hair covering the face, hands, or other objects blocking facial features. hair | glasses | sunglasses | mask | hand | clothing | hat | other objects | None

Focus: That means the clarity of fine facial details such as eyes, eyelashes, eyebrows, skin texture, and edges of facial features. focused | slightly blurry | blurry | slightly noisy | noisy | other
 
Resolution: That means if the face has sufficient pixel detail for accurate recognition or analysis considering face size in the image and pixel density. Excellent | Good | Medium | Low | Poor | other

Quality: Return a single floating-point number in the range (0.00, 1.00) representing the overall face image quality for biometric usability based on the following criteria: Resolution & Face Size (The face must be sufficiently large and detailed, with clear pixel density suitable for facial analysis.), Sharpness & Focus (Facial features (especially eyes, nose, mouth, and contours) must be sharp and in focus, with no motion blur or defocus.), Occlusion (Penalize occlusions such as masks, sunglasses, hat, hair covering the face, hands, or other objects blocking facial features. The penalty increases with the number of occlusions and the area covered.), Noise & Compression Artifacts (Penalize visible noise, pixelation, or compression artifacts.), Lighting & Exposure (Penalize low-light, uneven lighting, shadows, overexposure, or excessive brightness that reduce facial detail.), Pose & Alignment (The face should be frontal and well-centered, with minimal yaw, pitch, and roll.), Expression Neutrality (Neutral expressions are preferred; extreme expressions reduce quality.), Background & Contrast (The face should be clearly distinguishable from the background with adequate contrast.), Eye Visibility & Distance (Both eyes must be visible, open, and properly spaced for biometric analysis.) Return only the numeric quality score.

Full_Forgery_reasoning: Analyze whether there are any abnormalities compared with real face images and where the abnormalities are.
Let’s think step by step.
First, check for the following signs that indicate a possible PAD attack one by one:
- solid black bars/region, clearly separating the actual face region from the surrounding frame along image edges
- unnatural darkened region at the edge of the image, which does not correspond to natural lighting in the scene
- faint dark gradient band near the border, inconsistent with natural lighting
- noticeable dark band present along the edge of the image (indicating the image area is within the physical monitor boundaries)
- black bar at the very top/bottom/left/right edge of the image
- UI content (including application icons and window borders)
- flat depth, reduced depth cues, lack of real 3D facial depth
- uniform illumination
- lighting inconsistencies
- absence of real-world light interaction
- surface reflections and gloss
- screen-induced smoothing
- re-capture blur
- logo/watermark text overlays
- strangely overlapping layers
- compression artifacts, noise patterns
- moiré patterns, periodic noise and horizontal banding patterns to repeat at regular intervals
- signs of holding or mounting a real photo
- If the image is blurry, out of focus, obscured, insufficiently illuminated or excessively bright, consider the forgery type to be a PAD attack.
- Slight blur, flat lighting, and lack of real-world depth variation indicate PAD attacks.

Then, check whether there are any abnormalities in the facial organs, such as being too smooth, blurred, pixelation, asymmetry or uncoordinated. 
Then, check whether there are any fusion artifacts, inconsistent skin color and other issues that often occur at forged boundaries.
Next, check whether the various features of the face have been tampered with or added in terms of attributes, such as color, texture, naturalness, consistency, expressions, etc.
And, check whether the entire face is normally integrated with the hair, decorations, and surrounding environment.
Finally, check for signs of heavy makeup, such as cosmetic, impersonation, obfuscation, or concealment, rather than normal makeup.
In addition, you can provide other useful clues.

Full_Forgery_reasoning: Answer in one paragraph of 50 words.

Long_Forgery_reasoning: Answer the main forgery reasons in one paragraph of 30 words. Especially emphasis where abnormalities are. Avoid giving general explanation like "suggest/indicate/hint" and focus the main forgery reasons.

Short_Forgery_reasoning: Answer the main forgery reasons in one paragraph of 20 words. Especially emphasis where abnormalities are. Avoid giving general explanation like "suggest/indicate/hint" and focus the main forgery reasons.

Forgery_hint: Summarize the forgery reasons in exactly 7 words. Avoid giving general explanation like "suggest/indicate/hint" and focus the main forgery reasons.

Result: real | fake
Score: a floating-point number in (0.0, 1.0)
Forgery_type: Entire face synthesis | Identity exchange | Facial attribute manipulation | PAD attacks | Makeup manipulation | None
If forgery type is not "None", Analysis result is "fake".

Below are requirements for answers in the conversation:
1. Strictly follow the json format and requirements of answers, and do not say anything else irrelevant.
2. Avoid giving general explanations. Focus on and analyze the specific image. Then make a reasonable and specific explanation.
3. Do not bold the output content.


{}
{}
"""


# -------------------------------------------
# Step 1. Generate JSONL request files
# -------------------------------------------
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
        if source_path != '':
            head, fname = ntpath.split(source_path)
            pp = fname.split('.')
            source_path = head + f'/{pp[0]}.jpg'
            if not os.path.exists(source_path):
                file_name_without_ext, ext = os.path.splitext(fname)
                source_path = head + f'/{file_name_without_ext}.jpg'
                if not os.path.exists(source_path):
                    source_path = ""
    else: # "v"
        pp = org_path.split(':')
        path = (img_root + pp[0]).replace('\\', '/')
        frame = int(pp[1])
        path = get_frame(path, frame)
        if source_path != '':
            source_path = get_frame(source_path, frame)
        else:
            source_path = ""

    be = elem[5].split(':')
    bbox[0], bbox[1], bbox[2], bbox[3] = int(be[0]), int(be[1]), int(be[2]), int(be[3]),
    le = elem[6].split(':')
    for i in range(landmark.shape[0]):
        xy = le[i].split('_')
        landmark[i][0], landmark[i][1] = xy[0], xy[1]

    return path, {'labels': class_label, 'vf': video_or_frame, 'frame': frame, 'landmark': landmark, 'bbox': bbox,
                        'source_path': source_path,
                        'quality': quality}

def create_request_file(start_id, info_lines):
    fpath = f"{splited_info_dir}/{start_id}.txt"
    f_a = open(fpath, 'w')
    f_a.writelines(info_lines)
    f_a.close()

    fpath = f"{batches_dir}/{start_id}.jsonl"
    with open(fpath, "w") as f:
        for i, info in enumerate(info_lines):

            path, label_meta = parse_info(info)

            image = cv2.imread(path)
            if label_meta["source_path"] != "":
                src_image = cv2.imread(label_meta["source_path"])
            if CROP:
                # face_bbox, kps = get_face(image)
                face_bbox = label_meta['bbox']
                # if len(face_bbox) == 0:
                #     print('No faces')
                #     continue
                image = get_cropped(image, face_bbox, scale=crop_scale)
                # cv2.imwrite("crop.jpg", image)
                if label_meta["source_path"] != "":
                    src_image = get_cropped(image, face_bbox, scale=crop_scale)

            forgery_specific_prompt = get_forgery_specific_prompt2(path, label_meta["source_path"], label_meta["labels"])
            question_prompt = random.choice(prompt_list)
            final_prompt = system_prompt.format(forgery_specific_prompt, question_prompt)
            final_prompt = final_prompt + "\n\n" + final_prompt #zzzzz
            base64_image = image_to_base64(image)

            custom_id = f"req-{start_id+i}"
            entry = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.2",
                    "messages": [
                        {
                           "role": "user",
                           "content": [
                               {"type": "text",
                                "text": final_prompt},
                               {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                           ],
                       },
                   ],
                   "temperature": 0.0
                }
            }

            f.write(json.dumps(entry) + "\n")

    return fpath


# Split into chunks
def create_request_files():
    request_files = []
    # for i in range(0, len(image_urls), CHUNK_SIZE):
    start_idx = 0
    while True:
        end_idx = start_idx+CHUNK_SIZE
        if end_idx >= len(info_meta):
            end_idx = len(info_meta) - 1
        if start_idx >= len(info_meta)-1 or end_idx <= start_idx:
            break
        chunk = info_meta[start_idx:end_idx]
        filename = create_request_file(start_idx+bias_start_idx, chunk)
        request_files.append(filename)
        print(filename)
        start_idx = end_idx

    print(f"✅ Created {len(request_files)} batch request files.")

    return request_files


# Function to process a single request from the JSONL file with customizable retry logic
def process_request(request_body, retries=10, max_retry_time=60, retry_delay=10):
    """
    Process the request and ensure you get a response, with retries on failure.
    - retries: maximum number of retry attempts
    - max_retry_time: maximum time in seconds to keep retrying (total duration of retries)
    - retry_delay: delay between retries (in seconds, can be fixed or linear)
    """
    attempt = 0
    start_time = time.time()

    while attempt < retries:
        try:
            # Send the request to OpenAI GPT-4o
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=request_body["body"]["messages"],
                temperature=request_body["body"].get("temperature", 0.0),
                # max_tokens=2000  # Set a reasonable token limit
            )
            # Return the response or process as needed
            return response

        except openai.RateLimitError as e:
            # Rate limit exceeded, retry with exponential backoff
            print(f"Rate limit exceeded for {request_body['custom_id']}. Retrying... {e}")

        # except openai.Timeout as e:
        # #     # Timeout error, retry
        #     print(f"Request timeout for {request_body['custom_id']}. Retrying... {e}")

        except openai.APIConnectionError as e:
            # API connection error, retry
            print(f"API connection error for {request_body['custom_id']}. Retrying...{e}")

        except Exception as e:
            # Generic exception, retry
            print(f"Error processing request {request_body['custom_id']}: {e}")

        # Calculate how much time has passed since starting the retries
        # elapsed_time = time.time() - start_time
        # if elapsed_time >= max_retry_time:
        #     print(f"Max retry time exceeded for {request_body['custom_id']}. Stopping retries.")
        #     break

        # Delay before retrying
        print(f"Retrying after {retry_delay} seconds...")
        time.sleep(retry_delay)  # Linear backoff or fixed delay
        attempt += 1

    # If all attempts fail, return None or handle the error accordingly
    print(f"Failed to process request {request_body['custom_id']} after {retries} attempts.")
    return None


# Read the JSONL file and process each request synchronously
def process_jsonl_file(file_path):

    with open(file_path, "r") as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                request_body = json.loads(line)  # Parse JSON object from line
                # print(f"Processing request {request_body['custom_id']}...")

                start_t = time.time()

                # Process the request and get the response
                response = process_request(request_body)
                if response:
                    # Handle the response (e.g., save to a file, print, etc.)
                    token_num = response.usage.total_tokens
                    res_text = response.choices[0].message.content
                    if "{" in res_text:
                        res_text = res_text[res_text.find("{"): res_text.rfind("}") + 1]
                    res_text = res_text.replace("\n", "#")

                    if not "sorry" in res_text:
                        output_path = file_path.replace(batches_dir, results_dir)
                        f_a = open(output_path, 'w')
                        f_a.write(res_text)
                        f_a.close()

                        dest_filepath = file_path.replace(batches_dir, completed_dir)
                        shutil.move(file_path, dest_filepath)

                        end_t = time.time()
                        print(f"{request_body['custom_id']}: token: {token_num}, {end_t-start_t}, {res_text}")

                        return True

                    end_t = time.time()
                    print(f"{request_body['custom_id']}: token: {token_num}, {end_t-start_t}, {res_text}")
                else:
                    print(f"Failed to process {request_body['custom_id']}")

                # Optional: Add a delay to avoid rate-limiting
                # time.sleep(1)  # Sleep for 1 second (adjust as needed)

    return False


def process_jsonl_worker(file_path):
    """
    This function will be executed by each worker in the pool.
    It processes a single JSONL file.
    """
    # print(f"Processing file: {file_path}")
    process_jsonl_file(file_path)
    # print(f"Finished processing file: {file_path}")


def process_in_parallel(request_files, num_workers=cpu_count()):
    """
    Process the request files in parallel using multiprocessing.
    :param request_files: List of JSONL file paths to process
    :param num_workers: Number of workers (processes) to use
    """
    with Pool(num_workers) as pool:
        pool.map(process_jsonl_worker, request_files)


if __name__ == '__main__':

    # request_files = create_request_files()
    request_files = get_filepaths(batches_dir)
    random.shuffle(request_files)
    # request_files = request_files[:20]
    print(len(request_files))

    # start_idx = 0
    # end_idx = len(request_files)
    # # end_idx = 396000 * 3
    
    # for idx in range(start_idx, end_idx):
    #     batch_path = request_files[idx]
    #     process_jsonl_file(batch_path)


    num_workers = 20  # Adjust based on your system's resources

    # Process the files in parallel
    process_in_parallel(request_files, num_workers)

