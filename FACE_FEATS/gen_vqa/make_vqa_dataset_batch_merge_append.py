import json
import random
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils import *
import shutil


load_dotenv()
client = OpenAI()

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
img_root = '/datasets/newout'
# save_root = '/datasets/newout/vqa_info'
# save_root = '/datasets/newout/vqa_info/miss&makeup'
save_root = '/datasets/newout/vqa_info/add_bordertest&pad&df'
# info_path = '/datasets/newout/vqa_candidates.txt'
# info_path = '/datasets/newout/2025-09-23.txt'
# info_path = '/datasets/newout/vqa_info/miss&makeup/miss&makeup_0.txt'
# info_path = '/datasets/newout/vqa_info/miss&makeup/miss&makeup_1.txt'
info_path = '/datasets/newout/vqa_info/add_bordertest&pad&df/add_bordertest&pad&df.txt'
# img_root = 'D:/zLiveness_data/out'
# save_root = 'D:/zLiveness_data/out/vqa_info'
# info_path = 'D:/zLiveness_data/out/vqa_candidates.txt'

f = open(info_path, 'r+')
info_meta = f.readlines()
f.close()
random.shuffle(info_meta)

# bias_start_idx=3000000
bias_start_idx=3055000

TOTAL_REQUESTS = len(info_meta)     # total images
CHUNK_SIZE = 1                   # requests per batch file (tune for <100MB)
MERGE_SIZE = 28#100
MAX_ACTIVE_JOBS = 1#50                # how many batch jobs to keep running in parallel
CHECK_INTERVAL = 5                 # seconds between polling
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

Full_Description: Describe this image in terms of lighting, resolution, quality, face orientation, gender, age, skin color, expression, face decoration, etc. Answer in one paragraph of 30 words.

Short_description: Describe this image in terms of lighting, resolution, quality, face orientation, gender, age, skin color, expression, face decoration, etc. Answer in one paragraph of 20 words.

Age: 3~9 | 10~19 | 20~29 | 30~39 | 40~49 | 50~59 | 60~69 | 70+

Gender: male | female

Expression: neutral | smiling | frowning | surprised

Pose: frontal | profile | slightly tilted

Ethnicity: White | Black | Indian | East Asian | Southeast Asian | Middle Eastern | Latino

Skin: smoothness | wrinkles | scars | freckles | moles | tattoos | normal

Hair_color: Black | Dark brown | Light brown | Blonde | Red/Auburn | Gray/White | Dyed colors | Multi-color/highlights | Bald

Hat: true/false

Face_decoration: output decoration and location like the following format: makeup/eyes
makeup | jewelry | paint | sticker | tattoo | cultural | medical | digital | normal | None
eyes | lips | cheeks | nose | forehead | None

Occlusion: That means whether there is an obstacle that obscures the face, such as covering the face with the hand or covering the face with hair. hair | glasses | mask | hand | other | None

Quality: a floating-point number in (0.0, 1.0)

Full_Forgery_reasoning: Analyze whether there are any abnormalities compared with real face images and where the abnormalities are.
Let’s think step by step.
First, check whether there are any abnormalities in the facial organs, such as being too smooth, blurred, pixelation, asymmetry or uncoordinated.
Then, check whether there are any fusion artifacts, inconsistent skin color and other issues that often occur at forged boundaries.
Next, check whether the various features of the face have been tampered with or added in terms of attributes, such as color, texture, naturalness, consistency, expressions, etc.
And, check for signs of heavy makeup, such as cosmetic, impersonation, obfuscation, or concealment, rather than normal makeup.
And then, check for the following signs that indicate a possible PAD attack one by one:
- solid black bars/region, clearly separating the actual face region from the surrounding frame along image edges
- unnatural darkened region at the edge of the image, which does not correspond to natural lighting in the scene
- faint dark gradient band near the border, inconsistent with natural lighting
- noticeable dark band present along the edge of the image (indicating the image area is within the physical monitor boundaries)
- black bar at the very top/bottom/left/right edge of the image
- UI content (including application icons and window borders)
- logo/watermark text overlays
- strangely overlapping layers
- lack of depth
- moiré patterns, periodic noise and horizontal banding patterns to repeat at regular intervals
- lighting inconsistencies
- surface reflections and gloss
- compression artifacts, noise patterns
- signs of holding or mounting a real photo
- If the image is blurry, out of focus, obscured, insufficiently illuminated or excessively bright, consider the forgery type to be a PAD attack.
Finally, check whether the entire face is normally integrated with the hair, decorations, and surrounding environment.
In addition, you can provide other useful clues.
Answer in one paragraph of 100 words.

Long_Forgery_reasoning: Answer the main forgery reasons in one paragraph of 50 words. Especially emphasis where abnormalities are. Avoid giving general explanation like "suggest/indicate/hint" and focus the main forgery reasons.

Short_Forgery_reasoning: Answer the main forgery reasons in one paragraph of 30 words. Especially emphasis where abnormalities are. Avoid giving general explanation like "suggest/indicate/hint" and focus the main forgery reasons.

Forgery_hint: Summarize the Forgery_reasoning in 5~10 words. Avoid giving general explanation like "suggest/indicate/hint" and focus the main forgery reasons. If Forgery_type is "None", None.

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
            base64_image = image_to_base64(image)

            custom_id = f"req-{start_id+i}"
            entry = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
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


# request_files = create_request_files()
request_files = get_filepaths(batches_dir)
# exit(0)
# request_files = request_files[:5] ###zzzzzz

# random.shuffle(request_files)

# -------------------------------------------
# Step 2. Orchestrate batch jobs
# -------------------------------------------
active_batches = {}
completed_batches = set()

total_prompt_tokens = 0
total_completion_tokens = 0
total_failed_requests = 0

def get_token_usage(fpath):
    global total_prompt_tokens
    global total_completion_tokens
    global total_failed_requests
    prompt_tokens = 0
    completion_tokens = 0
    failed_requests = 0
    with open(fpath, "r") as f:
        for line in f:
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            status_code = obj["response"]["status_code"]

            if status_code == 200:
                # content = obj["response"]["body"]["choices"][0]["message"]["content"]
                usage = obj["response"]["body"]["usage"]
                prompt_tokens += usage["prompt_tokens"]
                completion_tokens += usage["completion_tokens"]
            else:
                # content = f"Error: {status_code}"
                failed_requests += 1

    print(f"📂   prompt_tokens:{prompt_tokens}, completion_tokens:{completion_tokens}, failed_requests:{failed_requests}")
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens
    total_failed_requests += failed_requests
    print(f"📂   total_prompt_tokens:{total_prompt_tokens}, total_completion_tokens:{total_completion_tokens}, total_failed_requests:{total_failed_requests}")

    return


start_idx = 0
while True:
    # Keep only MAX_ACTIVE_JOBS running at a time
    while len(active_batches) >= MAX_ACTIVE_JOBS:
        print("⏳ Waiting for free slots...")
        time.sleep(CHECK_INTERVAL)

        # Poll active jobs
        finished = []
        for bid, batch_path in list(active_batches.items()):
            try:
                status = client.batches.retrieve(bid)
                if status.status in ["completed", "failed", "cancelled"]:
                    finished.append(bid)
                    print(f"✅ Batch {bid} finished with status: {status.status}")
                    if status.status == "failed" or status.status == "cancelled":
                        if status.errors:
                            print(f"⚠️     Error Message: {str(status.errors)}")
                    elif status.status == "completed":
                        # filename = f"{batches_dir}/{start_id}.jsonl"
                        head, fname = ntpath.split(batch_path)
                        fn_idx = fname.split(".")[0]
                        temp_path = f"res-{fn_idx}.jsonl"
                        output_file = client.files.content(status.output_file_id)
                        with open(temp_path, "wb") as f:
                            f.write(output_file.content)

                        content = ""
                        with open(temp_path, "r") as f:
                            for line in f:
                                obj = json.loads(line)
                                custom_id = obj["custom_id"]
                                status_code = obj["response"]["status_code"]

                                if status_code == 200:
                                    content = obj["response"]["body"]["choices"][0]["message"]["content"]
                                    if "{" in content:
                                        content = content[content.find("{") : content.rfind("}") + 1]
                                    content = content.replace("\n", "#")

                                    if content != "":
                                        # f"req-{start_id}"
                                        fn_idx = custom_id.split("-")[1]
                                        result_path = f"{results_dir}/{fn_idx}.jsonl"
                                        with open(result_path, "w") as f:
                                            f.write(content)
                                        print(f"📂 Saved results to {result_path}")
                                        src_filepath = f"{batches_dir}/{fn_idx}.jsonl"
                                        dest_filepath = f"{completed_dir}/{fn_idx}.jsonl"
                                        shutil.move(src_filepath, dest_filepath)

                        os.remove(temp_path)

                    else:
                        print(f"⚠️ Batch {bid} status is: {status.status}")

                    completed_batches.add(bid)
                    os.remove(batch_path)
            except Exception as e:
                print(f"Error: {e}")

        # Remove finished jobs from active list
        for bid in finished:
            active_batches.pop(bid, None)

    # Upload next file and start new batch
    batches = client.batches.list(limit=100)
    enqueued_tokens = 0
    for batch in batches.data:
        if batch.status not in ['completed', 'failed', 'expired', 'cancelled']:
            enqueued_tokens += 2000*MERGE_SIZE
    print(f"🚀 enqueued_tokens: ~{enqueued_tokens}")

    end_idx = start_idx+MERGE_SIZE
    if end_idx >= len(request_files):
        end_idx = len(request_files)
    if start_idx > len(request_files)-1 or end_idx < start_idx:
        break

    try:
        request_filepath = f"{start_idx}.jsonl"
        with open(request_filepath, "w") as f_merge:
            for index in range(start_idx, end_idx):
                with open(request_files[index], "r") as f_line:
                    for line in f_line:
                        f_merge.write(line)

        start_idx = end_idx

        uploaded = client.files.create(file=open(request_filepath, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        active_batches[batch.id] = request_filepath
        print(f"🚀 Started batch {batch.id} for {request_filepath}")
    except Exception as ee:
        print(f"Error: {ee}")

# -------------------------------------------
# Step 3. Wait for remaining jobs to finish
# -------------------------------------------
while len(active_batches) > 0:
    print("⏳ Waiting for free slots...")
    time.sleep(CHECK_INTERVAL)

    # Poll active jobs
    finished = []
    for bid, batch_path in list(active_batches.items()):
        try:
            status = client.batches.retrieve(bid)
            if status.status in ["completed", "failed", "cancelled"]:
                finished.append(bid)
                print(f"✅ Batch {bid} finished with status: {status.status}")
                if status.status == "failed" or status.status == "cancelled":
                    if status.errors:
                        print(f"⚠️     Error Message: {str(status.errors)}")
                elif status.status == "completed":
                    # filename = f"{batches_dir}/{start_id}.jsonl"
                    head, fname = ntpath.split(batch_path)
                    fn_idx = fname.split(".")[0]
                    temp_path = f"res-{fn_idx}.jsonl"
                    output_file = client.files.content(status.output_file_id)
                    with open(temp_path, "wb") as f:
                        f.write(output_file.content)

                    content = ""
                    with open(temp_path, "r") as f:
                        for line in f:
                            obj = json.loads(line)
                            custom_id = obj["custom_id"]
                            status_code = obj["response"]["status_code"]

                            if status_code == 200:
                                content = obj["response"]["body"]["choices"][0]["message"]["content"]
                                if "{" in content:
                                    content = content[content.find("{"): content.rfind("}") + 1]
                                content = content.replace("\n", "#")

                                if content != "":
                                    # f"req-{start_id}"
                                    fn_idx = custom_id.split("-")[1]
                                    result_path = f"{results_dir}/{fn_idx}.jsonl"
                                    with open(result_path, "w") as f:
                                        f.write(content)
                                    print(f"📂 Saved results to {result_path}")
                                    src_filepath = f"{batches_dir}/{fn_idx}.jsonl"
                                    dest_filepath = f"{completed_dir}/{fn_idx}.jsonl"
                                    shutil.move(src_filepath, dest_filepath)

                    os.remove(temp_path)

                else:
                    print(f"⚠️ Batch {bid} status is: {status.status}")

                completed_batches.add(bid)
                os.remove(batch_path)
        except Exception as e:
            print(f"Error: {e}")

    # Remove finished jobs from active list
    for bid in finished:
        active_batches.pop(bid, None)

print("🎉 All batches finished.")
