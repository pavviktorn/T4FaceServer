from openai import OpenAI
from dotenv import load_dotenv
from utils import *


load_dotenv()
client = OpenAI()


system_prompt = """You are an AI master specialized in face forgery analysis. You’ll be provided with a human face image.
You need to think step by step to determine whether the face is real or fake.
Below are the four types of face forgery:
1. Entire face synthesis: Generate an entire non-existent face through some technologies.
2. Identity exchange: Use existing videos or images to replace the source character's face with the target character's face, thereby changing the source character's identity.
3. Facial attribute manipulation: This involves altering facial expressions or features without changing the identity, such as skin color, hair color, gender, age, decorations, etc.
4. PAD attacks: This involves PAD attack types such as printed photo (matte and glossy), phone screen, tablet screen, high-resolution monitor, replay video, 3D mask, or cutout with eye/mouth holes, etc.
Below are the format and requirements of answers:
Image description: Describe this image in terms of lighting, resolution, face orientation, gender, age, skin color, expression, face decoration, etc. Answer in one paragraph of 50 to 100 words;
Forgery reasoning: Analyze whether there are any abnormalities compared with real face images and where the abnormalities are.
Let’s think step by step.
First, check whether there are any abnormalities in the facial organs, such as being too smooth, blurred, pixelation, asymmetry or uncoordinated.
Then, check whether there are any fusion artifacts, inconsistent skin color and other issues that often occur at forged boundaries.
Next, check whether the various features of the face have been tampered with or added in terms of attributes, such as color, texture, naturalness, consistency, expressions, etc.
And then, check whether there are any lighting inconsistencies, lack of depth, strange overlapping layers, compression artifacts, noise patterns, surface reflections and gloss, and signs of holding or mounting a real photo.
Finally, check whether the entire face is normally integrated with the hair, decorations, and surrounding environment.
In addition, you can provide other useful clues. Answer in one paragraph of 100 to 150 words;
Analysis result: real | fake
Probability: a floating-point number in (0.0, 1.0)
Forgery type: Entire face synthesis | Identity exchange | Facial attribute manipulation | None
Below are requirements for answers in the conversation:
1. Strictly follow the format and requirements of answers, and do not say anything else irrelevant.
2. Avoid giving general explanations. Focus on and analyze the specific image. Then make a reasonable and specific explanation.
3. Do not bold the output content.

{}
{}
"""


def openai_request_image(img, final_prompt):
    base64_image = image_to_base64(img)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
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
        # max_tokens=200,
        temperature=0.0
    )
    token_num = response.usage.total_tokens
    # print("\n\nUsed token: ", token_num)

    res = response.choices[0].message.content
    # print(res)

    return res, token_num


def main():

    img_path = '/datasets/test_vqa'
    crop_flag = False

    full_file_paths = get_filepaths(img_path)

    prompt_list = read_txt_file("prompts.txt")

    for src_path in full_file_paths:
        head, fname = ntpath.split(src_path)
        if os.path.isdir(src_path) is False and (fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg') or fname.endswith('.jfif')):
            src_path = src_path.replace("\\", "/")
            image = cv2.imread(src_path)
            if crop_flag:
                face_bbox, kps = get_face(image)
                if len(face_bbox) == 0:
                    print('No faces')
                    continue
                crop_scale = 1.4
                image = get_cropped(image, face_bbox, scale=crop_scale)
                cv2.imwrite("crop.jpg", image)

            forgery_specific_prompt = get_forgery_specific_prompt(src_path)
            question_prompt = random.choice(prompt_list)
            final_prompt = system_prompt.format(forgery_specific_prompt, question_prompt)

            resized = resize_image_keep_aspect_opencv(image)

            cv2.imshow('Image1', resized)

            start_t = time.time()

            res, token_num = openai_request_image(image, final_prompt)

            end_t = time.time()

            print("\n\nUsed token: ", token_num)
            print(res)
            print(end_t-start_t)

            # Wait for 3 seconds before closing the display window
            key = cv2.waitKey(1) & 0xFF

            # Check if the ESC key was pressed
            if key == 27:
                # Close all windows and break the loop
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main()


