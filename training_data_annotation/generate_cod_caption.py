import time
import base64
import json
from os.path import join, exists
from PIL import Image
from openai import OpenAI
import argparse
import io
import os
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import random
from eval_utils import read_json, save_jsonl
import copy

MAX_TRY_TIMES = 2
sleep_times = [10, 10, 10, 10, 10]


def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# Define the function to generate the cod response
def generate_cod_response(question, image, caption):
    # Define the initial messages
    messages = [
        {
            "role": "system",
            "content": """You are an expert AI assistant tasked with analyzing an image and generating a detailed, step-by-step description. You are provided with an original description as a reference. Your goal is to ensure accuracy, clarity, and logical progression in your response. Follow these guidelines:

### Guidelines:
1. **Ensure Comprehensive Coverage**: Identify and include all relevant details visible in the image. Avoid unnecessary repetition or irrelevant information.  
2. **Avoid Adding Imaginary Details**: Base your reasoning strictly on what is visible in the image or provided in the description. Do not include fabricated or unverifiable details.  
3. **Incorporate Relevant Context**: Add factual, relevant context to enhance understanding where appropriate, but ensure it aligns strictly with the visible or provided content.  
4. **Prevent Inaccuracies**: Stick to the given data. Avoid assumptions or deviations from the available evidence.  

---

### Step-by-Step Process:

**Step 1**: Extract salient content by identifying the key elements that define the image or document.
*Example*:
The image is a monochrome photocopy of a document that appears to be a page of meeting or project notes. It contains both typed and handwritten text, with a focus on tasks and progress updates related to paper-related issues. The document includes a reference number at the bottom and a source URL.

**Step 2**: Analyze detailed information, focusing on instance-level attributes such as low-level and fine-grained details, such as specific text, layout.
*Example*:
The document lists several tasks, such as checking with "KC" on the possibility of putting bands "long-ways," which is marked as "In progress." Other tasks include checking on "shrinking" paper, which is also "In progress," and checking the commercial viability of banded papers, marked as "Okay." There are handwritten notes and checks next to some points, indicating their status.

**Step 3**: Consider relational-level attributes, analyzing interactions between elements and their spatial organization.
*Example*:
The tasks are organized in a list format, with some items having associated handwritten notes that indicate completion or ongoing status. The name "Jimmy Wu" is associated with an action item regarding a DC work request with KC banded papers, awaiting approval for banded additives. The document also mentions running "GPC KS and KOOL KS on RIP-4 (LCC)" and notes that KC is running "cross-hatch" papers.

**Step 4**: Examine marginal or peripheral content to ensure no important information is missed.
*Example*:
The document specifies that the next meeting is scheduled for Monday, February 7, at 9:00 a.m. in the International Conference Room. The reference number "584100571" is located at the bottom of the page, and the source URL is included at the bottom.

**Step 5**: Organize all observations into a detailed, cohesive description.
*Example*:
The image is a monochrome photocopy of a document that appears to be a page of meeting or project notes, containing both typed and handwritten text. The document lists several tasks related to paper-related issues, such as checking with "KC" on the possibility of putting bands "long-ways," which is marked as "In progress," and checking the commercial viability of banded papers, marked as "Okay." Handwritten notes and checks next to some points indicate their status. The name "Jimmy Wu" is associated with an action item regarding a DC work request with KC banded papers, awaiting approval for banded additives. Other items include running "GPC KS and KOOL KS on RIP-4 (LCC)" and KC running "cross-hatch" papers. The next meeting is scheduled for Monday, February 7, at 9:00 a.m. in the International Conference Room. The document is marked with a reference number "584100571" at the bottom, and a source URL is included.

**Important Notes**:

---

### Notes:
- **Steps 1–4**: Write concise observations in one or two sentences each.  
- **Step 5**: Summarize all observations into a detailed paragraph or two, as descriptive as necessary."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Question: {question}\n\nOriginal description: {caption}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Thank you! I will now think step-by-step, starting by extracting the salient content from the image."
        }
    ]
    # steps = []
    # step_count = 1

    # while True:
    step_data = make_api_call(messages, 2048)
    print("step_data: ", step_data)

    return step_data


def make_api_call(messages, max_tokens, is_final_answer=False, custom_client=None):
    if custom_client != None:
        client = custom_client
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", None)
        )
        client.base_url = ''
    for attempt in range(3):
        try:
            # if is_final_answer:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content
            # else:
            #     response = client.chat.completions.create(
            #         model="gpt-4o",
            #         messages=messages,
            #         max_tokens=max_tokens,
            #         temperature=0.2,
            #         response_format={"type": "json_object"}
            #     )
            #     return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                return {"content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                        "next_action": "final_answer"}
            time.sleep(1)


def make_one_data(idx, end_id, samples):
    tic = time.time()
    try_time = 0
    success = False
    while try_time < MAX_TRY_TIMES:
        try:
            data = samples[idx]
            image_file = os.path.join(args.image_dir, data['image'])
            problem_decoded_image = Image.open(image_file)
            base64_image = encode_image_to_base64(problem_decoded_image)
            question = data['conversations'][0]['value']
            print("question: ", question)
            caption = data['caption']
            print("original caption: ", caption)

            # generate true cod
            steps = generate_cod_response(question, base64_image, caption)
            print("revised_caption: ", steps)
            # save sft data
            sft_data = []
            data_ = copy.deepcopy(data)
            data_["conversations"][0]["value"] += " Describe the image step by step."
            data_["cod_caption"] = steps
            sft_data.append(data_)
            save_jsonl(join(args.output_cod_dir, f"{idx}_caption.jsonl"), sft_data)
            sft_dd_data = []
            dd_data = copy.deepcopy(data)
            dd_data["dd_caption"] = steps.split("Organize all observations into a detailed, cohesive description.\n\n")[-1]
            sft_dd_data.append(dd_data)
            save_jsonl(join(args.output_dd_dir, f"{idx}_caption.jsonl"), sft_dd_data)
            success = True
            break
        except Exception as e:
            print(f"index {idx}, failed because {e}")
            try_time += 1
            time.sleep(sleep_times[try_time])
            print("retry {}/{}".format(try_time, MAX_TRY_TIMES))
    toc = time.time()
    if success:
        print("[{}]/[{}] Done in {:.2f} seconds".format(idx, end_id, toc - tic))
    else:
        print("[{}]/[{}] Failed. {}".format(idx, end_id, samples[idx]))


def run_parallel(args):
    test_data = read_json(args.input_file)
    test_data = test_data[:35000]
    random.seed(42)
    # test_data = random.sample(test_data, 100)
    if not exists(args.output_cod_dir):
        os.makedirs(args.output_cod_dir)
    num_workers = multiprocessing.cpu_count()
    process_func = partial(make_one_data,
                           end_id=len(test_data),
                           samples=test_data,
                           )
    with ThreadPoolExecutor(num_workers) as exe:
        exe.map(process_func, list(range(0, len(test_data))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="annotations.json")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--output_cod_dir", type=str,
                        default="output_cod_dir")
    parser.add_argument("--output_dd_dir", type=str,
                    default="output_dd_dir")
    args = parser.parse_args()
    run_parallel(args)