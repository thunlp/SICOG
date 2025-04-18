import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import get_context
from tqdm import tqdm
from accelerate import Accelerator
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

import warnings
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.slice_process import slice_image_minicpm, split_image, resize_image_keep_ratio
from llava.utils import disable_torch_init
from PIL import Image
import requests
import copy
import torch

import sys
import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")


def clean_llm_output(text):
    """Cleans and processes the model output text."""
    return text.strip()  # Add proper cleaning logic if necessary


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = max(1, len(lst) // n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from n chunks."""
    chunks = split_list(lst, n)
    return chunks[k]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument('--annotation-file', type=str, required=True, help="Path to the input annotation JSON file.")
    parser.add_argument('--result-file', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--image-folder', type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument('--num-chunks', type=int, default=1, help="Number of chunks to split the dataset into.")
    parser.add_argument('--chunk-idx', type=int, default=0, help="Index of the dataset chunk to process.")
    parser.add_argument('--temperature', type=float, default=1, help="Sampling temperature for text generation.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Sampling temperature for text generation.")
    parser.add_argument('--num-beams', type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument('--max-new-tokens', type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument('--conv-mode', type=str, default="qwen_1_5", help="Conversation mode.")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for data loading.")
    parser.add_argument('--default-prompt', type=str, default="cot-default", help="Use default prompts from conversations.")
    return parser.parse_args()


class CustomDataset(Dataset):
    """Custom dataset for processing image-text pairs."""

    def __init__(self, data, image_folder, tokenizer, image_processor, model_config, default_prompt=False):
        self.data = data
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.default_prompt = default_prompt

    def __getitem__(self, index):
        line = self.data[index]
        image_file = line["image"]
        if self.default_prompt == "cot-default":
            qs = "Please generate a detailed caption of this image. Explain your description step-by-step."
            # qs = "Please generate a detailed caption of this image. Please be as descriptive as possible."
        elif self.default_prompt == "cot-conv":
            qs = line["conversations"][0]["value"].replace('<image>', '').strip() + " Explain your description step-by-step."
        elif self.default_prompt == "default":
            qs = "Please generate a detailed caption of this image. Please be as descriptive as possible."
        else:
            qs = line["conversations"][0]["value"].replace('<image>', '').strip()
        # + "\nInstead of describing the imaginary content, only describing the content one can determine confidently from the image."
        # qs = 'Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Minimize aesthetic descriptions as much as possible.'

        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        ## recaption
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        # "Please generate a detailed caption of this image. Please be as descriptive as possible."
        # print("current prompt: ", qs)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print("prompt: ", prompt)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image = resize_image_keep_ratio(image, max_size=1024)

        source_image, patches, best_grid, ind_tokens = slice_image_minicpm(
            image, max_slice_nums=7, scale_resolution=336, patch_size=14, never_split=False
        )

        source_tensors = self.image_processor.preprocess(
            source_image, do_resize=False, do_center_crop=False, do_rescale=True, do_normalize=True, return_tensors='pt'
        )["pixel_values"]
        patch_tensors = self.image_processor.preprocess(
            patches, do_resize=False, do_center_crop=False, do_rescale=True, do_normalize=True, return_tensors='pt'
        )["pixel_values"] if best_grid else None

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, source_tensors[0], image.size, patch_tensors, ind_tokens

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids, image_tensors, image_sizes, patch_images, ind_tokens = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)

    return {
        "input_ids": input_ids,
        "image_tensors": image_tensors,
        "image_sizes": image_sizes,
        "patch_images": patch_images,
        "ind_tokens": ind_tokens,
    }


def recap(args):

    llava_model_args = {
        "attn_implementation": "flash_attention_2"
    }
    model_name = "llava_qwen"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, model_name, device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

    accelerator = Accelerator(mixed_precision="no")  # Support bf16 precision
    disable_torch_init()

    # model_path = os.path.expanduser(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path, None, get_model_name_from_path(model_path), _args=args
    # )

    with open(os.path.expanduser(args.annotation_file), "r") as file:
        data = json.load(file)
    # data = data[:10]
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    dataset = CustomDataset(data, args.image_folder, tokenizer, image_processor, model.config, args.default_prompt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    model = accelerator.prepare(model)
    model.eval()

    all_results = []
    final_outputs = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        # for key, item in batch.items():
        #     print("key: ", key)
        #     print("item: ", item)
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
        # print("input_ids device:", input_ids.device)
        # print("input_ids, size: ", input_ids.size())
        image_tensors = batch["image_tensors"].to(accelerator.device, non_blocking=True)
        # print("image_tensors device:", image_tensors.device)
        # print("image_tensors.size()", image_tensors.size())
        if batch["patch_images"][0] is None:
            continue
        patch_images = batch["patch_images"][0].unsqueeze(0).to(accelerator.device, non_blocking=True)
        # print("patch_images device:", patch_images.device)
        # print(" patch_images size(): ", patch_images.size()) #patch_images size():  torch.Size([1, 6, 3, 308, 350])
        
        ind_tokens = [batch["ind_tokens"][0]]
        
        # # .to(accelerator.device, non_blocking=True)
        # print("ind_tokens", ind_tokens)
        # image_sizes = [batch["image_sizes"][0]]
        # # .to(accelerator.device, non_blocking=True)
        # print("image_sizes: ", image_sizes)
        # Convert `ind_tokens` to a tensor and move to the same device as `accelerator`
        ind_tokens = torch.tensor(batch["ind_tokens"]).to(accelerator.device, non_blocking=True)
        # print("ind_tokens device:", ind_tokens.device)

        # Convert `image_sizes` to a tensor and move to the same device as `accelerator`
        image_sizes = torch.tensor(batch["image_sizes"]).to(accelerator.device, non_blocking=True)
        # print("image_sizes device:", image_sizes.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                patch_images=patch_images,
                ind_tokens=ind_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print("outputs: ", outputs)
        outputs = [clean_llm_output(output) for output in outputs]
        final_outputs.extend(outputs)

    for result, line in zip(final_outputs, data):
        line = copy.deepcopy(line)
        line["generated_caption"] = result
        all_results.append(line)

    if accelerator.is_local_main_process:
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    recap(args)