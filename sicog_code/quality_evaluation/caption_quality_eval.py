import os
import json
import random
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
from eval_longbench_qa import drqa_metric_max_over_ground_truths, substring_exact_match_score
import copy
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
import re

def encode_sentences(dataset, encoder_model_name_or_path, accelerator):
    """Encode sentences using the specified embedding model with Accelerate."""
    sentences = sum([item["predictions"] for item in dataset], [])

    accelerator.print(f"Encoding {len(sentences)} sentences...")
    torch.cuda.empty_cache()

    if "jinaai/jina-embeddings-v3" in encoder_model_name_or_path:
        model = AutoModel.from_pretrained(encoder_model_name_or_path, trust_remote_code=True).to(accelerator.device)
        sentence_embeddings = []
        for sentence in tqdm(sentences, desc="Encoding sentences"):
            sentence_embeddings.append(model.encode([sentence], task="text-matching", max_length=8192).tolist()[0])

    elif "nvidia/NV-Embed-v2" in encoder_model_name_or_path:
        model = SentenceTransformer(encoder_model_name_or_path, trust_remote_code=True).to(accelerator.device)
        model.max_seq_length = 1024
        model.tokenizer.padding_side = "right"

        def add_eos(input_examples):
            return [example + model.tokenizer.eos_token for example in input_examples]

        sentence_embeddings = model.encode(
            list(tqdm(add_eos(sentences), desc="Encoding sentences")),
            batch_size=4,
            normalize_embeddings=True
        )

    elif "mixedbread-ai/mxbai-embed-large-v1" in encoder_model_name_or_path:
        model = SentenceTransformer(encoder_model_name_or_path).to(accelerator.device)
        sentence_embeddings = model.encode(
            list(tqdm(sentences, desc="Encoding sentences")),
            convert_to_tensor=True
        )
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).tolist()

    elif "openbmb/MiniCPM-Embedding" in encoder_model_name_or_path:
        model = SentenceTransformer(
            encoder_model_name_or_path,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16}
        )
        model = accelerator.prepare(model)
        sentence_embeddings = model.encode(
            list(tqdm(sentences, desc="Encoding sentences"))
        ).tolist()

    return sentence_embeddings


def calculate_mbr_scores(args, dataset, accelerator):
    """Calculate Minimum Bayes Risk scores based on sentence embeddings."""
    sentence_embeddings = encode_sentences(dataset, args.encoder_model_name_or_path, accelerator)

    num_predictions = len(dataset[0]["predictions"])
    assert len(sentence_embeddings) == len(dataset) * num_predictions

    def calculate_scores(item, idx):
        embeddings = np.array(sentence_embeddings[idx * num_predictions: (idx + 1) * num_predictions])
        similarity_matrix = embeddings @ embeddings.T
        item["mbr_scores_embedding"] = similarity_matrix.mean(axis=1).tolist()
        item["prediction"] = item["predictions"][np.argmax(item["mbr_scores_embedding"])]
        item["prediction"] = item["candidates"][np.argmax(item["mbr_scores_embedding"])]
        return item

    # Use tqdm to track the mapping process
    dataset = dataset.map(
        calculate_scores,
        with_indices=True,
        desc="Calculating MBR scores"
    )
    dataset.save_to_disk(args.score_dataset_output_path)

    return dataset


def evaluate_and_save(args, dataset, accelerator):
    """Evaluate predictions and save results."""
    accelerator.print("Evaluating predictions and saving results...")

    def evaluate_item(item):
        sorted_indices = np.argsort(item["mbr_scores_embedding"])
        item["chosen"] = item["prediction"]
        # item["rejected"] = item["predictions"][random.choice(sorted_indices[:len(sorted_indices) // 2])]
        item["rejected"] = item["candidates"][random.choice(sorted_indices[:len(sorted_indices) // 2])]
        return item

    # Use tqdm to track the evaluation process
    dataset = dataset.map(
        evaluate_item,
        desc="Evaluating items"
    )
    
    # Generate final dataset with chosen and rejected predictions
    def format_final_item(item):
        return {
            "id": item["id"],
            "image": item["image"],
            "conversations": item["conversations"],
            "prompt": item["conversations"][0]["value"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "candidates": item["candidates"]
        }

    final_dataset = dataset.map(
        format_final_item,
        remove_columns=[col for col in dataset.column_names if col not in ["id", "image", "conversations", "prompt", "chosen", "rejected", "mbr_scores_embedding", "candidates"]],
        desc="Formatting final dataset"
    )
    os.makedirs(os.path.dirname(args.dataset_output_path), exist_ok=True)
    # Write the final dataset to a JSON file
    final_dataset = final_dataset.to_list()
    # sim_scores = []
    new_final = []
    final085 = []
    final08 = []
    for item in final_dataset:
        # print("score_embed: ", item["mbr_scores_embedding"])
        if np.max(item["mbr_scores_embedding"]) >= 0.9:
            new_final.append(item)
        if np.max(item["mbr_scores_embedding"]) >= 0.85:
            final085.append(item)
        if np.max(item["mbr_scores_embedding"]) >= 0.8:
            final08.append(item)

    # new_final = [item for item in final_dataset if np.max(item["mbr_scores_embedding"]) >= 0.9]
    # for item in final_dataset:
    #     item_sim = np.max(item["mbr_scores_embedding"])
    #     sim_scores.append(item_sim)
    # print("list sim scores: ", sim_scores)
    # avg_sim = np.mean(sim_scores)
    # final_dataset = [{"avg_sim": np.mean(sim_scores)}] + final_dataset
    # print("avg_sim score: ", avg_sim)
    # accelerator.print(json.dumps(new_final indent=4))
    args.dataset_output_path = args.dataset_output_path.split('.json')[0] + "_len_" + str(len(final_dataset)) + ".json"
    args.filtered09_output_path = args.filtered_output_path.split('.json')[0] + "09_len_" + str(len(new_final)) + ".json"
    args.filtered085_output_path = args.filtered_output_path.split('.json')[0] + "085_len_" + str(len(final085)) + ".json"
    args.filtered08_output_path = args.filtered_output_path.split('.json')[0] + "08_len_" + str(len(final08)) + ".json"
    with open(args.dataset_output_path, "w") as output_file:
        json.dump(final_dataset, output_file, indent=4)
    with open(args.filtered09_output_path, "w") as output_file2:
        json.dump(new_final, output_file2, indent=4)
    with open(args.filtered085_output_path, "w") as output_file3:
        json.dump(final085, output_file3, indent=4)
    with open(args.filtered08_output_path, "w") as output_file4:
        json.dump(final08, output_file4, indent=4)


def extract_answer(text):
    print("text: ", text)
    pattern1 = r"Step 5: .*?\n\n(.*)"
    pattern2 = r"Step 5: .*?\n(.*)"
    pattern3 = r"Step 1: .*?\n\n(.*)"
    pattern4 = r"Step 1: .*?\n(.*)"
    if "Step" in text:
        match1 = re.search(pattern1, text, re.DOTALL)
        match2 = re.search(pattern2, text, re.DOTALL)
        match3 = re.search(pattern3, text, re.DOTALL)
        match4 = re.search(pattern4, text, re.DOTALL)
        if match1:
            content_after_step_5 = match1.group(1).strip()
        elif match2:
            content_after_step_5 = match2.group(1).strip()
        elif match3:
            content_after_step_5 = match3.group(1).strip()
        elif match4:
            content_after_step_5 = match4.group(1).strip()
        else:
            content_after_step_5 = text
        return content_after_step_5
    else:
        return text


def read_json(input_file_path):
    """Reads a JSON file and extracts candidate predictions."""
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        input_data = json.load(input_file)

    # Only process the first 10 items for debugging or testing purposes
    # input_data = input_data[:100]
    candidate_list = []

    # # Regex pattern to match content after "Step 5:"
    # pattern1 = r"Step 5: .*?\n\n(.*)"
    # pattern2 = r"Step 5: .*?\n(.*)"

    for item in input_data:
        item_ = copy.deepcopy(item)
        # print("item: ", item_)  # Optional debug print statement
        # item_["candidates"] = [item for item in item_["candidates"] if item != ""]
        item_["predictions"] = [extract_answer(item) for item in item_["candidates"]]
        # item_["predictions"] = [item.split("Step 5: Organize all observations into a detailed, cohesive description.")[-1].strip() for item in item_["candidates"]]
        for item in item_["predictions"]:
            if "Step" in item:
                print("item preds: ", item_["predictions"])
        candidate_list.append(item_)
    # print("candidatelist: ", candidate_list)

    assert len(candidate_list) == len(input_data)
    return candidate_list

def synthesize(args):
    """Main pipeline for dataset synthesis."""
    accelerator = Accelerator()

    # Load input datasets
    accelerator.print("Loading datasets...")
    sample_dataset = read_json(args.sample_dataset)

    # Convert to Hugging Face Dataset
    sample_dataset = Dataset.from_list(sample_dataset)

    # Calculate MBR scores
    scored_dataset = calculate_mbr_scores(args, sample_dataset, accelerator)

    # Evaluate and save results
    evaluate_and_save(args, scored_dataset, accelerator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dataset", type=str, required=True, help="Path to the sample dataset.")
    parser.add_argument("--score_dataset_output_path", type=str, required=True, help="Path to save MBR-scored dataset.")
    parser.add_argument("--dataset_output_path", type=str, required=True, help="Path to save final dataset.")
    parser.add_argument("--filtered_output_path", type=str, required=True, help="Path to save filtered dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save evaluation metrics.")
    parser.add_argument("--encoder_model_name_or_path", type=str, required=True, help="Path to the sentence embedding model.")
    args = parser.parse_args()

    synthesize(args)