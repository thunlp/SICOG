import os
import json
import random
import copy
import argparse
# Set a fixed random seed for reproducibility
random.seed(2024)

def read_jsonl(input_file_path):
    """
    Reads a JSONL (JSON Lines) file and returns a list of JSON objects.
    """
    data = []
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            data.append(json.loads(line.strip()))
    return data

def split_text_into_parts(content):
    """
    Splits the input text into four parts: SUMMARY, CAPTION, REASONING, and CONCLUSION.
    Returns a dictionary with the corresponding parts.
    """
    parts = {}
    try:
        # Extract <SUMMARY>
        summary_start = content.find("<SUMMARY>") + len("<SUMMARY>")
        summary_end = content.find("</SUMMARY>")
        parts["SUMMARY"] = content[summary_start:summary_end].strip()
        
        # Extract <CAPTION>
        caption_start = content.find("<CAPTION>") + len("<CAPTION>")
        caption_end = content.find("</CAPTION>")
        parts["CAPTION"] = content[caption_start:caption_end].strip()
        
        # Extract <REASONING>
        reasoning_start = content.find("<REASONING>") + len("<REASONING>")
        reasoning_end = content.find("</REASONING>")
        parts["REASONING"] = content[reasoning_start:reasoning_end].strip()
        
        # Extract <CONCLUSION>
        conclusion_start = content.find("<CONCLUSION>") + len("<CONCLUSION>")
        conclusion_end = content.find("</CONCLUSION>")
        parts["CONCLUSION"] = content[conclusion_start:conclusion_end].strip()
    except Exception as e:
        print(f"Error while splitting text: {e}")
        return None

    return parts

def convert_short_template(content):
    """
    Extracts only the CONCLUSION part from the text.
    """
    result = split_text_into_parts(content)
    if result:
        return result.get("CONCLUSION", "")
    print("Error: Failed to extract CONCLUSION.")
    return ""

def convert_long_template(content):
    """
    Converts the text into a step-by-step explanation format.
    """
    result = split_text_into_parts(content)
    if result:
        return (
            f"Step 1: Clarify the task objective.\n{result['SUMMARY']}\n\n"
            f"Step 2: Extract the crucial visual information from the image.\n{result['CAPTION']}\n\n"
            f"Step 3: Generate detailed reasoning to solve the task.\n{result['REASONING']}\n\n"
            f"Step 4: Conclude the task with an answer.\n{result['CONCLUSION']}"
        )
    print("Error: Failed to convert content to long template.")
    return ""

def process_short_data(data):
    """
    Processes the input data:
    - Updates human questions to request step-by-step answers.
    - Converts GPT responses into a step-by-step format.
    - Splits conversations into single-turn interactions.
    """
    final_data = []
    for i, item in enumerate(data):  # Iterate over each item in the dataset
        conv_content = item.get("conversations", [])
        
        # Iterate over each conversation turn using enumerate()
        for j, turn in enumerate(conv_content):
            
            # Convert GPT responses to step-by-step format
            if turn.get("from") == "gpt":
                try:
                    # turn["value"] = convert_long_template(turn["value"])
                    turn["value"] = convert_short_template(turn["value"])
                except Exception as e:
                    print(f"Error processing item {i}, conversation turn {j}: {e}")
        
        # Save the updated conversations back to the item
        item["conversations"] = conv_content
        final_data.append(item)
    
    return final_data

def process_long_data(data):
    """
    Processes the input data:
    - Updates human questions to request step-by-step answers.
    - Converts GPT responses into a step-by-step format.
    - Splits conversations into single-turn interactions.
    """
    final_data = []
    for i, item in enumerate(data):  # Iterate over each item in the dataset
        conv_content = item.get("conversations", [])
        
        # Iterate over each conversation turn using enumerate()
        for j, turn in enumerate(conv_content):
            # Update human questions to request step-by-step reasoning
            ### convert long
            if turn.get("from") == "human":
                question = turn["value"]
                if "Answer the question using a single word or phrase." in question:
                    question = question.replace("Answer the question using a single word or phrase.", "Answer the question step-by-step.")
                else:
                    question += " Answer the question step-by-step."
                turn["value"] = question
            ### convert long
            
            # Convert GPT responses to step-by-step format
            if turn.get("from") == "gpt":
                try:
                    turn["value"] = convert_long_template(turn["value"])
                    # turn["value"] = convert_short_template(turn["value"])
                except Exception as e:
                    print(f"Error processing item {i}, conversation turn {j}: {e}")
        
        # Save the updated conversations back to the item
        item["conversations"] = conv_content
        final_data.append(item)
    
    return final_data

def save_data(data, output_path, sample_size=None):
    """
    Saves the processed data to a JSON file. Allows optional sampling of the data.
    """
    if sample_size and sample_size < len(data):
        data = random.sample(data, sample_size)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path_new = output_path.split(".json")[0] + str(len(data)) + ".json"

    with open(output_path_new, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)

    print(f"Processed data saved to: {output_path_new}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="annotations.json")
    parser.add_argument("--output_cot_dir", type=str,
                        default="output_dir")
    parser.add_argument("--output_short_dir", type=str,
                        default="output_dir")
    args = parser.parse_args()
    print("Loading data...")
    data = read_jsonl(args.input_file)
    # Process data
    print("Processing data...")
    processed_long_data = process_long_data(data)
    processed_short_data = process_short_data(data)

    # Save the final data to a JSON file
    print("Saving data...")
    save_data(processed_long_data, args.output_cot_dir)
    save_data(processed_short_data, args.output_short_dir)
